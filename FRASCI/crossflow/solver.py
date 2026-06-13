from __future__ import annotations

from typing import Optional

import numpy as np

from FRASCI.core.fragment import extract_fragment_integrals, fragment_electron_count
from FRASCI.core.trimci_adapter import solve_fragment_trimci
from FRASCI.crossflow.coupling import (
    apply_damped_delta_h,
    classify_fragment_spin_orbitals,
    compute_all_pair_coupling,
)
from FRASCI.crossflow.embedding import (
    apply_delta_h,
    correlation_total_energy,
    dress_fragment_h1,
    mf_embedded_energy,
    mf_global_energy,
)
from FRASCI.crossflow.io import (
    load_fcidump,
    load_or_compute_gamma,
    load_or_derive_ref_det,
)
from FRASCI.crossflow.partition import (
    load_fragment_orbs_json,
    make_equal_nonoverlapping_partition,
    parse_coupled_pairs,
    validate_fragment_partition,
)
from FRASCI.crossflow.types import CrossflowIteration, CrossflowResult


def _validate_solver_inputs(
    max_iters: int,
    n_fragments: int,
    damping: float,
    energy_conv_threshold: float,
    partition_strategy: str,
) -> None:
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    if n_fragments < 1:
        raise ValueError(f"n_fragments must be >= 1, got {n_fragments}")
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    if energy_conv_threshold <= 0.0:
        raise ValueError(
            f"energy_conv_threshold must be > 0, got {energy_conv_threshold}"
        )
    if partition_strategy not in {"h1diag", "balanced"}:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy!r}")


def run_cross_coupled_solver(
    fcidump_path: str,
    *,
    gamma_path: Optional[str] = None,
    ref_dets_path: Optional[str] = None,
    n_fragments: int = 3,
    partition_strategy: str = "h1diag",
    fragment_orbs_json: Optional[str] = None,
    coupled_pairs_spec: str = "all",
    trimci_config: Optional[dict] = None,
    max_iters: int = 1,
    energy_conv_threshold: float = 1e-4,
    damping: float = 1.0,
    reference_energy: Optional[float] = None,
    brute_force_dets: Optional[int] = None,
) -> CrossflowResult:
    """Run the cross-fragment PT2/self-energy coupled TrimCI solver."""
    _validate_solver_inputs(
        max_iters=max_iters,
        n_fragments=n_fragments,
        damping=damping,
        energy_conv_threshold=energy_conv_threshold,
        partition_strategy=partition_strategy,
    )

    h1, eri, _n_elec, n_orb, E_nuc, n_alpha, n_beta = load_fcidump(fcidump_path)
    gamma_result = load_or_compute_gamma(gamma_path, h1, eri, n_alpha, n_beta, n_orb)
    gamma = gamma_result.gamma

    ref_alpha_bits, ref_beta_bits, ref_det_mode = load_or_derive_ref_det(
        ref_dets_path,
        h1,
        eri,
        n_alpha,
        n_beta,
        n_orb,
        uhf_cache=gamma_result.uhf_cache,
    )

    if fragment_orbs_json is None:
        fragments = make_equal_nonoverlapping_partition(
            h1,
            n_orb,
            n_fragments,
            strategy=partition_strategy,
            ref_alpha_bits=ref_alpha_bits,
            ref_beta_bits=ref_beta_bits,
        )
        validate_fragment_partition(fragments, n_orb, ref_alpha_bits, ref_beta_bits)
    else:
        fragments = load_fragment_orbs_json(
            fragment_orbs_json,
            n_orb,
            ref_alpha_bits,
            ref_beta_bits,
        )

    coupled_pairs = parse_coupled_pairs(coupled_pairs_spec, len(fragments))
    spin_classes = [
        classify_fragment_spin_orbitals(ref_alpha_bits, ref_beta_bits, frag)
        for frag in fragments
    ]

    h1_mfa_dressed: list[np.ndarray] = []
    eri_fragments: list[np.ndarray] = []
    gamma_fragments: list[np.ndarray] = []
    n_alpha_by_frag: list[int] = []
    n_beta_by_frag: list[int] = []

    for frag in fragments:
        h1_bare_frag, eri_frag = extract_fragment_integrals(h1, eri, frag)
        h1_mfa_dressed.append(dress_fragment_h1(h1_bare_frag, eri, frag, gamma, n_orb))
        eri_fragments.append(eri_frag)

        frag_idx = np.array(frag, dtype=np.intp)
        gamma_fragments.append(gamma[np.ix_(frag_idx, frag_idx)])

        n_alpha_frag, n_beta_frag = fragment_electron_count(
            ref_alpha_bits,
            ref_beta_bits,
            frag,
        )
        n_alpha_by_frag.append(n_alpha_frag)
        n_beta_by_frag.append(n_beta_frag)

    E_mf_global, _E_mf_elec, _fock = mf_global_energy(h1, eri, gamma, E_nuc)
    trimci_config_effective = dict(trimci_config or {})
    delta_h_prev = [np.zeros(len(frag), dtype=float) for frag in fragments]
    iteration_history: list[CrossflowIteration] = []

    E_trimci, n_dets, E_mf_emb = _solve_all_fragments(
        h1_solver_by_frag=h1_mfa_dressed,
        eri_by_frag=eri_fragments,
        gamma_by_frag=gamma_fragments,
        n_alpha_by_frag=n_alpha_by_frag,
        n_beta_by_frag=n_beta_by_frag,
        trimci_config=trimci_config_effective,
    )
    E_total_prev, _E_corr = correlation_total_energy(E_mf_global, E_trimci, E_mf_emb)

    iteration_history.append(
        CrossflowIteration(
            iter=0,
            E_total=E_total_prev,
            delta_E=None,
            fragment_n_dets=list(n_dets),
            E_pt2_cross=None,
            E_total_postprocessed=None,
            max_abs_delta_h=None,
            min_gap=None,
            converged=False,
        )
    )

    baseline_n_dets = list(n_dets)
    final_coupling = None
    converged = False

    failure_diagnostics = None
    for coupling_round in range(1, max_iters + 1):
        h1_solver_prev = {
            frag_idx: apply_delta_h(h1_mfa_dressed[frag_idx], delta_h_prev[frag_idx])
            for frag_idx in range(len(fragments))
        }
        try:
            coupling = compute_all_pair_coupling(
                eri,
                h1_solver_prev,
                fragments,
                spin_classes,
                coupled_pairs,
            )
        except ValueError as exc:
            failure_diagnostics = {
                "failure_stage": "pt2_pair_coupling",
                "error": str(exc),
                "coupling_round": coupling_round,
                "n_pairs": len(coupled_pairs),
            }
            break
        final_coupling = coupling

        delta_h_applied = [
            apply_damped_delta_h(
                coupling.delta_h_by_frag[frag_idx],
                delta_h_prev[frag_idx],
                damping,
            )
            for frag_idx in range(len(fragments))
        ]
        h1_solver_cur = [
            apply_delta_h(h1_mfa_dressed[frag_idx], delta_h_applied[frag_idx])
            for frag_idx in range(len(fragments))
        ]

        E_trimci, n_dets, E_mf_emb = _solve_all_fragments(
            h1_solver_by_frag=h1_solver_cur,
            eri_by_frag=eri_fragments,
            gamma_by_frag=gamma_fragments,
            n_alpha_by_frag=n_alpha_by_frag,
            n_beta_by_frag=n_beta_by_frag,
            trimci_config=trimci_config_effective,
        )
        E_total_cur, _E_corr = correlation_total_energy(
            E_mf_global,
            E_trimci,
            E_mf_emb,
        )
        delta_E = E_total_cur - E_total_prev
        converged = abs(delta_E) < energy_conv_threshold

        iteration_history.append(
            CrossflowIteration(
                iter=coupling_round,
                E_total=E_total_cur,
                delta_E=delta_E,
                fragment_n_dets=list(n_dets),
                E_pt2_cross=coupling.E_pt2_cross_total,
                E_total_postprocessed=E_total_prev + coupling.E_pt2_cross_total,
                max_abs_delta_h=coupling.max_abs_delta_h,
                min_gap=coupling.min_gap_global,
                converged=converged,
            )
        )

        delta_h_prev = delta_h_applied
        E_total_prev = E_total_cur
        if converged:
            break

    final_iter = iteration_history[-1]
    if failure_diagnostics is not None:
        status = "FAILED"
    elif max_iters == 1:
        status = "SUCCESS_ONE_SHOT"
    elif converged:
        status = "CONVERGED"
    else:
        status = "MAX_ITERS_REACHED"

    coupling_diagnostics = None
    if failure_diagnostics is not None:
        coupling_diagnostics = failure_diagnostics
    elif final_coupling is not None:
        coupling_diagnostics = {
            "n_pairs": len(coupled_pairs),
            "n_terms_total": final_coupling.n_terms_total,
            "min_gap_global": final_coupling.min_gap_global,
            "max_abs_M_global": final_coupling.max_abs_M_global,
            "max_abs_delta_h": final_coupling.max_abs_delta_h,
        }

    error_vs_reference = (
        final_iter.E_total - reference_energy if reference_energy is not None else None
    )
    det_fraction = (
        sum(final_iter.fragment_n_dets) / brute_force_dets
        if brute_force_dets
        else None
    )

    return CrossflowResult(
        status=status,
        E_total_baseline=iteration_history[0].E_total,
        E_total_final=final_iter.E_total,
        E_total_postprocessed_final_round=final_iter.E_total_postprocessed,
        delta_E_final_vs_baseline=final_iter.E_total - iteration_history[0].E_total,
        E_pt2_cross_final=final_iter.E_pt2_cross,
        max_abs_delta_h_final=final_iter.max_abs_delta_h,
        min_gap_final=final_iter.min_gap,
        fragment_n_dets_baseline=baseline_n_dets,
        fragment_n_dets_final=list(final_iter.fragment_n_dets),
        total_dets_baseline=sum(baseline_n_dets),
        total_dets_final=sum(final_iter.fragment_n_dets),
        reference_energy=reference_energy,
        error_vs_reference_final=error_vs_reference,
        brute_force_dets=brute_force_dets,
        det_fraction_vs_bruteforce=det_fraction,
        coupling_diagnostics_final=coupling_diagnostics,
        iteration_history=iteration_history,
        n_iters_completed=len(iteration_history) - 1,
        converged=converged,
        gamma_source_mode=gamma_result.gamma_source_mode,
        ref_det_source_mode=ref_det_mode,
        gamma_load_mode=gamma_result.gamma_load_mode,
        partition_strategy=partition_strategy,
        fragment_orbs=fragments,
        coupled_pairs=list(coupled_pairs),
        E_mf_global=E_mf_global,
        n_orb=n_orb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        E_nuc=float(E_nuc),
        trimci_config=trimci_config_effective,
        max_iters=max_iters,
        damping=damping,
        energy_conv_threshold=energy_conv_threshold,
    )


def _solve_all_fragments(
    *,
    h1_solver_by_frag: list[np.ndarray],
    eri_by_frag: list[np.ndarray],
    gamma_by_frag: list[np.ndarray],
    n_alpha_by_frag: list[int],
    n_beta_by_frag: list[int],
    trimci_config: dict,
) -> tuple[list[float], list[int], list[float]]:
    E_trimci: list[float] = []
    n_dets: list[int] = []
    E_mf_emb: list[float] = []

    for frag_idx, h1_solver in enumerate(h1_solver_by_frag):
        result = solve_fragment_trimci(
            h1_solver,
            eri_by_frag[frag_idx],
            n_alpha_by_frag[frag_idx],
            n_beta_by_frag[frag_idx],
            n_orb_frag=h1_solver.shape[0],
            config=trimci_config,
        )
        E_trimci.append(result.energy)
        n_dets.append(result.n_dets)
        E_mf_emb.append(
            mf_embedded_energy(
                h1_solver,
                eri_by_frag[frag_idx],
                gamma_by_frag[frag_idx],
            )
        )

    return E_trimci, n_dets, E_mf_emb
