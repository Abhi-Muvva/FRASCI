from __future__ import annotations

from typing import Optional

import numpy as np

from FRASCI.core.fragment import extract_fragment_integrals
from FRASCI.crossflow.coupling import classify_fragment_spin_orbitals
from FRASCI.crossflow.embedding import dress_fragment_h1, mf_global_energy
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
from FRASCI.crossflow.types import FragmentSpinOrbitalClasses


SPIN_CHANNELS = ("aa", "bb", "ab", "ba")


def summarize_gap_values(gaps: list[float]) -> dict:
    if not gaps:
        return {
            "n_terms": 0,
            "n_positive": 0,
            "n_zero": 0,
            "n_negative": 0,
            "negative_fraction": 0.0,
            "min_gap": None,
            "max_gap": None,
            "median_gap": None,
            "p05_gap": None,
            "p95_gap": None,
        }

    arr = np.asarray(gaps, dtype=float)
    n_negative = int(np.sum(arr < 0.0))
    n_zero = int(np.sum(arr == 0.0))
    n_positive = int(np.sum(arr > 0.0))
    return {
        "n_terms": int(arr.size),
        "n_positive": n_positive,
        "n_zero": n_zero,
        "n_negative": n_negative,
        "negative_fraction": float(n_negative / arr.size),
        "min_gap": float(np.min(arr)),
        "max_gap": float(np.max(arr)),
        "median_gap": float(np.median(arr)),
        "p05_gap": float(np.percentile(arr, 5)),
        "p95_gap": float(np.percentile(arr, 95)),
    }


def _channel_specs(
    spin_I: FragmentSpinOrbitalClasses,
    spin_J: FragmentSpinOrbitalClasses,
) -> list[tuple[str, list[int], list[int], list[int], list[int], bool]]:
    return [
        ("aa", spin_I.occ_alpha, spin_I.virt_alpha, spin_J.occ_alpha, spin_J.virt_alpha, True),
        ("bb", spin_I.occ_beta, spin_I.virt_beta, spin_J.occ_beta, spin_J.virt_beta, True),
        ("ab", spin_I.occ_alpha, spin_I.virt_alpha, spin_J.occ_beta, spin_J.virt_beta, False),
        ("ba", spin_I.occ_beta, spin_I.virt_beta, spin_J.occ_alpha, spin_J.virt_alpha, False),
    ]


def _matrix_element(
    eri_full: np.ndarray,
    channel: str,
    same_spin: bool,
    i: int,
    a: int,
    j: int,
    b: int,
) -> float:
    direct = float(eri_full[a, i, b, j])
    if same_spin:
        return direct - float(eri_full[a, j, b, i])
    return direct


def analyze_denominator_model(
    eri_full: np.ndarray,
    eps_by_frag: dict[int, np.ndarray],
    frag_orbs_list: list[list[int]],
    spin_classes_list: list[FragmentSpinOrbitalClasses],
    coupled_pairs: list[tuple[int, int]],
    *,
    max_examples: int = 20,
) -> dict:
    gaps_all: list[float] = []
    abs_m_all: list[float] = []
    gaps_by_pair: dict[str, list[float]] = {}
    gaps_by_channel: dict[str, list[float]] = {channel: [] for channel in SPIN_CHANNELS}
    gaps_by_pair_channel: dict[str, list[float]] = {}
    negative_examples: list[dict] = []

    for frag_I, frag_J in coupled_pairs:
        pair_key = f"{frag_I}-{frag_J}"
        gaps_by_pair.setdefault(pair_key, [])
        local_I = {orb: idx for idx, orb in enumerate(frag_orbs_list[frag_I])}
        local_J = {orb: idx for idx, orb in enumerate(frag_orbs_list[frag_J])}
        eps_I = eps_by_frag[frag_I]
        eps_J = eps_by_frag[frag_J]

        for channel, occ_I, virt_I, occ_J, virt_J, same_spin in _channel_specs(
            spin_classes_list[frag_I], spin_classes_list[frag_J]
        ):
            pair_channel_key = f"{pair_key}:{channel}"
            gaps_by_pair_channel.setdefault(pair_channel_key, [])
            if not (occ_I and virt_I and occ_J and virt_J):
                continue

            for i in occ_I:
                li = local_I[i]
                for a in virt_I:
                    la = local_I[a]
                    for j in occ_J:
                        lj = local_J[j]
                        for b in virt_J:
                            lb = local_J[b]
                            gap = (float(eps_I[la]) + float(eps_J[lb])) - (
                                float(eps_I[li]) + float(eps_J[lj])
                            )
                            matrix_element = _matrix_element(
                                eri_full, channel, same_spin, i, a, j, b
                            )
                            abs_m = abs(matrix_element)

                            gaps_all.append(gap)
                            abs_m_all.append(abs_m)
                            gaps_by_pair[pair_key].append(gap)
                            gaps_by_channel[channel].append(gap)
                            gaps_by_pair_channel[pair_channel_key].append(gap)

                            if gap <= 0.0:
                                negative_examples.append(
                                    {
                                        "gap": float(gap),
                                        "pair": [frag_I, frag_J],
                                        "channel": channel,
                                        "i": int(i),
                                        "a": int(a),
                                        "j": int(j),
                                        "b": int(b),
                                        "local_i": int(li),
                                        "local_a": int(la),
                                        "local_j": int(lj),
                                        "local_b": int(lb),
                                        "M": float(matrix_element),
                                        "abs_M": float(abs_m),
                                    }
                                )

    negative_examples.sort(key=lambda item: (item["gap"], -item["abs_M"]))
    max_abs_m = max(abs_m_all) if abs_m_all else 0.0
    return {
        "overall": {
            **summarize_gap_values(gaps_all),
            "max_abs_M": float(max_abs_m),
        },
        "by_pair": {
            key: summarize_gap_values(values)
            for key, values in sorted(gaps_by_pair.items())
        },
        "by_channel": {
            key: summarize_gap_values(values)
            for key, values in sorted(gaps_by_channel.items())
        },
        "by_pair_channel": {
            key: summarize_gap_values(values)
            for key, values in sorted(gaps_by_pair_channel.items())
        },
        "worst_nonpositive_examples": negative_examples[:max_examples],
    }


def _build_embedded_fock_diag(
    h1_solver: np.ndarray,
    eri_frag: np.ndarray,
    gamma_frag: np.ndarray,
) -> np.ndarray:
    J = np.einsum("rs,pqrs->pq", gamma_frag, eri_frag)
    K = 0.5 * np.einsum("rs,psrq->pq", gamma_frag, eri_frag)
    return np.diag(h1_solver + J - K)


def run_denominator_diagnostic(
    fcidump_path: str,
    *,
    gamma_path: Optional[str] = None,
    ref_dets_path: Optional[str] = None,
    n_fragments: int = 3,
    partition_strategy: str = "h1diag",
    fragment_orbs_json: Optional[str] = None,
    coupled_pairs_spec: str = "all",
    denominator_models: Optional[list[str]] = None,
    max_examples: int = 20,
) -> dict:
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta = load_fcidump(fcidump_path)
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

    h1_bare_by_frag: dict[int, np.ndarray] = {}
    h1_mfa_by_frag: dict[int, np.ndarray] = {}
    eri_by_frag: dict[int, np.ndarray] = {}
    gamma_by_frag: dict[int, np.ndarray] = {}
    for frag_idx, frag in enumerate(fragments):
        h1_bare, eri_frag = extract_fragment_integrals(h1, eri, frag)
        h1_bare_by_frag[frag_idx] = h1_bare
        h1_mfa_by_frag[frag_idx] = dress_fragment_h1(h1_bare, eri, frag, gamma, n_orb)
        eri_by_frag[frag_idx] = eri_frag
        idx = np.asarray(frag, dtype=np.intp)
        gamma_by_frag[frag_idx] = gamma[np.ix_(idx, idx)]

    E_mf_global, _E_mf_elec, fock_global = mf_global_energy(h1, eri, gamma, E_nuc)

    requested = denominator_models or [
        "h1_mfa",
        "h1_bare",
        "global_fock",
        "embedded_fock",
    ]
    eps_models: dict[str, dict[int, np.ndarray]] = {}
    for model in requested:
        if model == "h1_mfa":
            eps_models[model] = {
                frag_idx: np.diag(h1_mfa_by_frag[frag_idx])
                for frag_idx in range(len(fragments))
            }
        elif model == "h1_bare":
            eps_models[model] = {
                frag_idx: np.diag(h1_bare_by_frag[frag_idx])
                for frag_idx in range(len(fragments))
            }
        elif model == "global_fock":
            eps_models[model] = {
                frag_idx: np.diag(
                    fock_global[np.ix_(np.asarray(frag), np.asarray(frag))]
                )
                for frag_idx, frag in enumerate(fragments)
            }
        elif model == "embedded_fock":
            eps_models[model] = {
                frag_idx: _build_embedded_fock_diag(
                    h1_mfa_by_frag[frag_idx],
                    eri_by_frag[frag_idx],
                    gamma_by_frag[frag_idx],
                )
                for frag_idx in range(len(fragments))
            }
        else:
            raise ValueError(f"Unknown denominator model: {model!r}")

    model_results = {
        model: analyze_denominator_model(
            eri,
            eps_by_frag,
            fragments,
            spin_classes,
            coupled_pairs,
            max_examples=max_examples,
        )
        for model, eps_by_frag in eps_models.items()
    }

    return {
        "status": "SUCCESS",
        "n_orb": n_orb,
        "n_elec": n_elec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "E_nuc": float(E_nuc),
        "E_mf_global": float(E_mf_global),
        "gamma_source_mode": gamma_result.gamma_source_mode,
        "gamma_load_mode": gamma_result.gamma_load_mode,
        "ref_det_source_mode": ref_det_mode,
        "partition_strategy": partition_strategy,
        "fragment_orbs": fragments,
        "coupled_pairs": coupled_pairs,
        "denominator_models": requested,
        "model_results": model_results,
    }
