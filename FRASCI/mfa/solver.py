"""
solver.py — MFA-TrimCI embedding solver (Phase D).
Functions added in Tasks 2-4: load_gamma_mixed, load_ref_det,
make_nonoverlapping_partition, dress_fragment_h1_mfa, run_mfa_d1, run_mfa_d2
"""
from __future__ import annotations
import os, json, subprocess
from datetime import datetime
from typing import Optional
import numpy as np
import trimci
from FRASCI.core.fragment import (
    fragment_by_sliding_window, extract_fragment_integrals, fragment_electron_count,
)
from FRASCI.core.trimci_adapter import solve_fragment_trimci
from FRASCI.crossflow.partition import load_fragment_orbs_json
from FRASCI.mfa.energy import (
    build_fock, mf_global_energy, mf_rowpartition_energy,
    mf_embedded_energy, correlation_total_energy,
)


def load_gamma_mixed(
    gamma_path: str,
    expected_n_orb: int,
    allow_diagonal_vector: bool = False,
) -> np.ndarray:
    """
    Load gamma_mixed_final.npy and validate shape.

    Raises
    ------
    FileNotFoundError : gamma_path does not exist (message must contain "gamma_mixed not found")
    ValueError        : shape is incompatible (message must contain "gamma_mixed shape")
    """
    if not os.path.exists(gamma_path):
        raise FileNotFoundError(f"gamma_mixed not found: {gamma_path}")
    gamma = np.load(gamma_path)
    if allow_diagonal_vector and gamma.shape == (expected_n_orb,):
        return np.diag(gamma)
    if gamma.shape != (expected_n_orb, expected_n_orb):
        raise ValueError(
            f"gamma_mixed shape {gamma.shape} != expected "
            f"({expected_n_orb}, {expected_n_orb})"
        )
    return gamma


def _gamma_load_mode(gamma_path: str, expected_n_orb: int) -> str:
    shape = np.load(gamma_path, mmap_mode="r").shape
    if shape == (expected_n_orb,):
        return "diagonal_vector_promoted_to_matrix"
    if shape == (expected_n_orb, expected_n_orb):
        return "full_matrix"
    return f"unexpected_shape_{shape}"


def load_ref_det(ref_dets_path: str, row: int = 0) -> tuple[int, int]:
    """
    Load reference determinant bitstrings from dets.npz.

    data["dets"] has shape (n_dets, 2) uint64.
    Column 0 = alpha bitstring, column 1 = beta bitstring.
    row=0 is the correlated ground-state determinant.

    Returns (ref_alpha_bits, ref_beta_bits) as Python ints.
    """
    data = np.load(ref_dets_path)
    dets = data["dets"]
    return int(dets[row, 0]), int(dets[row, 1])


def make_nonoverlapping_partition(h1: np.ndarray, n_orb: int) -> list[list[int]]:
    """
    Three non-overlapping groups of n_orb//3 orbitals, ordered by h1 diagonal.

    Asserts n_orb % 3 == 0. Same orbital ordering as DMET 1-shot.
    """
    assert n_orb % 3 == 0, f"n_orb={n_orb} not divisible by 3"
    order = np.argsort(np.diag(h1))
    n_frag = n_orb // 3
    return [
        sorted(order[0          : n_frag  ].tolist()),
        sorted(order[n_frag     : 2*n_frag].tolist()),
        sorted(order[2*n_frag   : n_orb   ].tolist()),
    ]


def make_balanced_nonoverlapping_partition(
    h1: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
    n_orb: int,
    n_fragments: int = 3,
) -> list[list[int]]:
    """
    Build non-overlapping fragments balanced by reference occupation class.

    Orbitals are classified by the reference determinant as doubly occupied,
    alpha-only, beta-only, or virtual. Each class is sorted by h1 diagonal, then
    the concatenated list is distributed round-robin across fragments.
    """
    if n_orb % n_fragments != 0:
        raise ValueError(f"n_orb={n_orb} not divisible by n_fragments={n_fragments}")

    classes = {
        "docc": [],
        "socc_alpha": [],
        "socc_beta": [],
        "virt": [],
    }
    for orb in range(n_orb):
        alpha_occ = bool((ref_alpha_bits >> orb) & 1)
        beta_occ = bool((ref_beta_bits >> orb) & 1)
        if alpha_occ and beta_occ:
            classes["docc"].append(orb)
        elif alpha_occ:
            classes["socc_alpha"].append(orb)
        elif beta_occ:
            classes["socc_beta"].append(orb)
        else:
            classes["virt"].append(orb)

    h1_diag = np.diag(h1)
    ordered = []
    for name in ("docc", "socc_alpha", "socc_beta", "virt"):
        ordered.extend(sorted(classes[name], key=lambda orb: (h1_diag[orb], orb)))

    fragments = [sorted(ordered[i::n_fragments]) for i in range(n_fragments)]
    expected_size = n_orb // n_fragments
    if any(len(frag) != expected_size for frag in fragments):
        raise RuntimeError(f"Balanced partition did not make {expected_size}-orb fragments")
    return fragments


def dress_fragment_h1_mfa(
    h1_bare_frag: np.ndarray,
    eri_full: np.ndarray,
    frag_orbs: list[int],
    gamma_mixed: np.ndarray,
    n_orb: int,
) -> np.ndarray:
    """
    Add mean-field environment dressing to bare fragment h1.

    v_ext[p,q] = Σ_{(r,s) not both in frag_orbs} gamma_mixed[r,s]
                 * (eri_full[p,q,r,s] - 0.5 * eri_full[p,s,r,q])

    Covers env-env AND cross-block terms (one index in frag, one outside).
    Intra-frag pairs are excluded — eri_frag handles them exactly.

    Parameters
    ----------
    h1_bare_frag : (n_frag, n_frag)
    eri_full     : (n_orb,)*4, chemist notation (pq|rs)
    frag_orbs    : full-system orbital indices for this fragment
    gamma_mixed  : (n_orb, n_orb) Phase B spin-summed density
    n_orb        : total number of orbitals

    Returns
    -------
    h1_dressed : (n_frag, n_frag), symmetric
    """
    fa       = np.array(frag_orbs, dtype=np.intp)
    all_orbs = np.arange(n_orb)

    gamma_ext = gamma_mixed.copy()
    gamma_ext[np.ix_(fa, fa)] = 0.0

    J_frag = np.einsum(
        "rs,pqrs->pq", gamma_ext,
        eri_full[np.ix_(fa, fa, all_orbs, all_orbs)],
    )

    eri_psrq = eri_full.transpose(0, 3, 2, 1)
    K_frag = 0.5 * np.einsum(
        "rs,pqrs->pq", gamma_ext,
        eri_psrq[np.ix_(fa, fa, all_orbs, all_orbs)],
    )

    return h1_bare_frag + J_frag - K_frag


def _get_git_commit() -> Optional[str]:
    """Return short git commit hash, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _write_summary_d1(results: dict, output_dir: str) -> None:
    lines = [
        "# MFA-TrimCI D1 Summary",
        "",
        f"**Status:** {results['status']}",
        f"**Timestamp:** {results['timestamp']}",
        f"**Runtime:** {results['runtime_sec']:.1f} s",
        "",
        "## Determinant counts",
        f"| Fragment | MFA-D1 | Phase C baseline |",
        "| --- | --- | --- |",
    ]
    for i, (nd, bc) in enumerate(
        zip(results["fragment_n_dets"], results["phase_c_baseline_fragment_n_dets"])
    ):
        lines.append(f"| {i} | {nd} | {bc} |")
    lines += [
        f"| **Total** | **{results['total_dets']}** | **{results['phase_c_baseline_total_dets']}** |",
        "",
        f"**Matches Phase C baseline:** {results['matches_phase_c_baseline']}",
        f"**delta_dets_vs_phase_c:** {results['delta_dets_vs_phase_c']:+d}",
        f"**det_fraction_vs_bruteforce:** {results['det_fraction_vs_bruteforce']:.4f} "
        f"({results['det_reduction_vs_bruteforce']:.2f}% reduction)",
        "",
        "> **No total energy is reported for D1 because fragments overlap.**",
        "> The determinant count is the headline metric.",
        "> Compare against Phase C/B baseline [51, 51, 16] / 118 dets.",
    ]
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_mfa_d1(
    fcidump_path: str,
    gamma_path: str,
    ref_dets_path: str,
    trimci_config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    D1 benchmark: overlapping W=15 S=10 fragments with MFA dressing.

    Claims determinant counts only. No total energy (overlapping fragments
    cannot be cleanly energy-partitioned).

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file
    gamma_path    : path to gamma_mixed_final.npy (Phase B converged density)
    ref_dets_path : path to dets.npz (row 0 = correlated reference det)
    trimci_config : optional TrimCI overrides (threshold, max_final_dets, ...)
    output_dir    : if provided, writes results.json + summary.md here

    Returns
    -------
    results dict matching D1 results.json schema from the spec
    """
    import warnings
    t0 = datetime.now()

    # ── 1. Read FCIDUMP ──────────────────────────────────────────────────────
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    print(f"[MFA-D1] n_orb={n_orb}, n_elec={n_elec}, E_nuc={E_nuc}")
    if abs(E_nuc) > 1e-10:
        warnings.warn(f"[MFA-D1] E_nuc={E_nuc:.6e} (non-zero; included once in any global energy)")

    # ── 2. Load gamma_mixed ──────────────────────────────────────────────────
    gamma_load_mode = _gamma_load_mode(gamma_path, n_orb)
    gamma_mixed = load_gamma_mixed(gamma_path, n_orb, allow_diagonal_vector=True)
    print(f"[MFA-D1] gamma_mixed: shape={gamma_mixed.shape}, Tr={np.trace(gamma_mixed):.4f}")
    if gamma_load_mode != "full_matrix":
        print(f"[MFA-D1 WARNING] gamma_source loaded as {gamma_load_mode}")

    # ── 3. Load reference determinant ────────────────────────────────────────
    ref_alpha_bits, ref_beta_bits = load_ref_det(ref_dets_path, row=0)

    # ── 4. Fragment partition: overlapping W=15, S=10 ────────────────────────
    order = np.argsort(np.diag(h1))
    fragments = fragment_by_sliding_window(n_orb, order, window_size=15, stride=10)
    print(f"[MFA-D1] {len(fragments)} overlapping fragments (W=15, S=10)")

    # ── 5. Per-fragment solve ────────────────────────────────────────────────
    fragment_n_dets       = []
    fragment_n_alpha      = []
    fragment_n_beta       = []
    per_fragment_E_trimci = []

    for frag_idx, frag_orbs in enumerate(fragments):
        h1_bare, eri_frag = extract_fragment_integrals(h1, eri, frag_orbs)
        h1_dressed = dress_fragment_h1_mfa(h1_bare, eri, frag_orbs, gamma_mixed, n_orb)
        n_alpha_I, n_beta_I = fragment_electron_count(
            ref_alpha_bits, ref_beta_bits, frag_orbs)

        result_I = solve_fragment_trimci(
            h1_dressed, eri_frag,
            n_alpha_I, n_beta_I,
            n_orb_frag=len(frag_orbs),
            config=trimci_config,
        )
        fragment_n_dets.append(int(result_I.n_dets))
        fragment_n_alpha.append(int(n_alpha_I))
        fragment_n_beta.append(int(n_beta_I))
        per_fragment_E_trimci.append(float(result_I.energy))
        print(f"  [MFA-D1] frag {frag_idx}: orbs={len(frag_orbs)}, "
              f"n_dets={result_I.n_dets}, E={result_I.energy:.6f} Ha")

    # ── 6. Build results ─────────────────────────────────────────────────────
    total_dets = sum(fragment_n_dets)
    phase_c_baseline_frag = [51, 51, 16]
    phase_c_total = 118

    runtime_sec = (datetime.now() - t0).total_seconds()
    results = {
        "status": "SUCCESS",
        "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": runtime_sec,
        "method": "MFA-TrimCI",
        "phase": "D1",
        "partition": "overlapping_W15_S10",
        "gamma_source": str(gamma_path),
        "gamma_shape": list(gamma_mixed.shape),
        "gamma_load_mode": gamma_load_mode,
        "ref_dets_source": str(ref_dets_path),
        "ref_dets_row": 0,
        "trimci_config": trimci_config or {},
        "git_commit": _get_git_commit(),
        "E_nuc": float(E_nuc),
        "E_total": None,
        "energy_status": "diagnostic_only_overlapping_no_total",
        "electron_count_status": "overlapping_fragments_not_summed",
        "fragment_n_dets": fragment_n_dets,
        "total_dets": total_dets,
        "phase_c_baseline_fragment_n_dets": phase_c_baseline_frag,
        "phase_c_baseline_total_dets": phase_c_total,
        "matches_phase_c_baseline": (fragment_n_dets == phase_c_baseline_frag),
        "delta_dets_vs_phase_c": total_dets - phase_c_total,
        "det_reduction_vs_bruteforce": round((1 - total_dets / 10095) * 100, 2),
        "det_fraction_vs_bruteforce": round(total_dets / 10095, 4),
        "brute_force_dets": 10095,
        "per_fragment_E_trimci": per_fragment_E_trimci,
        "fragment_orbs": fragments,
        "fragment_n_alpha": fragment_n_alpha,
        "fragment_n_beta": fragment_n_beta,
    }

    # ── 7. Write outputs ─────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        _write_summary_d1(results, output_dir)
        print(f"[MFA-D1] Output written to {output_dir}")

    return results


def _write_summary_d2(results: dict, output_dir: str) -> None:
    lines = [
        "# MFA-TrimCI D2 Summary",
        "",
        f"**Status:** {results['status']}",
        f"**Timestamp:** {results['timestamp']}",
        f"**Runtime:** {results['runtime_sec']:.1f} s",
        "",
        "## Energy",
        "| Quantity | Value (Ha) |",
        "| --- | --- |",
        f"| E_mf_global | {results['E_mf_global']:.6f} |",
        f"| **E_total** | **{results['E_total']:.6f}** |",
        f"| Error vs reference (-327.1920) | {results['error_vs_reference']:+.4f} |",
        f"| E_DMET_1shot_B (-258.957062) | {results['E_total_minus_dmet_1shot_b']:+.4f} difference |",
        "",
        "## Determinant counts",
        "| Fragment | n_dets |",
        "| --- | --- |",
    ]
    for i, nd in enumerate(results["fragment_n_dets"]):
        lines.append(f"| {i} | {nd} |")
    lines += [
        f"| **Total** | **{results['total_dets']}** |",
        "",
        "> **E_total is a non-overlapping mean-field-plus-local-correlation correction.**",
        "> Formula: E_total = E_mf_global + sum_I (E_TrimCI_I - E_mf_emb_I).",
        "> E_nuc is included in E_mf_global and not re-added in fragment corrections.",
    ]
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_mfa_d2(
    fcidump_path: str,
    gamma_path: str,
    ref_dets_path: str,
    trimci_config: Optional[dict] = None,
    output_dir: Optional[str] = None,
    partition: str = "h1diag",
    fragment_orbs_json: Optional[str] = None,
) -> dict:
    """
    D2 benchmark: non-overlapping 12+12+12 fragments with MFA dressing.

    Reports the correlation-corrected total energy:
        E_total = E_mf_global + sum_I (E_TrimCI_I - E_mf_emb_I)

    E_nuc is included once in E_mf_global and is not re-added inside fragment
    corrections.
    """
    import warnings
    t0 = datetime.now()

    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    print(f"[MFA-D2] n_orb={n_orb}, n_elec={n_elec}, E_nuc={E_nuc}")
    if abs(E_nuc) > 1e-10:
        warnings.warn(f"[MFA-D2] E_nuc={E_nuc:.6e} (non-zero; included once in E_mf_global)")

    gamma_load_mode = _gamma_load_mode(gamma_path, n_orb)
    gamma_mixed = load_gamma_mixed(gamma_path, n_orb, allow_diagonal_vector=True)
    print(f"[MFA-D2] gamma_mixed: shape={gamma_mixed.shape}, Tr={np.trace(gamma_mixed):.4f}")
    if gamma_load_mode != "full_matrix":
        print(f"[MFA-D2 WARNING] gamma_source loaded as {gamma_load_mode}")

    ref_alpha_bits, ref_beta_bits = load_ref_det(ref_dets_path, row=0)

    if fragment_orbs_json is not None:
        fragments = load_fragment_orbs_json(
            fragment_orbs_json, n_orb, ref_alpha_bits, ref_beta_bits
        )
        partition_label = "nonoverlapping_explicit_fragment_orbs_json"
        partition_mode = "explicit_json"
    elif partition == "h1diag":
        fragments = make_nonoverlapping_partition(h1, n_orb)
        partition_label = "nonoverlapping_12_12_12_h1diag"
        partition_mode = partition
    elif partition == "balanced":
        fragments = make_balanced_nonoverlapping_partition(
            h1, ref_alpha_bits, ref_beta_bits, n_orb
        )
        partition_label = "nonoverlapping_12_12_12_balanced_refdet_roundrobin"
        partition_mode = partition
    else:
        raise ValueError(f"Unknown D2 partition: {partition!r}")

    print(
        f"[MFA-D2] {len(fragments)} non-overlapping fragments "
        f"(partition={partition_mode})"
    )

    E_mf_global, E_mf_global_elec, F_full = mf_global_energy(h1, eri, gamma_mixed, E_nuc)
    print(f"[MFA-D2] E_mf_global = {E_mf_global:.6f} Ha (E_nuc={E_nuc})")

    rowpart = [mf_rowpartition_energy(h1, F_full, gamma_mixed, f, n_orb) for f in fragments]
    rowpart_sum = sum(rowpart)
    rowpart_ok = bool(np.isclose(rowpart_sum, E_mf_global_elec, atol=1e-10))
    print(
        f"[MFA-D2] Row-partition sum={rowpart_sum:.6f}, "
        f"E_mf_global_elec={E_mf_global_elec:.6f}, match={rowpart_ok}"
    )

    fragment_n_dets = []
    fragment_n_alpha = []
    fragment_n_beta = []
    per_fragment_E_trimci = []
    per_fragment_E_mf_emb = []

    for frag_idx, frag_orbs in enumerate(fragments):
        h1_bare, eri_frag = extract_fragment_integrals(h1, eri, frag_orbs)
        h1_dressed = dress_fragment_h1_mfa(h1_bare, eri, frag_orbs, gamma_mixed, n_orb)
        n_alpha_I, n_beta_I = fragment_electron_count(
            ref_alpha_bits, ref_beta_bits, frag_orbs
        )

        result_I = solve_fragment_trimci(
            h1_dressed,
            eri_frag,
            n_alpha_I,
            n_beta_I,
            n_orb_frag=len(frag_orbs),
            config=trimci_config,
        )

        frag_idx_array = np.array(frag_orbs)
        gamma_frag_I = gamma_mixed[np.ix_(frag_idx_array, frag_idx_array)]
        E_mf_emb_I = mf_embedded_energy(h1_dressed, eri_frag, gamma_frag_I)

        fragment_n_dets.append(int(result_I.n_dets))
        fragment_n_alpha.append(int(n_alpha_I))
        fragment_n_beta.append(int(n_beta_I))
        per_fragment_E_trimci.append(float(result_I.energy))
        per_fragment_E_mf_emb.append(float(E_mf_emb_I))
        print(
            f"  [MFA-D2] frag {frag_idx}: n_dets={result_I.n_dets}, "
            f"E_TrimCI={result_I.energy:.6f}, E_mf_emb={E_mf_emb_I:.6f}, "
            f"E_corr={result_I.energy - E_mf_emb_I:.6f} Ha"
        )

    sum_alpha = sum(fragment_n_alpha)
    sum_beta = sum(fragment_n_beta)
    elec_check_passed = (sum_alpha == n_alpha and sum_beta == n_beta)
    if not elec_check_passed:
        raise RuntimeError(
            f"Electron count mismatch: sum_alpha={sum_alpha} (expected {n_alpha}), "
            f"sum_beta={sum_beta} (expected {n_beta})"
        )

    E_total, E_corr_list = correlation_total_energy(
        E_mf_global, per_fragment_E_trimci, per_fragment_E_mf_emb
    )
    ref_energy = -327.1920
    error_vs_ref = E_total - ref_energy
    E_dmet_1shot_b = -258.957062

    print(f"\n{'=' * 60}")
    print(f"=== MFA-TrimCI D2 (non-overlapping, {partition_mode}) ===")
    print(f"  E_mf_global              = {E_mf_global:.6f} Ha")
    print(f"  E_total (corr-corrected) = {E_total:.6f} Ha  <- canonical")
    print(f"  Reference (brute-force)  = {ref_energy} Ha")
    print(f"  Error                    = {error_vs_ref:+.4f} Ha")
    print(f"  vs DMET 1-shot B         = {E_total - E_dmet_1shot_b:+.4f} Ha")
    print(f"  Total dets: {sum(fragment_n_dets)}  (Phase C: 118, brute-force: 10095)")
    print(f"{'=' * 60}")

    runtime_sec = (datetime.now() - t0).total_seconds()
    results = {
        "status": "SUCCESS",
        "timestamp": t0.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": runtime_sec,
        "method": "MFA-TrimCI",
        "phase": "D2",
        "partition": partition_label,
        "partition_mode": partition_mode,
        "fragment_orbs_json": str(fragment_orbs_json) if fragment_orbs_json else None,
        "gamma_source": str(gamma_path),
        "gamma_shape": list(gamma_mixed.shape),
        "gamma_load_mode": gamma_load_mode,
        "ref_dets_source": str(ref_dets_path),
        "ref_dets_row": 0,
        "trimci_config": trimci_config or {},
        "git_commit": _get_git_commit(),
        "E_nuc": float(E_nuc),
        "E_nuc_included": True,
        "energy_status": "correlation_corrected_nonoverlapping_total",
        "E_mf_global_elec": float(E_mf_global_elec),
        "E_mf_global": float(E_mf_global),
        "E_total": float(E_total),
        "error_vs_reference": float(error_vs_ref),
        "reference_energy": ref_energy,
        "E_dmet_1shot_b": E_dmet_1shot_b,
        "E_total_minus_dmet_1shot_b": float(E_total - E_dmet_1shot_b),
        "electron_count_check": {
            "sum_n_alpha": int(sum_alpha),
            "sum_n_beta": int(sum_beta),
            "passed": elec_check_passed,
        },
        "fragment_n_dets": fragment_n_dets,
        "total_dets": sum(fragment_n_dets),
        "brute_force_dets": 10095,
        "fragment_orbs": fragments,
        "fragment_n_alpha": fragment_n_alpha,
        "fragment_n_beta": fragment_n_beta,
        "per_fragment_E_trimci": per_fragment_E_trimci,
        "per_fragment_E_mf_embedded": per_fragment_E_mf_emb,
        "per_fragment_E_corr": E_corr_list,
        "E_mf_row_partition_by_frag_elec": rowpart,
        "E_mf_row_partition_sum_elec": float(rowpart_sum),
        "E_mf_row_partition_sum_matches_global_elec": rowpart_ok,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        _write_summary_d2(results, output_dir)
        print(f"[MFA-D2] Output written to {output_dir}")

    return results
