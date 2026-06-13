"""
mfa/extract_full_gamma.py
=========================
Phase D v2: Extract the full 36×36 gamma (1-RDM) from the converged
Phase B self-consistent solution.

Phase B's ``assemble_global_rdm1_diag`` discards all off-diagonal elements
of each fragment 1-RDM, saving only a (36,) orbital-occupation vector.
This script re-runs the Phase B extraction pass — same dressed Hamiltonians,
same TrimCI config — but captures the full (n_frag × n_frag) 1-RDM from
``compute_fragment_rdm1`` and assembles the pieces into a 36×36 matrix.

Coverage note
-------------
With the overlapping W=15, S=10 partition on h1-diag-ordered orbitals:

  Frag 0: orbs 0..14   Frag 1: orbs 10..24   Frag 2: orbs 20..35
  Overlap 0∩1: {10..14}  Overlap 1∩2: {20..24}  No triple overlap.

Pairs (r, s) where r ∈ {0..9} and s ∈ {25..35} are not covered by any
single fragment and remain zero in gamma_full. All within-fragment
off-diagonals (the new information vs Phase B diagonal-only) are captured
and averaged for overlapping orbital pairs.

Usage
-----
From /home/unfunnypanda/Proj_Flow/ with the qflowenv active::

    python frasci/mfa/extract_full_gamma.py [--output-dir PATH]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_DATA               = ROOT / "FRASCI" / "data"
FCIDUMP_DEFAULT     = _DATA / "fcidump_cycle_6"
DETS_NPZ_DEFAULT    = _DATA / "dets.npz"
GAMMA_DIAG_DEFAULT  = (
    ROOT / "FRASCI" / "Outputs" / "mfa"
    / "outs_extract_full_gamma_20260417_002006" / "gamma_mixed_diag.npy"
)
OUTDIR_BASE = ROOT / "FRASCI" / "Outputs" / "mfa"

# Same config as Phase B's extraction step (outs_extraction_autodets)
EXTRACTION_CONFIG = {
    "threshold": 0.06,
    "max_final_dets": "auto",
    "max_rounds": 2,
    "num_runs": 1,
    "pool_build_strategy": "heat_bath",
    "verbose": False,
}


def assemble_global_gamma_full(
    rdm1_list: list[np.ndarray],
    frag_orbs_list: list[list[int]],
    n_orb: int,
) -> np.ndarray:
    """
    Assemble a full n_orb × n_orb 1-RDM from per-fragment 1-RDMs.

    For each pair (r, s) that appears in the same fragment, all contributing
    fragment matrices are averaged. Pairs not covered by any single fragment
    remain zero.

    Parameters
    ----------
    rdm1_list      : list of (n_frag_i, n_frag_i) spin-summed 1-RDM arrays
    frag_orbs_list : list of full-system orbital index lists, parallel to rdm1_list
    n_orb          : total number of orbitals in the full system

    Returns
    -------
    gamma_full : (n_orb, n_orb) float64, symmetric, zeros for uncovered pairs
    """
    gamma_total  = np.zeros((n_orb, n_orb), dtype=np.float64)
    count_matrix = np.zeros((n_orb, n_orb), dtype=np.int64)

    for rdm1, frag_orbs in zip(rdm1_list, frag_orbs_list):
        fa = np.asarray(frag_orbs, dtype=np.intp)
        gamma_total[np.ix_(fa, fa)]  += rdm1
        count_matrix[np.ix_(fa, fa)] += 1

    covered    = count_matrix > 0
    gamma_full = np.zeros((n_orb, n_orb), dtype=np.float64)
    gamma_full[covered] = gamma_total[covered] / count_matrix[covered]
    return gamma_full


def extract_full_gamma(
    fcidump_path: str | None = None,
    gamma_diag_path: str | None = None,
    dets_path: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """
    Run the extraction pass and save the full 36×36 gamma.

    Returns a dict with keys:
    - ``gamma_path`` : absolute path to the saved ``gamma_mixed_full.npy``
    - ``diagnostics``: dict with max |off-diagonal|, trace, symmetry check
    - ``fragments``  : per-fragment metadata
    """
    import trimci
    from FRASCI.core.fragment import (
        extract_fragment_integrals,
        fragment_by_sliding_window,
        fragment_electron_count,
    )
    from FRASCI.core.trimci_adapter import solve_fragment_trimci
    from FRASCI.mfa.helpers import (
        compute_fragment_rdm1,
        dress_integrals_meanfield,
    )
    from FRASCI.mfa.solver import load_ref_det

    fcidump_path     = str(fcidump_path    or FCIDUMP_DEFAULT)
    gamma_diag_path  = str(gamma_diag_path or GAMMA_DIAG_DEFAULT)
    dets_path        = str(dets_path       or DETS_NPZ_DEFAULT)
    timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = str(OUTDIR_BASE / f"outs_extract_full_gamma_{timestamp}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load FCIDUMP ---
    print(f"[extract_full_gamma] FCIDUMP: {fcidump_path}")
    h1, eri, n_elec, n_orb, _e_nuc, _n_alpha, _n_beta, _psym = trimci.read_fcidump(fcidump_path)
    h1  = np.asarray(h1,  dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)

    # --- Load converged diagonal gamma (Phase B output) ---
    gamma_diag = np.load(gamma_diag_path)
    if gamma_diag.ndim != 1 or gamma_diag.shape[0] != n_orb:
        raise ValueError(
            f"Expected 1D gamma of length {n_orb}, got shape {gamma_diag.shape}"
        )
    print(f"[extract_full_gamma] gamma_diag sum={gamma_diag.sum():.4f}  (n_elec={n_elec})")

    # --- Reference determinant ---
    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_path, row=0)

    # --- Build overlapping partition — identical to Phase B ---
    order     = np.argsort(np.diag(h1))
    fragments = [
        fo for fo in fragment_by_sliding_window(n_orb, order, 15, 10)
        if (lambda na, nb: na > 0 and nb > 0 and na <= len(fo) and nb <= len(fo))(
            *fragment_electron_count(ref_alpha_bits, ref_beta_bits, fo)
        )
    ]
    print(f"[extract_full_gamma] {len(fragments)} valid fragments")
    for i, fo in enumerate(fragments):
        print(f"  F{i}: orbs [{fo[0]}..{fo[-1]}]  ({len(fo)} orbs)")

    # --- Extraction pass: solve and capture full 1-RDMs ---
    rdm1_list    = []
    fragment_info = []
    for idx, frag_orbs in enumerate(fragments):
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
        ext_orbs  = [r for r in range(n_orb) if r not in set(frag_orbs)]
        ext_gamma = gamma_diag[np.asarray(ext_orbs, dtype=np.intp)]
        h1_use    = dress_integrals_meanfield(h1_f, eri, frag_orbs, ext_gamma, ext_orbs)
        res       = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), EXTRACTION_CONFIG)
        rdm1      = compute_fragment_rdm1(res.dets, res.coeffs, res.n_orb_frag)
        rdm1_list.append(rdm1)
        max_od = float(np.max(np.abs(rdm1 - np.diag(np.diag(rdm1)))))
        print(f"  [F{idx}] n_dets={res.n_dets}  trace={np.trace(rdm1):.4f}"
              f"  max|off-diag|={max_od:.4e}")
        fragment_info.append({
            "fragment": idx,
            "orbs": list(map(int, frag_orbs)),
            "n_alpha": int(na), "n_beta": int(nb),
            "n_dets": int(res.n_dets),
            "rdm1_trace": float(np.trace(rdm1)),
            "rdm1_max_offdiag": max_od,
        })

    # --- Assemble 36×36 global gamma ---
    gamma_full = assemble_global_gamma_full(rdm1_list, fragments, n_orb)

    # --- Diagnostics ---
    diag_err     = float(np.max(np.abs(np.diag(gamma_full) - gamma_diag)))
    offdiag_max  = float(np.max(np.abs(gamma_full - np.diag(np.diag(gamma_full)))))
    trace_full   = float(np.trace(gamma_full))
    is_symmetric = bool(np.allclose(gamma_full, gamma_full.T, atol=1e-12))
    n_nonzero    = int(np.sum(gamma_full != 0.0))

    print(f"\n[extract_full_gamma] Assembly diagnostics:")
    print(f"  diagonal vs Phase B:      max|Δ| = {diag_err:.2e}")
    print(f"  max |off-diagonal|               = {offdiag_max:.4f}")
    print(f"  gamma_full trace (≈ n_elec={n_elec}) = {trace_full:.4f}")
    print(f"  symmetric: {is_symmetric}   nonzero entries: {n_nonzero}/{n_orb**2}")

    # --- Save ---
    out = Path(output_dir)
    np.save(str(out / "gamma_mixed_full.npy"), gamma_full)
    np.save(str(out / "gamma_mixed_diag.npy"), gamma_diag)

    metadata = {
        "timestamp": timestamp,
        "fcidump": fcidump_path,
        "gamma_diag_source": gamma_diag_path,
        "n_orb": int(n_orb),
        "n_elec": int(n_elec),
        "extraction_config": EXTRACTION_CONFIG,
        "fragments": fragment_info,
        "diagnostics": {
            "diagonal_vs_phase_b_max_diff": diag_err,
            "offdiag_max_abs": offdiag_max,
            "gamma_full_trace": trace_full,
            "symmetric": is_symmetric,
            "nonzero_entries": n_nonzero,
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\n[extract_full_gamma] Saved to {output_dir}")
    print(f"  gamma_mixed_full.npy — shape {gamma_full.shape}")
    print(f"\nTo run D2 with full gamma:")
    print(f"  python frasci/mfa/runners/run_d2_nonoverlapping.py \\")
    print(f"    --gamma-path {out / 'gamma_mixed_full.npy'}")

    return {
        "gamma_path": str(out / "gamma_mixed_full.npy"),
        "output_dir": output_dir,
        "diagnostics": metadata["diagnostics"],
        "fragments": fragment_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full 36×36 gamma from Phase B converged solution"
    )
    parser.add_argument("--fcidump",          default=None)
    parser.add_argument("--gamma-diag-path",  default=None)
    parser.add_argument("--dets-path",        default=None)
    parser.add_argument("--output-dir",       default=None)
    args = parser.parse_args()

    extract_full_gamma(
        fcidump_path=args.fcidump,
        gamma_diag_path=args.gamma_diag_path,
        dets_path=args.dets_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
