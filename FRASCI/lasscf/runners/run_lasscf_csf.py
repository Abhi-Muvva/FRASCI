"""
run_lasscf_csf.py
=================
Phase 2 CLI runner: FCIDUMP → mock SCF → LASSCF with mrh's default csf_solver.

This is a CONTROL run to confirm the orchestration path (FCIDUMP → mock SCF →
LASSCF) works end-to-end before TrimCI is injected in Phase 3.  No TrimCI is
used here; each fragment uses mrh's built-in CSF/FCI solver.

Usage
-----
./FRASCIenv/bin/python -m FRASCI.lasscf.runners.run_lasscf_csf \\
    [--fcidump PATH] [--partition h1diag] [--output-dir DIR] [--max-cycle N]

Defaults:
    --fcidump   data/fcidump_cycle_6
    --partition h1diag
    --output-dir Outputs/lasscf/csf_h1diag_<YYYYMMDD>
    --max-cycle 50

Output
------
result.json in the output directory with:
    e_tot, converged, n_macro_iters, per_fragment_ci_energies,
    wall_clock_sec, localize_init_guess_path, orbital_lists, nelec_per_frag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import numpy as np


# ---------------------------------------------------------------------------
# localize_init_guess path selection
# ---------------------------------------------------------------------------
# R4 (design doc §7): localize_init_guess in mrh needs a real AO basis for its
# meta-Lowdin orthogonalisation (orth.orth_ao). For the mock FCIDUMP mol (fake
# atoms, no real basis), this call raises AssertionError deep in pyscf/lo/nao.py.
#
# Fallback: build mo_coeff manually by permuting the 36×36 identity matrix so
# columns are ordered (frag0_orbs | frag1_orbs | frag2_orbs).  LASSCF interprets
# column order as (core | active_frag0 | active_frag1 | ... | virtual), so with
# ncore=0 the first 12 columns belong to frag0, next 12 to frag1, last 12 to frag2.
#
# This is the "MANUAL_PERMUTATION" path documented in result.json.
# ---------------------------------------------------------------------------

_LOCALIZE_PATH_AUTO   = "localize_init_guess(frags_by_AOs=True)"
_LOCALIZE_PATH_MANUAL = "manual_permutation_of_identity_columns"


def _build_mo_manual(orbital_lists: list[list[int]], n_orb: int) -> np.ndarray:
    """
    Permute the n_orb×n_orb identity matrix so columns are ordered as:
        [frag0_orbs | frag1_orbs | frag2_orbs]

    This feeds LASSCF the correct initial active-orbital assignment
    without calling localize_init_guess (which requires a real AO basis).
    """
    col_order = []
    for frag_orbs in orbital_lists:
        col_order.extend(frag_orbs)
    mo = np.eye(n_orb)[:, col_order]
    return mo


def _try_localize_init_guess(las, orbital_lists, mo_coeff):
    """
    Attempt mrh's localize_init_guess with frags_by_AOs=True.

    Returns (mo, path_label).
    Falls back to manual permutation if localize_init_guess raises an exception
    (expected for FCIDUMP mock mol because orth.orth_ao requires a real AO basis).
    """
    try:
        mo = las.localize_init_guess(orbital_lists, mo_coeff, frags_by_AOs=True)
        return mo, _LOCALIZE_PATH_AUTO
    except Exception as exc:
        print(
            f"[run_lasscf_csf] localize_init_guess(frags_by_AOs=True) failed "
            f"({type(exc).__name__}: {exc}). "
            f"Falling back to manual column-permutation of identity MOs."
        )
        n_orb = mo_coeff.shape[0]
        mo = _build_mo_manual(orbital_lists, n_orb)
        return mo, _LOCALIZE_PATH_MANUAL


def _save_checkpoint(las, output_dir: str, result: dict) -> str:
    """
    Save converged LASSCF orbitals and (optionally) CI vectors to
    output_dir/checkpoint.npz and output_dir/checkpoint_metadata.json.

    Only mo_coeff is mandatory; CI vectors from csf_solver may be wrapped
    in custom types — we save those defensively and log a warning on failure.

    Returns the absolute path to the checkpoint directory (output_dir).
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- mo_coeff (mandatory) -------------------------------------------
    mo_coeff = las.mo_coeff
    save_arrays = {"mo_coeff": mo_coeff}

    # -- CI vectors (optional, best-effort) ------------------------------
    ci_saved_frags = []
    ci = getattr(las, "ci", None)
    if ci is not None:
        try:
            for i, ci_frag in enumerate(ci):
                # ci_frag may be a list-of-arrays [state0, state1, ...] or a single array
                if isinstance(ci_frag, (list, tuple)):
                    ci_root = ci_frag[0]
                else:
                    ci_root = ci_frag

                arr = np.asarray(ci_root, dtype=float)
                save_arrays[f"ci_{i}"] = arr
                ci_saved_frags.append(i)
        except Exception as exc:
            print(
                f"[run_lasscf_csf] WARNING: could not convert las.ci to numpy arrays "
                f"(fragments saved so far: {ci_saved_frags}). "
                f"Reason: {type(exc).__name__}: {exc}. "
                f"Continuing — only mo_coeff will be in the checkpoint.",
                file=sys.stderr,
            )
            # Remove any partial ci keys if conversion blew up mid-loop
            for key in [f"ci_{i}" for i in ci_saved_frags]:
                save_arrays.pop(key, None)
            ci_saved_frags = []

    # -- Write checkpoint.npz --------------------------------------------
    checkpoint_npz = os.path.join(output_dir, "checkpoint.npz")
    np.savez(checkpoint_npz, **save_arrays)
    saved_keys = list(save_arrays.keys())
    print(
        f"[run_lasscf_csf] checkpoint.npz written to {checkpoint_npz} "
        f"(keys: {saved_keys}, mo_coeff.shape={mo_coeff.shape})",
        file=sys.stderr,
    )

    # -- Write checkpoint_metadata.json ----------------------------------
    metadata = {
        "partition": result.get("partition"),
        "nelec_per_frag": result.get("nelec_per_frag"),
        "spin_sub": result.get("spin_sub"),
        "ncas_sub": result.get("ncas_sub"),
        "e_tot": result.get("e_tot"),
        "converged": result.get("converged"),
        "n_orb": result.get("n_orb"),
        "orbital_lists": result.get("orbital_lists"),
        "mo_coeff_shape": list(mo_coeff.shape),
        "ci_fragments_saved": ci_saved_frags,
        "method": result.get("method"),
        "output_dir": output_dir,
    }
    metadata_path = os.path.join(output_dir, "checkpoint_metadata.json")
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    print(
        f"[run_lasscf_csf] checkpoint_metadata.json written to {metadata_path}",
        file=sys.stderr,
    )

    return output_dir


def run(
    fcidump_path: str,
    partition: str,
    output_dir: str,
    max_cycle: int,
    explicit_orbital_lists: list[list[int]] | None = None,
    partition_description: str | None = None,
) -> dict:
    """
    Full Phase 2 LASSCF run.  Returns result dict (also written to result.json).
    """
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF

    from FRASCI.lasscf.fragments import h1diag_fragments
    from FRASCI.lasscf.support import (
        fragment_electron_count,
        load_ref_det,
        validate_fragment_partition,
    )
    from FRASCI.lasscf.mock_scf import build_mock_scf_from_fcidump

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: FCIDUMP → mock SCF
    # ------------------------------------------------------------------
    print(f"[run_lasscf_csf] Loading FCIDUMP: {fcidump_path}")
    mf = build_mock_scf_from_fcidump(fcidump_path)
    # mol.nao_nr() returns 0 for FCIDUMP mock mol (natm=0); use _nao or h1 shape
    n_orb = int(mf.mol._nao) if hasattr(mf.mol, "_nao") and mf.mol._nao else mf.get_hcore().shape[0]
    print(
        f"[run_lasscf_csf] mol: n_orb={n_orb}, nelectron={mf.mol.nelectron}, "
        f"spin={mf.mol.spin}, E_nuc={mf.mol.energy_nuc():.6f}"
    )

    # ------------------------------------------------------------------
    # Step 2: Fragment orbital lists + nelec + spin_sub
    # ------------------------------------------------------------------
    dets_path = os.path.join(os.path.dirname(fcidump_path), "dets.npz")

    if explicit_orbital_lists is not None:
        orbital_lists = [sorted(map(int, frag)) for frag in explicit_orbital_lists]
        ref_alpha_bits, ref_beta_bits = load_ref_det(dets_path, row=0)
        validate_fragment_partition(orbital_lists, n_orb, ref_alpha_bits, ref_beta_bits)
        nelec_per_frag = [
            tuple(map(int, fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag)))
            for frag in orbital_lists
        ]
        spin_sub = [int(abs(na - nb)) + 1 for na, nb in nelec_per_frag]
    elif partition == "h1diag":
        orbital_lists, nelec_per_frag, spin_sub = h1diag_fragments(
            fcidump_path=fcidump_path,
            dets_path=dets_path,
        )
    else:
        raise ValueError(
            f"Partition {partition!r} not supported in Phase 2 "
            f"(only 'h1diag' accepted until Phase 5 wires up other partitions)"
        )

    ncas_sub = tuple(len(f) for f in orbital_lists)
    print(f"[run_lasscf_csf] Partition: {partition}")
    for i, (f, ne) in enumerate(zip(orbital_lists, nelec_per_frag)):
        print(f"  F{i}: orbs={f}, nelec={ne}")
    print(f"  ncas_sub={ncas_sub}, nelec_per_frag={nelec_per_frag}, spin_sub={spin_sub}")

    # ------------------------------------------------------------------
    # Step 3: LASSCF object
    # ------------------------------------------------------------------
    las = LASSCF(mf, ncas_sub, nelec_per_frag, spin_sub=spin_sub)
    las.max_cycle_macro = max_cycle
    # Slightly higher gradient threshold than default is fine for Phase 2 control
    las.conv_tol_grad = 1e-4

    # ------------------------------------------------------------------
    # Step 4: Initial MO coefficients
    # R4 mitigation: try localize_init_guess, fall back to manual permutation
    # ------------------------------------------------------------------
    mo, localize_path = _try_localize_init_guess(las, orbital_lists, mf.mo_coeff)
    print(f"[run_lasscf_csf] Initial MO path: {localize_path}")

    # ------------------------------------------------------------------
    # Step 5: LASSCF kernel
    # ------------------------------------------------------------------
    print(f"[run_lasscf_csf] Starting LASSCF kernel (max_cycle={max_cycle}) ...")
    las.kernel(mo)
    t_elapsed = time.perf_counter() - t_start

    e_tot = float(las.e_tot)
    converged = bool(las.converged)

    # ------------------------------------------------------------------
    # Step 6: Collect per-fragment CI energies
    # ------------------------------------------------------------------
    per_fragment_ci_energies = None
    if hasattr(las, "e_states") and las.e_states is not None:
        try:
            per_fragment_ci_energies = [float(e) for e in las.e_states]
        except Exception:
            pass

    if per_fragment_ci_energies is None and las.ci is not None:
        # Fall back: read the last logged CI eigenvalues from the CI vectors
        # (LASCI stores ci as list-of-lists: ci[frag][state])
        try:
            per_fragment_ci_energies = []
            for frag_idx, ci_frag in enumerate(las.ci):
                ci_root = ci_frag[0] if isinstance(ci_frag, list) else ci_frag
                # Use LASSCF's own h1e_for_las / get_h2eff machinery
                # to compute the CI energy is complex; just leave as None per frag
                per_fragment_ci_energies.append(None)
        except Exception:
            per_fragment_ci_energies = None

    # Count macro iterations from las object
    n_macro_iters = getattr(las, "niter", None)
    if n_macro_iters is None:
        # Not directly stored; count is printed but not saved — report as None
        n_macro_iters = None

    # ------------------------------------------------------------------
    # Step 7: Write result.json
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "status": "SUCCESS" if converged else "NOT_CONVERGED",
        "method": "LASSCF_csf_solver",
        "phase": "Phase2_control",
        "partition": partition,
        "partition_description": partition_description,
        "fcidump": fcidump_path,
        "n_orb": n_orb,
        "n_elec": mf.mol.nelectron,
        "E_nuc": float(mf.mol.energy_nuc()),
        "ncas_sub": list(ncas_sub),
        "nelec_per_frag": [list(ne) for ne in nelec_per_frag],
        "spin_sub": list(spin_sub),
        "orbital_lists": [list(f) for f in orbital_lists],
        "e_tot": e_tot,
        "converged": converged,
        "n_macro_iters": n_macro_iters,
        "per_fragment_ci_energies": per_fragment_ci_energies,
        "wall_clock_sec": round(t_elapsed, 3),
        "localize_init_guess_path": localize_path,
        "max_cycle": max_cycle,
        "output_dir": output_dir,
    }

    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_lasscf_csf] result.json written to {result_path}")

    # ------------------------------------------------------------------
    # Step 7b: Save checkpoint (mo_coeff + ci vectors + metadata)
    # ------------------------------------------------------------------
    checkpoint_path = _save_checkpoint(las, output_dir, result)
    result["checkpoint_path"] = checkpoint_path

    # Re-write result.json with the checkpoint_path key
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run_lasscf_csf] result.json (with checkpoint_path) re-written to {result_path}")

    # ------------------------------------------------------------------
    # Step 8: Headline print
    # ------------------------------------------------------------------
    headline = (
        f"LASSCF (csf_solver) e_tot = {e_tot:.6f} Ha | "
        f"converged={converged} | "
        f"iters={n_macro_iters}"
    )
    print(headline)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    today = date.today().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 LASSCF control run: FCIDUMP → mock SCF → LASSCF "
            "with mrh default csf_solver (no TrimCI)."
        )
    )
    parser.add_argument(
        "--fcidump",
        default="data/fcidump_cycle_6",
        help="Path to FCIDUMP file (default: data/fcidump_cycle_6)",
    )
    parser.add_argument(
        "--partition",
        default="h1diag",
        help=(
            "Fragment partition label. Built-in 'h1diag' is supported; pass "
            "--fragment-orbs-json for arbitrary explicit partitions."
        ),
    )
    parser.add_argument(
        "--fragment-orbs-json",
        default=None,
        help="Optional JSON list of explicit non-overlapping fragment orbital groups.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"Outputs/lasscf/csf_h1diag_{today}",
        help="Directory for result.json output.",
    )
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=50,
        help="Maximum number of LASSCF macro iterations (default: 50).",
    )

    args = parser.parse_args()

    explicit_orbital_lists = None
    partition_description = None
    if args.fragment_orbs_json is not None:
        with open(args.fragment_orbs_json, encoding="utf-8") as fp:
            explicit_orbital_lists = json.load(fp)
        partition_description = f"explicit orbital JSON: {args.fragment_orbs_json}"

    result = run(
        fcidump_path=args.fcidump,
        partition=args.partition,
        output_dir=args.output_dir,
        max_cycle=args.max_cycle,
        explicit_orbital_lists=explicit_orbital_lists,
        partition_description=partition_description,
    )

    # Exit with non-zero if not converged so CI/callers can detect failure
    if not result.get("converged", False):
        print("[run_lasscf_csf] WARNING: LASSCF did not converge.")
        sys.exit(1)


if __name__ == "__main__":
    main()
