"""
run_lasscf_trimci.py
====================
Phase 4 CLI runner: FCIDUMP → mock SCF → LASSCF+TrimCI (lasscf_rdm.LASSCF).

Injects TrimCI as the per-fragment CI solver via mrh's make_fcibox hook,
then runs the LASSCF orbital+CI optimization.  Writes result.json and
kernel_calls.json to the output directory.

Usage
-----
# Production (Fe4S4 pilot):
./FRASCIenv/bin/python -m FRASCI.lasscf.runners.run_lasscf_trimci \\
    [--fcidump PATH] [--partition h1diag] \\
    [--trimci-threshold 0.06] [--max-cycle-macro 20] \\
    [--output-dir DIR]

# Smoke test (H4/STO-3G, must return exit 0 + "SMOKE TEST PASS"):
./FRASCIenv/bin/python -m FRASCI.lasscf.runners.run_lasscf_trimci \\
    --smoke-test

Defaults:
    --fcidump             data/fcidump_cycle_6
    --partition           h1diag
    --trimci-threshold    0.06
    --max-cycle-macro     20
    --output-dir          Outputs/lasscf/<partition>_thr<XXX>_<YYYYMMDD>
                          (auto-constructed when not supplied)

Design doc reference: §5 Phase 4, §6.5, §7 R3, §8
LASSCF class: mrh.my_pyscf.mcscf.lasscf_rdm.LASSCF
              (function factory returning LASSCFNoSymm; supports make_fcibox injection)
MO init: manual identity-permutation (R4 mitigation from Phase 2)
         localize_init_guess raises AssertionError on FCIDUMP mock mol (natm=0,
         no real AO basis for meta-Löwdin orthogonalisation in pyscf/lo/nao.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# MO init helper (duplicated from run_lasscf_csf.py:_build_mo_manual; kept
# inline to avoid touching Phase 2 runner)
# ---------------------------------------------------------------------------

def _build_identity_permutation_mo(orbital_lists: List[List[int]], n_orb: int) -> np.ndarray:
    """
    Build mo_coeff by permuting the n_orb×n_orb identity matrix so columns
    are ordered: [frag0_orbs | frag1_orbs | ... | fragN_orbs].

    LASSCF interprets columns as (core | active_frag0 | active_frag1 | ... | virtual).
    With ncore=0 the first len(frag0_orbs) columns belong to fragment 0, etc.

    This is the R4 fallback: localize_init_guess fails on the FCIDUMP mock mol
    because pyscf/lo/nao.py's meta-Löwdin step requires a real AO basis (natm>0).

    # duplicated from run_lasscf_csf.py:_build_mo_manual
    """
    col_order: List[int] = []
    for frag_orbs in orbital_lists:
        col_order.extend(frag_orbs)
    return np.eye(n_orb)[:, col_order]


# ---------------------------------------------------------------------------
# Warm-start loader: read mo_coeff from a prior csf checkpoint
# ---------------------------------------------------------------------------

def _load_warm_start_mo(
    checkpoint_dir: str,
    partition: str,
    orbital_lists,
    nelec_per_frag,
    spin_sub,
    ncas_sub,
    n_orb: int,
) -> np.ndarray:
    """
    Load mo_coeff from checkpoint_dir/checkpoint.npz.

    Validates that the checkpoint metadata matches the current run's parameters.
    Raises SystemExit (with a clear message) on any mismatch.

    Returns mo_coeff (ndarray, shape (n_orb, n_orb)).
    """
    npz_path = os.path.join(checkpoint_dir, "checkpoint.npz")
    meta_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")

    if not os.path.exists(npz_path):
        print(
            f"ERROR: checkpoint.npz not found in {checkpoint_dir!r}. "
            f"Run run_lasscf_csf.py first to produce the checkpoint.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(meta_path):
        print(
            f"ERROR: checkpoint_metadata.json not found in {checkpoint_dir!r}. "
            f"The checkpoint directory is incomplete.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(meta_path) as fp:
        meta = json.load(fp)

    # -- Sanity check: partition ------------------------------------------
    ck_partition = meta.get("partition")
    if ck_partition != partition:
        print(
            f"ERROR: Cannot warm-start --partition {partition!r} from a checkpoint "
            f"produced for --partition {ck_partition!r}. "
            f"Re-run run_lasscf_csf.py with --partition {partition} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Sanity check: exact fragment orbital list ------------------------
    ck_orbital_lists = meta.get("orbital_lists")
    if ck_orbital_lists is not None:
        cur_orbital_lists = [list(map(int, frag)) for frag in orbital_lists]
        if ck_orbital_lists != cur_orbital_lists:
            print(
                f"ERROR: orbital_lists mismatch. Checkpoint has "
                f"{ck_orbital_lists}, current run has {cur_orbital_lists}. "
                f"Checkpoint is incompatible.",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- Sanity check: nelec_per_frag -------------------------------------
    ck_nelec = meta.get("nelec_per_frag")
    if ck_nelec is not None:
        cur_nelec = [list(ne) for ne in nelec_per_frag]
        if ck_nelec != cur_nelec:
            print(
                f"ERROR: nelec_per_frag mismatch. Checkpoint has {ck_nelec}, "
                f"current run has {cur_nelec}. Checkpoint is incompatible.",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- Sanity check: spin_sub -------------------------------------------
    ck_spin_sub = meta.get("spin_sub")
    if ck_spin_sub is not None:
        cur_spin_sub = list(spin_sub)
        if ck_spin_sub != cur_spin_sub:
            print(
                f"ERROR: spin_sub mismatch. Checkpoint has {ck_spin_sub}, "
                f"current run has {cur_spin_sub}. Checkpoint is incompatible.",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- Sanity check: ncas_sub -------------------------------------------
    ck_ncas_sub = meta.get("ncas_sub")
    if ck_ncas_sub is not None:
        cur_ncas_sub = list(ncas_sub)
        if ck_ncas_sub != cur_ncas_sub:
            print(
                f"ERROR: ncas_sub mismatch. Checkpoint has {ck_ncas_sub}, "
                f"current run has {cur_ncas_sub}. Checkpoint is incompatible.",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- Load mo_coeff (numpy .npz, no pickle) ----------------------------
    data = np.load(npz_path)
    if "mo_coeff" not in data:
        print(
            f"ERROR: 'mo_coeff' key not found in {npz_path}. "
            f"The checkpoint is corrupt or incomplete.",
            file=sys.stderr,
        )
        sys.exit(1)

    mo_coeff = data["mo_coeff"]

    # -- Sanity check: shape ----------------------------------------------
    if mo_coeff.shape != (n_orb, n_orb):
        print(
            f"ERROR: mo_coeff.shape {mo_coeff.shape} from checkpoint does not match "
            f"expected ({n_orb}, {n_orb}) from FCIDUMP. "
            f"Are you using the right FCIDUMP + checkpoint pair?",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[run_lasscf_trimci] Warm-start: loaded mo_coeff {mo_coeff.shape} "
        f"from {npz_path}",
    )
    return mo_coeff


# ---------------------------------------------------------------------------
# Smoke test: 1-fragment H4/STO-3G LASSCF+TrimCI vs full PySCF FCI
# ---------------------------------------------------------------------------

def _run_smoke_test() -> int:
    """
    Run a 1-fragment H4/STO-3G LASSCF+TrimCI smoke test.

    Asserts |las.e_tot - e_fci| <= 1e-6 Ha.
    Prints "SMOKE TEST PASS" on success, "SMOKE TEST FAIL" on failure.
    Returns 0 on pass, 1 on fail.
    """
    from pyscf import gto, scf, ao2mo
    from pyscf.fci import direct_spin1
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox
    from FRASCI.lasscf.trimci_kernel import make_trimci_kernel_for_fragment

    print("[smoke] Building H4/STO-3G mol + RHF ...")
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    norb = mol.nao_nr()          # 4
    na = mol.nelectron // 2      # 2
    nb = mol.nelectron // 2      # 2
    nelec = (na, nb)

    # Ground truth: full PySCF FCI (includes E_nuc)
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False)
    h2 = eri.reshape(norb, norb, norb, norb)
    e_fci, _ = direct_spin1.kernel(h1, h2, norb, nelec, ecore=mol.energy_nuc())
    print(f"[smoke] PySCF full FCI e_tot = {e_fci:.10f} Ha")

    # LASSCF + TrimCI: 1 fragment = full 4-orbital space
    # threshold=1e-8 ensures near-exact TrimCI (all dets included at this tolerance)
    print("[smoke] Building LASSCF+TrimCI (1 fragment, threshold=1e-8) ...")
    las = LASSCF(mf, ncas_sub=(norb,), nelecas_sub=[nelec], spin_sub=(1,))
    las.verbose = 0

    kernel_fn = make_trimci_kernel_for_fragment(threshold=1e-8)
    # spin = |na - nb| = 0, smult = 1 (singlet)
    las.fciboxes[0] = make_fcibox(mol, kernel=kernel_fn, spin=0, smult=1)

    # 1-fragment: identity MO (no permutation needed — whole space is one fragment)
    mo = mf.mo_coeff

    t0 = time.perf_counter()
    las.kernel(mo)
    wall = time.perf_counter() - t0

    e_las = float(las.e_tot)
    delta = abs(e_las - e_fci)

    print(f"[smoke] LASSCF+TrimCI  e_tot = {e_las:.10f} Ha")
    print(f"[smoke] PySCF full FCI e_tot = {e_fci:.10f} Ha")
    print(f"[smoke] |ΔE|                 = {delta:.2e} Ha  (threshold 1e-6)")
    print(f"[smoke] Wall time            = {wall:.2f}s")

    if delta <= 1e-6:
        print(f"SMOKE TEST PASS  |ΔE| = {delta:.2e} Ha ≤ 1e-6 Ha")
        return 0
    else:
        print(f"SMOKE TEST FAIL  |ΔE| = {delta:.2e} Ha > 1e-6 Ha")
        return 1


# ---------------------------------------------------------------------------
# Production: Fe4S4 LASSCF+TrimCI
# ---------------------------------------------------------------------------

def run(
    fcidump_path: str,
    partition: str,
    trimci_threshold: float,
    max_cycle_macro: int,
    output_dir: str,
    trimci_max_dets="auto",
    trimci_max_rounds: int = 2,
    init_from: str = None,
    explicit_orbital_lists: list[list[int]] | None = None,
    partition_description: str | None = None,
) -> dict:
    """
    Full Phase 4 LASSCF+TrimCI run on the FCIDUMP system.
    Returns result dict (also written to result.json in output_dir).
    """
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox
    from FRASCI.lasscf.support import (
        fragment_electron_count,
        load_ref_det,
        validate_fragment_partition,
    )
    from FRASCI.lasscf.fragments import (
        h1diag_fragments,
        mi_min_cut_fragments,
        strong_pair_fragments,
    )
    from FRASCI.lasscf.mock_scf import build_mock_scf_from_fcidump
    from FRASCI.lasscf.trimci_kernel import make_trimci_kernel_for_fragment

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: FCIDUMP → mock SCF
    # ------------------------------------------------------------------
    print(f"[run_lasscf_trimci] Loading FCIDUMP: {fcidump_path}")
    mf = build_mock_scf_from_fcidump(fcidump_path)

    # nao_nr() returns 0 for FCIDUMP mock mol (natm=0); use _nao or h1 shape
    n_orb = (
        int(mf.mol._nao)
        if hasattr(mf.mol, "_nao") and mf.mol._nao
        else mf.get_hcore().shape[0]
    )
    print(
        f"[run_lasscf_trimci] mol: n_orb={n_orb}, "
        f"nelectron={mf.mol.nelectron}, spin={mf.mol.spin}, "
        f"E_nuc={mf.mol.energy_nuc():.6f}"
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
        total_na = sum(na for na, _nb in nelec_per_frag)
        total_nb = sum(nb for _na, nb in nelec_per_frag)
        if total_na + total_nb != mf.mol.nelectron:
            raise RuntimeError(
                f"Explicit partition electron mismatch: "
                f"sum(nelec)=({total_na},{total_nb}) but mol.nelectron={mf.mol.nelectron}"
            )
        spin_sub = [int(abs(na - nb)) + 1 for na, nb in nelec_per_frag]
    elif partition == "h1diag":
        orbital_lists, nelec_per_frag, spin_sub = h1diag_fragments(
            fcidump_path=fcidump_path,
            dets_path=dets_path,
        )
    elif partition == "strong_pair":
        orbital_lists, nelec_per_frag, spin_sub = strong_pair_fragments(
            fcidump_path=fcidump_path,
            dets_path=dets_path,
        )
    elif partition == "mi_min_cut":
        orbital_lists, nelec_per_frag, spin_sub = mi_min_cut_fragments(
            fcidump_path=fcidump_path,
            dets_path=dets_path,
        )
    else:
        raise ValueError(
            f"Partition {partition!r} not supported. Use a built-in partition "
            f"or pass explicit_orbital_lists."
        )

    ncas_sub = tuple(len(f) for f in orbital_lists)
    n_frags = len(orbital_lists)
    print(f"[run_lasscf_trimci] Partition: {partition}")
    for i, (f, ne) in enumerate(zip(orbital_lists, nelec_per_frag)):
        print(f"  F{i}: orbs={f}, nelec={ne}, smult={spin_sub[i]}")
    print(f"  ncas_sub={ncas_sub}, nelec_per_frag={nelec_per_frag}, spin_sub={spin_sub}")

    # ------------------------------------------------------------------
    # Step 3: Build MO — warm-start from checkpoint or identity-permutation
    # ------------------------------------------------------------------
    if init_from is not None:
        mo = _load_warm_start_mo(
            checkpoint_dir=init_from,
            partition=partition,
            orbital_lists=orbital_lists,
            nelec_per_frag=nelec_per_frag,
            spin_sub=spin_sub,
            ncas_sub=ncas_sub,
            n_orb=n_orb,
        )
        mo_init_label = "warm_start_from_csf_checkpoint"
        print(f"[run_lasscf_trimci] MO init: warm_start_from_csf_checkpoint ({init_from})")
    else:
        mo = _build_identity_permutation_mo(orbital_lists, n_orb)
        mo_init_label = "manual_identity_permutation"
        print("[run_lasscf_trimci] MO init: manual_identity_permutation")

    # ------------------------------------------------------------------
    # Step 4: LASSCF object (lasscf_rdm.LASSCF — has make_fcibox hook)
    # ------------------------------------------------------------------
    las = LASSCF(mf, ncas_sub, nelec_per_frag, spin_sub=spin_sub)
    las.max_cycle_macro = max_cycle_macro
    las.verbose = 4  # standard mrh verbosity; prints per-iter convergence info
    print(
        f"[run_lasscf_trimci] LASSCF: {n_frags} fragments, "
        f"max_cycle_macro={max_cycle_macro}, TrimCI threshold={trimci_threshold}"
    )

    # ------------------------------------------------------------------
    # Step 5: Per-fragment logging side-channel
    # kernel_calls collects one row per kernel invocation with:
    #   fragment_idx, n_dets, energy_electronic, wall_time, timestamp
    # We pass a closure over a shared list; each fragment kernel
    # captures its own fragment_idx at construction time.
    # ------------------------------------------------------------------
    kernel_calls: list[dict] = []

    def _make_log_callback(frag_idx: int):
        def _cb(norb, nelec, n_dets, energy_electronic, wall_time_sec):
            kernel_calls.append(
                {
                    "fragment_idx": frag_idx,
                    "n_dets": int(n_dets),
                    "energy_electronic": float(energy_electronic),
                    "wall_time": round(wall_time_sec, 6),
                    "timestamp": time.time(),
                }
            )
            print(
                f"  [TrimCI F{frag_idx}] n_dets={n_dets:6d}  "
                f"e_elec={energy_electronic:.8f}  t={wall_time_sec:.3f}s"
            )
        return _cb

    # ------------------------------------------------------------------
    # Step 6: Inject TrimCI kernel into each fragment's fcibox
    # spin = |na - nb|, smult = spin + 1 (from h1diag_fragments spin_sub)
    # h1diag_fragments returns spin_sub as multiplicities (2S+1).
    # ------------------------------------------------------------------
    for i in range(n_frags):
        na_i, nb_i = nelec_per_frag[i]
        smult_i = spin_sub[i]              # 2S+1 per fragments.py
        spin_i = smult_i - 1               # 2S = smult - 1

        kernel_fn = make_trimci_kernel_for_fragment(
            threshold=trimci_threshold,
            log_callback=_make_log_callback(i),
            max_final_dets=trimci_max_dets,
            max_rounds=trimci_max_rounds,
        )
        las.fciboxes[i] = make_fcibox(
            mf.mol,
            kernel=kernel_fn,
            spin=spin_i,
            smult=smult_i,
        )
        print(
            f"[run_lasscf_trimci] F{i}: injected TrimCI kernel "
            f"(spin={spin_i}, smult={smult_i}, threshold={trimci_threshold})"
        )

    # ------------------------------------------------------------------
    # Step 7: LASSCF kernel
    # ------------------------------------------------------------------
    print(f"[run_lasscf_trimci] Starting LASSCF kernel ...")
    t0 = time.perf_counter()
    las.kernel(mo)
    wall_time_total = time.perf_counter() - t0

    e_tot = float(las.e_tot)
    converged = bool(getattr(las, "converged", None))

    # ------------------------------------------------------------------
    # Step 8: Collect output metrics
    # ------------------------------------------------------------------

    # Per-fragment final TrimCI det counts: last row per fragment in kernel_calls
    last_by_frag: dict[int, dict] = {}
    for row in kernel_calls:
        last_by_frag[row["fragment_idx"]] = row
    dets_per_frag = [
        last_by_frag[i]["n_dets"] if i in last_by_frag else None
        for i in range(n_frags)
    ]

    # Macro iter count heuristic:
    # mrh calls each fragment kernel once per micro-iteration (inner loop) but
    # the number of outer macro iters groups these into blocks.  A conservative
    # lower bound: count the number of times fragment 0's call count increases
    # relative to the previous count for fragment 1, etc.  Simpler: total calls
    # for any one fragment = # macro iters × # micro iters per macro.  We can't
    # disentangle them reliably post-hoc.  Best estimate: total_calls / n_frags.
    frag0_calls = [r for r in kernel_calls if r["fragment_idx"] == 0]
    n_macro_iters_estimate = len(frag0_calls)  # 1 call per frag per outer iteration
    n_macro_iters_heuristic = (
        "calls_to_fragment_0 (one per macro iter × micro iter; "
        "lower bound for macro iter count — mrh may call each fragment "
        "multiple times per macro iter during orbital response)"
    )

    # Wall time aggregated per fragment
    wall_per_frag = {}
    for row in kernel_calls:
        fi = row["fragment_idx"]
        wall_per_frag[fi] = wall_per_frag.get(fi, 0.0) + row["wall_time"]
    wall_time_per_fragment_aggregated = {
        f"F{i}": round(wall_per_frag.get(i, 0.0), 3) for i in range(n_frags)
    }

    # ------------------------------------------------------------------
    # Step 9: Write outputs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_npz = os.path.join(output_dir, "checkpoint.npz")
    np.savez(checkpoint_npz, mo_coeff=las.mo_coeff)
    checkpoint_metadata = {
        "partition": partition,
        "partition_description": partition_description,
        "nelec_per_frag": [list(ne) for ne in nelec_per_frag],
        "spin_sub": list(spin_sub),
        "ncas_sub": list(ncas_sub),
        "n_orb": n_orb,
        "orbital_lists": [list(f) for f in orbital_lists],
        "e_tot": e_tot,
        "converged": converged,
        "method": "LASSCF_TrimCI",
        "mo_coeff_shape": list(las.mo_coeff.shape),
        "output_dir": output_dir,
    }
    checkpoint_metadata_path = os.path.join(output_dir, "checkpoint_metadata.json")
    with open(checkpoint_metadata_path, "w") as fp:
        json.dump(checkpoint_metadata, fp, indent=2)
    print(f"[run_lasscf_trimci] checkpoint.npz written to {checkpoint_npz}")

    # kernel_calls.json
    kernel_calls_path = os.path.join(output_dir, "kernel_calls.json")
    with open(kernel_calls_path, "w") as fp:
        json.dump(kernel_calls, fp, indent=2)
    print(f"[run_lasscf_trimci] kernel_calls.json written to {kernel_calls_path}")

    # result.json
    result = {
        "status": "SUCCESS" if converged else "NOT_CONVERGED",
        "method": "LASSCF_TrimCI",
        "phase": "Phase4_production",
        "lasscf_class": "mrh.my_pyscf.mcscf.lasscf_rdm.LASSCF",
        "mo_init": mo_init_label,
        "warm_start_source": init_from,
        # args
        "partition": partition,
        "partition_description": partition_description,
        "trimci_threshold": trimci_threshold,
        "max_cycle_macro": max_cycle_macro,
        "fcidump": fcidump_path,
        # system
        "n_orb": n_orb,
        "n_elec": mf.mol.nelectron,
        "E_nuc": float(mf.mol.energy_nuc()),
        "ncas_sub": list(ncas_sub),
        "nelec_per_frag": [list(ne) for ne in nelec_per_frag],
        "spin_sub": list(spin_sub),
        "orbital_lists": [list(f) for f in orbital_lists],
        # results
        "e_tot": e_tot,
        "converged": converged,
        "n_macro_iters_estimate": n_macro_iters_estimate,
        "n_macro_iters_heuristic": n_macro_iters_heuristic,
        "dets_per_frag_final": dets_per_frag,
        "wall_time_total": round(wall_time_total, 3),
        "wall_time_per_fragment_aggregated": wall_time_per_fragment_aggregated,
        "output_dir": output_dir,
        "kernel_calls_json": kernel_calls_path,
        "checkpoint_path": output_dir,
        "checkpoint_npz": checkpoint_npz,
        "checkpoint_metadata_json": checkpoint_metadata_path,
    }

    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"[run_lasscf_trimci] result.json written to {result_path}")

    # ------------------------------------------------------------------
    # Step 10: Headline print
    # ------------------------------------------------------------------
    headline = (
        f"LASSCF+TrimCI [{partition} thr={trimci_threshold}] "
        f"e_tot = {e_tot:.6f} Ha | "
        f"converged={converged} | "
        f"macro iters={n_macro_iters_estimate} | "
        f"dets/frag={dets_per_frag} | "
        f"wall {wall_time_total:.1f}s"
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
            "Phase 4 LASSCF+TrimCI runner: FCIDUMP → mock SCF → "
            "LASSCF (lasscf_rdm.LASSCF) with TrimCI injected via make_fcibox."
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
            "Fragment partition label. Built-ins cover h1diag, strong_pair, "
            "and mi_min_cut when no explicit JSON is supplied. Use "
            "--fragment-orbs-json for arbitrary sweep candidates."
        ),
    )
    parser.add_argument(
        "--fragment-orbs-json",
        default=None,
        help=(
            "Optional JSON list of explicit non-overlapping fragment orbital groups. "
            "When supplied, this overrides the built-in --partition orbital lists "
            "but keeps --partition as the run label."
        ),
    )
    parser.add_argument(
        "--trimci-threshold",
        type=float,
        default=0.06,
        help="TrimCI selection threshold (default: 0.06; pilot value).",
    )
    parser.add_argument(
        "--max-cycle-macro",
        type=int,
        default=20,
        help="Maximum number of LASSCF macro iterations (default: 20).",
    )
    parser.add_argument(
        "--trimci-max-dets",
        default="auto",
        help=(
            "TrimCI max_final_dets cap per fragment. 'auto' (default) lets TrimCI "
            "pick; SC MFA at thr=0.01 uses 1000 — match that for fair comparison."
        ),
    )
    parser.add_argument(
        "--trimci-max-rounds",
        type=int,
        default=2,
        help="TrimCI max_rounds (default: 2, matching SC MFA convention).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for result.json and kernel_calls.json. "
            "Auto-constructed as "
            "Outputs/lasscf/<partition>_thr<XXX>_<YYYYMMDD> "
            "if not provided."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a fast 1-fragment H4/STO-3G smoke test instead of the "
            "Fe4S4 production run.  Asserts |ΔE| ≤ 1e-6 Ha vs PySCF full FCI. "
            "No output dir written.  Prints 'SMOKE TEST PASS' or 'SMOKE TEST FAIL'."
        ),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        dest="init_from",
        metavar="CHECKPOINT_DIR",
        help=(
            "Directory containing a checkpoint.npz and checkpoint_metadata.json "
            "produced by run_lasscf_csf.py.  When supplied, mo_coeff from the "
            "converged CSF run is used as the initial orbital guess, bypassing "
            "the identity-permutation default (R3 mitigation, design doc §7). "
            "Mutually exclusive with the default identity-permutation init. "
            "Partition, nelec_per_frag, spin_sub, ncas_sub, and n_orb must match."
        ),
    )

    args = parser.parse_args()

    if args.smoke_test:
        rc = _run_smoke_test()
        sys.exit(rc)

    # Coerce trimci_max_dets: "auto" stays string, anything else is int
    trimci_max_dets = (
        "auto" if args.trimci_max_dets == "auto" else int(args.trimci_max_dets)
    )

    # Auto-construct output dir
    if args.output_dir is None:
        thr_str = f"{args.trimci_threshold:.6f}".rstrip("0").rstrip(".")
        output_dir = (
            f"Outputs/lasscf/"
            f"{args.partition}_thr{thr_str}_{today}"
        )
    else:
        output_dir = args.output_dir

    explicit_orbital_lists = None
    partition_description = None
    if args.fragment_orbs_json is not None:
        with open(args.fragment_orbs_json, encoding="utf-8") as fp:
            explicit_orbital_lists = json.load(fp)
        partition_description = f"explicit orbital JSON: {args.fragment_orbs_json}"

    result = run(
        fcidump_path=args.fcidump,
        partition=args.partition,
        trimci_threshold=args.trimci_threshold,
        max_cycle_macro=args.max_cycle_macro,
        output_dir=output_dir,
        trimci_max_dets=trimci_max_dets,
        trimci_max_rounds=args.trimci_max_rounds,
        init_from=args.init_from,
        explicit_orbital_lists=explicit_orbital_lists,
        partition_description=partition_description,
    )

    if not result.get("converged", False):
        print("[run_lasscf_trimci] WARNING: LASSCF did not converge.")
        sys.exit(1)


if __name__ == "__main__":
    main()
