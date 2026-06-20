"""
run_lassi_lassis.py
===================
CLI runner: load a converged LASSCF checkpoint, run LASSI and/or LASSIS on top
of it, and save JSON results.

Accepts checkpoints produced by either run_lasscf_csf.py or
run_lasscf_trimci.py (both write checkpoint.npz + checkpoint_metadata.json).

Usage
-----
./FRASCIenv/bin/python -m FRASCI.lasscf.runners.run_lassi_lassis \\
    --lasscf-checkpoint-dir Outputs/lasscf/<run_dir> \\
    [--fcidump PATH] [--output-dir DIR] \\
    [--skip-lassi] [--skip-lassis] \\
    [--opt INT] [--lassis-ncharge STR] [--lassis-nspin INT]

Defaults
--------
--fcidump        data/fcidump_cycle_6
--output-dir     auto: Outputs/lasscf/lassi_lassis_<partition>_<YYYYMMDD>
--opt            1  (Davidson solver)
--lassis-ncharge s  (auto: all singles)
--lassis-nspin   0  (charge hops only, no spin flips)

Outputs
-------
lassi_result.json   — LASSI energies and SI vectors (if not --skip-lassi)
lassis_result.json  — LASSIS energies and SI vectors (if not --skip-lassis)
summary.json        — combined energy table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date

import numpy as np


def _build_ct_rootspaces(nelec_per_frag, spin_sub):
    """
    Build LASSI rootspace dicts for one neutral + all single charge-transfer states.

    Neutral state: charges=[0,...], smults=spin_sub, spins=spin_sub[k]-1 for each k.
    CT states: for each ordered pair (i,j) with i!=j, one electron hops from i to j.
      Donor i: charge +1, new_smult = |na-nb| + 2 (one electron removed from lowest-spin state)
      Acceptor j: charge -1, new_smult = |na-nb| + 2 (one electron added)
      Spectators: charge 0, smult = spin_sub[k], spins = spin_sub[k] - 1

    Returns list of dicts: [{charges, spins, smults}, ...]
    """
    n_frags = len(nelec_per_frag)
    rootspaces = []

    # Neutral
    neutral = {
        "charges": [0] * n_frags,
        "spins": [int(spin_sub[k]) - 1 for k in range(n_frags)],
        "smults": [int(spin_sub[k]) for k in range(n_frags)],
    }
    rootspaces.append(neutral)

    # Charge-transfer: electron hops from fragment i to fragment j
    for i in range(n_frags):
        for j in range(n_frags):
            if i == j:
                continue
            na_i, nb_i = nelec_per_frag[i]
            na_j, nb_j = nelec_per_frag[j]

            # New smult after electron removal from i: |na-nb| changes by 1
            new_smult_i = int(abs(na_i - nb_i)) + 2
            # New smult after electron addition to j
            new_smult_j = int(abs(na_j - nb_j)) + 2

            charges = [0] * n_frags
            charges[i] = +1
            charges[j] = -1

            smults = list(spin_sub)
            smults[i] = new_smult_i
            smults[j] = new_smult_j

            # spins: max Sz for donor (+), max Sz for acceptor (-),
            # neutral Sz for spectators
            spins = [int(spin_sub[k]) - 1 for k in range(n_frags)]
            spins[i] = new_smult_i - 1
            spins[j] = -(new_smult_j - 1)

            rootspaces.append({"charges": charges, "spins": spins, "smults": smults})

    return rootspaces


def run(
    checkpoint_dir: str,
    fcidump_path: str,
    output_dir: str,
    skip_lassi: bool = False,
    skip_lassis: bool = False,
    opt: int = 1,
    lassis_ncharge="s",
    lassis_nspin: int = 0,
) -> dict:
    """
    Load a LASSCF checkpoint, run LASSI and LASSIS, save JSON results.
    Returns a summary dict.
    """
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    from mrh.my_pyscf.mcscf.lasci import state_average
    from mrh.my_pyscf.lassi.lassi import LASSI
    from mrh.my_pyscf.lassi.lassis import LASSIS
    from FRASCI.lasscf.mock_scf import build_mock_scf_from_fcidump

    # ------------------------------------------------------------------
    # Step 1: Load checkpoint
    # ------------------------------------------------------------------
    print(f"[run_lassi_lassis] Loading checkpoint from: {checkpoint_dir}")
    meta_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
    npz_path = os.path.join(checkpoint_dir, "checkpoint.npz")

    if not os.path.exists(meta_path):
        print(f"[run_lassi_lassis] ERROR: checkpoint_metadata.json not found in {checkpoint_dir!r}",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(npz_path):
        print(f"[run_lassi_lassis] ERROR: checkpoint.npz not found in {checkpoint_dir!r}",
              file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    data = np.load(npz_path)
    mo_coeff = data["mo_coeff"]

    partition = meta["partition"]
    nelec_per_frag = [tuple(ne) for ne in meta["nelec_per_frag"]]
    spin_sub = list(meta["spin_sub"])
    ncas_sub = tuple(meta["ncas_sub"])
    orbital_lists = meta.get("orbital_lists")
    e_lasscf = meta.get("e_tot")
    converged = meta.get("converged", False)

    if not converged:
        print(f"[run_lassi_lassis] WARNING: checkpoint reports converged=False. "
              f"Proceeding with unconverged orbitals.")

    print(f"[run_lassi_lassis] partition={partition}, ncas_sub={ncas_sub}, "
          f"nelec_per_frag={nelec_per_frag}, spin_sub={spin_sub}")
    print(f"[run_lassi_lassis] e_lasscf={e_lasscf}, converged={converged}")

    # ------------------------------------------------------------------
    # Step 2: Build mock SCF
    # ------------------------------------------------------------------
    print(f"[run_lassi_lassis] Loading FCIDUMP: {fcidump_path}")
    mf = build_mock_scf_from_fcidump(fcidump_path)

    # ------------------------------------------------------------------
    # Step 3: Build LASSCF with lasscf_o0 (supports state_average)
    # ------------------------------------------------------------------
    las = LASSCF(mf, ncas_sub, nelec_per_frag, spin_sub=spin_sub)
    las.verbose = 4

    # ------------------------------------------------------------------
    # Step 4: Run lasci in the converged MOs to populate CI vectors
    # lasci = LASCINoSymm.kernel; returns (converged, e_tot, e_states,
    # e_cas, e_lexc, ci) but does NOT set attributes on las.
    # ------------------------------------------------------------------
    print(f"[run_lassi_lassis] Running lasci in converged MOs ...")
    lasci_result = las.lasci(mo_coeff)
    _conv_lasci, e_lasci_val, e_states_lasci, _e_cas, _e_lexc, ci_lasci = lasci_result

    # Manually propagate results onto las so LASSI/LASSIS can read them
    las.mo_coeff = mo_coeff
    las.ci = ci_lasci
    las.e_tot = float(e_lasci_val)
    las.e_states = [float(e) for e in e_states_lasci]
    e_lasci = float(e_lasci_val)
    print(f"[run_lassi_lassis] LASCI energy (fixed MOs): {e_lasci:.8f} Ha")

    # ------------------------------------------------------------------
    # Step 5: LASSI
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    e_lassi = None
    delta_e_lassi = None
    lassi_result = {}

    if not skip_lassi:
        print(f"[run_lassi_lassis] Building LASSI rootspaces ...")
        try:
            rootspaces = _build_ct_rootspaces(nelec_per_frag, spin_sub)
            n_states = len(rootspaces)
            weights = [1.0 / n_states] * n_states
            print(f"[run_lassi_lassis] {n_states} rootspaces: 1 neutral + "
                  f"{n_states - 1} CT (nfrags={len(ncas_sub)})")

            las2 = state_average(
                las,
                weights=weights,
                charges=[rs["charges"] for rs in rootspaces],
                spins=[rs["spins"] for rs in rootspaces],
                smults=[rs["smults"] for rs in rootspaces],
            )
            las2.lasci()

            lsi = LASSI(las2, opt=opt)
            e_roots_lassi, si_lassi = lsi.kernel()

            e_lassi = float(e_roots_lassi[0])
            delta_e_lassi = float(e_lasci - e_lassi)
            print(f"[run_lassi_lassis] LASSI ground state: {e_lassi:.8f} Ha  "
                  f"(delta from LASCI: {delta_e_lassi:.6f} Ha)")

            lassi_result = {
                "e_lassi_roots": [float(e) for e in e_roots_lassi],
                "e_lassi_gs": e_lassi,
                "e_lasci": e_lasci,
                "delta_e_lassi": delta_e_lassi,
                "n_rootspaces": n_states,
                "rootspaces": rootspaces,
                "opt": opt,
            }
            lassi_path = os.path.join(output_dir, "lassi_result.json")
            with open(lassi_path, "w") as fp:
                json.dump(lassi_result, fp, indent=2)
            print(f"[run_lassi_lassis] lassi_result.json written to {lassi_path}")

        except Exception:
            print(f"[run_lassi_lassis] ERROR: LASSI step failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            e_lassi = None
            delta_e_lassi = None
            lassi_result = {"error": traceback.format_exc()}
            lassi_path = os.path.join(output_dir, "lassi_result.json")
            with open(lassi_path, "w") as fp:
                json.dump(lassi_result, fp, indent=2)
            print(f"[run_lassi_lassis] lassi_result.json (with error) written to {lassi_path}")
    else:
        print(f"[run_lassi_lassis] --skip-lassi: skipping LASSI step.")

    # ------------------------------------------------------------------
    # Step 6: LASSIS
    # ------------------------------------------------------------------
    e_lassis = None
    delta_e_lassis = None
    lassis_result_dict = {}

    if not skip_lassis:
        print(f"[run_lassi_lassis] Running LASSIS (ncharge={lassis_ncharge!r}, "
              f"nspin={lassis_nspin}) ...")
        try:
            lsi_s = LASSIS(las, opt=opt)
            lsi_s.ncharge = lassis_ncharge
            lsi_s.nspin = lassis_nspin
            e_roots_lassis, si_lassis = lsi_s.kernel()

            e_lassis = float(e_roots_lassis[0])
            delta_e_lassis = float(e_lasci - e_lassis)
            print(f"[run_lassi_lassis] LASSIS ground state: {e_lassis:.8f} Ha  "
                  f"(delta from LASCI: {delta_e_lassis:.6f} Ha)")

            lassis_result_dict = {
                "e_lassis_roots": [float(e) for e in e_roots_lassis],
                "e_lassis_gs": e_lassis,
                "e_lasci": e_lasci,
                "delta_e_lassis": delta_e_lassis,
                "ncharge": str(lassis_ncharge),
                "nspin": lassis_nspin,
                "opt": opt,
            }
            lassis_path = os.path.join(output_dir, "lassis_result.json")
            with open(lassis_path, "w") as fp:
                json.dump(lassis_result_dict, fp, indent=2)
            print(f"[run_lassi_lassis] lassis_result.json written to {lassis_path}")

        except Exception:
            print(f"[run_lassi_lassis] ERROR: LASSIS step failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            e_lassis = None
            delta_e_lassis = None
            lassis_result_dict = {"error": traceback.format_exc()}
            lassis_path = os.path.join(output_dir, "lassis_result.json")
            with open(lassis_path, "w") as fp:
                json.dump(lassis_result_dict, fp, indent=2)
            print(f"[run_lassi_lassis] lassis_result.json (with error) written to {lassis_path}")
    else:
        print(f"[run_lassi_lassis] --skip-lassis: skipping LASSIS step.")

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    delta_e_lassi_vs_lassis = None
    if delta_e_lassis is not None and delta_e_lassi is not None:
        delta_e_lassi_vs_lassis = float(delta_e_lassis - delta_e_lassi)

    summary = {
        "lasscf_checkpoint": checkpoint_dir,
        "partition": partition,
        "e_lasscf": e_lasscf,
        "e_lasci": e_lasci,
        "e_lassi": e_lassi,
        "delta_e_lassi": delta_e_lassi,
        "e_lassis": e_lassis,
        "delta_e_lassis": delta_e_lassis,
        "delta_e_lassi_vs_lassis": delta_e_lassi_vs_lassis,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[run_lassi_lassis] summary.json written to {summary_path}")

    # Headline table
    print()
    print("=" * 70)
    print(f"  Partition : {partition}")
    print(f"  E(LASSCF) : {e_lasscf}")
    print(f"  E(LASCI)  : {e_lasci:.8f} Ha")
    if e_lassi is not None:
        print(f"  E(LASSI)  : {e_lassi:.8f} Ha  (delta {delta_e_lassi:.6f} Ha from LASCI)")
    else:
        print(f"  E(LASSI)  : FAILED or skipped")
    if e_lassis is not None:
        print(f"  E(LASSIS) : {e_lassis:.8f} Ha  (delta {delta_e_lassis:.6f} Ha from LASCI)")
    else:
        print(f"  E(LASSIS) : FAILED or skipped")
    if delta_e_lassi_vs_lassis is not None:
        print(f"  LASSIS - LASSI correlation recovery: {delta_e_lassi_vs_lassis:.6f} Ha")
    print("=" * 70)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    today = date.today().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(
        description=(
            "Load a converged LASSCF checkpoint and run LASSI and/or LASSIS on top of it."
        )
    )
    parser.add_argument(
        "--lasscf-checkpoint-dir",
        required=True,
        metavar="DIR",
        help="Directory containing checkpoint.npz and checkpoint_metadata.json.",
    )
    parser.add_argument(
        "--fcidump",
        default="data/fcidump_cycle_6",
        help="Path to FCIDUMP file (default: data/fcidump_cycle_6)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for result JSON files. "
            "Auto-constructed as Outputs/lasscf/lassi_lassis_<partition>_<YYYYMMDD> "
            "if not supplied."
        ),
    )
    parser.add_argument(
        "--skip-lassi",
        action="store_true",
        help="Skip the LASSI step.",
    )
    parser.add_argument(
        "--skip-lassis",
        action="store_true",
        help="Skip the LASSIS step.",
    )
    parser.add_argument(
        "--opt",
        type=int,
        default=1,
        help="LASSI/LASSIS opt parameter (0=slower/exact, 1=Davidson; default: 1).",
    )
    parser.add_argument(
        "--lassis-ncharge",
        default="s",
        help="LASSIS ncharge parameter ('s'=auto, or an integer; default: 's').",
    )
    parser.add_argument(
        "--lassis-nspin",
        type=int,
        default=0,
        help=(
            "LASSIS nspin parameter (0=charge hops only, no spin flips; default: 0). "
            "Recommended for first run."
        ),
    )

    args = parser.parse_args()

    # Coerce ncharge: keep 's' as string, otherwise convert to int
    lassis_ncharge = (
        args.lassis_ncharge
        if args.lassis_ncharge == "s"
        else int(args.lassis_ncharge)
    )

    # Auto-construct output_dir after we know the partition (read metadata first)
    checkpoint_dir = args.lasscf_checkpoint_dir
    if args.output_dir is None:
        meta_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        try:
            with open(meta_path) as f:
                _meta = json.load(f)
            partition = _meta.get("partition", "unknown")
        except Exception:
            partition = "unknown"
        output_dir = (
            f"Outputs/lasscf/lassi_lassis_{partition}_{today}"
        )
    else:
        output_dir = args.output_dir

    summary = run(
        checkpoint_dir=checkpoint_dir,
        fcidump_path=args.fcidump,
        output_dir=output_dir,
        skip_lassi=args.skip_lassi,
        skip_lassis=args.skip_lassis,
        opt=args.opt,
        lassis_ncharge=lassis_ncharge,
        lassis_nspin=args.lassis_nspin,
    )

    if summary.get("e_lassi") is None and not args.skip_lassi:
        print("[run_lassi_lassis] WARNING: LASSI step failed.", file=sys.stderr)
    if summary.get("e_lassis") is None and not args.skip_lassis:
        print("[run_lassi_lassis] WARNING: LASSIS step failed.", file=sys.stderr)

    if (
        (summary.get("e_lassi") is None and not args.skip_lassi)
        or (summary.get("e_lassis") is None and not args.skip_lassis)
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
