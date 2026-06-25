"""
run_lasscf_coo.py
=================
LASSCF + COO-TrimCI runner — same layout as
``FRASCI/lasscf/runners/run_lasscf_trimci.py`` but injects the
COO-enabled fragment kernel (see ``FRASCI/lasscf/coo_kernel.py``).

Output is the same as ``run_lasscf_trimci`` (``result.json``,
``kernel_calls.json``, ``checkpoint.npz`` + metadata) so the existing notebook
helpers, LASSI/LASSIS runners, and comparison plotters work without changes.

Usage
-----
Production (Fe4S4 pilot):
    ./FRASCIenv/bin/python -m FRASCI.lasscf.runners.run_lasscf_coo \\
        [--fcidump PATH] [--partition h1diag] [--fragment-orbs-json FILE] \\
        [--trimci-threshold 0.01] [--trimci-max-dets 2000] \\
        [--coo-cycles 2] [--coo-bfgs-maxiter 20] \\
        [--max-cycle-macro 100] [--output-dir DIR]

Smoke test (H4/STO-3G):
    ./FRASCIenv/bin/python -m FRASCI.lasscf.runners.run_lasscf_coo \\
        --smoke-test
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


def _build_identity_permutation_mo(orbital_lists, n_orb):
    col_order = []
    for frag_orbs in orbital_lists:
        col_order.extend(frag_orbs)
    return np.eye(n_orb)[:, col_order]


def _load_warm_start_mo(checkpoint_dir, partition, orbital_lists,
                       nelec_per_frag, spin_sub, ncas_sub, n_orb):
    npz_path = os.path.join(checkpoint_dir, "checkpoint.npz")
    meta_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")

    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        print(f"ERROR: checkpoint missing in {checkpoint_dir!r}.", file=sys.stderr)
        sys.exit(1)

    with open(meta_path) as fp:
        meta = json.load(fp)

    for key, current in (
        ("partition", partition),
        ("orbital_lists", [list(map(int, f)) for f in orbital_lists]),
        ("nelec_per_frag", [list(ne) for ne in nelec_per_frag]),
        ("spin_sub", list(spin_sub)),
        ("ncas_sub", list(ncas_sub)),
    ):
        ck_val = meta.get(key)
        if ck_val is not None and ck_val != current:
            print(f"ERROR: checkpoint {key} mismatch.  Checkpoint={ck_val}, "
                  f"current={current}", file=sys.stderr)
            sys.exit(1)

    data = np.load(npz_path)
    if "mo_coeff" not in data:
        print(f"ERROR: 'mo_coeff' missing in {npz_path}", file=sys.stderr)
        sys.exit(1)

    mo_coeff = data["mo_coeff"]
    if mo_coeff.shape != (n_orb, n_orb):
        print(f"ERROR: mo_coeff.shape {mo_coeff.shape} != expected ({n_orb},{n_orb})",
              file=sys.stderr)
        sys.exit(1)

    print(f"[run_lasscf_coo] Warm-start: loaded mo_coeff {mo_coeff.shape} from {npz_path}")
    return mo_coeff


def _run_smoke_test(parallel_workers: int = 0, process_workers: int = 0) -> int:
    """1-fragment H4/STO-3G COO-LASSCF vs full FCI.  Optionally exercise the
    parallel-fragments context manager and/or the process-pool dispatch even
    on a 1-fragment system (they should install + uninstall cleanly with no
    behaviour change)."""
    from pyscf import gto, scf, ao2mo
    from pyscf.fci import direct_spin1
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox
    from FRASCI.lasscf.coo_kernel import make_coo_trimci_kernel_for_fragment
    from FRASCI.lasscf.parallel import parallel_fragments
    from FRASCI.lasscf.parallel_mp import (
        coo_worker_pool, make_proxy_kernel,
    )

    mol = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
                basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    norb = mol.nao_nr()
    na = mol.nelectron // 2; nb = mol.nelectron // 2

    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)
    e_fci, _ = direct_spin1.kernel(h1, h2, norb, (na, nb), ecore=mol.energy_nuc())

    las = LASSCF(mf, ncas_sub=(norb,), nelecas_sub=[(na, nb)], spin_sub=(1,))
    las.verbose = 0

    kernel_kwargs = dict(threshold=1e-8, n_coo_cycles=1)

    use_mp = process_workers and process_workers > 0
    t0 = time.perf_counter()
    if use_mp:
        # 1-fragment smoke -> 1 worker.  Bigger requests are clamped.
        n_workers_eff = min(process_workers, 1)
        with coo_worker_pool(n_workers_eff, kernel_kwargs,
                             omp_threads_per_worker=None,
                             log_banner=False) as workers:
            proxy = make_proxy_kernel(workers[0]["in_q"], workers[0]["out_q"],
                                      log_callback=None, frag_idx=0)
            las.fciboxes[0] = make_fcibox(mol, kernel=proxy, spin=0, smult=1)
            with parallel_fragments(parallel_workers, log_banner=False):
                las.kernel(mf.mo_coeff)
    else:
        kernel_fn = make_coo_trimci_kernel_for_fragment(**kernel_kwargs)
        las.fciboxes[0] = make_fcibox(mol, kernel=kernel_fn, spin=0, smult=1)
        with parallel_fragments(parallel_workers, log_banner=False):
            las.kernel(mf.mo_coeff)
    wall = time.perf_counter() - t0
    delta = abs(las.e_tot - e_fci)

    pieces = []
    if parallel_workers and parallel_workers > 1:
        pieces.append(f"thread({parallel_workers})")
    if use_mp:
        pieces.append(f"proc({process_workers})")
    label = "+".join(pieces) if pieces else "serial"
    print(f"[smoke {label}] LASSCF+COO e_tot = {las.e_tot:.10f}")
    print(f"[smoke {label}] PySCF FCI e_tot  = {e_fci:.10f}")
    print(f"[smoke {label}] |delta|          = {delta:.2e} (threshold 1e-6)  wall={wall:.2f}s")
    if delta <= 1e-6:
        print(f"SMOKE TEST PASS ({label})")
        return 0
    print(f"SMOKE TEST FAIL ({label})")
    return 1


def run(
    fcidump_path: str,
    partition: str,
    trimci_threshold: float,
    max_cycle_macro: int,
    output_dir: str,
    trimci_max_dets="auto",
    trimci_max_rounds: int = 2,
    coo_cycles: int = 2,
    coo_bfgs_maxiter: int = 20,
    coo_davidson_tol: float = 1e-7,
    coo_bfgs_ftol: float = 1e-8,
    warm_start_kappa: bool = True,
    init_from: str = None,
    explicit_orbital_lists=None,
    partition_description: str = None,
    parallel_workers: int = 0,
    omp_threads_per_frag: int = None,
    process_workers: int = 0,
) -> dict:
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox

    from FRASCI.lasscf.support import fragment_electron_count
    from FRASCI.lasscf.support import validate_fragment_partition
    from FRASCI.lasscf.coo_kernel import make_coo_trimci_kernel_for_fragment
    from FRASCI.lasscf.fragments import (
        h1diag_fragments, mi_min_cut_fragments, strong_pair_fragments,
    )
    from FRASCI.lasscf.mock_scf import build_mock_scf_from_fcidump
    from FRASCI.lasscf.parallel import parallel_fragments
    from FRASCI.lasscf.parallel_mp import (
        coo_worker_pool, make_proxy_kernel,
    )
    from FRASCI.lasscf.support import load_ref_det

    t_start = time.perf_counter()

    print(f"[run_lasscf_coo] Loading FCIDUMP: {fcidump_path}")
    mf = build_mock_scf_from_fcidump(fcidump_path)
    n_orb = (int(mf.mol._nao) if hasattr(mf.mol, "_nao") and mf.mol._nao
             else mf.get_hcore().shape[0])
    print(f"[run_lasscf_coo] mol: n_orb={n_orb}, "
          f"nelectron={mf.mol.nelectron}, spin={mf.mol.spin}, "
          f"E_nuc={mf.mol.energy_nuc():.6f}")

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
            fcidump_path=fcidump_path, dets_path=dets_path)
    elif partition == "strong_pair":
        orbital_lists, nelec_per_frag, spin_sub = strong_pair_fragments(
            fcidump_path=fcidump_path, dets_path=dets_path)
    elif partition == "mi_min_cut":
        orbital_lists, nelec_per_frag, spin_sub = mi_min_cut_fragments(
            fcidump_path=fcidump_path, dets_path=dets_path)
    else:
        raise ValueError(f"Partition {partition!r} not supported")

    ncas_sub = tuple(len(f) for f in orbital_lists)
    n_frags = len(orbital_lists)
    print(f"[run_lasscf_coo] Partition: {partition}")
    for i, (f, ne) in enumerate(zip(orbital_lists, nelec_per_frag)):
        print(f"  F{i}: orbs={f}, nelec={ne}, smult={spin_sub[i]}")
    print(f"  ncas_sub={ncas_sub}, nelec_per_frag={nelec_per_frag}, spin_sub={spin_sub}")
    print(f"[run_lasscf_coo] COO knobs: cycles={coo_cycles}  bfgs_maxiter={coo_bfgs_maxiter}  "
          f"davidson_tol={coo_davidson_tol}  ftol={coo_bfgs_ftol}")

    if init_from is not None:
        mo = _load_warm_start_mo(
            checkpoint_dir=init_from, partition=partition,
            orbital_lists=orbital_lists, nelec_per_frag=nelec_per_frag,
            spin_sub=spin_sub, ncas_sub=ncas_sub, n_orb=n_orb,
        )
        mo_init_label = "warm_start_from_csf_checkpoint"
        print(f"[run_lasscf_coo] MO init: warm_start_from_csf_checkpoint ({init_from})")
    else:
        mo = _build_identity_permutation_mo(orbital_lists, n_orb)
        mo_init_label = "manual_identity_permutation"
        print("[run_lasscf_coo] MO init: manual_identity_permutation")

    las = LASSCF(mf, ncas_sub, nelec_per_frag, spin_sub=spin_sub)
    las.max_cycle_macro = max_cycle_macro
    las.verbose = 4

    kernel_calls: list[dict] = []

    def _make_log_callback(frag_idx: int):
        def _cb(norb, nelec, n_dets, energy_electronic, wall_time_sec, extras=None):
            row = {
                "fragment_idx": frag_idx,
                "n_dets": int(n_dets),
                "energy_electronic": float(energy_electronic),
                "wall_time": round(wall_time_sec, 6),
                "timestamp": time.time(),
            }
            if extras:
                row.update({f"coo_{k}": v for k, v in extras.items()})
            kernel_calls.append(row)
            extras_str = (
                f"  COO[U|={extras['U_offdiag_norm']:.3f} "
                f"dE={extras['dE_from_coo_mHa']:+.2f} mHa "
                f"seed_dets={extras['n_dets_seed']}]"
                if extras else ""
            )
            print(f"  [COO-TrimCI F{frag_idx}] n_dets={n_dets:6d}  "
                  f"e_elec={energy_electronic:.8f}  t={wall_time_sec:.3f}s{extras_str}")
        return _cb

    # Kernel kwargs shared by every fragment (per-fragment log_callback is
    # injected separately).  In process mode the workers re-build the same
    # kernel from these kwargs inside their own interpreters.
    kernel_kwargs_shared = dict(
        threshold=trimci_threshold,
        n_coo_cycles=coo_cycles,
        bfgs_maxiter=coo_bfgs_maxiter,
        davidson_tol=coo_davidson_tol,
        ftol=coo_bfgs_ftol,
        warm_start_kappa=warm_start_kappa,
        max_final_dets=trimci_max_dets,
        max_rounds=trimci_max_rounds,
    )

    use_mp = bool(process_workers and process_workers > 0)
    if use_mp:
        # Clamp to active-fragment count; F0/F1 are trivial but they still
        # need a worker for the proxy kernel.  One worker per fragment is the
        # right default.
        n_workers_eff = min(process_workers, n_frags)
        print(f"[run_lasscf_coo] Process-pool fragment dispatch: "
              f"workers={n_workers_eff} (requested {process_workers})  "
              f"omp_threads_per_frag={omp_threads_per_frag}")
    else:
        n_workers_eff = 0

    print("[run_lasscf_coo] Starting LASSCF kernel ...")
    if parallel_workers and parallel_workers > 1:
        print(f"[run_lasscf_coo] Parallel fragment dispatch (threads): "
              f"workers={parallel_workers}  omp_threads_per_frag={omp_threads_per_frag}")

    def _inject_inproc_kernels():
        for i in range(n_frags):
            smult_i = spin_sub[i]
            spin_i = smult_i - 1
            kernel_fn = make_coo_trimci_kernel_for_fragment(
                log_callback=_make_log_callback(i),
                **kernel_kwargs_shared,
            )
            las.fciboxes[i] = make_fcibox(mf.mol, kernel=kernel_fn,
                                          spin=spin_i, smult=smult_i)
            print(f"[run_lasscf_coo] F{i}: injected in-proc COO-TrimCI kernel "
                  f"(spin={spin_i}, smult={smult_i})")

    def _inject_proxy_kernels(workers):
        # Round-robin fragments onto workers in case n_workers < n_frags.
        # With default n_workers_eff == n_frags this is just one-to-one.
        for i in range(n_frags):
            smult_i = spin_sub[i]
            spin_i = smult_i - 1
            w = workers[i % len(workers)]
            proxy = make_proxy_kernel(
                w["in_q"], w["out_q"],
                log_callback=_make_log_callback(i),
                frag_idx=i,
            )
            las.fciboxes[i] = make_fcibox(mf.mol, kernel=proxy,
                                          spin=spin_i, smult=smult_i)
            print(f"[run_lasscf_coo] F{i}: injected proxy kernel "
                  f"-> worker {i % len(workers)}")

    t0 = time.perf_counter()
    try:
        if use_mp:
            with coo_worker_pool(
                n_workers_eff,
                kernel_kwargs_shared,
                omp_threads_per_worker=omp_threads_per_frag,
            ) as workers:
                _inject_proxy_kernels(workers)
                with parallel_fragments(
                    n_workers=parallel_workers or 0,
                    omp_threads_per_worker=omp_threads_per_frag,
                    log_banner=bool(parallel_workers and parallel_workers > 1),
                ):
                    las.kernel(mo)
        else:
            _inject_inproc_kernels()
            with parallel_fragments(
                n_workers=parallel_workers or 0,
                omp_threads_per_worker=omp_threads_per_frag,
                log_banner=bool(parallel_workers and parallel_workers > 1),
            ):
                las.kernel(mo)
    except Exception as exc:
        print(f"[run_lasscf_coo] LASSCF kernel raised: {exc}")
        raise
    wall_time_total = time.perf_counter() - t0

    e_tot = float(las.e_tot)
    converged = bool(getattr(las, "converged", None))

    last_by_frag: dict[int, dict] = {}
    for row in kernel_calls:
        last_by_frag[row["fragment_idx"]] = row
    dets_per_frag = [
        last_by_frag[i]["n_dets"] if i in last_by_frag else None
        for i in range(n_frags)
    ]

    frag0_calls = [r for r in kernel_calls if r["fragment_idx"] == 0]
    n_macro_iters_estimate = len(frag0_calls)
    wall_per_frag = {}
    for row in kernel_calls:
        fi = row["fragment_idx"]
        wall_per_frag[fi] = wall_per_frag.get(fi, 0.0) + row["wall_time"]
    wall_time_per_fragment_aggregated = {
        f"F{i}": round(wall_per_frag.get(i, 0.0), 3) for i in range(n_frags)
    }

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
        "method": "LASSCF_COO_TrimCI",
        "mo_coeff_shape": list(las.mo_coeff.shape),
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "checkpoint_metadata.json"), "w") as fp:
        json.dump(checkpoint_metadata, fp, indent=2)

    with open(os.path.join(output_dir, "kernel_calls.json"), "w") as fp:
        json.dump(kernel_calls, fp, indent=2)

    result = {
        "status": "SUCCESS" if converged else "NOT_CONVERGED",
        "method": "LASSCF_COO_TrimCI",
        "phase": "Phase4_production_coo",
        "lasscf_class": "mrh.my_pyscf.mcscf.lasscf_rdm.LASSCF",
        "mo_init": mo_init_label,
        "warm_start_source": init_from,
        "partition": partition,
        "partition_description": partition_description,
        "trimci_threshold": trimci_threshold,
        "coo_cycles": coo_cycles,
        "coo_bfgs_maxiter": coo_bfgs_maxiter,
        "coo_davidson_tol": coo_davidson_tol,
        "coo_bfgs_ftol": coo_bfgs_ftol,
        "warm_start_kappa": warm_start_kappa,
        "parallel_workers": parallel_workers,
        "process_workers": process_workers,
        "omp_threads_per_frag": omp_threads_per_frag,
        "max_cycle_macro": max_cycle_macro,
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
        "n_macro_iters_estimate": n_macro_iters_estimate,
        "dets_per_frag_final": dets_per_frag,
        "wall_time_total": round(wall_time_total, 3),
        "wall_time_per_fragment_aggregated": wall_time_per_fragment_aggregated,
        "output_dir": output_dir,
        "checkpoint_path": output_dir,
        "checkpoint_npz": checkpoint_npz,
        "checkpoint_metadata_json": os.path.join(output_dir, "checkpoint_metadata.json"),
    }

    with open(os.path.join(output_dir, "result.json"), "w") as fp:
        json.dump(result, fp, indent=2)

    headline = (
        f"LASSCF+COO-TrimCI [{partition} thr={trimci_threshold} coo={coo_cycles}] "
        f"e_tot = {e_tot:.6f} Ha | converged={converged} | "
        f"macro iters={n_macro_iters_estimate} | dets/frag={dets_per_frag} | "
        f"wall {wall_time_total:.1f}s"
    )
    print(headline)
    return result


def main():
    today = date.today().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description="LASSCF + COO-TrimCI runner (Phase 4 with orbital optimization)"
    )
    parser.add_argument("--fcidump", default="data/fcidump_cycle_6")
    parser.add_argument("--partition", default="h1diag")
    parser.add_argument("--fragment-orbs-json", default=None)
    parser.add_argument("--trimci-threshold", type=float, default=0.01)
    parser.add_argument("--max-cycle-macro", type=int, default=20)
    parser.add_argument("--trimci-max-dets", default="auto")
    parser.add_argument("--trimci-max-rounds", type=int, default=2)
    parser.add_argument("--coo-cycles", type=int, default=2,
                        help="COO outer cycles per fragment kernel call")
    parser.add_argument("--coo-bfgs-maxiter", type=int, default=20,
                        help="BFGS steps per COO orbital-opt call")
    parser.add_argument("--coo-davidson-tol", type=float, default=1e-7)
    parser.add_argument("--coo-bfgs-ftol", type=float, default=1e-8)
    parser.add_argument(
        "--no-warm-start-kappa", action="store_false", dest="warm_start_kappa",
        help=("Disable warm-starting the COO orbital rotation across LASSCF "
              "macro iters.  By default the kernel caches U_total from the "
              "previous call and pre-rotates the next call's integrals so BFGS "
              "starts near the previous optimum -- this is what makes the per-"
              "call COO gains accumulate.  Disable only for diagnosing whether "
              "warm-start is helping or hurting."),
    )
    parser.set_defaults(warm_start_kappa=True)
    parser.add_argument("--parallel-fragments", type=int, default=0,
                        dest="parallel_workers",
                        help=("Number of fragments to dispatch concurrently inside "
                              "LASSCF's ci_cycle.  0 (default) = serial.  2 is a "
                              "good fit for the 4-fragment H1 partition since F0/F1 "
                              "are 1-determinant and only F2/F3 matter."))
    parser.add_argument("--omp-threads-per-frag", type=int, default=None,
                        dest="omp_threads_per_frag",
                        help=("OpenMP threads each fragment's TrimCI/OrbitalOptimizer "
                              "may use.  None = floor(os.cpu_count() / parallel-fragments). "
                              "Only used when --parallel-fragments > 1."))
    parser.add_argument("--process-fragments", type=int, default=0,
                        dest="process_workers",
                        help=("Number of fragments to dispatch into separate worker "
                              "processes (sidesteps the GIL contention thread-based "
                              "--parallel-fragments hits inside the COO BFGS phase).  "
                              "0 (default) = in-process kernels.  Recommended setting "
                              "for H1 4-frag partition: --process-fragments 4 (one "
                              "worker per fragment) combined with --parallel-fragments 4."))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--init-from", default=None, dest="init_from",
                        metavar="CHECKPOINT_DIR")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(_run_smoke_test(
            parallel_workers=args.parallel_workers,
            process_workers=args.process_workers,
        ))

    trimci_max_dets = ("auto" if args.trimci_max_dets == "auto"
                       else int(args.trimci_max_dets))

    if args.output_dir is None:
        thr_str = f"{args.trimci_threshold:.6f}".rstrip("0").rstrip(".")
        output_dir = (f"Outputs/lasscf/"
                      f"{args.partition}_coo{args.coo_cycles}_thr{thr_str}_{today}")
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
        coo_cycles=args.coo_cycles,
        coo_bfgs_maxiter=args.coo_bfgs_maxiter,
        coo_davidson_tol=args.coo_davidson_tol,
        coo_bfgs_ftol=args.coo_bfgs_ftol,
        warm_start_kappa=args.warm_start_kappa,
        init_from=args.init_from,
        explicit_orbital_lists=explicit_orbital_lists,
        partition_description=partition_description,
        parallel_workers=args.parallel_workers,
        omp_threads_per_frag=args.omp_threads_per_frag,
        process_workers=args.process_workers,
    )

    if not result.get("converged", False):
        print("[run_lasscf_coo] WARNING: LASSCF did not converge.")
        sys.exit(1)


if __name__ == "__main__":
    main()
