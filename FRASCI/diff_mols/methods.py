"""Method adapters: each wraps an existing runner and writes a standardized run-folder."""
from __future__ import annotations

import contextlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from FRASCI.diff_mols.config import MoleculeConfig
from FRASCI.diff_mols.integrals_builder import IntegralBundle
from FRASCI.diff_mols.run_writer import (
    make_run_dir, write_run_outputs, append_runs_index_row, RUNS_INDEX_COLUMNS,
)


@dataclass
class RunResult:
    e_tot: float | None
    n_dets: int | None
    wall_s: float
    run_dir: Path
    result_json: dict
    converged: bool


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@contextlib.contextmanager
def _capture_to_log(log_path: Path):
    """Redirect file descriptor 1 (stdout) AND fd 2 (stderr) at the OS level into log_path.

    Unlike Python's ``contextlib.redirect_stdout`` (which only swaps ``sys.stdout`` and
    leaves libc's stdout going to wherever fd 1 was originally pointing), this uses
    ``os.dup2`` to redirect the file descriptors themselves. The motivation is the
    ipykernel deadlock:

    * In Jupyter, ipykernel's stdout-capture machinery wires fd 1 to a finite-buffer
      pipe drained by a Python-side reader thread.
    * TrimCI/LASSCF emit thousands of ``printf`` lines per run via C++. The pipe fills
      faster than ipykernel can drain it, ``printf`` blocks waiting for buffer space,
      and the kernel hangs at 0% CPU.
    * A Python-level ``redirect_stdout`` doesn't help because C++ printf bypasses
      ``sys.stdout`` entirely — it writes directly to fd 1.

    By swapping fd 1 and fd 2 to point at our log file BEFORE the runner is called,
    TrimCI's printf is written straight into log.txt and never touches ipykernel's
    pipe. After the runner returns, we restore the original fds.

    Side effects:
      * The user does NOT see runner output live in the Jupyter cell during the run.
        They DO see the ▶/✓ banners (printed by ``benchmark.run_all`` BEFORE this
        context is entered).
      * After the run, log.txt has the full runner stdout for post-mortem inspection.
      * ``report.plot_macro_trajectory`` (which parses log.txt for LASSCF macro-iter
        lines) works.
    """
    import os
    log_path = Path(log_path)
    log_fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
        yield
    finally:
        # Flush whatever Python sys.stdout/stderr are pointing at before swapping fds back.
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def _summary_readme(method: str, e_tot: float, n_dets: int, wall_s: float,
                    converged: bool, knobs: dict) -> str:
    return (
        f"# {method} run\n\n"
        f"- e_tot: {e_tot:.6f} Ha\n"
        f"- n_dets: {n_dets}\n"
        f"- wall_s: {wall_s:.2f}\n"
        f"- converged: {converged}\n"
        f"- knobs: {json.dumps(knobs)}\n"
    )


def _save_trimci_dets(dets: list, fcidump_dir: Path, n_orb: int) -> None:
    """Persist TrimCI dets list as dets.npz next to the FCIDUMP (row 0 = correlated reference)."""
    if not dets:
        return
    arr = np.zeros((len(dets), 2), dtype=np.uint64)
    for i, d in enumerate(dets):
        arr[i, 0] = int(d.alpha) if hasattr(d, "alpha") else int(d[0])
        arr[i, 1] = int(d.beta) if hasattr(d, "beta") else int(d[1])
    np.savez_compressed(fcidump_dir / "dets.npz", dets=arr)


def _read_e_ref(bundle: IntegralBundle) -> float | None:
    """Pull the inline-FCI reference energy (if any) from scf_summary.json next to the FCIDUMP.

    Returns None if the integrals were built without inline FCI (e.g., AS too big) or if the
    scf_summary.json file is from an older builder that didn't write the e_fci_inline field.
    """
    try:
        summary = json.loads(bundle.scf_summary_path.read_text())
        ref = summary.get("e_fci_inline")
        return float(ref) if ref is not None else None
    except Exception:
        return None


def _error_mha(e_tot: float | None, e_ref: float | None) -> float | None:
    """Method energy error vs the inline-FCI reference, in milliHartree.

    Positive = method above FCI (i.e., correlation NOT recovered). Negative would indicate
    a non-variational result or a converged-to-wrong-state issue."""
    if e_tot is None or e_ref is None:
        return None
    return (float(e_tot) - float(e_ref)) * 1000.0


def _emit_run(results_root: Path, slug: str, row: dict, bundle: IntegralBundle) -> None:
    """Wrap ``append_runs_index_row`` and auto-inject ``e_ref`` + ``error_mha`` derived
    from the integral bundle's inline FCI reference. One-stop helper so every method
    adapter contributes consistent populated columns."""
    e_ref = _read_e_ref(bundle)
    e_tot_val = row.get("e_tot")
    err_mha = _error_mha(
        float(e_tot_val) if e_tot_val is not None else None,
        e_ref,
    )
    append_runs_index_row(results_root, slug, {**row, "e_ref": e_ref, "error_mha": err_mha})


def _parent_lasscf_metadata(lasscf_run_dir: Path) -> dict:
    """Read parent LASSCF identity from result.json, with folder-layout fallbacks."""
    lasscf_run_dir = Path(lasscf_run_dir)
    result_path = lasscf_run_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            return {
                "method": data.get("method") or lasscf_run_dir.parent.parent.parent.name,
                "partition": data.get("base_partition") or data.get("partition") or lasscf_run_dir.parent.parent.name,
                "protocol": data.get("protocol") or data.get("tag") or lasscf_run_dir.parent.name,
                "geom_tag": data.get("geom_tag") or lasscf_run_dir.parent.parent.parent.parent.name,
                "n_dets_total": data.get("result", {}).get("n_dets_total", 0),
            }
        except Exception:
            pass

    parts = lasscf_run_dir.name.split("__")
    if len(parts) >= 3:
        return {"method": parts[0], "partition": parts[1],
                "protocol": parts[2], "geom_tag": lasscf_run_dir.parent.name}

    # Nested fallback: <geom>/<method>/<partition>/<protocol-axis...>/run_<ts>
    geom_dir = None
    for ancestor in lasscf_run_dir.parents:
        if (ancestor / "integrals").exists():
            geom_dir = ancestor
            break
    if geom_dir is not None:
        rel = lasscf_run_dir.relative_to(geom_dir)
        rel_parts = rel.parts
        if len(rel_parts) >= 4:
            method, partition = rel_parts[0], rel_parts[1]
            protocol = "_".join(rel_parts[2:-1]) or rel_parts[2]
            return {
                "method": method,
                "partition": partition,
                "protocol": protocol,
                "geom_tag": geom_dir.name,
            }

    return {
        "method": lasscf_run_dir.parent.parent.parent.name,
        "partition": lasscf_run_dir.parent.parent.name,
        "protocol": lasscf_run_dir.parent.name,
        "geom_tag": lasscf_run_dir.parent.parent.parent.parent.name,
    }


def run_plain_trimci(
    config: MoleculeConfig,
    bundle: IntegralBundle,
    *,
    results_root: Path,
    tag: str = "default",
    timestamp: str | None = None,
    **overrides,
) -> RunResult:
    """Run vanilla TrimCI on the integral bundle. Writes dets.npz for downstream LASSCF methods."""
    from FRASCI.coo.coo_adapter import run_baseline

    spec = config.methods.plain_trimci
    if not spec.enabled and not overrides.get("force_enable", False):
        raise RuntimeError("plain_trimci.enabled=false in config; pass force_enable=True to override")

    timestamp = timestamp or _ts()
    geom_tag = bundle.geom_tag                      # per-geom layout: bundle carries the tag

    knobs = {
        "threshold": overrides.get("threshold", spec.threshold or 0.01),
        "max_dets": overrides.get("max_dets", spec.max_dets or 200),
        "num_runs": overrides.get("num_runs", spec.num_runs or 1),
        "max_rounds": overrides.get("max_rounds", spec.max_rounds or 2),
    }

    run_dir = make_run_dir(results_root, config.slug, "plain_trimci",
                           "full", geom_tag, tag, timestamp)

    t0 = time.time()
    with _capture_to_log(run_dir / "log.txt"):
        baseline = run_baseline(
            str(bundle.fcidump_path),
            det_budget=knobs["max_dets"],
            trimci_config={
                "threshold": knobs["threshold"],
                "num_runs": knobs["num_runs"],
                "max_rounds": knobs["max_rounds"],
            },
            quiet=False,
        )
    wall = time.time() - t0

    # Persist dets for downstream LASSCF (correlated reference)
    _save_trimci_dets(baseline.dets, bundle.fcidump_path.parent, bundle.n_orb)

    result_json = {
        "molecule": config.slug,
        "method": "plain_trimci",
        "partition": "full",
        "geom_tag": geom_tag,
        "tag": tag,
        "protocol": tag,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "system": {
            "n_orb": bundle.n_orb, "n_elec": bundle.n_elec,
            "e_nuc": bundle.e_nuc, "e_hf": bundle.e_hf,
        },
        "config_snapshot": knobs,
        "result": {
            "e_tot": float(baseline.energy),
            "e_corr": float(baseline.energy - bundle.e_hf),
            "converged": True,
            "n_dets_total": int(baseline.n_dets),
            "wall_time_total": wall,
        },
    }

    write_run_outputs(
        run_dir,
        result_json=result_json,
        config_snapshot=knobs,
        log_text=None,    # log.txt is streamed live by _capture_to_log during the run
        readme_summary=_summary_readme("plain_trimci", baseline.energy,
                                       baseline.n_dets, wall, True, knobs),
    )

    _emit_run(results_root, config.slug, bundle=bundle, row={
        "timestamp": timestamp, "molecule": config.slug, "method": "plain_trimci",
        "stage": "base", "base_partition": "full",
        "partition": "full", "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "e_tot": baseline.energy, "converged": True,
        "n_dets_total": baseline.n_dets, "wall_s": wall,
        "trimci_threshold": knobs["threshold"],
        "trimci_max_dets": knobs["max_dets"],
        "run_dir": str(run_dir),
    })

    return RunResult(
        e_tot=float(baseline.energy), n_dets=int(baseline.n_dets),
        wall_s=wall, run_dir=run_dir, result_json=result_json, converged=True,
    )


def run_trimci_coo(
    config: MoleculeConfig,
    bundle: IntegralBundle,
    *,
    results_root: Path,
    tag: str = "default",
    timestamp: str | None = None,
    **overrides,
) -> RunResult:
    """Run TrimCI + COO outer loop on the FCIDUMP."""
    from FRASCI.coo.coo_adapter import run_outer_loop

    spec = config.methods.trimci_coo
    if not spec.enabled and not overrides.get("force_enable", False):
        raise RuntimeError("trimci_coo.enabled=false in config")

    timestamp = timestamp or _ts()
    geom_tag = bundle.geom_tag

    knobs = {
        "threshold": overrides.get("threshold", spec.threshold or 0.01),
        "max_dets": overrides.get("max_dets", spec.max_dets or 200),
        "coo_cycles": overrides.get("coo_cycles", spec.coo_cycles or 4),
        "bfgs_maxiter": overrides.get("bfgs_maxiter", spec.bfgs_maxiter or 30),
        "bfgs_ftol": overrides.get("bfgs_ftol", spec.bfgs_ftol or 1e-8),
        "davidson_tol": overrides.get("davidson_tol", spec.davidson_tol or 1e-7),
    }

    run_dir = make_run_dir(results_root, config.slug, "trimci_coo",
                           "full", geom_tag, tag, timestamp)

    t0 = time.time()
    with _capture_to_log(run_dir / "log.txt"):
        outer = run_outer_loop(
            str(bundle.fcidump_path),
            det_budget=knobs["max_dets"],
            n_cycles=knobs["coo_cycles"],
            trimci_config={"threshold": knobs["threshold"]},
            opt_config={
                "maxiter": knobs["bfgs_maxiter"],
                "ftol": knobs["bfgs_ftol"],
                "davidson_tol": knobs["davidson_tol"],
            },
            quiet=False,
        )
    wall = time.time() - t0

    # outer.cycles[-1].e_trimci_out is the final energy
    cycles = outer.cycles if hasattr(outer, "cycles") else []
    e_final = float(cycles[-1].e_trimci_out) if cycles else 0.0
    n_dets_final = int(cycles[-1].n_dets_out) if cycles else 0

    result_json = {
        "molecule": config.slug, "method": "trimci_coo", "partition": "full",
        "geom_tag": geom_tag, "tag": tag, "timestamp": timestamp,
        "protocol": tag,
        "run_dir": str(run_dir),
        "system": {"n_orb": bundle.n_orb, "n_elec": bundle.n_elec,
                   "e_nuc": bundle.e_nuc, "e_hf": bundle.e_hf},
        "config_snapshot": knobs,
        "result": {
            "e_tot": e_final, "e_corr": e_final - bundle.e_hf, "converged": True,
            "n_dets_total": n_dets_final, "wall_time_total": wall,
            "n_cycles_run": len(cycles),
            "cycle_trace": [{"e_in": c.e_trimci_in, "e_after_orb": c.e_after_orb,
                             "e_out": c.e_trimci_out, "n_dets_in": c.n_dets_in,
                             "n_dets_out": c.n_dets_out}
                            for c in cycles],
        },
    }

    write_run_outputs(
        run_dir,
        result_json=result_json, config_snapshot=knobs,
        log_text=None,    # log.txt is streamed live by _capture_to_log during the run
        readme_summary=_summary_readme("trimci_coo", e_final, n_dets_final, wall, True, knobs),
    )
    _emit_run(results_root, config.slug, bundle=bundle, row={
        "timestamp": timestamp, "molecule": config.slug, "method": "trimci_coo",
        "stage": "base", "base_partition": "full",
        "partition": "full", "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "e_tot": e_final, "converged": True,
        "n_dets_total": n_dets_final, "wall_s": wall,
        "trimci_threshold": knobs["threshold"], "trimci_max_dets": knobs["max_dets"],
        "coo_cycles": knobs["coo_cycles"], "run_dir": str(run_dir),
    })

    return RunResult(e_tot=e_final, n_dets=n_dets_final, wall_s=wall,
                     run_dir=run_dir, result_json=result_json, converged=True)


def run_lasscf_cas(
    config: MoleculeConfig,
    bundle: IntegralBundle,
    partition_name: str,
    *,
    results_root: Path,
    tag: str = "default",
    timestamp: str | None = None,
    **overrides,
) -> RunResult:
    """Run LASSCF with mrh's CSF/FCI solver per fragment (no TrimCI)."""
    from FRASCI.lasscf.runners.run_lasscf_csf import run as run_csf
    from FRASCI.diff_mols.fragmentation import resolve_fragmentation

    spec = config.methods.lasscf_cas
    if not spec.enabled and not overrides.get("force_enable", False):
        raise RuntimeError("lasscf_cas.enabled=false in config")

    timestamp = timestamp or _ts()
    geom_tag = bundle.geom_tag

    parts = resolve_fragmentation(config, bundle)
    if partition_name not in parts:
        raise KeyError(f"partition {partition_name!r} not in config.fragmentation")
    part = parts[partition_name]

    knobs = {
        "max_cycle_macro": overrides.get("max_cycle_macro", spec.max_cycle_macro or 50),
        "partition": partition_name,
    }

    run_dir = make_run_dir(results_root, config.slug, "lasscf_cas",
                           partition_name, geom_tag, tag, timestamp)

    t0 = time.time()
    with _capture_to_log(run_dir / "log.txt"):
        runner_dict = run_csf(
            fcidump_path=str(bundle.fcidump_path),
            partition=partition_name,
            output_dir=str(run_dir),
            max_cycle=knobs["max_cycle_macro"],
            explicit_orbital_lists=part.orbital_lists,
            partition_description=part.description,
        )
    wall = time.time() - t0

    e_tot = float(runner_dict["e_tot"])
    converged = bool(runner_dict.get("converged", False))

    result_json = {
        "molecule": config.slug, "method": "lasscf_cas", "partition": partition_name,
        "geom_tag": geom_tag, "tag": tag, "protocol": tag, "timestamp": timestamp,
        "run_dir": str(run_dir),
        "system": {"n_orb": bundle.n_orb, "n_elec": bundle.n_elec,
                   "e_nuc": bundle.e_nuc, "e_hf": bundle.e_hf,
                   "ncas_sub": [len(f) for f in part.orbital_lists],
                   "nelec_per_frag": [list(ne) for ne in part.nelec_per_frag],
                   "spin_sub": part.spin_sub,
                   "orbital_lists": part.orbital_lists},
        "config_snapshot": knobs,
        "result": {
            "e_tot": e_tot, "e_corr": e_tot - bundle.e_hf,
            "converged": converged,
            "n_macro_iters": int(runner_dict.get("n_macro_iters") or 0),
            "wall_time_total": wall,
            "runner_inner": runner_dict,
        },
        "links": {"checkpoint_npz": "checkpoint.npz",
                  "checkpoint_metadata_json": "checkpoint_metadata.json"},
    }
    write_run_outputs(
        run_dir, result_json=result_json, config_snapshot=knobs,
        log_text=None,    # log.txt is streamed live by _capture_to_log during the run
        readme_summary=_summary_readme("lasscf_cas", e_tot, 0, wall, converged, knobs),
    )
    _emit_run(results_root, config.slug, bundle=bundle, row={
        "timestamp": timestamp, "molecule": config.slug, "method": "lasscf_cas",
        "stage": "lasscf", "base_partition": partition_name,
        "partition": partition_name, "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "e_tot": e_tot, "converged": converged,
        "wall_s": wall, "max_cycle_macro": knobs["max_cycle_macro"],
        "run_dir": str(run_dir),
    })
    return RunResult(e_tot=e_tot, n_dets=0, wall_s=wall,
                     run_dir=run_dir, result_json=result_json, converged=converged)


def run_lasscf_trimci(
    config: MoleculeConfig,
    bundle: IntegralBundle,
    partition_name: str,
    *,
    results_root: Path,
    tag: str = "default",
    timestamp: str | None = None,
    **overrides,
) -> RunResult:
    """Run LASSCF with vanilla TrimCI kernel per fragment."""
    from FRASCI.lasscf.runners.run_lasscf_trimci import run as run_trimci
    from FRASCI.diff_mols.fragmentation import resolve_fragmentation

    spec = config.methods.lasscf_trimci
    if not spec.enabled and not overrides.get("force_enable", False):
        raise RuntimeError("lasscf_trimci.enabled=false in config")

    timestamp = timestamp or _ts()
    geom_tag = bundle.geom_tag
    parts = resolve_fragmentation(config, bundle)
    if partition_name not in parts:
        raise KeyError(f"partition {partition_name!r} not in config.fragmentation")
    part = parts[partition_name]

    knobs = {
        "trimci_threshold": overrides.get("trimci_threshold", spec.trimci_threshold or 0.01),
        "trimci_max_dets": overrides.get("trimci_max_dets", spec.trimci_max_dets or "auto"),
        "trimci_max_rounds": overrides.get("trimci_max_rounds", spec.max_rounds or 2),
        "max_cycle_macro": overrides.get("max_cycle_macro", spec.max_cycle_macro or 50),
        "partition": partition_name,
    }

    run_dir = make_run_dir(results_root, config.slug, "lasscf_trimci",
                           partition_name, geom_tag, tag, timestamp)

    t0 = time.time()
    with _capture_to_log(run_dir / "log.txt"):
        runner_dict = run_trimci(
            fcidump_path=str(bundle.fcidump_path),
            partition=partition_name,
            trimci_threshold=knobs["trimci_threshold"],
            max_cycle_macro=knobs["max_cycle_macro"],
            output_dir=str(run_dir),
            trimci_max_dets=knobs["trimci_max_dets"],
            trimci_max_rounds=knobs["trimci_max_rounds"],
            explicit_orbital_lists=part.orbital_lists,
            partition_description=part.description,
        )
    wall = time.time() - t0

    e_tot = float(runner_dict["e_tot"])
    converged = bool(runner_dict.get("converged", False))
    dets_per_frag = runner_dict.get("dets_per_frag_final", []) or []
    n_dets_total = int(sum(d or 0 for d in dets_per_frag)) if dets_per_frag else 0

    result_json = {
        "molecule": config.slug, "method": "lasscf_trimci",
        "partition": partition_name, "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "timestamp": timestamp, "run_dir": str(run_dir),
        "system": {"n_orb": bundle.n_orb, "n_elec": bundle.n_elec,
                   "e_nuc": bundle.e_nuc, "e_hf": bundle.e_hf,
                   "ncas_sub": [len(f) for f in part.orbital_lists],
                   "nelec_per_frag": [list(ne) for ne in part.nelec_per_frag],
                   "spin_sub": part.spin_sub,
                   "orbital_lists": part.orbital_lists},
        "config_snapshot": knobs,
        "result": {
            "e_tot": e_tot, "e_corr": e_tot - bundle.e_hf,
            "converged": converged,
            "n_macro_iters": int(runner_dict.get("n_macro_iters_estimate", 0)),
            "dets_per_frag_final": list(map(int, dets_per_frag)) if dets_per_frag else [],
            "n_dets_total": n_dets_total,
            "wall_time_total": wall,
            "runner_inner": runner_dict,
        },
        "links": {"checkpoint_npz": "checkpoint.npz",
                  "checkpoint_metadata_json": "checkpoint_metadata.json",
                  "kernel_calls_json": "kernel_calls.json"},
    }
    write_run_outputs(
        run_dir, result_json=result_json, config_snapshot=knobs,
        log_text=None,
        readme_summary=_summary_readme("lasscf_trimci", e_tot, n_dets_total,
                                       wall, converged, knobs),
    )
    _emit_run(results_root, config.slug, bundle=bundle, row={
        "timestamp": timestamp, "molecule": config.slug, "method": "lasscf_trimci",
        "stage": "lasscf", "base_partition": partition_name,
        "partition": partition_name, "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "e_tot": e_tot, "converged": converged,
        "n_dets_total": n_dets_total, "wall_s": wall,
        "trimci_threshold": knobs["trimci_threshold"],
        "trimci_max_dets": knobs["trimci_max_dets"] if knobs["trimci_max_dets"] != "auto" else "",
        "trimci_max_rounds": knobs["trimci_max_rounds"],
        "max_cycle_macro": knobs["max_cycle_macro"],
        "run_dir": str(run_dir),
    })
    return RunResult(e_tot=e_tot, n_dets=n_dets_total, wall_s=wall,
                     run_dir=run_dir, result_json=result_json, converged=converged)


def run_lasscf_trimci_coo(
    config: MoleculeConfig,
    bundle: IntegralBundle,
    partition_name: str,
    *,
    results_root: Path,
    tag: str = "default",
    timestamp: str | None = None,
    **overrides,
) -> RunResult:
    """Run LASSCF with COO-enabled TrimCI kernel per fragment ('the best')."""
    from FRASCI.lasscf.runners.run_lasscf_coo import run as run_coo
    from FRASCI.diff_mols.fragmentation import resolve_fragmentation

    spec = config.methods.lasscf_trimci_coo
    if not spec.enabled and not overrides.get("force_enable", False):
        raise RuntimeError("lasscf_trimci_coo.enabled=false in config")

    timestamp = timestamp or _ts()
    geom_tag = bundle.geom_tag
    parts = resolve_fragmentation(config, bundle)
    if partition_name not in parts:
        raise KeyError(f"partition {partition_name!r} not in config.fragmentation")
    part = parts[partition_name]

    knobs = {
        "trimci_threshold": overrides.get("trimci_threshold", spec.trimci_threshold or 0.01),
        "trimci_max_dets": overrides.get("trimci_max_dets", spec.trimci_max_dets or "auto"),
        "trimci_max_rounds": overrides.get("trimci_max_rounds", spec.max_rounds or 2),
        "max_cycle_macro": overrides.get("max_cycle_macro", spec.max_cycle_macro or 50),
        "coo_cycles": overrides.get("coo_cycles", spec.coo_cycles or 2),
        "coo_bfgs_maxiter": overrides.get("bfgs_maxiter", spec.bfgs_maxiter or 20),
        "coo_bfgs_ftol": overrides.get("bfgs_ftol", spec.bfgs_ftol or 1e-8),
        "coo_davidson_tol": overrides.get("davidson_tol", spec.davidson_tol or 1e-7),
        "warm_start_kappa": overrides.get(
            "warm_start_kappa",
            True if spec.warm_start_kappa is None else spec.warm_start_kappa,
        ),
        "parallel_workers": overrides.get("parallel_workers", spec.parallel_workers or 0),
        "process_workers": overrides.get("process_workers", spec.process_workers or 0),
        "omp_threads_per_frag": overrides.get("omp_threads_per_frag", spec.omp_threads_per_frag),
        "partition": partition_name,
    }

    run_dir = make_run_dir(results_root, config.slug, "lasscf_trimci_coo",
                           partition_name, geom_tag, tag, timestamp)

    t0 = time.time()
    with _capture_to_log(run_dir / "log.txt"):
        runner_dict = run_coo(
            fcidump_path=str(bundle.fcidump_path),
            partition=partition_name,
            trimci_threshold=knobs["trimci_threshold"],
            max_cycle_macro=knobs["max_cycle_macro"],
            output_dir=str(run_dir),
            trimci_max_dets=knobs["trimci_max_dets"],
            trimci_max_rounds=knobs["trimci_max_rounds"],
            coo_cycles=knobs["coo_cycles"],
            coo_bfgs_maxiter=knobs["coo_bfgs_maxiter"],
            coo_bfgs_ftol=knobs["coo_bfgs_ftol"],
            coo_davidson_tol=knobs["coo_davidson_tol"],
            warm_start_kappa=knobs["warm_start_kappa"],
            explicit_orbital_lists=part.orbital_lists,
            partition_description=part.description,
            parallel_workers=knobs["parallel_workers"],
            process_workers=knobs["process_workers"],
            omp_threads_per_frag=knobs["omp_threads_per_frag"],
        )
    wall = time.time() - t0

    e_tot = float(runner_dict["e_tot"])
    converged = bool(runner_dict.get("converged", False))
    dets_per_frag = runner_dict.get("dets_per_frag_final", []) or []

    n_dets_total = int(sum(d or 0 for d in dets_per_frag)) if dets_per_frag else 0

    result_json = {
        "molecule": config.slug, "method": "lasscf_trimci_coo",
        "partition": partition_name, "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "timestamp": timestamp, "run_dir": str(run_dir),
        "system": {"n_orb": bundle.n_orb, "n_elec": bundle.n_elec,
                   "e_nuc": bundle.e_nuc, "e_hf": bundle.e_hf,
                   "ncas_sub": [len(f) for f in part.orbital_lists],
                   "nelec_per_frag": [list(ne) for ne in part.nelec_per_frag],
                   "spin_sub": part.spin_sub,
                   "orbital_lists": part.orbital_lists},
        "config_snapshot": knobs,
        "result": {
            "e_tot": e_tot, "e_corr": e_tot - bundle.e_hf,
            "converged": converged,
            "n_macro_iters": int(runner_dict.get("n_macro_iters_estimate", 0)),
            "dets_per_frag_final": list(map(int, dets_per_frag)) if dets_per_frag else [],
            "n_dets_total": n_dets_total,
            "wall_time_total": wall,
            "runner_inner": runner_dict,
        },
        "links": {"checkpoint_npz": "checkpoint.npz",
                  "checkpoint_metadata_json": "checkpoint_metadata.json",
                  "kernel_calls_json": "kernel_calls.json"},
    }
    write_run_outputs(
        run_dir, result_json=result_json, config_snapshot=knobs,
        log_text=None,    # log.txt is streamed live by _capture_to_log during the run
        readme_summary=_summary_readme("lasscf_trimci_coo", e_tot, n_dets_total,
                                       wall, converged, knobs),
    )
    _emit_run(results_root, config.slug, bundle=bundle, row={
        "timestamp": timestamp, "molecule": config.slug, "method": "lasscf_trimci_coo",
        "stage": "lasscf", "base_partition": partition_name,
        "partition": partition_name, "geom_tag": geom_tag, "tag": tag,
        "protocol": tag,
        "e_tot": e_tot, "converged": converged,
        "n_dets_total": n_dets_total, "wall_s": wall,
        "trimci_threshold": knobs["trimci_threshold"],
        "trimci_max_dets": knobs["trimci_max_dets"] if knobs["trimci_max_dets"] != "auto" else "",
        "trimci_max_rounds": knobs["trimci_max_rounds"],
        "coo_cycles": knobs["coo_cycles"],
        "coo_bfgs_maxiter": knobs["coo_bfgs_maxiter"],
        "coo_bfgs_ftol": knobs["coo_bfgs_ftol"],
        "coo_davidson_tol": knobs["coo_davidson_tol"],
        "warm_start_kappa": knobs["warm_start_kappa"],
        "parallel_workers": knobs["parallel_workers"],
        "process_workers": knobs["process_workers"],
        "omp_threads_per_frag": "" if knobs["omp_threads_per_frag"] is None else knobs["omp_threads_per_frag"],
        "max_cycle_macro": knobs["max_cycle_macro"],
        "run_dir": str(run_dir),
    })
    return RunResult(e_tot=e_tot, n_dets=n_dets_total, wall_s=wall,
                     run_dir=run_dir, result_json=result_json, converged=converged)


def run_lassis(
    config: MoleculeConfig,
    bundle: IntegralBundle,
    lasscf_run_dir: Path,
    *,
    results_root: Path,
    tag: str = "default",
    timestamp: str | None = None,
    **overrides,
) -> RunResult:
    """Run LASSIS on top of a LASSCF checkpoint."""
    from FRASCI.lasscf.runners.run_lassi_lassis import run as run_lassi
    from FRASCI.diff_mols.lassis_states import build_lassis_kwargs

    spec = config.methods.lassis
    if not spec.enabled and not overrides.get("force_enable", False):
        raise RuntimeError("lassis.enabled=false in config")

    timestamp = timestamp or _ts()
    geom_tag = bundle.geom_tag

    lasscf_run_dir = Path(lasscf_run_dir)
    parent_meta = _parent_lasscf_metadata(lasscf_run_dir)
    parent_method = parent_meta["method"]
    partition_name = parent_meta["partition"]
    parent_protocol = parent_meta["protocol"]
    parent_n_dets = int(parent_meta.get("n_dets_total") or 0)

    base_kwargs = build_lassis_kwargs(config, lasscf_run_dir)
    knobs = {**base_kwargs, **{k: v for k, v in overrides.items()
                               if k in ("lassis_ncharge", "lassis_nspin", "opt")}}
    lassis_protocol = overrides.get("lassis_protocol", tag)

    display_method = f"lassis_on_{parent_method}"
    run_dir = make_run_dir(results_root, config.slug, display_method,
                           partition_name, geom_tag, tag, timestamp)

    t0 = time.time()
    with _capture_to_log(run_dir / "log.txt"):
        runner_dict = run_lassi(
            checkpoint_dir=str(lasscf_run_dir),
            fcidump_path=str(bundle.fcidump_path),
            output_dir=str(run_dir),
            # The benchmark measures LASSIS on each LASSCF parent. Building a
            # separate explicit LASSI CT space is unrelated and can generate
            # impossible fragment occupations for small partitions.
            skip_lassi=True,
            skip_lassis=False,
            opt=knobs["opt"],
            lassis_ncharge=knobs["lassis_ncharge"],
            lassis_nspin=knobs["lassis_nspin"],
        )
    wall = time.time() - t0

    # Runner summary dict keys: e_lassis, e_lassi, e_lasci (NOT lassis_e_ground / lassi_e_ground).
    # Prefer LASSIS; fall back to LASSI; then LASCI; then 0.0.
    _e_lassis = runner_dict.get("e_lassis")
    _e_lassi = runner_dict.get("e_lassi")
    _e_lasci = runner_dict.get("e_lasci")
    if _e_lassis is not None:
        e_tot = float(_e_lassis)
        _energy_key = "e_lassis"
    elif _e_lassi is not None:
        e_tot = float(_e_lassi)
        _energy_key = "e_lassi"
    elif _e_lasci is not None:
        e_tot = float(_e_lasci)
        _energy_key = "e_lasci"
    else:
        e_tot = float(runner_dict.get("e_tot", 0.0))
        _energy_key = "e_tot"
    converged = _e_lassis is not None

    result_json = {
        "molecule": config.slug, "method": "lassis",
        "stage": "lassis",
        "parent_method": parent_method,
        "base_partition": partition_name,
        "partition": f"{partition_name}_on_{parent_method}",
        "display_method": display_method,
        "parent_protocol": parent_protocol,
        "n_dets_total": parent_n_dets,
        "lassis_protocol": lassis_protocol,
        "protocol": tag,
        "geom_tag": geom_tag, "tag": tag, "timestamp": timestamp,
        "run_dir": str(run_dir),
        "system": {"n_orb": bundle.n_orb, "n_elec": bundle.n_elec,
                   "e_nuc": bundle.e_nuc, "e_hf": bundle.e_hf,
                   "parent_lasscf_run_dir": str(lasscf_run_dir)},
        "config_snapshot": knobs,
        "result": {
            "e_tot": e_tot, "e_corr": e_tot - bundle.e_hf,
            "converged": converged, "wall_time_total": wall,
            "n_dets_total": parent_n_dets,
            "energy_key_used": _energy_key,
            "runner_inner": runner_dict,
        },
    }
    write_run_outputs(
        run_dir, result_json=result_json, config_snapshot=knobs,
        log_text=None,    # log.txt is streamed live by _capture_to_log during the run
        readme_summary=_summary_readme("lassis", e_tot, parent_n_dets, wall, converged, knobs),
    )
    _emit_run(results_root, config.slug, bundle=bundle, row={
        "timestamp": timestamp, "molecule": config.slug, "method": "lassis",
        "stage": "lassis", "parent_method": parent_method,
        "base_partition": partition_name,
        "partition": f"{partition_name}_on_{parent_method}",
        "geom_tag": geom_tag, "tag": tag,
        "protocol": tag, "parent_protocol": parent_protocol,
        "lassis_protocol": lassis_protocol,
        "e_tot": e_tot, "converged": converged,
        "wall_s": wall, "n_spin": knobs["lassis_nspin"],
        "n_dets_total": parent_n_dets,
        "run_dir": str(run_dir),
    })
    return RunResult(e_tot=e_tot, n_dets=parent_n_dets, wall_s=wall,
                     run_dir=run_dir, result_json=result_json, converged=converged)
