"""
coo_adapter.py
==============

Thin wrappers around ``trimci.run_full_calculation`` (vanilla TrimCI) and
``trimci.orblab.OrbitalOptimizer`` (the COO inner block) so that a notebook
can step through Core-Optimized Orbitals one cycle at a time.

Public entry points
-------------------
* ``run_baseline(...)``      -- vanilla TrimCI, no orbital rotation.
* ``run_outer_loop(...)``    -- manual COO loop with per-cycle trace.
* ``run_end_to_end(...)``    -- passthrough to ``trimci.run_full_calculation``
                                with ``orbital_optimization=True``.

The manual loop is the one the notebook actually uses for visualisation;
the end-to-end version is included as a sanity reference.

Conventions
-----------
* All integrals are in CHEMIST notation: ``eri[p,q,r,s] = (pq|rs)``.
* ``e_nuc`` is taken from the FCIDUMP (Fe4S4 has it absorbed -> ``0.0``).
* TrimCI's stdout is *very* noisy; ``silent(...)`` is provided so notebooks
  can capture both stdout and stderr without it leaking into cell output.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import trimci
from pyscf.tools.fcidump import from_integrals
from trimci.orblab import OrbitalOptimizer


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BaselineResult:
    """Outcome of a vanilla TrimCI run (no orbital optimisation)."""
    energy: float
    n_dets: int
    dets: list
    coeffs: list
    wall_s: float
    config: dict
    details: dict = field(default_factory=dict)


@dataclass
class COOCycle:
    """One outer iteration of Core-Optimized Orbitals.

    Records (a) the TrimCI energy at the start of the cycle in the current
    orbital basis, (b) the energy after the orbital optimization is applied
    to *the same* determinant set (this is the variational drop attributable
    purely to the basis change), and (c) the energy after re-running TrimCI
    in the new basis (which can wander because the next pool build sees a
    different Hamiltonian).
    """
    cycle: int
    n_dets_in: int                # determinants going into the cycle
    e_trimci_in: float            # E from the TrimCI solve in cycle's input basis
    e_after_orb: float            # E from OrbitalOptimizer.optimize at fixed det set
    e_trimci_out: float           # E from TrimCI re-solve in rotated basis
    n_dets_out: int               # determinants found in rotated basis
    bfgs_iters: int               # BFGS steps that actually ran
    converged_orb: bool           # OrbitalOptimizer convergence flag
    grad_norm: float | None       # ||grad||_inf if reported, else None
    wall_orb_s: float             # wall time for the OrbitalOptimizer call
    wall_trimci_s: float          # wall time for the TrimCI re-solve
    U_step: np.ndarray            # (n_orb, n_orb) rotation applied this cycle
    U_total: np.ndarray           # (n_orb, n_orb) cumulative rotation since start


@dataclass
class COOOuterLoopResult:
    """Full manual COO run with per-cycle trace."""
    fcidump_path: str
    n_orb: int
    n_elec: int
    e_nuc: float
    config: dict
    final_energy: float
    final_n_dets: int
    final_h1: np.ndarray
    final_eri: np.ndarray
    final_dets: list
    final_coeffs: list
    U_total: np.ndarray
    cycles: list[COOCycle] = field(default_factory=list)
    total_wall_s: float = 0.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def silent(capture: bool = True):
    """Suppress TrimCI's stdout/stderr.  Yields a StringIO of the trapped log."""
    if not capture:
        yield None
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def _write_temp_fcidump(h1: np.ndarray, eri: np.ndarray, n_orb: int,
                       n_elec: int, ms: int, e_nuc: float) -> str:
    """Write integrals to a temp FCIDUMP and return the path."""
    fd, path = tempfile.mkstemp(suffix=".fcidump", prefix="coo_")
    os.close(fd)
    from_integrals(path, h1, eri, nmo=n_orb, nelec=n_elec, nuc=e_nuc, ms=ms)
    return path


def _trimci_config_defaults(det_budget: int, overrides: Optional[dict]) -> dict:
    """Build a TrimCI config_dict suitable for a single fragment-style run."""
    base = {
        "max_final_dets": det_budget,
        "threshold": 0.01,
        "num_runs": 1,
        "max_rounds": 2,
        "verbose": False,
    }
    if overrides:
        base.update(overrides)
    return base


def _opt_config_defaults(overrides: Optional[dict]) -> dict:
    """OrbitalOptimizer options suited to a single COO cycle."""
    base = {
        "optimizer": "cpp_bfgs",
        "gradient_mode": "analytical",
        "maxiter": 30,
        "davidson_tol": 1e-7,
        "ftol": 1e-8,
        "verbose": False,
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Baseline TrimCI
# ---------------------------------------------------------------------------

def run_baseline(fcidump_path: str, *, det_budget: int = 200,
                 trimci_config: Optional[dict] = None,
                 quiet: bool = True) -> BaselineResult:
    """Run vanilla TrimCI on ``fcidump_path`` -- no orbital optimisation."""
    cfg = _trimci_config_defaults(det_budget, trimci_config)
    t0 = time.time()
    with silent(quiet):
        energy, dets, coeffs, details, _ = trimci.run_full_calculation(
            fcidump_path=fcidump_path,
            config_dict=cfg,
        )
    wall = time.time() - t0
    return BaselineResult(
        energy=float(energy),
        n_dets=len(dets),
        dets=list(dets),
        coeffs=list(coeffs),
        wall_s=wall,
        config=cfg,
        details=details if isinstance(details, dict) else {},
    )


# ---------------------------------------------------------------------------
# Manual COO outer loop
# ---------------------------------------------------------------------------

def _run_trimci_from_integrals(h1: np.ndarray, eri: np.ndarray, *, n_orb: int,
                              n_elec: int, n_alpha: int, n_beta: int,
                              e_nuc: float, config: dict, quiet: bool):
    """Helper: run TrimCI on in-memory integrals via a temp FCIDUMP."""
    ms = n_alpha - n_beta
    path = _write_temp_fcidump(h1, eri, n_orb, n_elec, ms, e_nuc)
    try:
        with silent(quiet):
            energy, dets, coeffs, details, _ = trimci.run_full_calculation(
                fcidump_path=path,
                config_dict=config,
            )
        return float(energy), list(dets), list(coeffs), details
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _orb_grad_norm_from_trace(opt: OrbitalOptimizer) -> float | None:
    """Best-effort extraction of the final gradient norm from the optimizer."""
    for attr in ("final_grad_norm", "grad_norm", "last_grad_norm"):
        if hasattr(opt, attr):
            try:
                return float(getattr(opt, attr))
            except (TypeError, ValueError):
                continue
    trace = getattr(opt, "trace_data", None) or getattr(opt, "optimization_trace", None)
    if isinstance(trace, dict):
        for key in ("grad_norm", "gradient_norm"):
            if key in trace and trace[key]:
                try:
                    return float(trace[key][-1])
                except (TypeError, ValueError, IndexError):
                    continue
    return None


def run_outer_loop(fcidump_path: str, *, det_budget: int = 200,
                  n_cycles: int = 4,
                  trimci_config: Optional[dict] = None,
                  opt_config: Optional[dict] = None,
                  early_stop_dE: float | None = 1e-5,
                  quiet: bool = True,
                  progress_callback=None) -> COOOuterLoopResult:
    """Run a manual COO outer loop with per-cycle trace.

    Cycle ``i`` does (in this order):
      a. TrimCI solve in the current basis      -> ``(dets, coeffs, e_in)``
      b. OrbitalOptimizer on that fixed det set -> ``U_step, e_after_orb``
      c. Apply ``U_step`` to integrals          -> ``(h1', eri')``
      d. Re-run TrimCI in the new basis         -> ``(dets', coeffs', e_out)``

    ``e_in`` for cycle 0 is the baseline TrimCI energy.  ``e_after_orb``
    minus ``e_in`` measures the variational drop attributable purely to the
    orbital rotation; ``e_out`` minus ``e_after_orb`` measures whatever
    additional gain comes from re-running the determinant search.

    If ``early_stop_dE`` is not ``None``, the loop stops as soon as
    ``|e_out - e_in| < early_stop_dE``.

    ``progress_callback(cycle_dict)`` is invoked at the end of each cycle
    with a dict describing the cycle -- useful for live logging in notebooks.
    """
    # 1. Read FCIDUMP and run the very first TrimCI in the input basis.
    h1, eri, n_elec, n_orb, e_nuc, n_alpha, n_beta, _psym = trimci.read_fcidump(fcidump_path)
    trimci_cfg = _trimci_config_defaults(det_budget, trimci_config)
    opt_cfg = _opt_config_defaults(opt_config)

    t_total = time.time()

    # Initial TrimCI on the unrotated FCIDUMP -- this is the starting point.
    with silent(quiet):
        e_in, dets, coeffs, details0, _ = trimci.run_full_calculation(
            fcidump_path=fcidump_path, config_dict=trimci_cfg,
        )
    e_in = float(e_in); dets = list(dets); coeffs = list(coeffs)
    n_dets = len(dets)

    U_total = np.eye(n_orb)
    cycles: list[COOCycle] = []

    h1_curr, eri_curr = h1.copy(), eri.copy()

    for k in range(n_cycles):
        # (a) is `e_in / dets / coeffs` from the previous cycle (or the seed run).
        # (b) Orbital optimization on the fixed determinant set.
        opt = OrbitalOptimizer(n_orb=n_orb, n_elec=n_elec,
                               mol_name=f"coo_cycle_{k}", verbose=False)
        opt.nuclear_repulsion = e_nuc

        t1 = time.time()
        with silent(quiet):
            h1_rot, eri_rot, e_after_orb, conv, U_step = opt.optimize(
                h1_curr, eri_curr, dets, coeffs,
                optimizer_options_dict=opt_cfg,
            )
        wall_orb = time.time() - t1
        bfgs_iters = int(getattr(opt, "n_iter", opt_cfg.get("maxiter", 0)))
        grad_norm = _orb_grad_norm_from_trace(opt)

        # (c, d) Re-run TrimCI in the new basis.
        t2 = time.time()
        e_out, dets_out, coeffs_out, details_out = _run_trimci_from_integrals(
            h1_rot, eri_rot,
            n_orb=n_orb, n_elec=n_elec, n_alpha=n_alpha, n_beta=n_beta,
            e_nuc=e_nuc, config=trimci_cfg, quiet=quiet,
        )
        wall_trimci = time.time() - t2

        U_total = U_step @ U_total
        cycles.append(COOCycle(
            cycle=k,
            n_dets_in=n_dets,
            e_trimci_in=e_in,
            e_after_orb=float(e_after_orb),
            e_trimci_out=e_out,
            n_dets_out=len(dets_out),
            bfgs_iters=bfgs_iters,
            converged_orb=bool(conv),
            grad_norm=grad_norm,
            wall_orb_s=wall_orb,
            wall_trimci_s=wall_trimci,
            U_step=U_step.copy(),
            U_total=U_total.copy(),
        ))

        if progress_callback is not None:
            progress_callback({
                "cycle": k,
                "e_trimci_in": e_in,
                "e_after_orb": float(e_after_orb),
                "e_trimci_out": e_out,
                "dE_orb_mHa": (float(e_after_orb) - e_in) * 1000,
                "dE_redetect_mHa": (e_out - float(e_after_orb)) * 1000,
                "n_dets_in": n_dets, "n_dets_out": len(dets_out),
                "wall_orb_s": wall_orb, "wall_trimci_s": wall_trimci,
                "bfgs_iters": bfgs_iters, "converged_orb": bool(conv),
            })

        # Roll forward.
        if early_stop_dE is not None and abs(e_out - e_in) < early_stop_dE and k > 0:
            h1_curr, eri_curr = h1_rot, eri_rot
            dets, coeffs = dets_out, coeffs_out
            e_in = e_out
            n_dets = len(dets_out)
            break

        h1_curr, eri_curr = h1_rot, eri_rot
        dets, coeffs = dets_out, coeffs_out
        e_in = e_out
        n_dets = len(dets_out)

    return COOOuterLoopResult(
        fcidump_path=fcidump_path,
        n_orb=n_orb,
        n_elec=n_elec,
        e_nuc=float(e_nuc),
        config={"trimci": trimci_cfg, "opt": opt_cfg, "n_cycles": n_cycles,
                "det_budget": det_budget, "early_stop_dE": early_stop_dE},
        final_energy=e_in,
        final_n_dets=n_dets,
        final_h1=h1_curr,
        final_eri=eri_curr,
        final_dets=dets,
        final_coeffs=coeffs,
        U_total=U_total,
        cycles=cycles,
        total_wall_s=time.time() - t_total,
    )


# ---------------------------------------------------------------------------
# End-to-end driver (built-in TrimCI orbital_optimization=True)
# ---------------------------------------------------------------------------

def run_end_to_end(fcidump_path: str, *, det_budget: int = 200,
                   n_cycles: int = 5,
                   trimci_config: Optional[dict] = None,
                   opt_config: Optional[dict] = None,
                   quiet: bool = True) -> dict:
    """Run ``trimci.run_full_calculation`` with ``orbital_optimization=True``.

    Returns a dict with ``energy``, ``n_dets``, ``wall_s``, ``details``
    (the TrimCI driver's own details dict, which includes a per-iteration
    trace of the *inner* selected-CI expansion, not the outer COO cycles).
    """
    trimci_cfg = _trimci_config_defaults(det_budget, trimci_config)
    opt_cfg = _opt_config_defaults(opt_config)
    opt_cfg["cycles"] = n_cycles  # the driver consumes this as the # of COO cycles

    cfg = dict(trimci_cfg)
    cfg["orbital_optimization"] = True
    cfg["optimizer_options_dict"] = opt_cfg

    t0 = time.time()
    with silent(quiet):
        energy, dets, coeffs, details, _ = trimci.run_full_calculation(
            fcidump_path=fcidump_path, config_dict=cfg,
        )
    return {
        "energy": float(energy),
        "n_dets": len(dets),
        "dets": list(dets),
        "coeffs": list(coeffs),
        "details": details if isinstance(details, dict) else {},
        "wall_s": time.time() - t0,
        "config": cfg,
    }
