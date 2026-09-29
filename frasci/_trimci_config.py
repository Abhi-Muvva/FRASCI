"""Map project knobs onto ``trimci.Config``, and silence TrimCI's C++/fd-level
stdout+stderr. Port of engine/coo/phased.py's build_coo_config/suppress_fd --
new module boundary, same logic. engine.coo.phased.normalize_result and
run_phased_fcidump are NOT ported: nothing in frasci imports them.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Optional

import trimci


@contextlib.contextmanager
def suppress_fd(quiet: bool = True):
    """Silence TrimCI's C++/fd-level stdout+stderr (it bypasses Python's
    ``redirect_stdout``).  ``quiet=False`` is a no-op passthrough."""
    if not quiet:
        yield
        return
    # Flush buffered Python stdout/stderr BEFORE the dup2, else pending prints get
    # flushed into /dev/null (silently lost) once fd 1/2 point there.
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        # Flush anything the block wrote (so it lands in /dev/null, not after restore).
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


def build_coo_config(
    *,
    threshold: Optional[float] = None,
    num_runs: Optional[int] = None,
    orbopt: Optional[bool] = None,
    orbopt_cycles: Optional[int] = None,
    save_fcidumps: Optional[bool] = None,
    max_dets_phase0: Optional[int] = None,
    enable_phase1: Optional[bool] = None,
    max_dets_phase1: Optional[int] = None,
    growth_factor_phase1: Optional[float] = None,
    enable_phase2: Optional[bool] = None,
    max_dets_phase2: Optional[int] = None,
    growth_factor_phase2: Optional[float] = None,
    pt2: Optional[bool] = None,
    pt2_only_last: Optional[bool] = None,
    seed: Optional[int] = None,
    verbosity: int = 0,
    overrides: Optional[dict] = None,
) -> "trimci.Config":
    """Map project knobs -> ``trimci.Config``.

    Only non-``None`` knobs are set; everything else keeps ``trimci.Config``'s
    paper-default (Tables S1-S4).  So callers can forward config fields straight
    through and get the paper protocol for anything they leave unset.

    ``pt2`` maps to the Config field ``pt2_correction``.  ``overrides`` is a
    dict of any advanced ``Config`` field (e.g. ``phase0_overrides``,
    ``n_cpu_workers``, ``backend_phase2``) applied verbatim after construction.
    """
    fields = dict(
        threshold=threshold,
        num_runs=num_runs,
        orbopt=orbopt,
        orbopt_cycles=orbopt_cycles,
        save_fcidumps=save_fcidumps,
        max_dets_phase0=max_dets_phase0,
        enable_phase1=enable_phase1,
        max_dets_phase1=max_dets_phase1,
        growth_factor_phase1=growth_factor_phase1,
        enable_phase2=enable_phase2,
        max_dets_phase2=max_dets_phase2,
        growth_factor_phase2=growth_factor_phase2,
        pt2_correction=pt2,
        pt2_only_last=pt2_only_last,
        seed=seed,
        verbosity=verbosity,
    )
    fields = {k: v for k, v in fields.items() if v is not None}
    cfg = trimci.Config(**fields)
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    return cfg
