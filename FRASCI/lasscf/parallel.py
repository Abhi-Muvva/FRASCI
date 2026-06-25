"""
parallel.py
===========
Opt-in parallelisation of the per-fragment CI dispatch inside mrh's LASSCF.

mrh's stock ``ci_cycle`` in ``mrh/my_pyscf/mcscf/lasscf_sync_o0.py`` iterates
fragments serially:

    for isub, (fcibox, ncas, nelecas, h1e, fcivec) in enumerate(zip(...)):
        ...
        e_sub, fcivec = fcibox.kernel(h1e, eri_cas, ncas, nelecas, ...)

Each call is independent given the current ``mo_coeff`` -- fragments don't share
mutable state during the CI step.  TrimCI and ``OrbitalOptimizer`` are C++ that
release the GIL during their work, so Python threading gives real parallelism.

This module exposes a single context manager,
``parallel_fragments(n_workers, omp_threads_per_worker)``, that:

  1. Snapshots ``lasscf_sync_o0.ci_cycle``.
  2. Replaces it with a ThreadPoolExecutor-driven version.
  3. Caps numpy/MKL/OpenMP threads via ``threadpoolctl.threadpool_limits`` and
     a temporary ``OMP_NUM_THREADS`` env override (TrimCI checks the env on
     first parallel region; the override is restored on exit).
  4. Restores the original ``ci_cycle`` and thread settings on exit.

Default behaviour of the runner is **serial**; this module activates only when
the runner explicitly calls into it (``run_lasscf_coo --parallel-fragments N``).

Why threading and not multiprocessing
-------------------------------------
Each fragment's ``(h1, eri)`` is a numpy view into the full LASSCF state.
Processes would copy these on fork/spawn (50+ MB per fragment for Fe4S4 -- not
huge but wasteful) and IPC the RDMs back.  Threading keeps everything in-process
and is fine because the inner numerical kernels (TrimCI, OrbitalOptimizer,
PySCF's ``direct_spin1.make_rdm12s``) all release the GIL.
"""

from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from pyscf import lib

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover -- threadpoolctl is a soft dep
    threadpool_limits = None


# ---------------------------------------------------------------------------
# Parallel ci_cycle replacement
# ---------------------------------------------------------------------------

def _build_ci_cycle_parallel(n_workers: int):
    """Return a parallel-fragment ci_cycle closure with ``n_workers``."""
    def _ci_cycle_parallel(las, mo, ci0, veff, h2eff_sub, casdm1frs, log):
        # --- Setup, identical to mrh's stock implementation ---
        if ci0 is None:
            ci0 = [None for _ in range(las.nfrags)]
        frozen_ci = las.frozen_ci if las.frozen_ci is not None else []

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        h1eff_sub = las.get_h1eff(mo, veff=veff, h2eff_sub=h2eff_sub,
                                  casdm1frs=casdm1frs)
        ncas_cum = np.cumsum([0] + las.ncas_sub.tolist()) + las.ncore
        orbsym_root = getattr(mo, 'orbsym', None)

        # --- Build per-fragment payload (cheap, serial) ---
        max_memory_base = max(400, las.max_memory - lib.current_memory()[0])
        per_frag = []  # list of (isub, fcibox, h1e, eri_cas, ncas, nelecas, fcivec, orbsym_sub)
        for isub, (fcibox, ncas, nelecas, h1e, fcivec) in enumerate(
            zip(las.fciboxes, las.ncas_sub, las.nelecas_sub, h1eff_sub, ci0)
        ):
            eri_cas = las.get_h2eff_slice(h2eff_sub, isub, compact=8)
            orbsym_sub = None
            if orbsym_root is not None:
                i = ncas_cum[isub]; j = ncas_cum[isub + 1]
                orbsym_sub = orbsym_root[i:j]
                log.info("LASSCF subspace %d (parallel) with orbsyms %s",
                         isub, str(orbsym_sub))
            else:
                log.info("LASSCF subspace %d (parallel) with no orbsym info", isub)
            per_frag.append((isub, fcibox, h1e, eri_cas, ncas, nelecas,
                             fcivec, orbsym_sub))

        # --- Submit non-frozen fragments to the thread pool ---
        e_cas = [0.0] * las.nfrags
        ci1: list = [None] * las.nfrags
        e0 = 0.0  # mrh's stock uses ecore=e0 with e0=0; we preserve that

        def _run_one(payload):
            isub, fcibox, h1e, eri_cas, ncas, nelecas, fcivec, orbsym_sub = payload
            t0 = time.perf_counter()
            e_sub, fcivec_out = fcibox.kernel(
                h1e, eri_cas, ncas, nelecas,
                ci0=fcivec, verbose=log, max_memory=max_memory_base,
                ecore=e0, orbsym=orbsym_sub,
            )
            return isub, e_sub, fcivec_out, time.perf_counter() - t0

        # Frozen fragments stay sequential (zero work anyway)
        for isub, _, _, _, _, _, fcivec, _ in [p for p in per_frag if p[0] in frozen_ci]:
            e_cas[isub] = 0.0
            ci1[isub] = fcivec

        active = [p for p in per_frag if p[0] not in frozen_ci]

        if active:
            with ThreadPoolExecutor(max_workers=n_workers,
                                    thread_name_prefix="lasscf-frag") as pool:
                futures = {pool.submit(_run_one, p): p[0] for p in active}
                for fut in as_completed(futures):
                    isub_done, e_sub, fcivec_out, wall = fut.result()
                    e_cas[isub_done] = e_sub
                    ci1[isub_done] = fcivec_out
                    log.timer(f"FCI box for subspace {isub_done} (par {wall:.2f}s)", *t1)
        # Sanity: every active fragment was filled
        for isub_check in range(las.nfrags):
            if isub_check not in frozen_ci and ci1[isub_check] is None:
                raise RuntimeError(
                    f"parallel ci_cycle: fragment {isub_check} was not solved"
                )
        return e_cas, ci1

    return _ci_cycle_parallel


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def parallel_fragments(
    n_workers: int = 2,
    omp_threads_per_worker: Optional[int] = None,
    *,
    log_banner: bool = True,
):
    """Activate parallel fragment dispatch inside mrh's LASSCF.

    Parameters
    ----------
    n_workers : int
        Number of fragments to dispatch concurrently.  ``0`` or ``1`` returns
        a no-op context (serial).  Typical setting is the number of
        non-trivial fragments in the partition (2 for the 4-fragment H1
        partition since F0/F1 are 1-determinant closed-shell).
    omp_threads_per_worker : int or None
        Threads each fragment's TrimCI/OrbitalOptimizer may use.  ``None``
        defaults to ``max(1, os.cpu_count() // n_workers)``.
    log_banner : bool
        Print a one-line banner with the active settings.

    Behaviour
    ---------
    * Replaces ``mrh.my_pyscf.mcscf.lasscf_sync_o0.ci_cycle`` with the parallel
      version for the duration of the ``with`` block.
    * Sets ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` env to
      ``omp_threads_per_worker`` (restored on exit).
    * Wraps the body in ``threadpool_limits(omp_threads_per_worker)`` if
      ``threadpoolctl`` is available.
    * Original function and env are restored even on exception.

    Restrictions
    ------------
    * Requires ``mrh.my_pyscf.mcscf.lasscf_sync_o0.ci_cycle`` to exist with the
      signature ``(las, mo, ci0, veff, h2eff_sub, casdm1frs, log)``.  If mrh
      renames it, the context raises a clear ``RuntimeError`` instead of
      silently doing nothing.
    """
    if n_workers <= 1:
        # Serial fast-path: no patching at all.
        if log_banner:
            print("[parallel_fragments] n_workers <= 1, running serially")
        yield
        return

    # Resolve OMP threads per worker
    cpu_total = os.cpu_count() or 1
    if omp_threads_per_worker is None:
        omp_threads_per_worker = max(1, cpu_total // n_workers)

    # Locate the patch target
    from mrh.my_pyscf.mcscf import lasscf_sync_o0
    original_ci_cycle = getattr(lasscf_sync_o0, "ci_cycle", None)
    if original_ci_cycle is None:
        raise RuntimeError(
            "mrh.my_pyscf.mcscf.lasscf_sync_o0.ci_cycle not found.  "
            "mrh may have been upgraded; the parallel patch needs an update."
        )

    if log_banner:
        print(f"[parallel_fragments] workers={n_workers}  "
              f"omp_threads_per_worker={omp_threads_per_worker}  "
              f"(cpu_count={cpu_total})")

    # Snapshot env we will mutate
    env_backup = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "TRIMCI_NUM_THREADS",
    )}

    # Install patch + thread caps
    lasscf_sync_o0.ci_cycle = _build_ci_cycle_parallel(n_workers)
    for key in env_backup:
        os.environ[key] = str(omp_threads_per_worker)

    tp_ctx = (threadpool_limits(omp_threads_per_worker)
              if threadpool_limits is not None
              else contextlib.nullcontext())

    try:
        with tp_ctx:
            yield
    finally:
        lasscf_sync_o0.ci_cycle = original_ci_cycle
        for key, val in env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


# ---------------------------------------------------------------------------
# Standalone smoke is intentionally omitted -- validating LASSCF + parallel on
# a real partition needs a localised initial MO, which lives in the runners.
# Validate via ``python -m FRASCI.lasscf.runners.run_lasscf_coo
# --smoke-test --parallel-fragments 2`` (covers the patch install/uninstall path
# even on a single-fragment system) or the timing comparison in the COO
# notebook (parallel vs serial Fe4S4 wall clock).
# ---------------------------------------------------------------------------
