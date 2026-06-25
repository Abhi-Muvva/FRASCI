"""
parallel_mp.py
==============
Process-based fragment dispatch for the COO-LASSCF kernel.

Sibling of ``parallel.py`` (thread-based).  Use this when the per-fragment
kernel is GIL-bound -- which the COO kernel is at H1Frag scale, because each
BFGS step re-runs Python-side list conversions (``h1_rot.tolist()`` on a
20736-element ERI tensor) and ``cpp_bfgs`` re-acquires the GIL on every
callback.  With two thread workers the two fragments time-slice on the GIL
and CPU saturates at ~1 core.  This module sidesteps that by giving each
fragment its own Python interpreter in a separate process.

Validated 2026-06-20 on F3-shaped 12-orb/1008-det fragment: two sequential
full-kernel calls = 39.17 s, two parallel via ``ProcessPoolExecutor(spawn)``
= 21.66 s -> 1.81x speedup (measurement was contended with another live
pipeline; isolation should be closer to 2x).

Architecture
------------
1. ``coo_worker_pool(n_workers, kernel_kwargs, omp_threads_per_worker)`` --
   context manager that spawns ``n_workers`` long-lived worker processes.
   Each worker:

     * Sets OMP env BEFORE importing trimci, so the C++ OpenMP runtime
       respects the per-worker cap.
     * Imports trimci/pyscf once (~1-2 s init).
     * Builds its own ``make_coo_trimci_kernel_for_fragment`` closure with a
       private ``U_cached`` state, so warm-starting kappa works per fragment
       across macro iters.
     * Serves tasks from its in-queue: receives
       ``(norb, nelec, h0, h1s, h2)``, runs the COO kernel, pushes
       ``(etot, dm1s, dm2, log_entries)`` back on its out-queue.

2. ``make_proxy_kernel(in_q, out_q, log_callback)`` -- returns a kernel
   closure with the mrh contract ``(norb, nelec, h0, h1s, h2) -> (etot,
   dm1s, dm2)``.  It sends the inputs over the in_q (GIL released during
   the IPC), blocks on the out_q (GIL released during the wait), replays
   log entries through the parent-side callback, and returns the RDMs.

3. Combine with ``parallel.parallel_fragments(n_workers=n_frags)`` (the
   thread-based ``ci_cycle`` patch).  Each fragment's proxy kernel call
   runs in its own dispatcher thread; that thread blocks on the worker's
   out_q (no GIL contention because Queue.get releases the GIL during the
   wait); the worker processes crunch genuinely in parallel.

Why we keep the thread patch
----------------------------
mrh's stock ``ci_cycle`` calls ``fcibox.kernel(...)`` serially for each
fragment.  To dispatch them concurrently we still need a Python-level
fan-out, but with proxy kernels the threads are essentially idle on IPC --
GIL contention is no longer the bottleneck.
"""
from __future__ import annotations

import contextlib
import os
import time
import traceback
from multiprocessing import get_context
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Worker loop (top-level so spawn can transport it)
# ---------------------------------------------------------------------------

def _worker_loop(in_q, out_q, kernel_kwargs: dict, frag_idx: int,
                 omp_env: dict) -> None:
    """Long-lived COO worker.  Builds one kernel, serves tasks until None.

    All heavy imports happen here (post-fork in the spawn context) so each
    process owns its own trimci / pyscf state, including the
    ``OrbitalOptimizer`` warm-start cache.
    """
    # 1. Apply OMP caps *before* the trimci import so libgomp picks them up.
    for k, v in omp_env.items():
        os.environ[k] = str(v)

    # 2. Heavy imports (per-process).  PYTHONPATH must include the project
    #    root; the runner sets that in the parent env which spawn inherits.
    from FRASCI.lasscf.coo_kernel import (
        make_coo_trimci_kernel_for_fragment,
    )

    # 3. Local log buffer; the kernel calls _local_log per fragment-call,
    #    we drain it after each task and ship the entries back.
    log_buffer: list = []

    def _local_log(norb, nelec, n_dets, e_elec, wall, extras=None):
        log_buffer.append((norb, nelec, int(n_dets), float(e_elec),
                           float(wall), dict(extras) if extras else None))

    # 4. Build the per-fragment kernel.  U_cached lives inside the closure,
    #    persistent across all tasks this worker handles.
    kernel_fn = make_coo_trimci_kernel_for_fragment(
        log_callback=_local_log,
        **kernel_kwargs,
    )

    # 5. Serve loop.
    while True:
        msg = in_q.get()
        if msg is None:
            break
        try:
            norb, nelec, h0, h1s, h2 = msg
            log_buffer.clear()
            etot, dm1s, dm2 = kernel_fn(norb, nelec, h0, h1s, h2)
            out_q.put(("ok", etot, dm1s, dm2, list(log_buffer)))
        except Exception:
            out_q.put(("err", traceback.format_exc()))


# ---------------------------------------------------------------------------
# Proxy kernel: turns a worker into something mrh's fcibox can call
# ---------------------------------------------------------------------------

def make_proxy_kernel(in_q, out_q, *, log_callback: Optional[Callable] = None,
                      frag_idx: int = -1) -> Callable:
    """Return a kernel closure with the mrh contract that round-trips one
    call through a worker process.

    Parameters
    ----------
    in_q, out_q
        multiprocessing.Queue handles for one dedicated worker.
    log_callback : callable or None
        ``log_callback(norb, nelec, n_dets, energy_electronic, wall, extras)``
        invoked in the parent for every log entry the worker produced.
    frag_idx : int
        Diagnostic label included in worker-error messages.
    """
    def kernel(norb, nelec, h0, h1s, h2):
        in_q.put((norb, nelec, h0, h1s, h2))
        result = out_q.get()
        if result[0] == "err":
            raise RuntimeError(
                f"COO worker (frag={frag_idx}) raised:\n{result[1]}"
            )
        _tag, etot, dm1s, dm2, log_entries = result
        if log_callback is not None:
            for entry in log_entries:
                log_callback(*entry)
        return etot, dm1s, dm2

    return kernel


# ---------------------------------------------------------------------------
# Worker-pool context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def coo_worker_pool(
    n_workers: int,
    kernel_kwargs: dict,
    *,
    omp_threads_per_worker: Optional[int] = None,
    log_banner: bool = True,
):
    """Spawn ``n_workers`` persistent COO worker processes for the ``with``
    block.  Yields a list of dicts ``[{"proc","in_q","out_q"}, ...]``.

    ``kernel_kwargs`` is forwarded verbatim to
    ``make_coo_trimci_kernel_for_fragment`` inside each worker, minus
    ``log_callback`` (workers use their own buffered local callback).
    """
    if n_workers <= 0:
        if log_banner:
            print("[coo_worker_pool] n_workers <= 0; no pool spawned")
        yield []
        return

    # Strip any log_callback the caller passed by mistake -- workers can't
    # call a parent-side callback directly.
    kernel_kwargs = {k: v for k, v in kernel_kwargs.items()
                     if k != "log_callback"}

    cpu_total = os.cpu_count() or 1
    if omp_threads_per_worker is None:
        omp_threads_per_worker = max(1, cpu_total // n_workers)
    omp_env = {
        "OMP_NUM_THREADS":      omp_threads_per_worker,
        "MKL_NUM_THREADS":      omp_threads_per_worker,
        "OPENBLAS_NUM_THREADS": omp_threads_per_worker,
        "TRIMCI_NUM_THREADS":   omp_threads_per_worker,
    }

    if log_banner:
        print(f"[coo_worker_pool] spawning {n_workers} workers  "
              f"OMP={omp_threads_per_worker}/worker  (cpu_count={cpu_total})")

    ctx = get_context("spawn")
    workers: List[dict] = []
    t0 = time.perf_counter()
    for i in range(n_workers):
        in_q = ctx.Queue()
        out_q = ctx.Queue()
        p = ctx.Process(
            target=_worker_loop,
            args=(in_q, out_q, kernel_kwargs, i, omp_env),
            name=f"coo-worker-{i}",
        )
        p.start()
        workers.append({"proc": p, "in_q": in_q, "out_q": out_q})

    if log_banner:
        print(f"[coo_worker_pool] {n_workers} workers up "
              f"({time.perf_counter()-t0:.2f}s)")

    try:
        yield workers
    finally:
        # Sentinel-shutdown, then join with a short grace period
        for w in workers:
            try:
                w["in_q"].put(None)
            except Exception:
                pass
        for w in workers:
            w["proc"].join(timeout=5)
            if w["proc"].is_alive():
                w["proc"].terminate()
        if log_banner:
            print(f"[coo_worker_pool] {n_workers} workers shut down")
