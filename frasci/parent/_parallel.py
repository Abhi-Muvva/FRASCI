"""Bounded process-pool fragment dispatch for frasci.parent (``n_cpu_workers > 0``).

Two mechanisms, same as the validated ``engine/lasscf/parallel_mp.py`` +
``engine/lasscf/parallel.py`` pair this replaces:

1. A ``ProcessPoolExecutor`` of ``min(n_cpu_workers, n_frags)`` long-lived spawn
   workers does the actual CI solves (the phased COO kernel is GIL-bound at
   fragment scale, so threads alone buy nothing).
2. ``mrh.my_pyscf.mcscf.lasscf_sync_o0.ci_cycle`` is monkeypatched for the
   duration of the solve, because mrh's stock version calls
   ``fcibox.kernel(...)`` for each fragment *serially*. The replacement fans all
   fragments out over ``n_frags`` dispatcher threads; each thread only blocks on
   its future (GIL released), so the real parallelism is the worker processes and
   the thread count does not need to be bounded.

Worker reuse vs. warm-start
---------------------------
``make_phased_kernel``'s closure carries a warm-start cache across macro cycles
(the cached COO rotation ``U_cached`` and the drift reference ``h1_ref`` that
decides discovery-vs-refine). The old engine code preserved it by pinning one
dedicated worker process per fragment forever, which forces ``n_workers ==
n_frags``.

Here the workers are instead **stateless**: the parent owns one warm-start dict
per fragment, ships it into the worker with the task, and takes the updated dict
back with the result. Any worker can therefore serve any fragment and still see
that fragment's full history, so a free worker takes whatever is next in the
queue rather than idling until "its" fragment comes round again -- and the
numerics stay identical to the in-process serial path, which the affinity-free
alternative (a per-worker per-fragment cache, warm-start preserved only when a
fragment happens to land on the same worker twice) would not give.

The cost is IPC: the state dict holds ``U_cached``/``h1_ref`` (norb^2) plus the
density-mixing carry ``dm1s_prev``/``dm2_prev``/``h2_ref_mix`` (norb^4), so a
round trip is a few MB at norb=24 -- the same order as the ``h2`` already sent
with every task and the ``dm2`` already sent back, against a CI solve measured
in seconds to minutes.
"""
from __future__ import annotations

import contextlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

from pyscf import lib

_OMP_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "TRIMCI_NUM_THREADS",
)


def _init_worker(omp_threads: int) -> None:
    """Cap each worker's OpenMP/BLAS threads. Runs before this process imports
    trimci (the pool initializer fires before the first task is unpickled, and
    it is that unpickling that pulls in ``frasci.parent._kernel`` -> trimci),
    which matters because libgomp reads the env once at first parallel region.
    """
    for key in _OMP_ENV_KEYS:
        os.environ[key] = str(omp_threads)


def _run_fragment(fragment_idx, warm_state, frame_packet, kernel_kwargs,
                  norb, nelec, h0, h1s, h2, capture_log=False):
    """One fragment CI-solve, in a worker process. Top level so spawn can pickle it.

    Stateless: ``warm_state`` is the fragment's warm-start cache as of its
    previous call (empty dict on the first), and the updated cache is returned
    for the parent to hold. Returns the pid and wall-clock span too, so the
    parent can see which worker served which fragment and when.
    """
    from frasci.parent._kernel import make_phased_kernel

    # A worker never draws its own bar (they would fight over the terminal, and
    # the parent owns the display anyway): it captures the kernel's log payload
    # and ships it home with the result, for the parent to render. Left off
    # unless asked, so the no-progress path stays byte-for-byte what it was --
    # log_callback=None is what makes the kernel skip building extras at all.
    payload = []
    kernel_fn = make_phased_kernel(
        fragment_idx=fragment_idx,
        frame_provider=(None if frame_packet is None else lambda: frame_packet),
        log_callback=(lambda *args: payload.append(args)) if capture_log else None,
        **kernel_kwargs)
    kernel_fn.state.update(warm_state)
    # time.time(), not perf_counter(): only wall clock is comparable across processes.
    t_start = time.time()
    etot, dm1s, dm2 = kernel_fn(norb, nelec, h0, h1s, h2)
    return (etot, dm1s, dm2, kernel_fn.state, os.getpid(), t_start, time.time(),
            payload[-1] if payload else None)


def _make_proxy_kernel(executor, fragment_idx, kernel_kwargs, warm_states, records,
                       on_progress=None, frame_packets=None):
    """Kernel closure with mrh's fcibox contract that round-trips one call
    through the pool. Only this fragment's dispatcher thread touches
    ``warm_states[fragment_idx]``, and mrh calls a fragment once per macro
    cycle, so the slot has a single writer and needs no lock.
    """
    def kernel(norb, nelec, h0, h1s, h2):
        future = executor.submit(
            _run_fragment, fragment_idx, warm_states[fragment_idx],
            None if frame_packets is None else frame_packets[fragment_idx],
            kernel_kwargs, norb, nelec, h0, h1s, h2,
            on_progress is not None)
        etot, dm1s, dm2, state, pid, t_start, t_end, payload = future.result()
        # Keep the parent-owned dict object stable: parent.solve also exposes
        # it as ``proxy.state`` for the post-convergence Phase-2 finalization
        # gate. Replacing the list slot would leave that view stale.
        warm_states[fragment_idx].clear()
        warm_states[fragment_idx].update(state)
        records.append(
            {"fragment": fragment_idx, "pid": pid, "start": t_start, "end": t_end})
        if on_progress is not None and payload is not None:
            on_progress(*payload)
        return etot, dm1s, dm2

    kernel.state = warm_states[fragment_idx]
    return kernel


def _ci_cycle_parallel(las, mo, ci0, veff, h2eff_sub, casdm1frs, log):
    """Fan-out replacement for mrh's serial ``lasscf_sync_o0.ci_cycle``.

    Mirrors mrh's own body (verified against the installed mrh 2026-08-29) with
    the per-fragment loop turned into a thread pool. The two features of mrh's
    version this drops are rejected loudly rather than ignored, because either
    one silently mishandled would corrupt the answer; ``frasci.parent`` sets
    neither.
    """
    if las.frozen_ci:
        raise NotImplementedError(
            "frasci.parent parallel dispatch does not handle las.frozen_ci")
    if getattr(mo, "orbsym", None) is not None:
        raise NotImplementedError(
            "frasci.parent parallel dispatch does not handle orbsym-tagged MOs")

    if ci0 is None:
        ci0 = [None] * las.nfrags
    h1eff_sub = las.get_h1eff(mo, veff=veff, h2eff_sub=h2eff_sub,
                              casdm1frs=casdm1frs)
    max_memory = max(400, las.max_memory - lib.current_memory()[0])
    # Slice serially: only fcibox.kernel is farmed out to threads.
    eri_cas_sub = [las.get_h2eff_slice(h2eff_sub, isub, compact=8)
                   for isub in range(las.nfrags)]

    def _one(isub):
        return las.fciboxes[isub].kernel(
            h1eff_sub[isub], eri_cas_sub[isub], las.ncas_sub[isub],
            las.nelecas_sub[isub], ci0=ci0[isub], verbose=log,
            max_memory=max_memory, ecore=0.0, orbsym=None)

    with ThreadPoolExecutor(max_workers=las.nfrags,
                            thread_name_prefix="frasci-frag") as pool:
        solved = list(pool.map(_one, range(las.nfrags)))
    return [e_sub for e_sub, _ in solved], [fcivec for _, fcivec in solved]


@contextlib.contextmanager
def parallel_fragment_dispatch(n_workers: int, n_frags: int, kernel_kwargs: dict,
                               *, log_banner: bool = True, on_progress=None,
                               initial_states: list[dict] | None = None,
                               frame_packets: list[dict] | None = None):
    """Run the ``with`` body with ``min(n_workers, n_frags)`` worker processes
    serving fragment CI solves, and mrh's ``ci_cycle`` patched to dispatch
    fragments concurrently.

    ``initial_states``, if given, is one warm-start dict per fragment seeding
    the cache the parent owns -- the cross-``solve()`` counterpart of the
    across-macro-cycle carry described above (see ``frasci.parent.solve``'s
    ``warm_start_from``).

    Yields ``(proxy_kernels, records)``: one proxy kernel per fragment to hand
    to ``make_fcibox``, and a list that grows one dict
    ``{"fragment", "pid", "start", "end"}`` per completed kernel call.

    Deliberately no ``n_workers == 1`` serial short-circuit: 1 worker is a valid
    request (run the solve out-of-process) and a fast-path there would leave the
    real dispatch untested, which is how the previous attempt at this shipped
    broken.
    """
    n_active = min(int(n_workers), n_frags)
    cpu_total = os.cpu_count() or 1
    omp_threads = max(1, cpu_total // n_active)

    if log_banner:
        print(f"[frasci.parent] parallel dispatch: {n_active} worker processes "
              f"for {n_frags} fragments, OMP={omp_threads}/worker "
              f"(cpu_count={cpu_total})")

    from mrh.my_pyscf.mcscf import lasscf_sync_o0
    original_ci_cycle = getattr(lasscf_sync_o0, "ci_cycle", None)
    if original_ci_cycle is None:
        raise RuntimeError(
            "mrh.my_pyscf.mcscf.lasscf_sync_o0.ci_cycle not found -- mrh may "
            "have been upgraded; frasci.parent's parallel dispatch needs an update.")

    warm_states: list[dict] = ([dict(s) for s in initial_states]
                               if initial_states is not None
                               else [{} for _ in range(n_frags)])
    records: list[dict] = []

    with ProcessPoolExecutor(
        max_workers=n_active,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(omp_threads,),
    ) as executor:
        proxy_kernels = [
            _make_proxy_kernel(executor, i, kernel_kwargs, warm_states, records,
                               on_progress=None if on_progress is None
                               else on_progress(i), frame_packets=frame_packets)
            for i in range(n_frags)
        ]
        lasscf_sync_o0.ci_cycle = _ci_cycle_parallel
        try:
            yield proxy_kernels, records
        finally:
            lasscf_sync_o0.ci_cycle = original_ci_cycle
