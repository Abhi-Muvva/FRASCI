"""frasci.parent -- the phased LASSCF-COO whole-system solve (Layer A).

``solve()`` is the public API: a SIMPLIFIED rewrite of
``engine/lasscf/runners/run_lasscf_coo.py::run()``'s ``phased=True`` path,
simplified because this library drops keyframe/resumability checkpointing,
merge/warm-start-from-different-partition validation, macro-history/kernel-call
telemetry JSON logging, the CLI, and warm-starting from a prior checkpoint (see
the plan's Global Constraints) -- it always starts from an identity-permutation
MO guess. Same numerics as the original for everything it keeps.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox

from frasci import _progress
from frasci._mock_scf import build_mock_scf_from_fcidump
from frasci.parent._config import FragmentResult, Parent, ParentConfig
from frasci.parent._convergence import (
    MacroRecord, TailConvergencePolicy, classify_trace, decide_extension,
    rdm_block_change)
from frasci.parent._frame import compute_frame_transport
from frasci.parent._kernel import make_phased_kernel, patch_large_fragment_aufbau
from frasci.parent._lassis_adapter import build_lassis_ready_las
from frasci.parent._parallel import parallel_fragment_dispatch

__all__ = ["solve", "ParentConfig", "FragmentResult", "Parent", "TailConvergencePolicy",
          "build_lassis_ready_las"]


def _identity_permutation_mo(orbital_lists, n_orb):
    """Block-diagonal identity MO guess: permute eye(n_orb)'s columns so each
    fragment's listed orbitals appear in order. Exact behaviour of the
    original's ``_build_identity_permutation_mo``
    (engine/lasscf/runners/run_lasscf_coo.py:545) -- it does NOT append any
    orbitals left unlisted by ``orbital_lists``; for a total partition (every
    orbital assigned to exactly one fragment) that never matters.
    """
    col_order = []
    for frag_orbs in orbital_lists:
        col_order.extend(frag_orbs)
    return np.eye(n_orb)[:, col_order]


def _load_warm_states(parent_dir, ncas_sub):
    """One warm-start dict per fragment from a prior solve's ``save_dets_to``
    directory: the converged COO rotation plus the embedding h1 it was solved
    at. Seeding these is what lets a fresh solve take the kernel's existing
    cheap ``refine`` path on its FIRST call instead of re-running Phase-0
    discovery (see ``solve``'s ``warm_start_from``).
    """
    state_dir = Path(parent_dir) / "phase1_state"
    states = []
    for i, norb in enumerate(ncas_sub):
        data = np.load(state_dir / f"fragment_{i}.npz")
        if int(data["n_orb"]) != norb:
            raise ValueError(
                f"warm_start_from fragment {i} was solved with "
                f"n_orb={int(data['n_orb'])}, but this partition's fragment {i} "
                f"has {norb} orbitals -- a saved rotation is only valid for the "
                f"partition it came from")
        states.append({"U_cached": data["orbital_rotation"],
                       "h1_ref": data["h1_input"]})
    return states


def _atomic_write_json(path: Path, data) -> None:
    """Write ``path`` via a same-directory temp file + ``os.replace`` so a
    reader never observes a partially-written trace/diagnostics file."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def _make_fragment_progress(bars):
    """Build the per-fragment ``log_callback`` that drives one progress chip.

    Returns a factory; ``factory(i)`` is ``None`` when progress is off, which is
    what ``make_phased_kernel`` already treats as "skip the whole extras block",
    so the default path costs nothing.

    Granularity note: the kernel's ``log_callback`` fires once per fragment per
    LASSCF macro cycle, at completion -- ``trimci.ground_state_from_integrals``
    is a single opaque call with no progress hook of its own, so nothing finer
    is observable from here. So there is no honest fragment-level completion
    fraction to draw (the det target is soft and gets overshot), and the chip
    reports state instead: phase, last energy, last determinant count, with the
    clock restarted so the elapsed figure times the solve now in flight.
    """
    if bars.mode == "off":
        return lambda i: None

    def factory(i):
        def record(norb, nelec, n_dets, energy, wall, extras):
            bars.update(
                f"F{i}", phase=str(extras.get("phased_mode", "solving")),
                reset_clock=True,
                E=f"{float(energy):.6f}", dets=int(n_dets))
        return record

    return factory


def _frame_packet(C_prev, C_curr, S_AO, frame_id_prev, frame_id_curr,
                  leakage_tol):
    transport = compute_frame_transport(
        C_prev, C_curr, S_AO, frame_id_prev=frame_id_prev,
        frame_id_curr=frame_id_curr, leakage_tol=leakage_tol)
    return {
        "frame_id_prev": frame_id_prev,
        "frame_id_curr": frame_id_curr,
        "C_prev": np.asarray(C_prev).copy(),
        "C_curr": np.asarray(C_curr).copy(),
        "transport": transport,
    }


def _install_frame_publisher(las, mo_initial, n_frags, leakage_tol):
    """Publish the accepted external fragment frame for the next CI cycle.

    MRH's fragment kernel never receives ``mo_coeff``.  Its orbital update is
    observable at ``H_op.update_mo_eri``, so a narrow ``get_hop`` wrapper
    publishes that before/after transition into mutable per-fragment packets.
    """
    S_AO = np.asarray(las._scf.get_ovlp())
    packets = [None] * n_frags
    current_id = [0]

    def blocks(mo_coeff):
        return [np.asarray(las.get_mo_slice(i, mo_coeff=mo_coeff)).copy()
                for i in range(n_frags)]

    initial_blocks = blocks(mo_initial)
    for i, C in enumerate(initial_blocks):
        packets[i] = _frame_packet(
            C, C, S_AO, current_id[0], current_id[0], leakage_tol)

    real_get_hop = las.get_hop

    def wrapped_get_hop(*args, **kwargs):
        mo_before = np.asarray(kwargs.get("mo_coeff")).copy()
        H_op = real_get_hop(*args, **kwargs)
        real_update = H_op.update_mo_eri
        transition_prev_id = current_id[0]

        def wrapped_update(x, h2eff_sub):
            mo_after, h2eff_new = real_update(x, h2eff_sub)
            prev_blocks = blocks(mo_before)
            curr_blocks = blocks(mo_after)
            next_id = transition_prev_id + 1
            for i, (C_prev, C_curr) in enumerate(zip(prev_blocks, curr_blocks)):
                packets[i] = _frame_packet(
                    C_prev, C_curr, S_AO, transition_prev_id, next_id,
                    leakage_tol)
            # Repeated scaled-step attempts from the same H_op overwrite the
            # same transition generation; only the final returned trial can be
            # consumed by the next CI cycle.
            current_id[0] = next_id
            return mo_after, h2eff_new

        H_op.update_mo_eri = wrapped_update
        return H_op

    las.get_hop = wrapped_get_hop
    return packets


_RE_MACRO_G_INT = re.compile(
    r"LASSCF macro (\d+) : E = ([-\d.eE+]+) ; \|g_int\| = ([-\d.eE+]+)")


class _GIntTap:
    """Forwards every write to whatever ``las.stdout`` already is (pyscf's
    own default, or ``_progress.MacroTap`` when progress display is on) and
    additionally parses the same "LASSCF macro N : E = ... ; |g_int| = ..."
    line MacroTap already parses for its own display -- purely to expose the
    values synchronously to the escape controller (``holder``) and/or record
    them into a full trace (``trace``, ``segment_holder``) for the tail
    -confirmation policy below. mrh prints this line at the START of a
    macro's convergence check, before that macro's own orbital step is
    computed.

    mrh keeps no per-macrocycle attribute on ``las`` itself (``E``/``g_int``
    live only as local variables inside its own ``kernel()``), so this
    stdout parse is the only existing reliable hook for a full trace --
    verified against ``lasscf_rdm.py``'s ``%.15g`` format (15 significant
    digits, effectively full float64 precision at the 1e-4..1e-5 Ha
    magnitudes this policy classifies).

    ``trace``/``segment_holder`` default to ``None`` (today's behavior,
    unchanged): a caller that only wants the escape controller's synchronous
    ``holder`` never allocates or appends to a trace at all.
    """

    def __init__(self, real_stream, holder, trace=None, segment_holder=None):
        self._real = real_stream
        self._holder = holder
        self._trace = trace
        self._segment_holder = segment_holder
        self._buf = ""

    def write(self, text):
        self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            m = _RE_MACRO_G_INT.search(line)
            if m:
                local_macro = int(m.group(1))
                e_tot = float(m.group(2))
                g_int = float(m.group(3))
                self._holder["macro"] = local_macro
                self._holder["g_int"] = g_int
                if self._trace is not None:
                    segment = (self._segment_holder["n"]
                              if self._segment_holder is not None else 0)
                    self._trace.append(MacroRecord(
                        segment=segment, local_macro=local_macro,
                        global_macro=len(self._trace), e_tot=e_tot, g_int=g_int))

    def flush(self):
        self._real.flush()


def _run_tail_confirmation(las, *, policy, conv_tol_grad, trace, segment_holder):
    """Bounded adaptive tail-confirmation protocol (2026-09-29 contract),
    run immediately after the initial ``las.kernel(mo)`` returns.

    Classifies ``trace`` (already fully captured by ``_GIntTap`` through the
    initial solve). In ``mode="extend"``, continues from the current
    orbitals/RDMs -- the SAME continuation call the parent's Phase-2
    finalization already uses -- ``extension_cycles`` macrocycles at a time,
    reclassifying after every block, snapshotting the fragment 1-RDMs
    (``las.make_casdm1s_sub()``) held by the LAS object at the end of each
    block for the plateau's RDM-stability check. Stops on strict
    convergence, a confirmed plateau, instability, or exhausting
    ``policy.max_extra_cycles`` -- never runs an unbounded loop. Does not
    touch ``las.get_hop``/``las.stdout`` (the escape controller and progress
    display, if installed, keep working across these continuation calls
    exactly as they already do for Phase-2 finalization).

    Returns ``(diagnostics, extra_cycles_used)``.
    """
    # Snapshot BEFORE the loop (not merely before the first block's own
    # classification) so a single extension block can already supply a
    # meaningful RDM *change* -- otherwise confirmation would silently need
    # two extension blocks no matter what `policy.confirmation_blocks` says,
    # since the very first change always has nothing prior to diff against.
    rdm_prev = las.make_casdm1s_sub()
    diagnostics = classify_trace(
        trace, conv_tol_grad=conv_tol_grad, policy=policy,
        las_declared_converged=bool(getattr(las, "converged", False)))

    extra_used = 0
    extension_blocks_run = 0
    plateau_streak = 0

    while True:
        next_cycles = decide_extension(diagnostics, policy=policy, extra_cycles_used=extra_used)
        if next_cycles <= 0:
            break
        segment_holder["n"] += 1
        las.max_cycle_macro = next_cycles
        las.kernel(las.mo_coeff, casdm1frs=las.casdm1frs, casdm2fr=las.casdm2fr)
        extra_used += next_cycles
        extension_blocks_run += 1

        rdm_curr = las.make_casdm1s_sub()
        rdm_change = rdm_block_change(rdm_prev, rdm_curr)
        rdm_prev = rdm_curr

        diagnostics = classify_trace(
            trace, conv_tol_grad=conv_tol_grad, policy=policy,
            las_declared_converged=bool(getattr(las, "converged", False)),
            extension_blocks_run=extension_blocks_run,
            consecutive_plateau_blocks=plateau_streak, rdm_change=rdm_change)
        plateau_streak = (
            plateau_streak + 1
            if diagnostics.status in ("stationary_plateau_candidate", "stationary_plateau")
            else 0)

    if (policy.mode == "extend" and extension_blocks_run > 0
            and diagnostics.status == "near_threshold_progressing"
            and extra_used >= policy.max_extra_cycles):
        # Only for the generic "still improving, just needs more time"
        # status: a candidate that already satisfies the numeric plateau
        # criteria but failed RDM confirmation is a MORE specific, more
        # useful diagnostic ("looked flat, RDMs kept moving") than the
        # generic budget-exhausted label, so it is reported as-is even
        # after the budget runs out. Any terminal status (strict/plateau/
        # unstable) already broke the loop above on its own.
        diagnostics = dataclasses.replace(
            diagnostics, status="budget_exhausted", needs_more_iterations=True,
            reasons=diagnostics.reasons + (
                "max_extra_cycles_exhausted_without_plateau_or_convergence",))

    return diagnostics, extra_used


def _derive_episode_seed(master_seed: int, n_attempt: int) -> int:
    """Reproducible per-attempt seed for the escape rule's forced
    rediscovery (2026-09-17 contract). Deliberately a DIFFERENT mixing path
    from TrimCI-FRASCI's own per-restart derivation inside one discovery
    call (``master_seed + r * 0x9e3779b97f4a7c15``, fix branch
    ``fix/cpu-rng-propagation``) so the two seed sequences can never
    coincide. ``n_attempt<=1`` (the original support) maps to ``master_seed``
    UNCHANGED: a solve that never hits the escape rule is byte-for-byte
    unaffected by this function existing. Without this distinct-per-attempt
    derivation, a forced rediscovery would reproduce the identical Phase-0
    bank (CPU Phase-0 discovery is now deterministic given a fixed seed) and
    the escape rule would invalidate the cache without exploring anything
    new -- the scientific blocker this function exists to close.
    """
    n_attempt = int(n_attempt)
    if n_attempt <= 1:
        return int(master_seed) & 0xFFFFFFFFFFFFFFFF
    x = (int(master_seed) + n_attempt * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 27)
    return x & 0xFFFFFFFFFFFFFFFF


def _phase2_started(kernel_fns) -> bool:
    """True if ANY fragment has begun Phase-2 finalization. The escape
    controller must never publish an invalidating certificate once this
    holds (2026-09-17 contract)."""
    return any(
        fn.state.get("force_phase2_next")
        or fn.state.get("post_phase2_active")
        or fn.state.get("phase2_promoted_ever")
        or fn.state.get("phase2_episode_complete")
        for fn in kernel_fns)


def decide_escape(*, any_downhill, ended_in_restore, g_int, conv_tol_grad,
                  attempts_used, max_attempts, phase2_started):
    """Pure decision function for the macro-boundary escape rule (2026-09-17
    contract) -- no I/O, no mrh/trimci imports, so it is testable in
    isolation. Trigger requires ALL of: zero accepted downhill trials this
    macro; the safeguard exhausted its halving budget and restored the
    keyframe; |g_int| still above conv_tol_grad (a converged fragment is
    structurally excluded, never mistaken for a deadlock); Phase-2
    finalization has not started for any fragment; and the global attempt
    budget not yet exhausted. Returns (should_invalidate, reason)."""
    if any_downhill:
        return False, "downhill_step_accepted_this_macro"
    if not ended_in_restore:
        return False, "orbital_step_not_rejected_restore"
    if g_int is None or g_int <= conv_tol_grad:
        return False, "g_int_at_or_below_conv_tol_grad"
    if phase2_started:
        return False, "phase2_started_escape_suppressed"
    if attempts_used >= max_attempts:
        return False, "retry_budget_exhausted"
    return True, "rejected_restore_keyframe_nonconverged_gint"


def _install_safeguarded_escape_controller(
        las, mo_initial, n_frags, leakage_tol, *, kernel_fns, master_seed,
        max_attempts, conv_tol_grad, g_int_holder, escape_log,
        max_extra_halvings=10):
    """Compose ``_install_frame_publisher`` (unmodified) with the
    already-validated energy-backtracking safeguard (previously report-only
    -- ``docs/reports/frasci_fe4s4_orbital_stability_ladder_2026-09-14.py``'s
    ``_install_energy_backtracking`` -- now production) and a bounded,
    global escape controller (2026-09-17 contract, see ``decide_escape``).

    Global lineage: one counter for the whole macro-boundary event (macro
    acceptance is one global orbital step; fragment episode caches invalidate
    atomically together), never per-fragment. ``kernel_fns`` (mutated in
    place by the caller after this returns, then read by reference here) is
    what lets the Phase-2 guard see every fragment's own state without this
    function owning fragment construction.
    """
    packets = _install_frame_publisher(las, mo_initial, n_frags, leakage_tol)
    for p in packets:
        p["episode_seed"] = _derive_episode_seed(master_seed, 1)

    lineage = {"attempts_used": 1, "exhausted": False, "last_macro_resolved": None}
    # mrh resets ITS OWN internal macro counter to 0 on every fresh
    # las.kernel() call (verified empirically, matching the same fact
    # already documented for the 2026-09-16 macro-boundary harness) -- so
    # g_int_holder["macro"] (parsed straight from mrh's own log line) is NOT
    # a globally-unique macro index across a caller that resumes las.kernel()
    # in chunks (this file's solve() never does; a chunked test harness can).
    # controller_get_hop itself is called exactly once per REAL macro
    # attempt regardless of chunking, so a counter local to ITS closure
    # (persisting across every later las.kernel() call on this same las,
    # since las.get_hop is set once, here) is the correct, caller-agnostic
    # globally-unique index.
    global_macro_counter = {"n": -1}
    real_get_hop = las.get_hop

    def controller_get_hop(*args, **kwargs):
        H_op = real_get_hop(*args, **kwargs)
        casdm1frs = kwargs.get("casdm1frs")
        casdm2fr = kwargs.get("casdm2fr")
        casdm1fs = las.make_casdm1s_sub(casdm1frs=casdm1frs)
        real_update = H_op.update_mo_eri
        keyframe_e = H_op.e_tot

        def controller_update(x, h2eff_sub):
            trial_x = x
            any_downhill = False
            mo_after = h2eff_after = None
            for _attempt in range(max_extra_halvings + 1):
                mo1, h2eff_new = real_update(trial_x, h2eff_sub)
                veff1 = las.get_veff(
                    dm=las.make_rdm1s(mo_coeff=mo1, casdm1s_sub=casdm1fs))
                e1 = las.energy_nuc() + las.energy_elec(
                    mo_coeff=mo1, h2eff=h2eff_new, casdm1frs=casdm1frs,
                    casdm2fr=casdm2fr, veff=veff1)
                if e1 < keyframe_e:
                    any_downhill = True
                    mo_after, h2eff_after = mo1, h2eff_new
                    break
                trial_x = trial_x * 0.5
            ended_in_restore = not any_downhill
            if ended_in_restore:
                mo_after, h2eff_after = real_update(np.zeros_like(x), h2eff_sub)

            global_macro_counter["n"] += 1
            macro_idx = global_macro_counter["n"]
            if macro_idx != lineage["last_macro_resolved"]:
                lineage["last_macro_resolved"] = macro_idx
                if any_downhill:
                    lineage["attempts_used"] = 1
                    lineage["exhausted"] = False
                g_int = g_int_holder.get("g_int")
                should, reason = (
                    (False, "exhausted") if lineage["exhausted"] else
                    decide_escape(
                        any_downhill=any_downhill, ended_in_restore=ended_in_restore,
                        g_int=g_int, conv_tol_grad=conv_tol_grad,
                        attempts_used=lineage["attempts_used"], max_attempts=max_attempts,
                        phase2_started=_phase2_started(kernel_fns)))
                record = {
                    "macro": macro_idx, "invalidated": should, "reason": reason,
                    "attempts_used": lineage["attempts_used"],
                    "master_seed": int(master_seed),
                    "episode_seed": _derive_episode_seed(
                        master_seed, lineage["attempts_used"]),
                    "g_int": g_int,
                }
                if reason == "retry_budget_exhausted":
                    lineage["exhausted"] = True
                if should:
                    lineage["attempts_used"] += 1
                    record["next_episode_seed"] = _derive_episode_seed(
                        master_seed, lineage["attempts_used"])
                escape_log.append(record)

                if should:
                    next_seed = record["next_episode_seed"]
                    for i in range(n_frags):
                        pkt = packets[i]
                        packets[i] = {
                            **pkt,
                            "episode_seed": next_seed,
                            "transport": pkt["transport"]._replace(
                                reusable=False,
                                leakage=max(leakage_tol * 10, 1.0)),
                        }

            return mo_after, h2eff_after

        H_op.update_mo_eri = controller_update
        return H_op

    las.get_hop = controller_get_hop
    return packets


def _keep_final_frame_uncanonicalized(las):
    """Keep the returned adaptive Parent in the last certified LASSCF frame.

    MRH canonicalizes fragment orbitals after its last CI call.  That is an
    additional within-fragment frame rotation, but no subsequent fragment
    kernel call exists to consume and compose it into ``U_cached``.  The
    adaptive parent does not use orbital energies or occupations, so retaining
    the already-optimized, noncanonical MOs is both sufficient and the only
    frame-consistent output without inventing an unverified post-hoc state
    transport.
    """
    def canonicalize(mo_coeff=None, casdm1frs=None, casdm2fr=None,
                     natorb_casdm1=None, veff=None, h2eff_sub=None,
                     orbsym=None):
        del natorb_casdm1, veff, orbsym
        mo_coeff = las.mo_coeff if mo_coeff is None else mo_coeff
        nmo = np.asarray(mo_coeff).shape[1]
        mo_energy = np.zeros(nmo)
        mo_occ = np.zeros(nmo)
        return (mo_coeff, mo_energy, mo_occ, casdm1frs, casdm2fr,
                h2eff_sub)

    las.canonicalize = canonicalize


def solve(
    fcidump_path: str,
    orbital_lists: list[list[int]],
    nelec_per_frag: list[tuple[int, int]],
    spin_sub: list[int],
    config: ParentConfig,
    *,
    save_dets_to: Path | None = None,
    warm_start_from: Path | None = None,
    progress: bool | str = False,
) -> Parent:
    """Run the phased LASSCF-COO solve on a fixed fragment partition.

    Fragment solves run in parallel by default, with one worker process per
    fragment. ``config.n_cpu_workers=N`` caps dispatch at ``min(N, n_frags)``;
    explicit ``0`` keeps the in-process serial path for diagnostics. Workers
    are reused and each fragment's warm-start cache travels with its task (see
    ``frasci.parent._parallel``).

    ``save_dets_to``, if given, persists ``checkpoint.npz``,
    ``checkpoint_metadata.json``, and ``phase1_state/fragment_{i}.npz`` to
    that directory (creating it if needed) -- the same schema
    ``frasci.recovery`` reads from a parent directory. Default (``None``)
    leaves no trace on disk, matching the scratch dir's own cleanup.

    To drive ``frasci.recovery.solve()`` with this ``Parent``'s output, pass
    ``save_dets_to=<a directory>`` and give that directory to
    ``frasci.recovery.solve()`` as its ``parent_dir`` argument.
    The in-memory ``Parent`` object returned here IS accepted directly by
    ``frasci.recovery.build_contracted_reference()`` alone, if you only need
    the contracted reference and not a full recovery solve.

    ``warm_start_from`` is a prior solve's ``save_dets_to`` directory. Each
    fragment's kernel then starts holding that run's converged COO rotation and
    the h1 it was solved at, so its first call takes the cheap ``refine`` path
    (``config.refine_num_runs``) instead of re-running Phase-0 discovery
    (``config.discovery_num_runs``, e.g. 64 stochastic restarts). Intended for
    sweeps that vary ``config.phase1_dets`` over one fixed partition, where the
    discovery is identical work every time. Note this reuses ORBITALS, not
    determinants -- the new cap's Phase 1 still grows its own determinant set.

    Because reuse routes through the existing drift check, it only takes effect
    when the incoming embedding is within ``config.rediscover_tol`` of the saved
    h1; the solve is correct either way, but across separate runs (which restart
    from the same identity-permutation MO guess while the saved h1 is a
    CONVERGED one) the drift is generally large, so the speedup lands with
    ``rediscover_tol=inf`` -- the setting the fe4s4 production sweeps already
    use -- and not with the 1e-2 default.

    ``progress`` (default ``False``, i.e. today's behaviour: mrh's logger
    scrolls to stdout) turns on the live display -- one bar tracking the LASSCF
    macro cycle against ``config.max_cycle_macro`` with the running energy and
    |g_int| beside it, plus a compact per-fragment status strip. In place on a
    TTY, a periodic one-line digest otherwise. It also routes mrh's logger
    through the display, since a bar cannot share a stream with an unrelated
    printer. See frasci._progress.
    """
    mf = build_mock_scf_from_fcidump(fcidump_path, max_memory_mb=config.max_memory_mb)
    n_orb = (int(mf.mol._nao) if hasattr(mf.mol, "_nao") and mf.mol._nao
             else mf.get_hcore().shape[0])

    all_orbs = sorted(o for frag in orbital_lists for o in frag)
    if all_orbs != list(range(n_orb)):
        raise ValueError(
            f"orbital_lists must partition all {n_orb} orbitals exactly once; "
            f"got {len(all_orbs)} entries, duplicates or gaps present")
    orbital_lists = [sorted(map(int, frag)) for frag in orbital_lists]

    mo = _identity_permutation_mo(orbital_lists, n_orb)

    ncas_sub = tuple(len(f) for f in orbital_lists)
    n_frags = len(orbital_lists)

    warm_states = (None if warm_start_from is None
                   else _load_warm_states(warm_start_from, ncas_sub))
    if config.adaptive_phase_policy and warm_start_from is not None:
        raise ValueError(
            "adaptive cached-support cross-run warm starts are not yet supported: "
            "the saved parent state has no external-frame certificate")

    las = LASSCF(mf, ncas_sub, nelec_per_frag, spin_sub=spin_sub)
    las.max_cycle_macro = config.max_cycle_macro
    if config.max_cycle_rdmjk is not None:
        las.max_cycle_rdmjk = int(config.max_cycle_rdmjk)
    las.conv_tol_grad = config.conv_tol_grad
    las.verbose = 4

    scratch_dir = tempfile.mkdtemp(prefix="frasci_parent_")
    kernel_kwargs_shared = dict(
        threshold=config.threshold,
        discovery_num_runs=config.discovery_num_runs,
        refine_num_runs=config.refine_num_runs,
        orbopt=config.orbopt,
        orbopt_cycles=config.orbopt_cycles,
        max_dets_phase0=config.phase0_dets,
        max_dets_phase1=config.phase1_dets,
        growth_factor_phase1=config.growth_factor_phase1,
        enable_phase2=config.enable_phase2,
        max_dets_phase2=config.phase2_dets,
        growth_factor_phase2=config.growth_factor_phase2,
        pt2=config.pt2,
        seed=config.rng_seed,
        rediscover_tol=config.rediscover_tol,
        warm_start_kappa=config.warm_start_kappa,
        warm_start_dets=config.adaptive_phase_policy,
        adaptive_phase_policy=config.adaptive_phase_policy,
        adaptive_fixed_support_tol=config.adaptive_fixed_support_tol,
        adaptive_phase0_tol=config.adaptive_phase0_tol,
        adaptive_phase2_after_plateau=config.adaptive_phase2_after_plateau,
        phase2_plateau_delta_e_tol=config.phase2_plateau_delta_e_tol,
        phase2_plateau_delta_rdm_tol=config.phase2_plateau_delta_rdm_tol,
        frame_leakage_tol=config.frame_leakage_tol,
        frame_strict=config.frame_strict,
        failure_capture_dir=config.failure_capture_dir,
        dm_mixing_alpha=config.dm_mixing_alpha,
        invariance_atol=config.invariance_atol,
        rdm_backend=config.rdm_backend,
        phase1_state_dir=scratch_dir,
    )

    bars = _progress.make(["parent"] + [f"F{i}" for i in range(n_frags)], progress,
                          headline="parent",
                          title=f"frasci.parent  {n_frags} fragments  "
                                f"macro<={config.max_cycle_macro}")
    frag_progress = _make_fragment_progress(bars)

    # kernel_fns is defined before the installer below (rather than after,
    # as it once was) so the escape controller's Phase-2 guard can close
    # over this SAME list object and read it by reference once populated --
    # the guard only ever runs later, during las.kernel(), by which point
    # the list below is fully populated.
    kernel_fns = []
    escape_log: list = []
    g_int_holder: dict = {}
    tail_policy = config.tail_convergence
    tail_enabled = tail_policy is not None and tail_policy.mode != "off"
    macro_trace: list = []
    segment_holder = {"n": 0}
    if config.escape_max_attempts is not None:
        frame_packets = _install_safeguarded_escape_controller(
            las, mo, n_frags, config.frame_leakage_tol, kernel_fns=kernel_fns,
            master_seed=config.rng_seed, max_attempts=config.escape_max_attempts,
            conv_tol_grad=config.conv_tol_grad, g_int_holder=g_int_holder,
            escape_log=escape_log)
    else:
        frame_packets = (_install_frame_publisher(
            las, mo, n_frags, config.frame_leakage_tol)
            if config.adaptive_phase_policy and config.orbopt else None)
    if config.orbopt:
        # EXPERIMENTAL, release-only, not yet validated for live: previously
        # gated on `config.adaptive_phase_policy and config.orbopt`, but
        # `_keep_final_frame_uncanonicalized`'s own rationale (mrh's terminal
        # canonicalize() rotates orbitals AFTER each fragment kernel's own
        # (determinants, coefficients, orbital_rotation) was already saved,
        # with no subsequent kernel call to compose the extra rotation into
        # U_cached) is generic to any orbopt=True solve, not adaptive-specific.
        _keep_final_frame_uncanonicalized(las)

    n_cpu_workers = n_frags if config.n_cpu_workers is None else config.n_cpu_workers
    if n_cpu_workers > 0:
        dispatch = parallel_fragment_dispatch(
            n_cpu_workers, n_frags, kernel_kwargs_shared,
            on_progress=frag_progress, initial_states=warm_states,
            frame_packets=frame_packets)
    else:
        dispatch = contextlib.nullcontext((None, None))

    with bars, bars.capture(), dispatch as (proxy_kernels, _records):
        if bars.mode != "off":
            las.stdout = _progress.MacroTap(bars, "parent", config.max_cycle_macro)
        if config.escape_max_attempts is not None or tail_enabled:
            # Compose AFTER MacroTap (if installed) rather than before, so
            # neither tap ever clobbers the other regardless of `progress`.
            las.stdout = _GIntTap(
                las.stdout, g_int_holder,
                trace=macro_trace if tail_enabled else None,
                segment_holder=segment_holder if tail_enabled else None)
        for i in range(n_frags):
            if proxy_kernels is None:
                kernel_fn = make_phased_kernel(
                    fragment_idx=i, log_callback=frag_progress(i),
                    frame_provider=(None if frame_packets is None
                                    else lambda i=i: frame_packets[i]),
                    **kernel_kwargs_shared)
                if warm_states is not None:
                    kernel_fn.state.update(warm_states[i])
            else:
                kernel_fn = proxy_kernels[i]
            kernel_fns.append(kernel_fn)
            las.fciboxes[i] = make_fcibox(
                mf.mol, kernel=kernel_fn, spin=spin_sub[i] - 1, smult=spin_sub[i])
        patch_large_fragment_aufbau(las, ncas_sub, nelec_per_frag)
        las.kernel(mo)

        convergence_diagnostics = None
        extra_macro_cycles = 0
        if tail_enabled:
            # Placed before Phase-2 finalization below: if this reaches
            # strict convergence, `las.converged` is already True by the
            # time the existing Phase-2 gate checks it, so that gate's
            # present semantics are unaffected -- it never learns or cares
            # how convergence was reached.
            convergence_diagnostics, extra_macro_cycles = _run_tail_confirmation(
                las, policy=tail_policy, conv_tol_grad=config.conv_tol_grad,
                trace=macro_trace, segment_holder=segment_holder)
            las.max_cycle_macro = config.max_cycle_macro

        # MRH ends an RDM-JK cycle immediately when its own density/energy
        # tolerances are met.  Therefore a policy that notices a plateau only
        # on the *next* fragment call can otherwise lose its promised final
        # Phase-2 promotion: that next call never happens.  Once the ordinary
        # LASSCF solve has provisionally converged, explicitly request Phase 2
        # and resume from the converged MOs and RDMs until both the promoted
        # support and LASSCF have reconverged.  Usually this is one short pass;
        # the bounded retries also cover the three-stable-call debounce.
        if (config.adaptive_phase_policy
                and config.adaptive_phase2_after_plateau
                and getattr(las, "converged", False)):
            def finalized():
                return (bool(getattr(las, "converged", False))
                        and all(bool(fn.state.get("phase2_episode_complete"))
                                for fn in kernel_fns))

            for _ in range(6):
                if finalized():
                    break
                for fn in kernel_fns:
                    state = fn.state
                    if (not state.get("phase2_promoted_ever")
                            and not state.get("post_phase2_active")
                            and not state.get("phase2_episode_complete")):
                        state["force_phase2_next"] = True
                las.kernel(
                    las.mo_coeff, casdm1frs=las.casdm1frs,
                    casdm2fr=las.casdm2fr)
            if not finalized():
                raise RuntimeError(
                    "adaptive Phase-2 finalization did not reconverge within "
                    "6 continuation passes")
        if bars.mode != "off":
            # MacroTap only fires on a "LASSCF macro N :" line, so once the
            # loop stops nothing else updates the bar. One plain line here
            # says whether it converged, rather than changing the bar itself.
            bars.write(
                "parent: converged" if getattr(las, "converged", None)
                else "parent: stopped (not converged)")

    e_tot = float(las.e_tot)
    converged = bool(getattr(las, "converged", None))

    convergence_status = convergence_diagnostics.status if convergence_diagnostics else None
    needs_more_iterations = (convergence_diagnostics.needs_more_iterations
                             if convergence_diagnostics else False)
    convergence_diagnostics_dict = (dataclasses.asdict(convergence_diagnostics)
                                    if convergence_diagnostics else None)
    initial_macro_budget = config.max_cycle_macro
    total_macro_cycles = initial_macro_budget + extra_macro_cycles

    fragments = []
    for i in range(n_frags):
        data = np.load(Path(scratch_dir) / f"fragment_{i}.npz")
        fragments.append(FragmentResult(
            determinants=data["determinants"],
            coefficients=data["coefficients"],
            orbital_rotation=data["orbital_rotation"],
            n_orb=int(data["n_orb"]),
            n_alpha=int(data["n_alpha"]),
            n_beta=int(data["n_beta"]),
            energy=float(data["energy"]),
        ))
    fragments = tuple(fragments)
    dets_per_frag = tuple(int(fr.determinants.shape[0]) for fr in fragments)

    if save_dets_to is not None:
        save_dets_to = Path(save_dets_to)
        save_dets_to.mkdir(parents=True, exist_ok=True)
        np.savez(save_dets_to / "checkpoint.npz", mo_coeff=las.mo_coeff)
        metadata = {
            "ncas_sub": list(ncas_sub),
            "nelec_per_frag": [list(ne) for ne in nelec_per_frag],
            "spin_sub": list(spin_sub),
            "e_tot": e_tot,
            "converged": converged,
            "method": "frasci.parent.solve (phased LASSCF-COO)",
            "convergence_status": convergence_status,
            "needs_more_iterations": needs_more_iterations,
            "initial_macro_budget": initial_macro_budget,
            "extra_macro_cycles": extra_macro_cycles,
            "total_macro_cycles": total_macro_cycles,
        }
        (save_dets_to / "checkpoint_metadata.json").write_text(
            json.dumps(metadata, indent=2))
        (save_dets_to / "result.json").write_text(json.dumps(
            {"result": {"e_tot": float(e_tot)}}, indent=2))
        shutil.copytree(scratch_dir, save_dets_to / "phase1_state")
        if tail_enabled:
            _atomic_write_json(
                save_dets_to / "g_int_trace.json",
                [dataclasses.asdict(r) for r in macro_trace])
            _atomic_write_json(
                save_dets_to / "convergence_diagnostics.json",
                convergence_diagnostics_dict)

    # Deliberately not a try/finally: cleanup only runs on success so a
    # failure's scratch dir (TrimCI checkpoints, fragment npz files) survives
    # for post-mortem debugging -- see frasci/recovery/__init__.py for the
    # same pattern and its own review history.
    shutil.rmtree(scratch_dir, ignore_errors=True)

    return Parent(
        mo_coeff=las.mo_coeff,
        fragments=fragments,
        ncas_sub=ncas_sub,
        nelec_per_frag=tuple(map(tuple, nelec_per_frag)),
        spin_sub=tuple(spin_sub),
        e_tot=e_tot,
        converged=converged,
        dets_per_frag=dets_per_frag,
        escape_exhausted=any(r.get("reason") == "retry_budget_exhausted"
                             for r in escape_log),
        escape_log=tuple(escape_log),
        convergence_status=convergence_status,
        needs_more_iterations=needs_more_iterations,
        initial_macro_budget=initial_macro_budget,
        extra_macro_cycles=extra_macro_cycles,
        total_macro_cycles=total_macro_cycles,
        convergence_diagnostics=convergence_diagnostics_dict,
    )
