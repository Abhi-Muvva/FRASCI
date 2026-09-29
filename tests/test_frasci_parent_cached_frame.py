"""Focused contracts for the opt-in adaptive cached-frame parent path."""
from types import SimpleNamespace

import numpy as np
import pytest

import trimci
from frasci.parent import (
    _frame_packet,
    _install_frame_publisher,
    _keep_final_frame_uncanonicalized,
)
from frasci.parent._kernel import make_phased_kernel
from frasci.parent._parallel import _make_proxy_kernel, parallel_fragment_dispatch


def _h4_integrals():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    n = mol.nao_nr()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n,) * 4)
    return n, h1, eri


def _adaptive_kernel(packet_slot, seen, **overrides):
    kwargs = dict(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=12, max_dets_phase2=24,
        orbopt=True, orbopt_cycles=3, warm_start_kappa=True,
        warm_start_dets=True, adaptive_phase_policy=True,
        adaptive_fixed_support_tol=1e-4, adaptive_phase0_tol=1e-2,
        frame_provider=lambda: packet_slot[0], frame_leakage_tol=5e-2,
        seed=1, log_callback=lambda *args: seen.append(args[-1]))
    kwargs.update(overrides)
    return make_phased_kernel(**kwargs)


def _rotate_eri(eri, U):
    return np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl", U, U, U, U, eri, optimize=True)


def test_reusable_external_rotation_preserves_support_and_common_frame_drift():
    n, h1, eri = _h4_integrals()
    C0 = np.eye(n)
    packets = [_frame_packet(C0, C0, np.eye(n), 0, 0, 5e-2)]
    seen = []
    kernel = _adaptive_kernel(packets, seen)
    args = (n, (2, 2), 0.0, np.stack([h1, h1]), eri)
    kernel(*args)
    U_cache = kernel.state["U_cached"].copy()
    support = kernel.state["dets_prev"].copy()

    theta = 0.13
    R = np.eye(n)
    R[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]
    C1 = C0 @ R
    packets[0] = _frame_packet(C0, C1, np.eye(n), 0, 1, 5e-2)
    h1_curr = R.T @ h1 @ R
    eri_curr = _rotate_eri(eri, R)
    kernel(n, (2, 2), 0.0, np.stack([h1_curr, h1_curr]), eri_curr)

    row = seen[-1]
    assert row["frame_reusable"] is True
    assert row["frame_invalidated"] is False
    assert row["phase_policy_action"] == "fixed_support"
    assert row["delta_h_rel"] < 1e-12  # raw h1 matrices are not equal
    assert not np.allclose(h1_curr, h1)
    assert np.array_equal(kernel.state["dets_prev"], support)
    assert kernel.state["U_cached"] == pytest.approx(R.T @ U_cache, abs=1e-10)


def test_run_expansion_continuation_composes_nonidentity_bfgs_step():
    """The low-level Phase-1-only continuation helper (preserved per the
    2026-09-14 production policy for future explicitly-justified use, but no
    longer wired into the ordinary per-call dispatch -- see
    make_phased_kernel's docstring) still genuinely runs BFGS orbital
    optimization and composes a nonidentity rotation when asked to, called
    directly rather than through the kernel's dispatch."""
    from frasci.parent._kernel import _run_expansion_continuation

    n, h1, eri = _h4_integrals()
    seen = []
    kernel = _adaptive_kernel([_frame_packet(np.eye(n), np.eye(n), np.eye(n), 0, 0, 5e-2)], seen)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)
    dets, coeffs = kernel.state["dets_prev"], kernel.state["coeffs_prev"]

    h1_mid = h1.copy()
    h1_mid[0, 0] += 5e-3 * np.linalg.norm(h1)
    result = _run_expansion_continuation(
        h1_mid, eri, dets, coeffs, n, phase_index=1, threshold=1e-6,
        growth_factor=1.1, max_n_dets=12, orbital_optimization=True,
        orbital_opt_max_iter=3)

    assert np.linalg.norm(np.asarray(result.orbital_rotation) - np.eye(n)) > 1e-8
    assert result.expansion_residual < 1e-6


def test_unsafe_subspace_transition_atomically_discards_cached_episode():
    n, h1, eri = _h4_integrals()
    C0 = np.eye(n + 1, n)
    packets = [_frame_packet(C0, C0, np.eye(n + 1), 0, 0, 5e-2)]
    seen = []
    kernel = _adaptive_kernel(packets, seen)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)

    C1 = C0.copy()
    theta = 0.3
    C1[:, 0] = 0.0
    C1[0, 0], C1[-1, 0] = np.cos(theta), np.sin(theta)
    packets[0] = _frame_packet(C0, C1, np.eye(n + 1), 0, 1, 5e-2)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)

    row = seen[-1]
    assert row["frame_reusable"] is False
    assert row["frame_invalidated"] is True
    assert row["frame_invalidations"] == 1
    assert row["phase_policy_action"] == "phase0_phase1"
    assert row["delta_h_rel"] is None
    assert row["support_jaccard"] is None
    assert kernel.state["external_frame_id"] == 1


def test_call_after_invalidation_reset_is_fixed_support():
    """Required test 7: the call(s) following an invalidating frame
    transition -- once a fresh support exists again -- go back to
    fixed_support, exactly like any other same-episode call."""
    n, h1, eri = _h4_integrals()
    C0 = np.eye(n + 1, n)
    packets = [_frame_packet(C0, C0, np.eye(n + 1), 0, 0, 5e-2)]
    seen = []
    kernel = _adaptive_kernel(packets, seen)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)

    C1 = C0.copy()
    theta = 0.3
    C1[:, 0] = 0.0
    C1[0, 0], C1[-1, 0] = np.cos(theta), np.sin(theta)
    packets[0] = _frame_packet(C0, C1, np.eye(n + 1), 0, 1, 5e-2)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)  # invalidates, forces phase0_phase1
    assert seen[-1]["phase_policy_action"] == "phase0_phase1"

    packets[0] = _frame_packet(C1, C1, np.eye(n + 1), 1, 1, 5e-2)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)
    kernel(n, (2, 2), 0.0, np.stack([h1, h1]), eri)

    assert seen[-2]["phase_policy_action"] == "fixed_support"
    assert seen[-1]["phase_policy_action"] == "fixed_support"
    assert seen[-1]["support_jaccard"] == pytest.approx(1.0)


def test_two_episodes_each_get_their_own_forced_promotion():
    """An invalidating external frame transition is the only thing that opens
    a genuinely new episode (a bare h1 jump alone does not, see
    test_adaptive_kernel_phase2_promotion_persists_and_reconverges in
    test_frasci_parent_kernel.py) -- and Phase-2 tracking resets with it, so
    the parent's finalizer may request -- and get -- a fresh promotion in the
    new episode after a previous one already completed in the old episode."""
    n, h1, eri = _h4_integrals()
    C0 = np.eye(n + 1, n)
    packets = [_frame_packet(C0, C0, np.eye(n + 1), 0, 0, 5e-2)]
    seen = []
    kernel = _adaptive_kernel(
        packets, seen, adaptive_phase2_after_plateau=True,
        phase2_plateau_delta_e_tol=1e-3, phase2_plateau_delta_rdm_tol=1e-2)
    args = (n, (2, 2), 0.0, np.stack([h1, h1]), eri)
    kernel(*args)  # episode 1 cold start

    kernel.state["force_phase2_next"] = True
    kernel(*args)  # episode 1 promotion
    assert seen[-1]["phase_policy_action"] == "phase2_promote"
    assert kernel.state["phase2_promoted_ever"] is True

    # Invalidate: a genuinely different subspace, not representable in the
    # cached frame.
    C1 = C0.copy()
    theta = 0.3
    C1[:, 0] = 0.0
    C1[0, 0], C1[-1, 0] = np.cos(theta), np.sin(theta)
    packets[0] = _frame_packet(C0, C1, np.eye(n + 1), 0, 1, 5e-2)
    kernel(*args)  # episode 2 cold start (forced by invalidation)
    assert seen[-1]["phase_policy_action"] == "phase0_phase1"
    assert kernel.state["phase2_promoted_ever"] is False  # reset for the new episode
    assert kernel.state["phase2_episode_complete"] is False

    packets[0] = _frame_packet(C1, C1, np.eye(n + 1), 1, 1, 5e-2)
    kernel.state["force_phase2_next"] = True
    kernel(*args)  # episode 2 promotion -- a fresh promotion, not a repeat

    assert seen[-1]["phase_policy_action"] == "phase2_promote"
    actions = [row["phase_policy_action"] for row in seen]
    assert actions.count("phase2_promote") == 2
    assert actions.count("phase0_phase1") == 2


def test_completed_phase2_episode_is_debounced_until_rediscovery():
    """A forced promotion mid-episode debounces correctly: it fires exactly
    once even though the caller could in principle set force_phase2_next
    again immediately (the kernel does not re-arm on its own), and
    post-Phase-2 convergence is declared exactly once."""
    n, h1, eri = _h4_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, max_dets_phase0=6,
        max_dets_phase1=6, max_dets_phase2=12, orbopt=False,
        warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, adaptive_phase2_after_plateau=True,
        phase2_plateau_delta_e_tol=1e-3,
        phase2_plateau_delta_rdm_tol=1e-2,
        log_callback=lambda *args: seen.append(args[-1]), seed=2)
    args = (n, (2, 2), 0.0, np.stack([h1, h1]), eri)
    kernel(*args)
    kernel.state["force_phase2_next"] = True
    for _ in range(11):
        kernel(*args)

    actions = [row["phase_policy_action"] for row in seen]
    assert actions.count("phase2_promote") == 1
    assert sum(bool(row["post_phase2_converged_now"]) for row in seen) == 1
    assert kernel.state["phase2_episode_complete"] is True


def test_parent_can_force_pending_phase2_at_lasscf_convergence_boundary():
    n, h1, eri = _h4_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, max_dets_phase0=6,
        max_dets_phase1=6, max_dets_phase2=12, orbopt=False,
        warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, adaptive_phase2_after_plateau=True,
        adaptive_fixed_support_tol=1e-4, adaptive_phase0_tol=1e-2,
        # Deliberately impossible natural plateau: this isolates the explicit
        # parent finalization request from the ordinary tolerance path.
        phase2_plateau_delta_e_tol=0.0,
        phase2_plateau_delta_rdm_tol=0.0,
        log_callback=lambda *args: seen.append(args[-1]), seed=2)
    args = (n, (2, 2), 0.0, np.stack([h1, h1]), eri)
    kernel(*args)
    kernel.state["force_phase2_next"] = True
    kernel(*args)

    assert seen[-1]["phase_policy_action"] == "phase2_promote"
    assert kernel.state["force_phase2_next"] is False
    assert kernel.state["post_phase2_active"] is True


def test_finalizer_promotes_each_correlated_fragment_once_and_skips_filled_fragment(tmp_path):
    """Required test 12: frasci.parent.solve()'s post-convergence finalizer
    (frasci/parent/__init__.py's retry loop, gated on las.converged) promotes
    each correlated fragment exactly once. A filled (1-determinant) fragment's
    promotion request is satisfied as an immediate no-op instead of running
    Phase-2 continuation on it (make_phased_kernel's ci_dim<=1 short-circuit)
    -- "must not run for a one-determinant filled fragment as expensive no-op
    work."""
    from pyscf import ao2mo, gto, scf
    from pyscf.tools.fcidump import from_integrals

    import frasci.parent as parent_module
    from frasci.parent import ParentConfig

    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
                basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    assert mf.converged
    n_orb = mol.nao_nr()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False)

    fcidump_path = tmp_path / "fcidump"
    from_integrals(str(fcidump_path), h1, eri, n_orb, mol.nelectron, nuc=0.0, ms=0)

    # Fragment 0: 2 orbitals, fully filled (na=nb=2 -> ci_dim=1, nothing to
    # correlate). Fragment 1: the other 4 orbitals, correlated (na=nb=1,
    # ci_dim=C(4,1)^2=16). 4+2=6 electrons matches H6 neutral.
    orbital_lists = [[0, 1], [2, 3, 4, 5]]
    nelec_per_frag = [(2, 2), (1, 1)]
    spin_sub = [1, 1]

    calls = []
    real_factory = parent_module._make_fragment_progress

    def capturing_factory(bars):
        def for_fragment(i):
            def record(norb, nelec, n_dets, energy, wall, extras):
                calls.append({"fragment_idx": i, "n_dets": n_dets, **extras})
            return record
        return for_fragment

    parent_module._make_fragment_progress = capturing_factory
    try:
        config = ParentConfig(
            threshold=1e-6, max_cycle_macro=25, max_cycle_rdmjk=20,
            orbopt=True, orbopt_cycles=2, phase0_dets=8, phase1_dets=8,
            discovery_num_runs=2, refine_num_runs=1,
            adaptive_phase_policy=True, adaptive_phase2_after_plateau=True,
            phase2_dets=16, rng_seed=1,
        )
        parent = parent_module.solve(
            str(fcidump_path), orbital_lists, nelec_per_frag, spin_sub, config)
    finally:
        parent_module._make_fragment_progress = real_factory

    assert parent.converged is True

    by_fragment = {0: [], 1: []}
    for row in calls:
        by_fragment[row["fragment_idx"]].append(row)

    # parent.converged is True (asserted above) only because solve()'s
    # finalizer successfully drove EVERY fragment's live state to
    # phase2_episode_complete=True within its 6-continuation-pass budget (it
    # raises otherwise) -- so that overall contract is already proven. The
    # checks below additionally pin down the two DIFFERENT ways each fragment
    # type is required to reach it.
    for i in (0, 1):
        actions = [row["phase_policy_action"] for row in by_fragment[i]]
        # Real orbital optimization can genuinely invalidate the cached frame
        # more than once before outer convergence (each is a legitimate new
        # episode, not a bug) -- this test's focus is the finalizer's
        # Phase-2 behavior, which is unaffected by how many rediscoveries
        # happened on the way to convergence.
        assert actions.count("phase0_phase1") >= 1
        assert actions.count("phase2_promote") == (0 if i == 0 else 1), (
            i, actions)

    # Fragment 0 (filled, ci_dim<=1): promotion is satisfied as an immediate
    # no-op the moment it is requested -- no "phase2_promote" action, no
    # growth, and phase2_episode_complete becomes True on that very same call
    # (the short-circuit runs before this call's own telemetry is built,
    # unlike the genuine promotion path below).
    assert all(row["n_dets"] <= 1 for row in by_fragment[0])
    assert any(row["phase2_episode_complete"] for row in by_fragment[0])

    # Fragment 1 (correlated): a genuine promotion, followed by the
    # three-stable-call contract actually completing (post_phase2_converged_now
    # is a post-call-fresh field, unlike phase2_episode_complete/post_phase2_active
    # in the same row, which are pre-call snapshots -- see make_phased_kernel).
    assert any(row["post_phase2_converged_now"] for row in by_fragment[1])


def test_adaptive_parent_skips_unreported_final_canonical_rotation():
    class FakeLas:
        mo_coeff = np.eye(4)

    las = FakeLas()
    _keep_final_frame_uncanonicalized(las)
    dm1 = [np.ones((1, 1, 2, 2))]
    dm2 = [np.ones((1, 2, 2, 2, 2))]
    result = las.canonicalize(
        mo_coeff=las.mo_coeff, casdm1frs=dm1, casdm2fr=dm2,
        h2eff_sub="h2")

    assert result[0] is las.mo_coeff
    assert result[3] is dm1
    assert result[4] is dm2
    assert result[5] == "h2"


def test_frame_publisher_tracks_the_update_mo_eri_boundary():
    class FakeHop:
        def update_mo_eri(self, x, h2):
            return x, h2

    class FakeLas:
        def __init__(self):
            self._scf = SimpleNamespace(get_ovlp=lambda: np.eye(4))

        def get_mo_slice(self, i, mo_coeff):
            return mo_coeff[:, 2 * i:2 * i + 2]

        def get_hop(self, *args, **kwargs):
            return FakeHop()

    las = FakeLas()
    mo0 = np.eye(4)
    packets = _install_frame_publisher(las, mo0, 2, 5e-2)
    assert all(p["frame_id_curr"] == 0 for p in packets)
    hop = las.get_hop(mo_coeff=mo0)
    theta = 0.2
    mo1 = np.eye(4)
    mo1[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]]
    hop.update_mo_eri(mo1, None)
    assert all(p["frame_id_prev"] == 0 for p in packets)
    assert all(p["frame_id_curr"] == 1 for p in packets)
    assert packets[0]["transport"].reusable is True


def test_process_proxy_sends_current_packet_and_roundtrips_state():
    packet = {"frame_id_prev": 2, "frame_id_curr": 3, "transport": "sentinel"}
    warm_states = [{"external_frame_id": 2}]
    records = []

    class Future:
        def result(self):
            return (1.0, "dm1", "dm2", {"external_frame_id": 3},
                    99, 1.0, 2.0, None)

    class Executor:
        submitted = None

        def submit(self, *args):
            self.submitted = args
            return Future()

    executor = Executor()
    proxy = _make_proxy_kernel(
        executor, 0, {}, warm_states, records, frame_packets=[packet])
    assert proxy(4, (2, 2), 0.0, "h1", "h2") == (1.0, "dm1", "dm2")
    assert executor.submitted[3] is packet
    assert warm_states[0]["external_frame_id"] == 3
    assert records[0]["pid"] == 99


def test_real_process_pool_roundtrips_cached_frame_state():
    n, h1, eri = _h4_integrals()
    C0 = np.eye(n)
    packets = [_frame_packet(C0, C0, np.eye(n), 0, 0, 5e-2)]
    seen = []
    kwargs = dict(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=12, orbopt=True,
        orbopt_cycles=3, warm_start_kappa=True, warm_start_dets=True,
        adaptive_phase_policy=True, adaptive_fixed_support_tol=1e-4,
        adaptive_phase0_tol=1e-2, frame_leakage_tol=5e-2,
        frame_strict=True, seed=1)

    def callback_factory(_i):
        return lambda *args: seen.append(args[-1])

    with parallel_fragment_dispatch(
            1, 1, kwargs, log_banner=False, on_progress=callback_factory,
        frame_packets=packets) as (proxies, records):
        args = (n, (2, 2), 0.0, np.stack([h1, h1]), eri)
        proxies[0](*args)
        theta = 0.07
        R = np.eye(n)
        R[:2, :2] = [[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]]
        packets[0] = _frame_packet(C0, C0 @ R, np.eye(n), 0, 1, 5e-2)
        h1_curr = R.T @ h1 @ R
        eri_curr = _rotate_eri(eri, R)
        proxies[0](n, (2, 2), 0.0, np.stack([h1_curr, h1_curr]), eri_curr)

    assert [row["phase_policy_action"] for row in seen] == [
        "phase0_phase1", "fixed_support"]
    assert seen[1]["external_frame_id"] == 1
    assert seen[1]["frame_reusable"] is True
    assert seen[1]["delta_h_rel"] == pytest.approx(0.0, abs=1e-14)
    assert len(records) == 2
