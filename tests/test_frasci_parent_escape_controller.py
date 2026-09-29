"""frasci.parent's opt-in macro-boundary escape rule (2026-09-17 contract):
the energy-backtracking safeguard (promoted from report-only to production)
plus a bounded, global escape controller with a reproducible per-attempt
episode-seed schedule. ``decide_escape``/``_derive_episode_seed``/
``_phase2_started`` are pure and covered directly; the composed installer's
wiring is covered with a minimal fake ``las``/``H_op``, mirroring the
already-validated pattern in
``docs/reports/frasci_fe4s4_macro_boundary_adaptive_rediscovery_2026-09-16.py``'s
``self_check_installer`` -- no real TrimCI/mrh time spent.
"""
from __future__ import annotations

import numpy as np

from frasci.parent import (
    _derive_episode_seed,
    _install_safeguarded_escape_controller,
    _phase2_started,
    decide_escape,
)
from frasci.parent._frame import FrameTransport


def test_decide_escape_triggers_on_full_rejection_nonconverged_gint():
    ok, reason = decide_escape(
        any_downhill=False, ended_in_restore=True, g_int=0.05,
        conv_tol_grad=1e-4, attempts_used=1, max_attempts=2, phase2_started=False)
    assert ok is True
    assert reason == "rejected_restore_keyframe_nonconverged_gint"


def test_decide_escape_never_triggers_when_any_trial_went_downhill():
    ok, reason = decide_escape(
        any_downhill=True, ended_in_restore=False, g_int=0.05,
        conv_tol_grad=1e-4, attempts_used=1, max_attempts=2, phase2_started=False)
    assert ok is False
    assert reason == "downhill_step_accepted_this_macro"


def test_decide_escape_never_triggers_on_genuine_convergence():
    ok, reason = decide_escape(
        any_downhill=False, ended_in_restore=True, g_int=1e-5,
        conv_tol_grad=1e-4, attempts_used=1, max_attempts=2, phase2_started=False)
    assert ok is False
    assert reason == "g_int_at_or_below_conv_tol_grad"


def test_decide_escape_suppressed_once_phase2_has_started():
    ok, reason = decide_escape(
        any_downhill=False, ended_in_restore=True, g_int=0.5,
        conv_tol_grad=1e-4, attempts_used=1, max_attempts=2, phase2_started=True)
    assert ok is False
    assert reason == "phase2_started_escape_suppressed"


def test_decide_escape_max_attempts_one_is_a_valid_safeguard_only_control():
    # max_attempts=1: the energy-backtracking safeguard still rejects and
    # restores (that happens upstream of decide_escape, unconditionally),
    # but the escape controller must report exhaustion on the VERY FIRST
    # qualifying rejection -- zero rediscoveries ever, not "after N misses".
    ok, reason = decide_escape(
        any_downhill=False, ended_in_restore=True, g_int=0.5,
        conv_tol_grad=1e-4, attempts_used=1, max_attempts=1, phase2_started=False)
    assert ok is False
    assert reason == "retry_budget_exhausted"


def test_decide_escape_reports_explicit_exhaustion_reason():
    ok, reason = decide_escape(
        any_downhill=False, ended_in_restore=True, g_int=0.5,
        conv_tol_grad=1e-4, attempts_used=2, max_attempts=2, phase2_started=False)
    assert ok is False
    assert reason == "retry_budget_exhausted"


def test_episode_seed_attempt_one_is_master_seed_unchanged():
    # A solve that never deadlocks must be byte-for-byte unaffected by this
    # function existing.
    assert _derive_episode_seed(12345, 1) == 12345
    assert _derive_episode_seed(12345, 0) == 12345


def test_episode_seed_attempt_two_is_distinct_and_reproducible():
    s2a = _derive_episode_seed(12345, 2)
    s2b = _derive_episode_seed(12345, 2)
    assert s2a == s2b, "must be a pure, reproducible function"
    assert s2a != 12345


def test_episode_seed_never_collides_with_trimci_frascis_own_restart_stride():
    # TrimCI-FRASCI's per-restart derivation inside ONE discovery call (fix
    # branch fix/cpu-rng-propagation, trimci_driver.py): master_seed + r *
    # 0x9e3779b97f4a7c15. The episode schedule must use a different mixing
    # path so the two sequences can never coincide.
    master = 12345
    for n_attempt in range(2, 6):
        episode = _derive_episode_seed(master, n_attempt)
        restart_stride = (master + n_attempt * 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
        assert episode != restart_stride


def test_phase2_started_true_if_any_fragment_has_begun_finalization():
    class Fake:
        def __init__(self, **state):
            self.state = state

    none_started = [Fake(force_phase2_next=False, post_phase2_active=False,
                         phase2_promoted_ever=False, phase2_episode_complete=False)
                    for _ in range(3)]
    assert _phase2_started(none_started) is False

    one_started = none_started[:2] + [Fake(force_phase2_next=True, post_phase2_active=False,
                                           phase2_promoted_ever=False,
                                           phase2_episode_complete=False)]
    assert _phase2_started(one_started) is True


class _FakeHOp:
    def __init__(self, e_tot):
        self.e_tot = e_tot
        self.update_mo_eri = None


class _FakeLAS:
    def __init__(self):
        self.mo_coeff = np.eye(2)
        self._keyframe_e = -1.0
        self.get_hop = None

    def make_casdm1s_sub(self, casdm1frs=None):
        return None

    def get_veff(self, dm=None):
        return None

    def make_rdm1s(self, mo_coeff=None, casdm1s_sub=None):
        return None

    def energy_nuc(self):
        return 0.0

    def energy_elec(self, mo_coeff=None, h2eff=None, casdm1frs=None,
                    casdm2fr=None, veff=None):
        # The fake's h2eff carries the trial's intended energy directly.
        return float(h2eff)


def _fake_install_frame_publisher(las, mo_initial, n_frags, leakage_tol):
    packet = {"frame_id_prev": 0, "frame_id_curr": 1,
             "transport": FrameTransport(
                 same_frame=False, Q=np.eye(2), singular_values=np.ones(2),
                 principal_angles_rad=np.zeros(2), orthogonality_err=0.0,
                 leakage=0.001, usable_transform=np.eye(2), reusable=True)}
    packets = [dict(packet) for _ in range(n_frags)]

    def real_get_hop(*args, **kwargs):
        H_op = _FakeHOp(las._keyframe_e)

        def real_update(x, h2eff_sub):
            return las.mo_coeff, h2eff_sub

        H_op.update_mo_eri = real_update
        return H_op

    las.get_hop = real_get_hop
    return packets


def _install(monkeypatch, *, kernel_fns, master_seed=12345, max_attempts=2,
            g_int_holder, escape_log):
    monkeypatch.setattr(
        "frasci.parent._install_frame_publisher", _fake_install_frame_publisher)
    las = _FakeLAS()
    packets = _install_safeguarded_escape_controller(
        las, None, n_frags=2, leakage_tol=0.05, kernel_fns=kernel_fns,
        master_seed=master_seed, max_attempts=max_attempts, conv_tol_grad=1e-4,
        g_int_holder=g_int_holder, escape_log=escape_log)
    return las, packets


def test_installer_leaves_packets_untouched_on_a_downhill_macro(monkeypatch):
    g_int_holder, escape_log = {"macro": 0, "g_int": 0.5}, []
    las, packets = _install(monkeypatch, kernel_fns=[], g_int_holder=g_int_holder,
                            escape_log=escape_log)
    assert packets[0]["episode_seed"] == 12345  # attempt 1 == master, unchanged

    H_op = las.get_hop(mo_coeff=las.mo_coeff)
    H_op.update_mo_eri(np.array([1.0]), -2.0)  # below keyframe -1.0: downhill
    assert packets[0]["transport"].reusable is True
    assert escape_log[-1]["invalidated"] is False
    assert escape_log[-1]["reason"] == "downhill_step_accepted_this_macro"


def test_installer_forces_distinct_seed_invalidation_on_uphill_exhausted_deadlock(monkeypatch):
    g_int_holder, escape_log = {"macro": 0, "g_int": 0.5}, []
    las, packets = _install(monkeypatch, kernel_fns=[], g_int_holder=g_int_holder,
                            escape_log=escape_log)

    g_int_holder["macro"] = 1
    g_int_holder["g_int"] = 0.3  # above conv_tol_grad=1e-4
    H_op = las.get_hop(mo_coeff=las.mo_coeff)
    H_op.update_mo_eri(np.array([1.0]), 5.0)  # every halving stays uphill of -1.0

    assert escape_log[-1]["invalidated"] is True
    assert escape_log[-1]["reason"] == "rejected_restore_keyframe_nonconverged_gint"
    for i in range(2):
        assert packets[i]["transport"].reusable is False
        assert packets[i]["episode_seed"] == escape_log[-1]["next_episode_seed"]
    assert escape_log[-1]["next_episode_seed"] != 12345


def test_installer_never_invalidates_once_phase2_has_started(monkeypatch):
    class FakeFragState:
        def __init__(self):
            self.state = {"force_phase2_next": True, "post_phase2_active": False,
                          "phase2_promoted_ever": False, "phase2_episode_complete": False}

    kernel_fns = [FakeFragState(), FakeFragState()]
    g_int_holder, escape_log = {"macro": 0, "g_int": 0.5}, []
    las, packets = _install(monkeypatch, kernel_fns=kernel_fns, g_int_holder=g_int_holder,
                            escape_log=escape_log)

    g_int_holder["macro"] = 1
    g_int_holder["g_int"] = 0.3
    H_op = las.get_hop(mo_coeff=las.mo_coeff)
    H_op.update_mo_eri(np.array([1.0]), 5.0)  # would otherwise deadlock-trigger

    assert escape_log[-1]["invalidated"] is False
    assert escape_log[-1]["reason"] == "phase2_started_escape_suppressed"
    assert packets[0]["transport"].reusable is True


def test_installer_exhausts_after_max_attempts_with_explicit_reason(monkeypatch):
    g_int_holder, escape_log = {"macro": 0, "g_int": 0.5}, []
    las, packets = _install(monkeypatch, kernel_fns=[], max_attempts=2,
                            g_int_holder=g_int_holder, escape_log=escape_log)

    for macro in (1, 2):
        g_int_holder["macro"] = macro
        g_int_holder["g_int"] = 0.3
        H_op = las.get_hop(mo_coeff=las.mo_coeff)
        H_op.update_mo_eri(np.array([1.0]), 5.0)

    assert escape_log[0]["invalidated"] is True   # macro 1: the one allowed rescue
    assert escape_log[1]["invalidated"] is False  # macro 2: budget now exhausted
    assert escape_log[1]["reason"] == "retry_budget_exhausted"
    assert escape_log[1]["attempts_used"] == 2
    assert escape_log[1]["master_seed"] == 12345
    assert escape_log[1]["episode_seed"] == _derive_episode_seed(12345, 2)
