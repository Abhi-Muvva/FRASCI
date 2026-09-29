"""Bounded adaptive tail-confirmation protocol (2026-09-29 contract).

Pure classification/config tests exercise ``frasci.parent._convergence``
directly (no mrh/pyscf import -- keeps these fast and dependency-light).
Control-flow tests use a minimal fake ``las``, mirroring the pattern already
validated in ``test_frasci_parent_escape_controller.py``. The physical
regression test reuses the small alfe2 fixture already used by
``test_frasci_parent_persistence.py``.

CORE SCIENTIFIC RULE under test throughout: the tail policy may extend the
macrocycle budget, but a ``stationary_plateau`` must never be reported as
``strict_converged``, and ``Parent.converged`` must stay ``False`` for it.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from frasci.parent._convergence import (
    ConvergenceDiagnostics,
    MacroRecord,
    TailConvergencePolicy,
    classify_trace,
    decide_extension,
    rdm_block_change,
)
from frasci.parent import _run_tail_confirmation

CONV_TOL_GRAD = 1e-4


def _trace(g_values, e_values, *, segment=0):
    assert len(g_values) == len(e_values)
    return tuple(
        MacroRecord(segment=segment, local_macro=i, global_macro=i,
                    e_tot=e, g_int=g)
        for i, (e, g) in enumerate(zip(e_values, g_values)))


def _flat_energies(n, start=-300.1234567, step=1e-8):
    return [start - i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Pure classification tests
# ---------------------------------------------------------------------------

def test_strictly_converged_trace():
    g = [9e-5, 8e-5, 7e-5, 6e-5, 5e-5, 4.5e-5, 4e-5, 3.5e-5, 3.2e-5, 3.0e-5]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=True)
    assert d.status == "strict_converged"
    assert d.needs_more_iterations is False


def test_near_threshold_monotonically_improving_trace():
    g = [4.9e-4 * (0.95 ** i) for i in range(10)]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False)
    assert d.status == "near_threshold_progressing"
    assert d.needs_more_iterations is True


def test_true_stationary_plateau_candidate():
    g = [2.00e-4, 2.01e-4, 1.99e-4, 2.02e-4, 1.98e-4,
        2.00e-4, 2.01e-4, 1.99e-4, 2.00e-4, 2.00e-4]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False,
                       extension_blocks_run=0)
    assert d.status == "stationary_plateau_candidate"
    assert d.needs_more_iterations is True


def test_confirmed_stationary_plateau_after_extension_and_rdm_check():
    g = [2.00e-4, 2.01e-4, 1.99e-4, 2.02e-4, 1.98e-4,
        2.00e-4, 2.01e-4, 1.99e-4, 2.00e-4, 2.00e-4]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False,
                       extension_blocks_run=1, consecutive_plateau_blocks=0,
                       rdm_change=1e-6)
    assert d.status == "stationary_plateau"
    assert d.needs_more_iterations is False


def test_large_oscillation_unstable_trace():
    g = [4.5e-4, 2.0e-4, 4.8e-4, 2.2e-4, 3.0e-4, 4.9e-4, 2.1e-4, 4.6e-4, 2.5e-4, 4.0e-4]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False)
    assert d.status == "unstable_nonconverged"
    assert d.needs_more_iterations is False


def test_above_band_unconverged_trace():
    g = [9.0e-4, 8.8e-4, 8.6e-4, 8.5e-4, 8.5e-4, 8.4e-4, 8.3e-4, 8.3e-4, 8.2e-4, 8.2e-4]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False)
    assert d.status == "budget_exhausted"
    assert d.needs_more_iterations is True


def test_insufficient_history_trace_is_never_a_plateau():
    g = [3.5e-4, 3.2e-4, 3.0e-4]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False)
    assert d.insufficient_history is True
    assert d.status not in ("stationary_plateau", "stationary_plateau_candidate")
    assert d.status == "near_threshold_progressing"
    assert "insufficient_history_for_plateau_confirmation" in d.reasons


def test_odd_even_sawtooth_with_descending_envelope_is_progressing_not_plateau():
    g = [4.5e-4 - i * 1e-5 + (1e-5 if i % 2 == 0 else -1e-5) for i in range(10)]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False)
    assert d.status == "near_threshold_progressing"
    # Raw gradient values are preserved unmodified in the window/diagnostics.
    assert [r.g_int for r in d.window] == g


def test_h1diag3_trace_replay_classifies_as_near_threshold_progressing():
    # Compact fixture: macros 90-99 of the real fe4s4_h1diag3 cap=500,
    # max_cycle_macro=100 convergence probe (2026-09-23) --
    # results/large_system_scaling/logs/fe4s4_h1diag3_macro100_test.log.
    # Final |g_int|=3.5e-4, "LASSCF not converged after 100 cycles". The
    # final ten cycles improve |g_int| by ~24%, so this is a direct guard
    # against calling a still-descending trajectory a plateau.
    e = [-326.945910196675, -326.945911897129, -326.945913436429, -326.945914897535,
        -326.945916214411, -326.945917460849, -326.945918573669, -326.945919625224,
        -326.945920558219, -326.94592143907]
    g = [0.000461984263513152, 0.000453697841027435, 0.000421784389158869,
        0.000432064826339662, 0.000389390172649424, 0.000404484075550516,
        0.000358863517271489, 0.000376942486563435, 0.000330545289223035,
        0.000350352463268102]
    improvement = (g[0] - g[-1]) / g[0]
    assert 0.20 < improvement < 0.28  # "roughly 24%"

    trace = _trace(g, e)
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False)
    assert d.status == "near_threshold_progressing"
    assert d.status != "strict_converged"
    assert d.status != "stationary_plateau"


def test_energy_stable_but_rdm_unstable_trace_must_not_pass():
    g = [2.00e-4, 2.01e-4, 1.99e-4, 2.02e-4, 1.98e-4,
        2.00e-4, 2.01e-4, 1.99e-4, 2.00e-4, 2.00e-4]
    trace = _trace(g, _flat_energies(len(g)))
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False,
                       extension_blocks_run=1, rdm_change=1e-2)
    assert d.status != "stationary_plateau"
    assert d.status == "stationary_plateau_candidate"
    assert d.rdm_status == "unstable"


def test_gradient_stable_but_energy_drifting_trace_must_not_pass():
    g = [2.00e-4, 2.01e-4, 1.99e-4, 2.02e-4, 1.98e-4,
        2.00e-4, 2.01e-4, 1.99e-4, 2.00e-4, 2.00e-4]
    e = [-300.001 - i * 6e-4 for i in range(10)]  # span >> energy_span_tol
    trace = _trace(g, e)
    d = classify_trace(trace, conv_tol_grad=CONV_TOL_GRAD,
                       policy=TailConvergencePolicy(), las_declared_converged=False,
                       extension_blocks_run=1, rdm_change=1e-6)
    assert d.status not in ("stationary_plateau", "stationary_plateau_candidate")
    assert d.status == "unstable_nonconverged"


def test_classify_trace_requires_at_least_one_record():
    with pytest.raises(ValueError):
        classify_trace((), conv_tol_grad=CONV_TOL_GRAD, policy=TailConvergencePolicy(),
                       las_declared_converged=False)


# ---------------------------------------------------------------------------
# decide_extension
# ---------------------------------------------------------------------------

def _diag(status, **kw):
    defaults = dict(
        needs_more_iterations=True, reasons=(), g_int_final=2e-4, g_int_median=2e-4,
        log_gradient_slope=0.0, gradient_relative_range=0.0, energy_span=0.0,
        median_gradient_change=0.0, rdm_change=None, rdm_status="not_yet_confirmed",
        insufficient_history=False, extension_blocks_run=0,
        consecutive_plateau_blocks=0, las_declared_converged=False, window=())
    defaults.update(kw)
    return ConvergenceDiagnostics(status=status, **defaults)


def test_decide_extension_off_mode_never_extends():
    d = _diag("near_threshold_progressing")
    assert decide_extension(d, policy=TailConvergencePolicy(mode="off"), extra_cycles_used=0) == 0


def test_decide_extension_signal_mode_never_extends():
    d = _diag("near_threshold_progressing")
    assert decide_extension(d, policy=TailConvergencePolicy(mode="signal"), extra_cycles_used=0) == 0


def test_decide_extension_extend_mode_returns_extension_cycles():
    d = _diag("near_threshold_progressing")
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=50)
    assert decide_extension(d, policy=policy, extra_cycles_used=0) == 10


def test_decide_extension_stops_on_strict_converged():
    d = _diag("strict_converged")
    policy = TailConvergencePolicy(mode="extend")
    assert decide_extension(d, policy=policy, extra_cycles_used=0) == 0


def test_decide_extension_stops_on_confirmed_plateau():
    d = _diag("stationary_plateau")
    policy = TailConvergencePolicy(mode="extend")
    assert decide_extension(d, policy=policy, extra_cycles_used=10) == 0


def test_decide_extension_stops_on_instability():
    d = _diag("unstable_nonconverged")
    policy = TailConvergencePolicy(mode="extend")
    assert decide_extension(d, policy=policy, extra_cycles_used=10) == 0


def test_decide_extension_respects_remaining_budget_partial_block():
    d = _diag("near_threshold_progressing")
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=25)
    assert decide_extension(d, policy=policy, extra_cycles_used=20) == 5


def test_decide_extension_stops_exactly_at_max_extra_cycles():
    d = _diag("near_threshold_progressing")
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=20)
    assert decide_extension(d, policy=policy, extra_cycles_used=20) == 0


# ---------------------------------------------------------------------------
# rdm_block_change
# ---------------------------------------------------------------------------

def test_rdm_block_change_none_when_no_previous_snapshot():
    current = [np.eye(2), np.eye(3)]
    assert rdm_block_change(None, current) is None


def test_rdm_block_change_is_max_frobenius_norm_over_fragments():
    prev = [np.zeros((2, 2)), np.zeros((2, 2))]
    curr = [np.eye(2) * 3.0, np.eye(2) * 1.0]  # frobenius norms: 3*sqrt(2), 1*sqrt(2)
    change = rdm_block_change(prev, curr)
    assert change == pytest.approx(3.0 * math.sqrt(2))


# ---------------------------------------------------------------------------
# TailConvergencePolicy validation
# ---------------------------------------------------------------------------

def test_default_construction_is_valid_and_off():
    p = TailConvergencePolicy()
    assert p.mode == "off"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        TailConvergencePolicy(mode="bogus")


@pytest.mark.parametrize("field", [
    "gradient_upper", "energy_span_tol", "log_gradient_slope_tol",
    "gradient_relative_range_tol", "median_gradient_change_tol", "rdm_change_tol"])
def test_non_positive_tolerances_rejected(field):
    with pytest.raises(ValueError):
        TailConvergencePolicy(**{field: 0.0})
    with pytest.raises(ValueError):
        TailConvergencePolicy(**{field: -1e-3})


def test_window_below_three_rejected():
    with pytest.raises(ValueError):
        TailConvergencePolicy(window=2)


def test_extension_cycles_below_one_rejected():
    with pytest.raises(ValueError):
        TailConvergencePolicy(extension_cycles=0)


def test_max_extra_cycles_below_extension_cycles_rejected():
    with pytest.raises(ValueError):
        TailConvergencePolicy(extension_cycles=10, max_extra_cycles=5)


def test_confirmation_blocks_below_one_rejected():
    with pytest.raises(ValueError):
        TailConvergencePolicy(confirmation_blocks=0)


def test_max_extra_cycles_need_not_be_a_multiple_of_extension_cycles():
    # Explicit final-partial-block handling (see decide_extension), not a
    # divisibility requirement.
    p = TailConvergencePolicy(extension_cycles=10, max_extra_cycles=25)
    assert p.max_extra_cycles == 25


# ---------------------------------------------------------------------------
# ParentConfig cross-field validation
# ---------------------------------------------------------------------------

def test_parent_config_legacy_default_construction_remains_valid():
    from frasci.parent._config import ParentConfig
    config = ParentConfig(threshold=1e-4, max_cycle_macro=50)
    assert config.tail_convergence is None


def test_parent_config_rejects_gradient_upper_at_or_below_conv_tol_grad():
    from frasci.parent._config import ParentConfig
    with pytest.raises(ValueError):
        ParentConfig(
            threshold=1e-4, max_cycle_macro=50, conv_tol_grad=1e-4,
            tail_convergence=TailConvergencePolicy(mode="signal", gradient_upper=1e-4))


def test_parent_config_accepts_a_configured_tail_policy():
    from frasci.parent._config import ParentConfig
    policy = TailConvergencePolicy(mode="extend", gradient_upper=5e-4)
    config = ParentConfig(threshold=1e-4, max_cycle_macro=50, conv_tol_grad=1e-4,
                          tail_convergence=policy)
    assert config.tail_convergence is policy


# ---------------------------------------------------------------------------
# Control-flow tests: _run_tail_confirmation with a fake LAS (mirrors the
# _FakeLAS pattern in test_frasci_parent_escape_controller.py -- no real
# TrimCI/mrh time spent).
# ---------------------------------------------------------------------------

class _FakeLAS:
    def __init__(self, *, trace, segment_holder, blocks, converged_flags, rdm_snapshots):
        self.mo_coeff = np.eye(2)
        self.casdm1frs = ("casdm1frs-marker",)
        self.casdm2fr = ("casdm2fr-marker",)
        self.max_cycle_macro = None
        self.converged = False
        self._trace = trace
        self._segment_holder = segment_holder
        self._blocks = blocks
        self._converged_flags = converged_flags
        self._rdm_snapshots = rdm_snapshots
        self._rdm_call_count = 0
        self.kernel_calls = []

    def kernel(self, mo_coeff, casdm1frs=None, casdm2fr=None):
        i = len(self.kernel_calls)
        self.kernel_calls.append({
            "mo_coeff": mo_coeff, "casdm1frs": casdm1frs, "casdm2fr": casdm2fr,
            "max_cycle_macro": self.max_cycle_macro})
        for local_i, (e, g) in enumerate(self._blocks[i]):
            self._trace.append(MacroRecord(
                segment=self._segment_holder["n"], local_macro=local_i,
                global_macro=len(self._trace), e_tot=e, g_int=g))
        self.converged = self._converged_flags[i]

    def make_casdm1s_sub(self):
        snap = self._rdm_snapshots[self._rdm_call_count]
        self._rdm_call_count += 1
        return snap


def _seed(g_values, e_values):
    trace = list(_trace(g_values, e_values, segment=0))
    return trace, {"n": 0}


def test_extend_mode_reaches_strict_convergence_in_one_block():
    g_init = [4.9e-4 * (0.95 ** i) for i in range(10)]
    trace, segment_holder = _seed(g_init, _flat_energies(len(g_init)))
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=50)

    g_block0 = [3.0e-4, 2.5e-4, 2.0e-4, 1.6e-4, 1.3e-4, 1.1e-4, 9e-5, 7e-5, 6e-5, 5e-5]
    fake = _FakeLAS(
        trace=trace, segment_holder=segment_holder, blocks=[list(zip(_flat_energies(10), g_block0))],
        converged_flags=[True], rdm_snapshots=[[np.eye(2)], [np.eye(2) * (1 + 1e-7)]])

    diagnostics, extra_used = _run_tail_confirmation(
        fake, policy=policy, conv_tol_grad=CONV_TOL_GRAD, trace=trace,
        segment_holder=segment_holder)

    assert diagnostics.status == "strict_converged"
    assert diagnostics.needs_more_iterations is False
    assert extra_used == 10
    assert len(fake.kernel_calls) == 1
    call = fake.kernel_calls[0]
    assert call["mo_coeff"] is fake.mo_coeff
    assert call["casdm1frs"] is fake.casdm1frs
    assert call["casdm2fr"] is fake.casdm2fr
    assert call["max_cycle_macro"] == 10
    assert [r.global_macro for r in trace] == list(range(len(trace)))
    assert all(r.segment == 0 for r in trace[:10])
    assert all(r.segment == 1 for r in trace[10:])


def test_progressing_trace_uses_multiple_bounded_blocks_and_respects_max_extra_cycles():
    g_full = [4.9e-4 * (0.97 ** i) for i in range(35)]
    e_full = _flat_energies(35)
    trace, segment_holder = _seed(g_full[:10], e_full[:10])
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=25)

    blocks = [
        list(zip(e_full[10:20], g_full[10:20])),
        list(zip(e_full[20:30], g_full[20:30])),
        list(zip(e_full[30:35], g_full[30:35])),
    ]
    fake = _FakeLAS(
        trace=trace, segment_holder=segment_holder, blocks=blocks,
        converged_flags=[False, False, False],
        rdm_snapshots=[[np.eye(2)]] * 4)

    diagnostics, extra_used = _run_tail_confirmation(
        fake, policy=policy, conv_tol_grad=CONV_TOL_GRAD, trace=trace,
        segment_holder=segment_holder)

    assert extra_used == 25  # 10 + 10 + 5, exactly max_extra_cycles
    assert len(fake.kernel_calls) == 3
    assert [c["max_cycle_macro"] for c in fake.kernel_calls] == [10, 10, 5]
    assert diagnostics.status == "budget_exhausted"
    assert diagnostics.needs_more_iterations is True
    assert [r.global_macro for r in trace] == list(range(len(trace)))
    assert [r.segment for r in trace] == [0] * 10 + [1] * 10 + [2] * 10 + [3] * 5


def test_plateau_confirmed_by_a_single_real_extension_block():
    g_plateau = [2.00e-4, 2.01e-4, 1.99e-4, 2.02e-4, 1.98e-4,
                2.00e-4, 2.01e-4, 1.99e-4, 2.00e-4, 2.00e-4]
    trace, segment_holder = _seed(g_plateau, _flat_energies(10))
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=50)

    fake = _FakeLAS(
        trace=trace, segment_holder=segment_holder,
        blocks=[list(zip(_flat_energies(10), g_plateau))],
        converged_flags=[False],
        rdm_snapshots=[[np.eye(2)], [np.eye(2) * (1 + 1e-7)]])  # tiny, well within rdm_change_tol

    diagnostics, extra_used = _run_tail_confirmation(
        fake, policy=policy, conv_tol_grad=CONV_TOL_GRAD, trace=trace,
        segment_holder=segment_holder)

    assert diagnostics.status == "stationary_plateau"
    assert diagnostics.needs_more_iterations is False
    assert extra_used == 10
    assert len(fake.kernel_calls) == 1  # confirmed on the FIRST real block


def test_plateau_candidate_not_confirmed_when_rdm_actually_moves():
    g_plateau = [2.00e-4, 2.01e-4, 1.99e-4, 2.02e-4, 1.98e-4,
                2.00e-4, 2.01e-4, 1.99e-4, 2.00e-4, 2.00e-4]
    trace, segment_holder = _seed(g_plateau, _flat_energies(10))
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=10)

    fake = _FakeLAS(
        trace=trace, segment_holder=segment_holder,
        blocks=[list(zip(_flat_energies(10), g_plateau))],
        converged_flags=[False],
        rdm_snapshots=[[np.eye(2)], [np.eye(2) * 1.5]])  # large jump, > rdm_change_tol

    diagnostics, extra_used = _run_tail_confirmation(
        fake, policy=policy, conv_tol_grad=CONV_TOL_GRAD, trace=trace,
        segment_holder=segment_holder)

    # Budget is also exhausted after this one block (max_extra_cycles=10),
    # but a still-unconfirmed candidate must be reported as such, not
    # papered over as budget_exhausted.
    assert diagnostics.status == "stationary_plateau_candidate"
    assert diagnostics.rdm_status == "unstable"


def test_unstable_trace_receives_no_extension():
    g_unstable = [4.5e-4, 2.0e-4, 4.8e-4, 2.2e-4, 3.0e-4, 4.9e-4, 2.1e-4, 4.6e-4, 2.5e-4, 4.0e-4]
    trace, segment_holder = _seed(g_unstable, _flat_energies(10))
    policy = TailConvergencePolicy(mode="extend", extension_cycles=10, max_extra_cycles=50)

    fake = _FakeLAS(
        trace=trace, segment_holder=segment_holder, blocks=[], converged_flags=[],
        rdm_snapshots=[[np.eye(2)]])

    diagnostics, extra_used = _run_tail_confirmation(
        fake, policy=policy, conv_tol_grad=CONV_TOL_GRAD, trace=trace,
        segment_holder=segment_holder)

    assert diagnostics.status == "unstable_nonconverged"
    assert extra_used == 0
    assert len(fake.kernel_calls) == 0


# ---------------------------------------------------------------------------
# Physical regression tests: small, fast, known-convergent fixture. Release
# isolation: self-contained pyscf-native H6/sto-3g molecule (no engine/mols
# dependency), mirroring the fixture pattern already used by
# test_frasci_parent_cached_frame.py and test_frasci_parent_lassis_adapter.py.
# ---------------------------------------------------------------------------

from pyscf import ao2mo, gto, scf  # noqa: E402
from pyscf.tools.fcidump import from_integrals  # noqa: E402

from frasci.parent import solve as frasci_solve  # noqa: E402
from frasci.parent._config import ParentConfig  # noqa: E402

_H6_ORBITAL_LISTS = [[0, 1], [2, 3, 4, 5]]
_H6_NELEC_PER_FRAG = [(2, 2), (1, 1)]
_H6_SPIN_SUB = [1, 1]
_H6_SMALL = dict(
    threshold=1e-6, max_cycle_macro=25, max_cycle_rdmjk=20,
    orbopt=True, orbopt_cycles=2, phase0_dets=8, phase1_dets=8,
    discovery_num_runs=2, refine_num_runs=1, rng_seed=1)
_H6_KNOWN_E_TOT = -9.31516485551766  # converges in 8 real macrocycles


@pytest.fixture(scope="module")
def h6_fcidump_path(tmp_path_factory):
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    n_orb = mol.nao_nr()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False)
    path = tmp_path_factory.mktemp("h6_tail") / "fcidump"
    from_integrals(str(path), h1, eri, n_orb, mol.nelectron, nuc=0.0, ms=0)
    return str(path)


def test_tail_policy_off_preserves_legacy_behavior(h6_fcidump_path):
    config = ParentConfig(**_H6_SMALL)  # tail_convergence defaults to None
    parent = frasci_solve(
        h6_fcidump_path, _H6_ORBITAL_LISTS, _H6_NELEC_PER_FRAG, _H6_SPIN_SUB, config)

    assert parent.converged is True
    assert parent.e_tot == pytest.approx(_H6_KNOWN_E_TOT, abs=1e-9)
    assert parent.dets_per_frag == (1, 10)

    # New fields stay at their neutral, backward-compatible defaults.
    assert parent.convergence_status is None
    assert parent.needs_more_iterations is False
    assert parent.extra_macro_cycles == 0
    assert parent.initial_macro_budget == 25
    assert parent.total_macro_cycles == 25
    assert parent.convergence_diagnostics is None


def test_tail_policy_signal_mode_classifies_a_real_converged_run(h6_fcidump_path):
    config = ParentConfig(
        **_H6_SMALL, tail_convergence=TailConvergencePolicy(mode="signal"))
    parent = frasci_solve(
        h6_fcidump_path, _H6_ORBITAL_LISTS, _H6_NELEC_PER_FRAG, _H6_SPIN_SUB, config)

    # Same physics, only the tail-policy diagnostics are additive -- signal
    # mode never runs an extra las.kernel() call.
    assert parent.converged is True
    assert parent.e_tot == pytest.approx(_H6_KNOWN_E_TOT, abs=1e-9)
    assert parent.convergence_status == "strict_converged"
    assert parent.needs_more_iterations is False
    assert parent.extra_macro_cycles == 0
    assert parent.convergence_diagnostics is not None
    assert parent.convergence_diagnostics["g_int_final"] < config.conv_tol_grad


def test_save_dets_to_writes_trace_and_diagnostics_only_when_tail_enabled(
        h6_fcidump_path, tmp_path):
    on_dir = tmp_path / "tail_on"
    config_on = ParentConfig(
        **_H6_SMALL, tail_convergence=TailConvergencePolicy(mode="signal"))
    frasci_solve(
        h6_fcidump_path, _H6_ORBITAL_LISTS, _H6_NELEC_PER_FRAG, _H6_SPIN_SUB,
        config_on, save_dets_to=on_dir)
    assert (on_dir / "g_int_trace.json").exists()
    assert (on_dir / "convergence_diagnostics.json").exists()
    trace = json.loads((on_dir / "g_int_trace.json").read_text())
    assert trace  # at least one captured macrocycle
    assert {"segment", "local_macro", "global_macro", "e_tot", "g_int"} <= trace[0].keys()
    meta = json.loads((on_dir / "checkpoint_metadata.json").read_text())
    assert meta["convergence_status"] == "strict_converged"

    off_dir = tmp_path / "tail_off"
    config_off = ParentConfig(**_H6_SMALL)
    frasci_solve(
        h6_fcidump_path, _H6_ORBITAL_LISTS, _H6_NELEC_PER_FRAG, _H6_SPIN_SUB,
        config_off, save_dets_to=off_dir)
    assert not (off_dir / "g_int_trace.json").exists()
    assert not (off_dir / "convergence_diagnostics.json").exists()
    meta_off = json.loads((off_dir / "checkpoint_metadata.json").read_text())
    assert meta_off["convergence_status"] is None


def test_signal_mode_never_calls_kernel():
    g = [4.9e-4 * (0.95 ** i) for i in range(10)]
    trace, segment_holder = _seed(g, _flat_energies(10))
    policy = TailConvergencePolicy(mode="signal")

    fake = _FakeLAS(
        trace=trace, segment_holder=segment_holder, blocks=[], converged_flags=[],
        rdm_snapshots=[[np.eye(2)]])

    diagnostics, extra_used = _run_tail_confirmation(
        fake, policy=policy, conv_tol_grad=CONV_TOL_GRAD, trace=trace,
        segment_holder=segment_holder)

    assert extra_used == 0
    assert len(fake.kernel_calls) == 0
    assert diagnostics.status == "near_threshold_progressing"
    assert diagnostics.needs_more_iterations is True
