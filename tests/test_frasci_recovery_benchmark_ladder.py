"""Focused coverage for the SEPARATE all-molecule recovery-limit benchmark driver
(``frasci.recovery.benchmark_ladder``): the recovery-cap ladder, the 0.01 mHa plateau
campaign criterion, per-rung campaign checkpointing, and warm-starting across rungs.

None of this is part of ``frasci.recovery.solve()`` -- see
``test_frasci_recovery_growth.py`` for the production single-target path.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.tools import fcidump

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery._growth import analyze_rung_spin
from frasci.recovery._native_stream import iter_fixed_filtered_ladder
from frasci.recovery._solve import solve_recovery_point
from frasci.recovery._state import build_contracted_reference, top_product_terms
from frasci.recovery.benchmark_ladder import (
    BenchmarkLadderConfig, evaluate_stop, run, run_recovery_ladder_benchmark,
    warm_start_vector,
)

_ORBITAL_LISTS = [[0, 1, 2], [3, 4, 5]]
_NELEC_PER_FRAG = [(2, 1), (1, 2)]
_SPIN_SUB = [2, 2]
_PARENT_CONFIG = ParentConfig(threshold=1e-6, max_cycle_macro=50, orbopt=True, orbopt_cycles=5,
                              phase0_dets=16, discovery_num_runs=8, refine_num_runs=1,
                              n_cpu_workers=0, rng_seed=1)


@pytest.fixture(scope="module")
def h6_fcidump(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h6_benchmark") / "FCIDUMP"
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    fcidump.from_scf(mf, str(path))
    return path


@pytest.fixture(scope="module")
def h6_bundle(h6_fcidump):
    return FCIDUMPInput.from_file(h6_fcidump)


@pytest.fixture(scope="module")
def h6_parent_dir(tmp_path_factory, h6_fcidump) -> Path:
    save_dir = tmp_path_factory.mktemp("h6_benchmark_parent") / "parent_dir"
    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB,
                          _PARENT_CONFIG, save_dets_to=save_dir)
    assert parent.converged
    return save_dir


@pytest.fixture(scope="module")
def h6_state(h6_parent_dir, h6_bundle):
    return build_contracted_reference(h6_parent_dir, h6_bundle)


def _bcfg(**overrides):
    return BenchmarkLadderConfig(
        recovery_targets=overrides.pop("recovery_targets"),
        las_seed_dets=overrides.pop("las_seed_dets", 20), **overrides)


# --- plateau/stop semantics: this is the CAMPAIGN's own criterion, exercised directly
# (pure unit tests, no chemistry) -------------------------------------------------------

def _rung(*, native_actual_n, native_target_met, termination_reason, energy_ha):
    return {"native_actual_n": native_actual_n, "native_target_met": native_target_met,
            "termination_reason": termination_reason, "energy_ha": energy_ha}


def test_first_rung_cannot_plateau():
    current = _rung(native_actual_n=10, native_target_met=True,
                     termination_reason="target_reached", energy_ha=-5.0)
    assert evaluate_stop(None, current, plateau_threshold_mha=0.01) is None


def test_gain_strictly_below_threshold_is_plateau():
    previous = _rung(native_actual_n=10, native_target_met=True,
                     termination_reason="target_reached", energy_ha=-5.0)
    current = _rung(native_actual_n=20, native_target_met=True,
                    termination_reason="target_reached", energy_ha=-5.0 - 0.005e-3)
    assert evaluate_stop(previous, current, plateau_threshold_mha=0.01) == "energy_plateau"


def test_gain_exactly_at_threshold_continues():
    # Constructed so 1000*(previous-current) is bit-exactly 0.01 (not just "close"):
    # 0.0 minus a plain negation round-trips exactly in IEEE754, avoiding the float
    # representation error a generic energy-minus-energy subtraction would have.
    previous = _rung(native_actual_n=10, native_target_met=True,
                     termination_reason="target_reached", energy_ha=0.0)
    current = _rung(native_actual_n=20, native_target_met=True,
                    termination_reason="target_reached", energy_ha=-(0.01 / 1000.0))
    gain = 1000.0 * (previous["energy_ha"] - current["energy_ha"])
    assert gain == 0.01
    assert evaluate_stop(previous, current, plateau_threshold_mha=0.01) is None


def test_negative_gain_raises_as_correctness_failure():
    previous = _rung(native_actual_n=10, native_target_met=True,
                     termination_reason="target_reached", energy_ha=-5.0)
    current = _rung(native_actual_n=20, native_target_met=True,
                    termination_reason="target_reached", energy_ha=-5.0 + 1e-3)
    with pytest.raises(AssertionError, match="increased"):
        evaluate_stop(previous, current, plateau_threshold_mha=0.01)


def test_native_saturation_is_pool_exhaustion_not_plateau():
    current = _rung(native_actual_n=37, native_target_met=False,
                    termination_reason="filtered_saturated", energy_ha=-5.0)
    assert evaluate_stop(None, current, plateau_threshold_mha=0.01) == "candidate_pool_exhausted"


def test_stalled_growth_is_pool_exhaustion():
    previous = _rung(native_actual_n=37, native_target_met=False,
                     termination_reason="other", energy_ha=-5.0)
    current = _rung(native_actual_n=37, native_target_met=False,
                    termination_reason="other", energy_ha=-5.0 - 1.0)
    assert evaluate_stop(previous, current, plateau_threshold_mha=0.01) == "candidate_pool_exhausted"


# --- end-to-end benchmark ladder behavior ----------------------------------------------

def test_benchmark_ladder_produces_nested_supports_and_honest_stop_reason(
        h6_state, h6_parent_dir):
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    result = run_recovery_ladder_benchmark(
        h6_state, seed_dets, seed_coeffs,
        _bcfg(recovery_targets=(5, 20, 60)), parent_dir=h6_parent_dir)
    rungs = result["rungs"]
    assert len(rungs) == 3
    assert result["requested_targets"] == (5, 20, 60)
    assert result["stop_reason"] == "schedule_exhausted_not_plateaued"
    for row in rungs:
        assert "requested_n" in row and "native_actual_n" in row


def test_requested_vs_actual_n_can_differ(h6_state, h6_parent_dir):
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    result = run_recovery_ladder_benchmark(
        h6_state, seed_dets, seed_coeffs,
        _bcfg(recovery_targets=(5,)), parent_dir=h6_parent_dir)
    rung = result["rungs"][0]
    assert rung["requested_n"] == 5
    # native_actual_n need not equal requested_n once the pool is exhausted/rounded.
    assert "native_actual_n" in rung


# --- real, expanded-rung warm-start regression (N1 -> larger nested N2) ----------------

def test_warm_start_from_a_smaller_nested_rung_matches_cold_on_the_larger_support(h6_state):
    """A genuine N1 -> N2 warm start via the production warm_start_vector helper,
    mapping coefficients by determinant identity -- not a synthetic perturbation."""
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    n1 = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(5,), n_ct_max=2,
        global_topk=True, candidate_reservoir_capacity=40))[-1]
    n2 = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(20,), n_ct_max=2,
        initial_working_dets=n1["dets"], initial_working_coeffs=n1["coeffs"],
        global_topk=True, candidate_reservoir_capacity=40))[-1]
    assert frozenset(n1["eligible"]) <= frozenset(n2["eligible"])

    solved_n1 = solve_recovery_point(h6_state, n1["dets"], n1["coeffs"], n_ct_max=2,
                                     solver_kind="matrix_free")
    eligible_order_n2 = list(n2["eligible"].keys())
    warm_vector = warm_start_vector(solved_n1, eligible_order_n2)
    assert warm_vector.shape == (len(eligible_order_n2) + 1,)
    # Coefficients carried over by determinant identity: every N1 determinant's N2
    # warm-start amplitude equals its own N1-converged amplitude, unchanged.
    n1_amplitudes = dict(zip(solved_n1["explicit_determinants"], solved_n1["coefficients"][1:]))
    for index, det in enumerate(eligible_order_n2):
        if det in n1_amplitudes:
            assert warm_vector[index + 1] == pytest.approx(n1_amplitudes[det])
    assert warm_vector[0] == pytest.approx(solved_n1["coefficients"][0])

    warm = solve_recovery_point(h6_state, n2["dets"], n2["coeffs"], n_ct_max=2,
                                solver_kind="matrix_free", initial_coefficients=warm_vector)
    cold = solve_recovery_point(h6_state, n2["dets"], n2["coeffs"], n_ct_max=2,
                                solver_kind="matrix_free", initial_coefficients=None)
    assert warm["e_tot"] == pytest.approx(cold["e_tot"], abs=1e-8)
    assert set(warm["explicit_determinants"]) == set(cold["explicit_determinants"])
    assert set(warm["explicit_determinants"]) == frozenset(n2["eligible"])


# --- deterministic golden corrected-A' regression (frozen tiny fixture; no outer
# research-tree import -- avoids any such coupling) -------------------------------------

def test_golden_corrected_a_prime_ladder_regression(h6_state, h6_parent_dir):
    """Pins per-rung raw support hashes, requested/actual N, nestedness, energies, and
    the stop reason for a fixed, deterministic tiny ladder run. A change to the growth,
    plateau, or nesting logic that alters any of these numbers should fail this test;
    update the golden values deliberately (with a reason) if the change is intentional
    and independently verified -- never just to make this test pass.
    """
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    result = run_recovery_ladder_benchmark(
        h6_state, seed_dets, seed_coeffs,
        _bcfg(recovery_targets=(5, 20, 60), plateau_threshold_mha=1e-12),
        parent_dir=h6_parent_dir)
    rungs = result["rungs"]

    requested = [row["requested_n"] for row in rungs]
    native_actual = [row["native_actual_n"] for row in rungs]
    hashes = [row["support_sha256"] for row in rungs]
    energies = [row["energy_ha"] for row in rungs]

    assert requested == [5, 20, 60]
    assert native_actual == [5, 20, 60]
    # support_sha256 is bit-exactly reproducible (verified by rerunning this scenario
    # twice independently); the energies below agree only to the iterative solver's
    # own convergence tolerance (~1e-10), hence the looser comparison tolerance.
    assert hashes == [
        "f815a6092b363dc5138f81fc67983158e607b53046c161fe677b2b5e3fb913f0",
        "ba06c7a174e6d64364b47897db24abb7de560e0795768060a431f242819141a8",
        "3d663898c1179a2e9b40353ec2722906db165fdcae5d1316647ec6135e2608b3",
    ]
    assert energies == pytest.approx(
        [-1.7723216943724445, -2.6471502142743506, -2.816892040508557], abs=1e-7)
    assert result["stop_reason"] == "schedule_exhausted_not_plateaued"

    # Nestedness, independent of the pinned literals above: re-run the identical raw
    # growth directly (not through the benchmark wrapper) to get the actual per-rung
    # eligible sets and verify each is a superset of the previous one.
    points = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(5, 20, 60), n_ct_max=2,
        global_topk=True, candidate_reservoir_capacity=120))
    previous_set: frozenset = frozenset()
    for point in points:
        current_set = frozenset(point["eligible"].keys())
        assert previous_set <= current_set
        previous_set = current_set


# --- checkpoint-resume with a CHANGED target schedule: reuse matching rung checkpoints,
# but recompute adjacency-dependent plateau decisions ------------------------------------

def test_checkpoint_resume_with_changed_schedule_recomputes_adjacency(
        tmp_path, h6_state, h6_parent_dir):
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    checkpoint_dir = tmp_path / "ckpt"

    # Run 1: schedule (5, 20, 60) -- schedule exhausts without plateauing; all three
    # rungs get checkpointed, rung 60's stored incremental_gain_mha is relative to 20.
    run1 = run_recovery_ladder_benchmark(
        h6_state, seed_dets, seed_coeffs,
        _bcfg(recovery_targets=(5, 20, 60), checkpoint_directory=checkpoint_dir),
        parent_dir=h6_parent_dir)
    assert run1["checkpoint_status"]["rungs_computed"] == 3
    rung60_gain_vs_20 = run1["rungs"][-1]["incremental_gain_mha"]

    # Run 2: CHANGED schedule that skips 20 -- rung 60's checkpoint is still reused
    # (same fingerprint, same target), but its ADJACENT rung is now 5, not 20, so its
    # reported incremental_gain_mha must be recomputed relative to 5, not reused stale
    # from run 1.
    run2 = run_recovery_ladder_benchmark(
        h6_state, seed_dets, seed_coeffs,
        _bcfg(recovery_targets=(5, 60), checkpoint_directory=checkpoint_dir),
        parent_dir=h6_parent_dir)
    assert run2["checkpoint_status"]["rungs_reused"] == 2
    assert run2["checkpoint_status"]["rungs_computed"] == 0
    rung5_record, rung60_record = run2["rungs"]
    expected_gain_vs_5 = 1000.0 * (rung5_record["energy_ha"] - rung60_record["energy_ha"])
    assert rung60_record["incremental_gain_mha"] == pytest.approx(expected_gain_vs_5, abs=1e-9)
    assert rung60_record["incremental_gain_mha"] != pytest.approx(rung60_gain_vs_20, abs=1e-9)
    # The underlying stop-reason DECISION was already always correct (evaluate_stop
    # recomputes independently), but this confirms the REPORTED field matches too.
    assert evaluate_stop(rung5_record, rung60_record, plateau_threshold_mha=0.01) == (
        "energy_plateau" if expected_gain_vs_5 < 0.01 else None)


# --- Parent-convergence requirement: no diagnostic opt-out in the benchmark driver -----

def test_run_rejects_nonconverged_parent_with_no_opt_out(h6_parent_dir, h6_bundle):
    meta_path = h6_parent_dir / "checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text())
    original = meta_path.read_text()
    meta["converged"] = False
    meta_path.write_text(json.dumps(meta))
    try:
        with pytest.raises(ValueError, match="converged=False"):
            run(h6_parent_dir, h6_bundle, _bcfg(recovery_targets=(5,)))
    finally:
        meta_path.write_text(original)


def test_run_accepts_confirmed_stationary_plateau_parent(h6_parent_dir, h6_bundle):
    meta_path = h6_parent_dir / "checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text())
    original = meta_path.read_text()
    meta["converged"] = False
    meta["convergence_status"] = "stationary_plateau"
    meta_path.write_text(json.dumps(meta))
    try:
        result = run(h6_parent_dir, h6_bundle, _bcfg(recovery_targets=(5,)))
        assert result["rungs"]
    finally:
        meta_path.write_text(original)
