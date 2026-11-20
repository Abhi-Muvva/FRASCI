"""Focused coverage for the PRODUCTION recovery path: one direct corrected-A' growth
call at a single user-selected ``recovery_target``, its reservoir checkpoint, the
diagnostic-only spin-orbit analysis, and the COO default-off migration.

Deliberately does NOT cover the recovery-cap ladder, the 0.01 mHa plateau rule, or any
per-rung campaign checkpointing -- those are the separate benchmark driver, covered in
``test_frasci_recovery_benchmark_ladder.py``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pyscf import gto, scf
from pyscf.tools import fcidump

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery import RecoveryConfig, Result, solve as solve_recovery
from frasci.recovery._growth import (
    RESERVOIR_CAPACITY_MULTIPLIER, analyze_rung_spin, build_fingerprint,
    fingerprint_hash, grow_recovery_target,
)
from frasci.recovery._native_stream import _recovery_policy, iter_fixed_filtered_ladder
from frasci.recovery._state import build_contracted_reference, top_product_terms

_ORBITAL_LISTS = [[0, 1, 2], [3, 4, 5]]
_NELEC_PER_FRAG = [(2, 1), (1, 2)]
_SPIN_SUB = [2, 2]
_PARENT_CONFIG = ParentConfig(threshold=1e-6, max_cycle_macro=50, orbopt=True, orbopt_cycles=5,
                              phase0_dets=16, discovery_num_runs=8, refine_num_runs=1,
                              n_cpu_workers=0, rng_seed=1)


@pytest.fixture(scope="module")
def h6_fcidump(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h6_growth") / "FCIDUMP"
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
    save_dir = tmp_path_factory.mktemp("h6_growth_parent") / "parent_dir"
    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB,
                          _PARENT_CONFIG, save_dets_to=save_dir)
    assert parent.converged
    return save_dir


@pytest.fixture(scope="module")
def h6_state(h6_parent_dir, h6_bundle):
    return build_contracted_reference(h6_parent_dir, h6_bundle)


def _rcfg(**overrides):
    return RecoveryConfig(recovery_target=overrides.pop("recovery_target", 60),
                          las_seed_dets=overrides.pop("las_seed_dets", 20), **overrides)


# --- A-prime growth policy (shared primitive, used by both the production path and
# the benchmark driver) ----------------------------------------------------------------

def test_recovery_policy_default_is_unbounded_per_round():
    policy = _recovery_policy(100, max_work_dets=None)
    assert policy.global_topk is False


def test_recovery_policy_corrected_global_topk_sets_reservoir_capacity():
    policy = _recovery_policy(100, max_work_dets=None, global_topk=True,
                              candidate_reservoir_capacity=250)
    assert policy.global_topk is True
    assert policy.candidate_reservoir_capacity == 250


def test_iterator_resume_reproduces_single_call_result(h6_state):
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    straight = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(5, 20),
        n_ct_max=2, global_topk=True, candidate_reservoir_capacity=40))[-1]
    first = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(5,),
        n_ct_max=2, global_topk=True, candidate_reservoir_capacity=40))[-1]
    resumed = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(20,),
        n_ct_max=2, initial_working_dets=first["dets"],
        initial_working_coeffs=first["coeffs"],
        global_topk=True, candidate_reservoir_capacity=40))[-1]
    assert set(resumed["dets"]) == set(straight["dets"])
    assert resumed["n_recovery"] == straight["n_recovery"]


# --- single-target production growth --------------------------------------------------

def test_solve_accepts_an_arbitrary_single_recovery_target(h6_parent_dir, h6_bundle):
    for target in (5, 37, 150):
        result = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(recovery_target=target))
        assert result.diagnostics["growth"]["requested_n"] == target
        assert result.diagnostics["protocol"] == (
            "complete_contracted_parent_plus_corrected_a_prime_recovery")


def test_no_public_ladder_or_plateau_fields_remain():
    field_names = {field.name for field in dataclasses.fields(RecoveryConfig)}
    assert "recovery_targets" not in field_names
    assert "plateau_threshold_mha" not in field_names
    assert field_names == {
        "recovery_target", "threshold", "n_ct_max", "max_memory_gb", "las_seed_dets",
        "growth_factor", "max_rounds", "solver_kind", "dense_dimension_threshold",
        "las_reconstruction_tol", "union_energy_tol", "las_top_k",
        "free_coo_orbital_opt_max_iter", "checkpoint_directory", "run_post_recovery_coo",
        "expand_parent_k", "allow_nonconverged_parent",
    }


def test_no_m_multiparent_selector_guide_or_boundary_shell_controls_exposed():
    field_names = {field.name for field in dataclasses.fields(RecoveryConfig)}
    forbidden = {"m", "selector", "selector_kind", "guide", "guide_kind", "target_two_s",
                "boundary_shell", "boundary_shell_kind", "multiparent", "native_multiroot",
                "candidate_reservoir_multiplier"}
    assert not (field_names & forbidden), field_names & forbidden


def test_result_has_no_native_multiroot_or_m_fields():
    field_names = {field.name for field in dataclasses.fields(Result)}
    assert "parent_kind" not in field_names
    assert "contracted_state_count" not in field_names


def test_production_size_path_dispatches_matrix_free_not_dense(h6_parent_dir, h6_bundle):
    result = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=60, dense_dimension_threshold=1, solver_kind="auto"))
    assert result.diagnostics["solver_kind"] == "matrix_free"


# --- diagnostic-only spin analysis: no false completeness claim -----------------------

def test_spin_diagnostics_never_claim_closure_and_report_owned_space_blocking(h6_state):
    seed_dets, seed_coeffs = top_product_terms(h6_state, 20)
    point = list(iter_fixed_filtered_ladder(
        h6_state, seed_dets, seed_coeffs, recovery_targets=(20,), n_ct_max=2,
        global_topk=True, candidate_reservoir_capacity=40))[-1]
    n_alpha, n_beta = int(h6_state["n_elec"][0]), int(h6_state["n_elec"][1])
    info = analyze_rung_spin(point, h6_state, n_alpha=n_alpha, n_beta=n_beta)
    assert "is_spin_group_closed" not in info
    assert info["spin_closure_status"] in (
        "complete", "blocked_by_owned_space", "incomplete_within_recovery_space")
    # The recovery determinant selection is NEVER changed by this analysis -- it
    # analyzes exactly point["eligible"], the unmodified corrected A' support.
    assert info["total_spin_orbits"] >= info["complete_spin_orbits"]
    # On this real H6 2+1/1+2 partition, dispersion determinants genuinely DO have
    # spin-exchange partners in the "owned" (parent) sector -- the architecturally
    # blocked case this module exists to report honestly, not paper over.
    assert info["orbits_intersecting_owned_space"] > 0
    assert info["spin_closure_status"] == "blocked_by_owned_space"


def test_solve_never_claims_spin_closure_and_spin_square_stays_none(h6_parent_dir, h6_bundle):
    result = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(recovery_target=20))
    assert result.spin_square is None
    spin = result.diagnostics["spin"]
    assert "is_spin_group_closed" not in spin
    assert spin["spin_closure_status"] in (
        "complete", "blocked_by_owned_space", "incomplete_within_recovery_space")
    assert "explicit_block_spin_square_partial" in spin
    assert "NON-CERTIFYING" in spin["explicit_block_spin_square_note"]
    assert spin["full_wavefunction_spin_square"] is None


# --- COO default-off migration ---------------------------------------------------------

def test_coo_defaults_off(h6_parent_dir, h6_bundle):
    result = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(recovery_target=20))
    assert result.free_coo_produced is False
    assert result.free_coo_energy is None
    assert result.diagnostics["coo"]["produced"] is False


def test_coo_opt_in_runs_once_with_matched_pre_post_energies(h6_parent_dir, h6_bundle):
    result = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=20, run_post_recovery_coo=True))
    assert result.free_coo_produced is True
    coo = result.diagnostics["coo"]
    assert coo["produced"] is True
    assert coo["matched_support_energy_before_ha"] >= result.free_coo_energy
    assert coo["gain_mha"] == pytest.approx(
        1000.0 * (coo["matched_support_energy_before_ha"] - result.free_coo_energy), abs=1e-6)


# --- production reservoir checkpoint (single target, distinct from the benchmark
# driver's per-rung checkpoints) -------------------------------------------------------

def test_checkpoint_reuses_a_smaller_reservoir_to_grow_a_larger_target(
        tmp_path, h6_parent_dir, h6_bundle):
    checkpoint_dir = tmp_path / "ckpt"
    first = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=20, checkpoint_directory=checkpoint_dir))
    assert first.diagnostics["growth"]["checkpoint_status"]["reservoir"] in (
        "computed_and_saved", "rebuilt")
    second = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=60, checkpoint_directory=checkpoint_dir))
    assert second.diagnostics["growth"]["checkpoint_status"]["reservoir"] == "reused"
    assert second.diagnostics["growth"]["requested_n"] == 60


def test_checkpoint_rejects_incompatible_settings_and_rebuilds(
        tmp_path, h6_parent_dir, h6_bundle):
    checkpoint_dir = tmp_path / "ckpt"
    solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=20, n_ct_max=2, checkpoint_directory=checkpoint_dir))
    changed = solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=20, n_ct_max=1, checkpoint_directory=checkpoint_dir))
    assert changed.diagnostics["growth"]["checkpoint_status"]["reservoir"] == "rebuilt"


def test_reservoir_checkpoint_distinct_from_benchmark_ladder_rung_checkpoints(
        tmp_path, h6_parent_dir, h6_bundle):
    checkpoint_dir = tmp_path / "ckpt"
    solve_recovery(h6_parent_dir, h6_bundle, _rcfg(
        recovery_target=20, checkpoint_directory=checkpoint_dir))
    written = {p.name for p in checkpoint_dir.iterdir()}
    assert any(name.startswith("reservoir.") for name in written)
    assert not any(name.startswith("benchmark_ladder_rung_") for name in written)


def test_reservoir_capacity_multiplier_is_derived_not_configurable():
    assert RESERVOIR_CAPACITY_MULTIPLIER == 2
    field_names = {field.name for field in dataclasses.fields(RecoveryConfig)}
    assert "candidate_reservoir_multiplier" not in field_names


def test_fingerprint_is_stable_for_identical_settings(h6_state, h6_parent_dir):
    config_a = RecoveryConfig(recovery_target=20)
    config_b = RecoveryConfig(recovery_target=20)
    assert fingerprint_hash(build_fingerprint(h6_state, h6_parent_dir, config_a)) == (
        fingerprint_hash(build_fingerprint(h6_state, h6_parent_dir, config_b)))
    config_c = RecoveryConfig(recovery_target=20, n_ct_max=1)
    assert fingerprint_hash(build_fingerprint(h6_state, h6_parent_dir, config_a)) != (
        fingerprint_hash(build_fingerprint(h6_state, h6_parent_dir, config_c)))
