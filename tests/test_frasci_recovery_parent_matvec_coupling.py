"""Validates the row-sliced-matvec coupling backend
(``frasci.recovery._state.h0d_batch_via_parent_matvec``) against the original
per-candidate neighbor-enumeration backend (``h0d_batch``) on small deterministic
samples from H4, FeS, AlFe2, and Fe4S4 -- the four systems explicitly named for this
check -- plus the preflight dispatch, duplicate-support handling, nonzero-e_const
behavior, and per-stage timing reporting.

Candidate selection, Parent contraction, A-prime ordering, and the generalized hybrid
solve are NOT touched by the feature under test here -- only how the coupling vector
feeding that solve is computed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.tools import fcidump

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery._native_stream import iter_fixed_filtered_ladder
from frasci.recovery._slater_condon import RealSlaterCondon
from frasci.recovery._solve import solve_recovery_point
from frasci.recovery._state import (
    build_contracted_reference, h0d, h0d_batch, h0d_batch_via_parent_matvec,
    parent_support_preflight, prepare_owned_coefficient_lookup, retain_candidates,
    top_product_terms,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "data"
_CHEAP_SETTINGS = dict(orbopt=True, orbopt_cycles=2, phase0_dets=16, phase1_dets=50,
                       discovery_num_runs=4, refine_num_runs=1, rng_seed=0,
                       n_cpu_workers=0)


def _grow(state, *, seed_size=20, target=60, n_ct_max=2):
    seed_dets, seed_coeffs = top_product_terms(state, seed_size)
    point = list(iter_fixed_filtered_ladder(
        state, seed_dets, seed_coeffs, recovery_targets=(target,), n_ct_max=n_ct_max,
        global_topk=True, candidate_reservoir_capacity=target * 2))[-1]
    eligible, _ = retain_candidates(
        point["dets"], point["coeffs"], state, n_ct_max=n_ct_max, include_same_sector=False)
    return point, eligible


def _assert_coupling_matches(state, eligible_dets, *, tol=1e-9):
    edge_kernel = RealSlaterCondon(state["h1"], state["eri"], e_const=state["e_const"])
    lookup_cache = prepare_owned_coefficient_lookup(state)
    old = h0d_batch(eligible_dets, state, edge_kernel, lookup_cache=lookup_cache)
    new, timings = h0d_batch_via_parent_matvec(eligible_dets, state)
    err = float(np.max(np.abs(old - new)))
    assert err <= tol, f"max abs coupling error {err:.3e} exceeds {tol:.0e} Ha"
    assert set(timings) == {
        "parent_reconstruction_s", "index_and_diagonal_s", "row_sliced_matvec_s"}
    assert all(v >= 0.0 for v in timings.values())
    return new, timings


# --- H4: fresh tiny fixture (no bundled FCIDUMP in this release) ----------------------

@pytest.fixture(scope="module")
def h4_state(tmp_path_factory):
    fcidump_path = tmp_path_factory.mktemp("h4") / "FCIDUMP"
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(4)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    fcidump.from_scf(mf, str(fcidump_path))
    cfg = ParentConfig(threshold=1e-6, max_cycle_macro=30, **_CHEAP_SETTINGS)
    parent_dir = tmp_path_factory.mktemp("h4_parent") / "parent_dir"
    parent = solve_parent(str(fcidump_path), [[0, 1], [2, 3]], [(1, 1), (1, 1)], [1, 1],
                          cfg, save_dets_to=parent_dir)
    bundle = FCIDUMPInput.from_file(fcidump_path)
    return build_contracted_reference(parent_dir, bundle)


def test_coupling_matches_h0d_batch_on_h4(h4_state):
    _, eligible = _grow(h4_state, seed_size=4, target=10)
    _assert_coupling_matches(h4_state, list(eligible))


# --- FeS dimer(3-): bundled FCIDUMP, cheap (non-production) parent settings for a fast
# correctness-only check at this system's real 20-orbital scale -----------------------

@pytest.fixture(scope="module")
def fes_state(tmp_path_factory):
    fcidump_path = DATA_ROOT / "fes_dimer_3minus" / "FCIDUMP"
    cfg = ParentConfig(threshold=1e-4, max_cycle_macro=5, **_CHEAP_SETTINGS)
    parent_dir = tmp_path_factory.mktemp("fes_parent") / "parent_dir"
    parent = solve_parent(
        str(fcidump_path), [list(range(0, 10)), list(range(10, 20))],
        [(5, 1), (0, 5)], [5, 6], cfg, save_dets_to=parent_dir)
    bundle = FCIDUMPInput.from_file(fcidump_path)
    return build_contracted_reference(parent_dir, bundle)


def test_coupling_matches_h0d_batch_on_fes(fes_state):
    _, eligible = _grow(fes_state, seed_size=20, target=60)
    _assert_coupling_matches(fes_state, list(eligible))


def test_fes_has_nonzero_e_const_general_case_exercised(fes_state):
    # A real molecular FCIDUMP's scalar constant is nuclear repulsion -- nonzero here
    # (contrast Fe4S4's bare FCIDUMP, where it is exactly 0 by convention). The match
    # above already exercises "Add e_const * overlap explicitly" for a genuinely
    # nonzero e_const, not just the normally-zero-overlap case.
    assert float(fes_state["e_const"]) != 0.0


# --- AlFe2(11o,10e): bundled FCIDUMP, already small/fast with normal settings ----------

@pytest.fixture(scope="module")
def alfe2_state(tmp_path_factory):
    fcidump_path = DATA_ROOT / "alfe2_11o10e" / "FCIDUMP"
    cfg = ParentConfig(threshold=1e-5, max_cycle_macro=20, **_CHEAP_SETTINGS)
    parent_dir = tmp_path_factory.mktemp("alfe2_parent") / "parent_dir"
    parent = solve_parent(
        str(fcidump_path), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        [(1, 5), (5, 0)], [5, 6], cfg, save_dets_to=parent_dir)
    bundle = FCIDUMPInput.from_file(fcidump_path)
    return build_contracted_reference(parent_dir, bundle)


def test_coupling_matches_h0d_batch_on_alfe2(alfe2_state):
    _, eligible = _grow(alfe2_state, seed_size=20, target=60)
    _assert_coupling_matches(alfe2_state, list(eligible))


# --- Fe4S4: bundled FCIDUMP, h1diag2 partition, cheap (non-production) parent settings
# for a fast correctness-only check at this system's real 36-orbital/18+18-fragment
# scale -- NOT a scientifically validated Fe4S4 parent (see test_frasci_recovery_growth
# / the benchmark campaign for that) ----------------------------------------------------

@pytest.fixture(scope="module")
def fe4s4_state(tmp_path_factory):
    fcidump_path = DATA_ROOT / "fe4s4" / "FCIDUMP"
    orbital_lists = [
        [2, 3, 4, 6, 7, 8, 9, 10, 11, 21, 24, 25, 26, 27, 28, 29, 32, 33],
        [0, 1, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 30, 31, 34, 35]]
    cfg = ParentConfig(threshold=1e-3, max_cycle_macro=3, orbopt=False,
                       phase0_dets=16, phase1_dets=40, discovery_num_runs=2,
                       refine_num_runs=1, rng_seed=0, n_cpu_workers=0)
    parent_dir = tmp_path_factory.mktemp("fe4s4_parent") / "parent_dir"
    parent = solve_parent(
        str(fcidump_path), orbital_lists, [(10, 10), (17, 17)], [1, 1],
        cfg, save_dets_to=parent_dir)
    bundle = FCIDUMPInput.from_file(fcidump_path)
    return build_contracted_reference(parent_dir, bundle)


def test_coupling_matches_h0d_batch_on_fe4s4(fe4s4_state):
    _, eligible = _grow(fe4s4_state, seed_size=10, target=30)
    _assert_coupling_matches(fe4s4_state, list(eligible))


def test_fe4s4_bare_fcidump_has_zero_e_const(fe4s4_state):
    # AGENTS.md's own correctness invariant: E_nuc = 0.0 for the Fe4S4 FCIDUMP,
    # already absorbed -- the "normally zero" branch of the overlap*e_const term.
    assert float(fe4s4_state["e_const"]) == 0.0


# --- final matched-support energy agreement between backends --------------------------

def test_final_energy_agrees_between_backends(fes_state):
    point, _ = _grow(fes_state, seed_size=20, target=60)
    via_h0d_batch = solve_recovery_point(
        fes_state, point["dets"], point["coeffs"], n_ct_max=2,
        solver_kind="matrix_free", coupling_backend="h0d_batch")
    via_matvec = solve_recovery_point(
        fes_state, point["dets"], point["coeffs"], n_ct_max=2,
        solver_kind="matrix_free", coupling_backend="row_sliced_matvec")
    assert via_h0d_batch["e_tot"] == pytest.approx(via_matvec["e_tot"], abs=1e-8)
    assert set(via_h0d_batch["eligible"]) == set(via_matvec["eligible"])
    assert via_matvec["coupling_backend"] == "row_sliced_matvec"
    assert via_h0d_batch["coupling_backend"] == "h0d_batch"


def test_coupling_backend_auto_is_reported_and_dispatches(fes_state):
    point, _ = _grow(fes_state, seed_size=20, target=60)
    solved = solve_recovery_point(
        fes_state, point["dets"], point["coeffs"], n_ct_max=2, solver_kind="matrix_free")
    assert solved["coupling_backend"] in ("row_sliced_matvec", "h0d_batch")
    if solved["coupling_backend"] == "row_sliced_matvec":
        assert set(solved["coupling_timings"]) == {
            "parent_reconstruction_s", "index_and_diagonal_s", "row_sliced_matvec_s"}


# --- duplicate-support handling: a candidate that coincides with a Parent determinant --

def test_duplicate_candidate_matches_exact_h0d_without_double_counting(fes_state):
    point, eligible = _grow(fes_state, seed_size=20, target=30)
    genuine_candidates = list(eligible)[:5]
    # Force a duplicate: inject one of the Parent's OWN determinants into the candidate
    # list (never produced by retain_candidates in practice, but handled generally).
    from frasci.recovery._las_reconstruct import certify_las_reconstruction
    certificate = certify_las_reconstruction(fes_state, tol=1e-6)
    owned_det = tuple(map(int, certificate["dets"][0]))
    candidates_with_duplicate = genuine_candidates + [owned_det]

    coupling, _ = h0d_batch_via_parent_matvec(candidates_with_duplicate, fes_state)
    exact_owned_h0d = h0d(owned_det, fes_state)
    assert coupling[-1] == pytest.approx(exact_owned_h0d, abs=1e-9)
    # The genuine candidates' own couplings are unaffected by the injected duplicate.
    coupling_without_duplicate, _ = h0d_batch_via_parent_matvec(
        genuine_candidates, fes_state)
    assert np.allclose(coupling[:-1], coupling_without_duplicate, atol=1e-9)


# --- preflight dispatch, both directions -----------------------------------------------

def test_preflight_prefers_row_sliced_matvec_when_support_is_small(fes_state):
    preflight = parent_support_preflight(fes_state, n_candidates=60)
    assert preflight["use_row_sliced_matvec"] is True
    assert preflight["cartesian_product_size"] > 0


def test_preflight_falls_back_when_configured_ceiling_is_exceeded(fes_state):
    preflight = parent_support_preflight(
        fes_state, n_candidates=60, max_combined_dets=1)
    assert preflight["use_row_sliced_matvec"] is False
