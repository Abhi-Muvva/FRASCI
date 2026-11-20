"""Regression coverage for the 2026-09-26 converged-parent safety guard.

Fe2S2 and Fe4S4 pilot runs (release notebooks 03/04) found that running
``frasci.recovery.solve()`` on top of a ``converged=False`` parent produces multi-Ha
"recovery" energies -- an artifact of the underlying non-convergence, not genuine
inter-fragment physics (confirmed: h00 in the -100 to -300 Ha range, with `generalized`
several hundred mHa to several Ha *below* h00, far outside the single-to-double-digit-mHa
range every genuinely converged system showed). ``solve()`` must refuse this by default,
in both the in-memory ``Parent`` and persisted-directory code paths, and must only proceed
under an explicit, conspicuously-flagged opt-out.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from pyscf import gto, scf
from pyscf.tools import fcidump

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery import RecoveryConfig, solve as solve_recovery

_ORBITAL_LISTS = [[0, 1, 2], [3, 4, 5]]
_NELEC_PER_FRAG = [(2, 1), (1, 2)]
_SPIN_SUB = [2, 2]
_CONFIG = ParentConfig(threshold=1e-6, max_cycle_macro=50, orbopt=True, orbopt_cycles=5,
                       phase0_dets=16, discovery_num_runs=8, refine_num_runs=1,
                       n_cpu_workers=0, rng_seed=1)


@pytest.fixture(scope="module")
def h6_fcidump(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h6_guard") / "FCIDUMP"
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    fcidump.from_scf(mf, str(path))
    return path


@pytest.fixture(scope="module")
def h6_parent(h6_fcidump):
    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB, _CONFIG)
    assert parent.converged, "fixture must converge for the positive-path tests to be meaningful"
    return parent


@pytest.fixture()
def h6_persisted_parent_dir(tmp_path, h6_fcidump):
    save_dir = tmp_path / "parent_dir"
    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB, _CONFIG,
                          save_dets_to=save_dir)
    assert parent.converged
    return save_dir


def _rcfg(**overrides):
    return RecoveryConfig(recovery_target=50, n_ct_max=2, las_top_k=50, **overrides)


def test_solve_rejects_nonconverged_in_memory_parent(h6_parent, h6_fcidump):
    unconverged = dataclasses.replace(h6_parent, converged=False)
    bundle = FCIDUMPInput.from_file(h6_fcidump)
    # solve()'s type hint says `parent_dir: Path`, but it forwards whatever it is given
    # straight to build_contracted_reference, which accepts Parent | Path via isinstance --
    # exercise the actual guard inside solve(), not just the state-dict internals.
    with pytest.raises(ValueError, match="converged=False"):
        solve_recovery(unconverged, bundle, _rcfg())


def test_state_carries_parent_converged_for_in_memory_parent(h6_parent, h6_fcidump):
    from frasci.recovery._state import build_contracted_reference
    unconverged = dataclasses.replace(h6_parent, converged=False)
    bundle = FCIDUMPInput.from_file(h6_fcidump)
    state = build_contracted_reference(unconverged, bundle)
    assert state["parent_converged"] is False
    state_ok = build_contracted_reference(h6_parent, bundle)
    assert state_ok["parent_converged"] is True


def test_solve_rejects_nonconverged_persisted_parent(tmp_path, h6_persisted_parent_dir, h6_fcidump):
    meta_path = h6_persisted_parent_dir / "checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["converged"] is True
    meta["converged"] = False
    meta_path.write_text(json.dumps(meta))

    bundle = FCIDUMPInput.from_file(h6_fcidump)
    with pytest.raises(ValueError, match="converged=False"):
        solve_recovery(h6_persisted_parent_dir, bundle, _rcfg())


def test_solve_accepts_converged_persisted_parent_and_flags_it_true(h6_persisted_parent_dir, h6_fcidump):
    bundle = FCIDUMPInput.from_file(h6_fcidump)
    result = solve_recovery(h6_persisted_parent_dir, bundle, _rcfg())
    assert result.parent_converged is True
    assert result.diagnostic_only is False


def test_allow_nonconverged_parent_opt_out_proceeds_and_flags_conspicuously(
        tmp_path, h6_persisted_parent_dir, h6_fcidump):
    meta_path = h6_persisted_parent_dir / "checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["converged"] = False
    meta_path.write_text(json.dumps(meta))

    bundle = FCIDUMPInput.from_file(h6_fcidump)
    result = solve_recovery(h6_persisted_parent_dir, bundle, _rcfg(allow_nonconverged_parent=True))
    assert result.parent_converged is False
    assert result.diagnostic_only is True


def test_allow_nonconverged_parent_defaults_to_false():
    assert RecoveryConfig(recovery_target=50).allow_nonconverged_parent is False


def test_result_defaults_reflect_the_ordinary_converged_path():
    from frasci.recovery._config import Result
    # Constructing with only the required fields should default to the "ordinary" state:
    # a caller who never touches the new fields gets exactly the old behavior's meaning.
    result = Result(hybrid_produced=True, energy=-1.0, n_dets=0, converged=True, gates={},
                    h00=-1.0, wall_s=0.0, diagnostics={})
    assert result.parent_converged is True
    assert result.parent_convergence_status is None
    assert result.diagnostic_only is False


# --- confirmed stationary_plateau: accepted like strict convergence (2026-09-30) --------
#
# ParentConfig.tail_convergence exists so a parent that plateaus (RDM-stable, confirmed by
# an actual extension block -- see frasci.parent._convergence) short of strict
# conv_tol_grad is still usable, not just diagnosable. Without this, the guard above
# refused a confirmed plateau exactly like genuine non-convergence, defeating the point of
# building the tail-confirmation protocol in the first place.

def test_solve_accepts_confirmed_stationary_plateau_in_memory_parent_without_override(
        h6_parent, h6_fcidump):
    plateaued = dataclasses.replace(h6_parent, converged=False,
                                    convergence_status="stationary_plateau")
    bundle = FCIDUMPInput.from_file(h6_fcidump)
    result = solve_recovery(plateaued, bundle, _rcfg())
    assert result.parent_converged is False
    assert result.parent_convergence_status == "stationary_plateau"
    assert result.diagnostic_only is False


def test_solve_accepts_confirmed_stationary_plateau_persisted_parent_without_override(
        h6_persisted_parent_dir, h6_fcidump):
    meta_path = h6_persisted_parent_dir / "checkpoint_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["converged"] = False
    meta["convergence_status"] = "stationary_plateau"
    meta_path.write_text(json.dumps(meta))

    bundle = FCIDUMPInput.from_file(h6_fcidump)
    result = solve_recovery(h6_persisted_parent_dir, bundle, _rcfg())
    assert result.parent_converged is False
    assert result.parent_convergence_status == "stationary_plateau"
    assert result.diagnostic_only is False


def test_solve_still_rejects_unconfirmed_plateau_candidate(h6_parent, h6_fcidump):
    # stationary_plateau_candidate is the NOT-YET-confirmed status (classify_trace never
    # promotes to stationary_plateau without an extension block + RDM stability) -- the
    # guard must not treat "still awaiting confirmation" the same as "confirmed".
    candidate = dataclasses.replace(h6_parent, converged=False,
                                    convergence_status="stationary_plateau_candidate")
    bundle = FCIDUMPInput.from_file(h6_fcidump)
    with pytest.raises(ValueError, match="converged=False"):
        solve_recovery(candidate, bundle, _rcfg())
