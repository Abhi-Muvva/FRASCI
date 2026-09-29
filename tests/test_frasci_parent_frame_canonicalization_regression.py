"""Regression test for the 2026-09-25 frame-provenance bug: mrh's terminal LASSCF
canonicalization was previously allowed to rotate Parent.mo_coeff AFTER each fragment's
persisted (determinants, coefficients, orbital_rotation) state had already been written
during the last kernel call -- but the no-op canonicalize() override that prevents this
(``_keep_final_frame_uncanonicalized``) was installed only under
``adaptive_phase_policy=True``, leaving every ordinary (non-adaptive) orbopt=True solve
exposed. On the H6 fixture below this produced a real, non-numerical-noise mismatch between
``parent.e_tot`` and ``frasci.recovery.build_contracted_reference(parent, ...).h00``
(~3.5 mHa, confirmed before the fix landed).

This must keep passing at machine precision, not a loosened tolerance -- a 1e-6 Ha gate
here is already far tighter than the bug it guards (mHa-scale), so a regression cannot
sneak back in under it.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.tools import fcidump

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery._las_reconstruct import certify_las_reconstruction
from frasci.recovery._state import build_contracted_reference

_ORBITAL_LISTS = [[0, 1, 2], [3, 4, 5]]
_NELEC_PER_FRAG = [(2, 1), (1, 2)]
_SPIN_SUB = [2, 2]
_CONFIG = ParentConfig(threshold=1e-6, max_cycle_macro=50, orbopt=True, orbopt_cycles=5,
                       phase0_dets=16, discovery_num_runs=8, refine_num_runs=1,
                       n_cpu_workers=0, rng_seed=1)


@pytest.fixture(scope="module")
def h6_fcidump(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h6_frame") / "FCIDUMP"
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    fcidump.from_scf(mf, str(path))
    return path


def test_terminal_canonicalization_is_real_and_nontrivial():
    """Sanity check on the bug's own precondition: mrh's canonicalize() really does rotate
    mo_coeff by a nontrivial (not phase-only) amount when NOT suppressed -- otherwise this
    whole regression would be vacuous. Verified once, directly, not assumed."""
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox
    from frasci._mock_scf import build_mock_scf_from_fcidump
    from frasci.parent._kernel import make_phased_kernel

    d = Path(tempfile.mkdtemp())
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf0 = scf.RHF(mol)
    mf0.verbose = 0
    mf0.kernel()
    fcidump.from_scf(mf0, str(d / "FCIDUMP"))

    mf = build_mock_scf_from_fcidump(str(d / "FCIDUMP"))
    n_orb = int(mf.mol._nao)
    las = LASSCF(mf, (3, 3), list(_NELEC_PER_FRAG), spin_sub=list(_SPIN_SUB))
    las.max_cycle_macro = 50
    las.verbose = 0

    captured = {}
    real_canonicalize = las.canonicalize

    def spy(mo_coeff=None, *a, **k):
        captured["pre"] = np.asarray(las.mo_coeff if mo_coeff is None else mo_coeff).copy()
        result = real_canonicalize(mo_coeff, *a, **k)
        captured["post"] = np.asarray(result[0]).copy()
        return result

    las.canonicalize = spy
    for i in range(2):
        kfn = make_phased_kernel(fragment_idx=i, threshold=1e-6, discovery_num_runs=8,
                                 refine_num_runs=1, max_dets_phase0=16, max_dets_phase1=1000,
                                 orbopt=True, orbopt_cycles=5, seed=1)
        las.fciboxes[i] = make_fcibox(mf.mol, kernel=kfn,
                                      spin=_SPIN_SUB[i] - 1, smult=_SPIN_SUB[i])
    las.kernel(np.eye(n_orb)[:, list(range(n_orb))])

    assert "pre" in captured, "canonicalize() was never called -- test fixture is stale"
    assert np.max(np.abs(captured["pre"] - captured["post"])) > 1e-3, (
        "terminal canonicalization did not rotate mo_coeff nontrivially on this fixture -- "
        "this regression test's premise no longer holds, investigate before trusting it")


def test_parent_e_tot_matches_build_contracted_reference_h00(h6_fcidump):
    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB, _CONFIG)
    assert parent.converged

    bundle = FCIDUMPInput.from_file(h6_fcidump)
    state = build_contracted_reference(parent, bundle)
    assert state["h00"] == pytest.approx(parent.e_tot, abs=1e-6), (
        f"H00 ({state['h00']}) disagrees with parent.e_tot ({parent.e_tot}) -- the "
        "canonicalization frame-provenance bug has regressed")

    certificate = certify_las_reconstruction(state, tol=1e-8)
    assert certificate["passed"]


def test_parent_object_and_persisted_path_branches_agree(tmp_path, h6_fcidump):
    save_dir = tmp_path / "saved_parent"
    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB, _CONFIG,
                          save_dets_to=save_dir)
    bundle = FCIDUMPInput.from_file(h6_fcidump)

    state_from_object = build_contracted_reference(parent, bundle)
    state_from_path = build_contracted_reference(save_dir, bundle)

    assert state_from_object["h00"] == pytest.approx(state_from_path["h00"], abs=1e-9)
    assert state_from_object["h00"] == pytest.approx(parent.e_tot, abs=1e-6)


def test_adapted_lassis_reference_energy_agrees_with_parent_and_h00(h6_fcidump):
    from mrh.my_pyscf.lassi.lassi import LASSI
    from frasci.parent import build_lassis_ready_las

    parent = solve_parent(str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB, _CONFIG)
    bundle = FCIDUMPInput.from_file(h6_fcidump)
    state = build_contracted_reference(parent, bundle)

    las = build_lassis_ready_las(parent, str(h6_fcidump))
    lsi = LASSI(las)
    ci_ref = lsi.get_ci_ref() if hasattr(lsi, "get_ci_ref") else None
    # LASSI's own reconstructed reference-state energy (its internal LASCI energy at the
    # assigned .ci/.mo_coeff) must agree with both parent.e_tot and the certified H00 -- the
    # three independent computations of the same physical quantity this whole bug was about.
    e_ref = float(np.asarray(lsi.e_states)[0]) if getattr(lsi, "e_states", None) is not None \
        else None
    if e_ref is None:
        pytest.skip("LASSI did not populate e_states before kernel(); covered indirectly by "
                    "test_build_lassis_ready_las_runs_lassis_kernel's variational-bound check")
    assert e_ref == pytest.approx(parent.e_tot, abs=1e-6)
    assert e_ref == pytest.approx(state["h00"], abs=1e-6)
