"""Tests for frasci.parent.build_lassis_ready_las (public API; implemented in
frasci.parent._lassis_adapter).

The positive-convergence fixture is a real H6/sto-3g frasci.parent.solve() run (2 fragments,
3 orbitals/(2,1)+(1,2) electrons each) -- deliberately NOT the H4/(2,2)e-(1,1)o config used
elsewhere in ad-hoc investigation scripts this session, which was driven directly through a raw
``lasscf_rdm.LASSCF.kernel()`` call (bypassing frasci.parent.solve()'s own convergence machinery)
and did not converge. This fixture goes through the real public API and is independently verified
to converge below.
"""
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf
from pyscf.fci import cistring, direct_spin1
from pyscf.tools import fcidump

from frasci.parent import ParentConfig, build_lassis_ready_las, solve as solve_parent
from frasci.recovery._state import densify_fragment_ci

_ORBITAL_LISTS = [[0, 1, 2], [3, 4, 5]]
_NELEC_PER_FRAG = [(2, 1), (1, 2)]
_SPIN_SUB = [2, 2]


@pytest.fixture(scope="module")
def h6_fcidump(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("h6") / "FCIDUMP"
    mol = gto.M(atom="; ".join(f"H 0 0 {1.4 * i}" for i in range(6)),
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    fcidump.from_scf(mf, str(path))
    return path


@pytest.fixture(scope="module")
def h6_parent(h6_fcidump):
    parent = solve_parent(
        str(h6_fcidump), _ORBITAL_LISTS, _NELEC_PER_FRAG, _SPIN_SUB,
        ParentConfig(threshold=1e-6, max_cycle_macro=50, orbopt=True, orbopt_cycles=5,
                    phase0_dets=16, discovery_num_runs=8, refine_num_runs=1,
                    n_cpu_workers=0, rng_seed=1))
    assert parent.converged, "fixture must converge for the positive-path tests to be meaningful"
    return parent


def test_densify_preserves_addresses_normalization_and_phase():
    """Pure structural check, independent of any real solve: a hand-built sparse fragment
    lands at the exact cistring addresses with the exact signed coefficients, and nothing
    else is populated."""
    norb, na, nb = 3, 2, 1
    dets = np.array([[0b011, 0b001], [0b101, 0b010]], dtype=np.int64)
    coeffs = np.array([0.6, -0.8])  # already normalized, deliberately includes a negative phase

    dense = densify_fragment_ci(dets, coeffs, norb=norb, na=na, nb=nb)

    ca = cistring.make_strings(range(norb), na)
    cb = cistring.make_strings(range(norb), nb)
    addr = {(int(a), int(b)): (ia, ib)
           for ia, a in enumerate(ca) for ib, b in enumerate(cb)}

    expected = np.zeros_like(dense)
    for (la, lb), c in zip(dets.tolist(), coeffs.tolist()):
        expected[addr[(la, lb)]] = c
    assert np.array_equal(dense, expected)
    assert np.isclose(np.linalg.norm(dense), np.linalg.norm(coeffs))
    # Phase preserved exactly, not just magnitude:
    ia, ib = addr[(0b101, 0b010)]
    assert dense[ia, ib] == -0.8


def test_densified_fragments_are_consistent_with_the_certified_parent_frame(h6_parent, h6_fcidump):
    """Fragment energy/RDM preservation in the final parent orbital frame, checked against the
    release's own already-established oracle for exactly this question
    (frasci.recovery._state.build_contracted_reference + certify_las_reconstruction, the same
    machinery frasci.recovery.solve() itself relies on) rather than a hand-rolled re-derivation
    of LASSCF's embedded fragment Hamiltonian (which needs the other fragments' mean-field
    contribution, not recoverable from bare restricted AO integrals -- attempting that
    independently here would just be a second, unvalidated implementation of what
    build_contracted_reference already does correctly).

    Also asserts state["h00"] == parent.e_tot: the 2026-09-25 frame-provenance bug (mrh's
    terminal canonicalization rotating Parent.mo_coeff after each fragment's persisted state
    was already written) previously made these disagree by ~3.5 mHa on this fixture; fixed in
    frasci.parent.solve() (see test_frasci_parent_frame_canonicalization_regression.py for the
    dedicated regression coverage) and confirmed to agree at machine precision here too.
    """
    from frasci import FCIDUMPInput
    from frasci.recovery._las_reconstruct import certify_las_reconstruction
    from frasci.recovery._state import build_contracted_reference

    bundle = FCIDUMPInput.from_file(h6_fcidump)
    state = build_contracted_reference(h6_parent, bundle)
    assert state["h00"] == pytest.approx(h6_parent.e_tot, abs=1e-6)
    certificate = certify_las_reconstruction(state, tol=1e-8)
    assert certificate["passed"]

    # densify_fragment_ci on the SAME sparse (determinants, coefficients) the adapter itself
    # assigns to las.ci must be shape/normalization-consistent for every fragment.
    for frag in h6_parent.fragments:
        dense = densify_fragment_ci(frag.determinants, frag.coefficients,
                                    norb=frag.n_orb, na=frag.n_alpha, nb=frag.n_beta)
        assert np.isclose(np.linalg.norm(dense), np.linalg.norm(frag.coefficients))


def test_build_lassis_ready_las_runs_lassis_kernel(h6_parent, h6_fcidump):
    las = build_lassis_ready_las(h6_parent, str(h6_fcidump))
    from mrh.my_pyscf.lassi.lassis import LASSIS
    lsi = LASSIS(las)
    e_roots, si = lsi.kernel()
    e_roots = np.atleast_1d(e_roots)
    assert np.all(np.isfinite(e_roots))
    assert e_roots.min() <= h6_parent.e_tot + 1e-6, (
        "LASSIS ground state must be variational relative to the bare LAS reference it was built from")


def test_build_lassis_ready_las_rejects_nonconverged_parent(h6_parent, h6_fcidump):
    unconverged = dataclasses.replace(h6_parent, converged=False)
    with pytest.raises(ValueError, match="converged"):
        build_lassis_ready_las(unconverged, str(h6_fcidump))
    # Explicit opt-out still works, for diagnostic/stress-test use:
    las = build_lassis_ready_las(unconverged, str(h6_fcidump), allow_nonconverged=True)
    assert las.converged is False


def test_build_lassis_ready_las_rejects_partition_mismatch(h6_parent, h6_fcidump):
    bad = dataclasses.replace(h6_parent, nelec_per_frag=((2, 1), (2, 2)))  # frag 1 now disagrees
    with pytest.raises(ValueError, match="nelec_per_frag"):
        build_lassis_ready_las(bad, str(h6_fcidump))


def test_build_lassis_ready_las_preflights_dense_memory(h6_parent, h6_fcidump):
    # Each fragment here is C(3,2)*C(3,1) = 9 or C(3,1)*C(3,2) = 9 dense elements; set the cap
    # below that so the preflight must reject before ever calling densify_fragment_ci.
    with pytest.raises(MemoryError, match="max_dense_elements"):
        build_lassis_ready_las(h6_parent, str(h6_fcidump), max_dense_elements=4)
