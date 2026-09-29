"""Milestone F: forming W_final = P_LAS union R and solving the ordinary
(uncontracted) Davidson eigenproblem in that space (frasci.recovery._union_solve).

Uses the same fast H4/STO-3G fixture pattern as the Milestone D/E test files. The
genuine real-molecule case (real recovery determinants from the native stream, a
real converged rebuild) is covered in test_frasci_recovery_solve_end_to_end.py,
which wires this module into frasci.recovery.solve() itself.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from frasci.recovery._las_reconstruct import reconstruct_las_vector
from frasci.recovery._state import selected_hamiltonian
from frasci.recovery._union_solve import (
    certify_final_union,
    form_final_union,
    solve_union_davidson,
)


def _h4_state():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {1.0 * i}" for i in range(4)),
               basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    def _frag(dets, coeffs):
        dets = np.asarray(dets, dtype=np.int64)
        coeffs = np.asarray(coeffs, dtype=float)
        return dict(dets=dets, coeffs=coeffs,
                   lookup=dict(zip(map(tuple, dets.tolist()), coeffs.tolist())))

    frags = [
        _frag([[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436]),
        _frag([[0b01, 0b01], [0b10, 0b10]], [0.7, 0.714]),
    ]
    h00 = None  # filled in by callers that need it via a direct h00 computation
    return dict(n_orb=4, n_elec=(2, 2), blocks=[[0, 1], [2, 3]],
               ref_sectors=[(1, 1), (1, 1)], frags=frags, mo_final=c.copy(),
               h1=h1, eri=eri, e_const=float(mol.energy_nuc()), h00=h00)


# LAS-present (owned) determinant -- must never enter R.
LAS_PRESENT = (0b0101, 0b0101)
# Genuine dispersion determinant relative to the (1,1)/(1,1) reference: charge per
# fragment unchanged (1 alpha + 1 beta each), spin sectors differ from the ref.
DISPERSION = (0b0011, 0b1100)
CT1 = (0b0011, 0b0101)


def test_form_final_union_excludes_owned_and_includes_dispersion_ct():
    state = _h4_state()
    recovery_dets = [LAS_PRESENT, DISPERSION, CT1]
    recovery_coeffs = [0.5, 0.3, 0.2]
    union = form_final_union(state, recovery_dets, recovery_coeffs, n_ct_max=2)

    las_dets, _ = reconstruct_las_vector(state)
    assert set(las_dets) <= set(union["dets"])
    assert DISPERSION in union["dets"]
    assert CT1 in union["dets"]
    # LAS_PRESENT is same-sector -- already inside P_LAS, and retain_candidates with
    # include_same_sector=False (hardcoded, not a passthrough of caller config) drops
    # it from R -- it must not be double-counted or separately added.
    assert union["n_recovery"] == 2  # DISPERSION + CT1, not LAS_PRESENT
    assert union["n_overlap"] == 0
    assert union["n_work"] == len(las_dets) + 2

    got = dict(zip(union["dets"], union["initial_guess"]))
    for det, coeff in zip(las_dets, reconstruct_las_vector(state)[1]):
        assert got[det] == pytest.approx(coeff, abs=1e-12)
    assert got[DISPERSION] == 0.0
    assert got[CT1] == 0.0


def test_solve_union_davidson_matches_dense_eigh_on_the_same_space():
    state = _h4_state()
    recovery_dets = [DISPERSION, CT1]
    recovery_coeffs = [0.3, 0.2]
    union = form_final_union(state, recovery_dets, recovery_coeffs, n_ct_max=2)

    solved = solve_union_davidson(state, union["dets"], union["initial_guess"])
    assert solved["converged"]

    ham = selected_hamiltonian(state["h1"], state["eri"], union["dets"], n_orb=4)
    ew, ev = eigh(ham, subset_by_index=[0, 0])
    e_dense = float(ew[0]) + float(state["e_const"])
    assert solved["e_final"] == pytest.approx(e_dense, abs=1e-8)
    assert solved["residual_norm"] < 1e-6


def test_solve_union_davidson_recovers_from_a_guess_orthogonal_to_the_true_ground_state():
    """Regression for a real bug caught while building this module, not a
    hypothetical: on this exact (state, recovery_dets) pair, the true ground state
    lives almost entirely on the CT1 determinant, which the LAS-weighted initial
    guess (mandated by the design doc: LAS amplitudes on P_LAS, zero on new R
    members) has zero weight on. davidson_solve_matfree reports converged=True with
    a near-zero residual for a genuine but non-lowest eigenpair in that case
    (verified directly: -3.7138223 Ha, the SECOND dense eigenvalue, not
    -4.0229844 Ha, the true lowest). solve_union_davidson must not be fooled by
    that native converged=True -- it must still return the true minimum."""
    state = _h4_state()
    union = form_final_union(state, [DISPERSION, CT1], [0.3, 0.2], n_ct_max=2)
    solved = solve_union_davidson(state, union["dets"], union["initial_guess"])

    ham = selected_hamiltonian(state["h1"], state["eri"], union["dets"], n_orb=4)
    ew, ev = eigh(ham)
    e_true_ground = float(ew[0]) + float(state["e_const"])
    e_second = float(ew[1]) + float(state["e_const"])
    # The pathological case is genuinely reproduced here (not silently fixed by an
    # unrelated change) -- the LAS-weighted guess alone converges to e_second.
    assert e_true_ground < e_second - 1e-3
    assert solved["e_final"] == pytest.approx(e_true_ground, abs=1e-8)
    assert solved["n_candidates_tried"] == 2

    c = solved["coefficients"]
    assert abs(float(c @ c) - 1.0) < 1e-8


def test_union_energy_is_at_or_below_the_las_only_energy():
    """Adding recovery determinants to a certified LAS space can only lower (or
    match) the variational energy relative to the LAS-only space -- a direct,
    independent sanity check distinct from the H00 certificate below."""
    state = _h4_state()
    las_dets, las_coeffs = reconstruct_las_vector(state)
    ham_las = selected_hamiltonian(state["h1"], state["eri"], las_dets, n_orb=4)
    e_las_only = (float(las_coeffs @ ham_las @ las_coeffs) / float(las_coeffs @ las_coeffs)
                 + float(state["e_const"]))

    union = form_final_union(state, [DISPERSION, CT1], [0.3, 0.2], n_ct_max=2)
    solved = solve_union_davidson(state, union["dets"], union["initial_guess"])
    assert solved["e_final"] <= e_las_only + 1e-10


def test_certify_final_union_passes_when_bound_holds():
    state = _h4_state()
    union = form_final_union(state, [DISPERSION, CT1], [0.3, 0.2], n_ct_max=2)
    solved = solve_union_davidson(state, union["dets"], union["initial_guess"])
    # h00 for this synthetic fixture: the LAS-only Rayleigh energy (no _assemble_
    # contracted_reference RDM formula wired up for this toy state) -- a legitimate
    # stand-in for H00 for this unit test's purpose (E_final <= "the LAS reference
    # energy" is exactly the bound being tested, regardless of which formula
    # produced that reference number for a real vs. toy state).
    las_dets, las_coeffs = reconstruct_las_vector(state)
    ham_las = selected_hamiltonian(state["h1"], state["eri"], las_dets, n_orb=4)
    h00 = (float(las_coeffs @ ham_las @ las_coeffs) / float(las_coeffs @ las_coeffs)
          + float(state["e_const"]))

    result = certify_final_union(solved, h00, tol=1e-6)
    assert result["passed"]
    assert result["margin"] >= -1e-6


def test_certify_final_union_raises_clearly_on_a_genuine_violation():
    solved = {"e_final": -1.0}
    with pytest.raises(AssertionError, match="exceeds H00"):
        certify_final_union(solved, h00=-2.0, tol=1e-6)


def test_solve_union_davidson_rejects_n_orb_above_64():
    """The boundary is n_orb > 64, not >= 64: alpha/beta strings are native
    uint64_t, which covers orbitals 0-63 exactly, so n_orb=64 itself is fully
    supported (see test_solve_union_davidson_accepts_n_orb_exactly_64 below) --
    only n_orb=65+ genuinely cannot be represented."""
    state = _h4_state()
    state = dict(state, n_orb=65)
    with pytest.raises(NotImplementedError, match="n_orb=65 exceeds 64"):
        solve_union_davidson(state, [(1, 1)], np.array([1.0]))


def test_solve_union_davidson_never_calls_selected_hamiltonian(monkeypatch):
    """Production correctness gate for the O(N^2) memory fix: monkeypatch
    selected_hamiltonian (at its defining module, frasci.recovery._state, since
    _union_solve no longer imports it at all) to raise if called, and prove
    solve_union_davidson still runs to completion with a correct result -- the
    energy/residual recompute must be entirely matrix-free (matvec_matfree), with
    no dense N_union x N_union Hamiltonian built anywhere in the production path."""
    import frasci.recovery._state as state_module
    import frasci.recovery._union_solve as union_module

    # A module-level `from frasci.recovery._state import selected_hamiltonian` in
    # _union_solve.py would bind its OWN reference at import time, before this
    # per-test monkeypatch runs, and silently escape the patch below -- so this is
    # not redundant with it, it catches a different regression (the import coming
    # back) that the patch alone would not.
    assert not hasattr(union_module, "selected_hamiltonian"), (
        "_union_solve.py imports selected_hamiltonian at module level -- that "
        "reference would bypass the monkeypatch below and reintroduce the dense "
        "O(N^2) path")

    def _forbidden(*args, **kwargs):
        raise AssertionError(
            "selected_hamiltonian was called from the production union-solve path "
            "-- it must only ever be used as a dense oracle in tests")

    monkeypatch.setattr(state_module, "selected_hamiltonian", _forbidden)

    state = _h4_state()
    union = form_final_union(state, [DISPERSION, CT1], [0.3, 0.2], n_ct_max=2)
    solved = solve_union_davidson(state, union["dets"], union["initial_guess"])

    # Independently verify correctness using the UN-patched dense oracle imported
    # directly into this test module (a separate reference object, not reached
    # through the monkeypatched frasci.recovery._state module attribute).
    ham = selected_hamiltonian(state["h1"], state["eri"], union["dets"], n_orb=4)
    ew, _ = eigh(ham, subset_by_index=[0, 0])
    e_dense = float(ew[0]) + float(state["e_const"])
    assert solved["e_final"] == pytest.approx(e_dense, abs=1e-8)


def test_solve_union_davidson_accepts_n_orb_exactly_64():
    """Direct proof, not assumption: n_orb=64 is accepted and gives the expected
    trivial one-electron energy, on a minimal synthetic 64-orbital system (h1 =
    identity, eri = 0) rather than real chemistry -- this is purely a boundary/
    dispatch check, not a physics correctness check (that is covered elsewhere by
    the H4/alfe2 fixtures, which stay well under 64 orbitals)."""
    n_orb = 64
    state = dict(
        n_orb=n_orb, e_const=0.0,
        h1=np.eye(n_orb), eri=np.zeros((n_orb, n_orb, n_orb, n_orb)))
    det = (0b0101, 0)  # electrons in orbitals 0 and 2 (bit 63 exists, unused here)
    solved = solve_union_davidson(state, [det], np.array([1.0]))
    assert solved["e_final"] == pytest.approx(2.0, abs=1e-10)  # h1[0,0] + h1[2,2]
    assert solved["residual_norm"] < 1e-10


def test_solve_union_davidson_guess_shape_is_one_by_n(monkeypatch):
    """Davidson guess-shape regression: the native contract is
    initial_guess.shape == (n_states, N); for one root that is (1, N), never
    (N, 1). Intercept the native call on the real LAS-weighted-guess path."""
    import trimci.trimci_core as tc

    fe = tc.fast_expansion
    real_davidson = fe.davidson_solve_matfree
    seen_shapes = []

    def spy(alpha, beta, h1c, eric, n_orb, params, guess):
        arr = np.asarray(guess)
        if arr.size > 0:
            seen_shapes.append(arr.shape)
        return real_davidson(alpha, beta, h1c, eric, n_orb, params, guess)

    monkeypatch.setattr(fe, "davidson_solve_matfree", spy)

    state = _h4_state()
    union = form_final_union(state, [DISPERSION, CT1], [0.3, 0.2], n_ct_max=2)
    solve_union_davidson(state, union["dets"], union["initial_guess"])

    assert seen_shapes, "davidson_solve_matfree was never called"
    for shape in seen_shapes:
        assert shape[0] == 1, (
            f"one-root guess shape {shape} has shape[0]={shape[0]} != 1 -- "
            "native contract is (n_states, N), not (N, n_states)")


def test_solve_union_davidson_rejects_energy_only_convergence(monkeypatch):
    """Davidson stopping regression, real production path: a nonzero
    energy_tol default (fe.DavidsonParams()'s own 1e-7) was found (2026-09-14,
    on the real alfe2_11o10e 103-det union space) to let a converged=True
    native result through with an independently recomputed residual failing
    this function's own residual_tol gate -- reproduce that shape directly by
    mocking the native call to report converged=True for a deliberately
    wrong eigenvector, and verify it is rejected rather than accepted on
    res.converged alone."""
    import trimci.trimci_core as tc

    fe = tc.fast_expansion

    class _FakeConverged:
        def __init__(self, eigenvectors):
            self.converged = True
            self.eigenvectors = eigenvectors
            self.n_iters = 1
            self.residual_norm = 0.0  # native's own (unreliable) scalar

    state = _h4_state()
    union = form_final_union(state, [DISPERSION, CT1], [0.3, 0.2], n_ct_max=2)
    bad_vec = np.zeros(len(union["dets"]))
    bad_vec[0] = 1.0

    def fake_davidson(alpha, beta, h1c, eric, n_orb, params, guess):
        return _FakeConverged(bad_vec.copy())

    monkeypatch.setattr(fe, "davidson_solve_matfree", fake_davidson)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_union_davidson(state, union["dets"], union["initial_guess"])
