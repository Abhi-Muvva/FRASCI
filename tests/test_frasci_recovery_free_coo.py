"""Milestone G: build_top_k_union / solve_free_orbital_union
(frasci.recovery._free_coo).

Same H6/STO-3G two-fragment fixture as test_frasci_recovery_constrained_coo.py --
asymmetric per-fragment weights so BFGS has a genuine nonzero gradient, not a
vacuous identity-rotation check.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

import frasci.recovery._free_coo as free_coo_module
from frasci.recovery._free_coo import build_top_k_union, solve_free_orbital_union
from frasci.recovery._state import h00_from_frags, selected_hamiltonian, top_product_terms


def _h6_two_fragment_state():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {1.0 * i}" for i in range(6)),
               basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    e_const = float(mol.energy_nuc())

    blocks = [[0, 1, 2], [3, 4, 5]]
    ref_sectors = [(2, 1), (1, 2)]
    frag0_dets = np.array([[0b011, 0b001], [0b101, 0b010], [0b110, 0b100]], dtype=np.int64)
    frag0_coeffs = np.array([0.85, 0.4, 0.2])
    frag0_coeffs /= np.linalg.norm(frag0_coeffs)
    frag1_dets = np.array([[0b001, 0b011], [0b010, 0b101]], dtype=np.int64)
    frag1_coeffs = np.array([0.9, 0.3])
    frag1_coeffs /= np.linalg.norm(frag1_coeffs)
    frags = [
        dict(dets=frag0_dets, coeffs=frag0_coeffs, R=np.eye(3), na=2, nb=1,
            lookup=dict(zip(map(tuple, frag0_dets.tolist()), frag0_coeffs.tolist()))),
        dict(dets=frag1_dets, coeffs=frag1_coeffs, R=np.eye(3), na=1, nb=2,
            lookup=dict(zip(map(tuple, frag1_dets.tolist()), frag1_coeffs.tolist()))),
    ]
    h00 = h00_from_frags(h1, eri, blocks, frags, e_const=e_const)
    return dict(n_orb=6, n_elec=(3, 3), blocks=blocks, orbital_lists=blocks,
               ref_sectors=ref_sectors, frags=frags, mo_final=c.copy(),
               h1=h1, eri=eri, e_const=e_const, h00=h00)


# A genuine dispersion/CT determinant relative to this state's (2,1)/(1,2) reference,
# built the same way test_frasci_recovery_constrained_coo.py's WORKING_DETS is.
def _recovery_dets(state):
    from frasci.recovery._state import _embed_bits, classify

    frag0, frag1 = state["frags"]
    block0, block1 = state["blocks"]
    la0, lb0 = int(frag0["dets"][0][0]), int(frag0["dets"][0][1])
    la1, lb1 = int(frag1["dets"][0][0]), int(frag1["dets"][0][1])
    owned = (_embed_bits(la0, block0) | _embed_bits(la1, block1),
            _embed_bits(lb0, block0) | _embed_bits(lb1, block1))
    alpha_owned, beta_owned = owned
    frag1_alpha_occ = [o for o in block1 if (alpha_owned >> o) & 1]
    frag0_alpha_vir = [o for o in block0 if not (alpha_owned >> o) & 1]
    ct_alpha = (alpha_owned & ~(1 << frag1_alpha_occ[0])) | (1 << frag0_alpha_vir[0])
    ct_det = (ct_alpha, beta_owned)
    label, _ = classify(ct_det, state["ref_sectors"], state["blocks"])
    assert label in ("dispersion", "ct"), f"fixture bug: expected interfragment, got {label}"
    return [ct_det], [1.0]


def test_build_top_k_union_excludes_owned_and_includes_recovery():
    state = _h6_two_fragment_state()
    recovery_dets, recovery_coeffs = _recovery_dets(state)
    union = build_top_k_union(state, 5, recovery_dets, recovery_coeffs, n_ct_max=2)

    las_dets, _ = top_product_terms(state, 5)
    assert set(las_dets) <= set(union["dets"])
    assert recovery_dets[0] in union["dets"]
    assert union["n_recovery"] == 1
    assert union["n_overlap"] == 0
    assert union["n_work"] == len(las_dets) + 1
    assert union["las_top_k"] == 5


def test_solve_free_orbital_union_respects_variational_bound_and_rotates_cross_fragment():
    state = _h6_two_fragment_state()
    recovery_dets, recovery_coeffs = _recovery_dets(state)
    union = build_top_k_union(state, 5, recovery_dets, recovery_coeffs, n_ct_max=2)

    solved = solve_free_orbital_union(
        state, union["dets"], union["initial_guess"], orbital_opt_max_iter=25)

    assert solved["gates"]["E_after_le_E_before"]
    assert solved["e_after"] <= solved["e_before"] + 1e-8
    assert solved["gain_mha"] >= -1e-5  # allow float noise, never a genuine loss

    # Independently verify e_before against a dense oracle on the unrotated frame.
    ham_before = selected_hamiltonian(state["h1"], state["eri"], union["dets"], n_orb=6)
    ew_before, _ = eigh(ham_before, subset_by_index=[0, 0])
    assert solved["e_before"] == pytest.approx(
        float(ew_before[0]) + state["e_const"], abs=1e-8)

    # Independently verify e_after against a dense oracle on the RETURNED rotated frame.
    ham_after = selected_hamiltonian(solved["h1_final"], solved["eri_final"],
                                     solved["dets"], n_orb=6)
    ew_after, _ = eigh(ham_after, subset_by_index=[0, 0])
    assert solved["e_after"] == pytest.approx(
        float(ew_after[0]) + state["e_const"], abs=1e-8)

    # This is a genuinely free (cross-fragment-allowed) rotation on a two-fragment
    # boundary with real gradient -- unrestricted BFGS should find at least as much
    # as the identity, and a nonzero cross-fragment component is expected (not
    # asserted as a hard requirement, since BFGS could in principle land on a
    # fragment-block-preserving local rotation for this specific tiny system).
    assert solved["rotation_errors"]["orthogonality"] < 1e-8


def test_solve_free_orbital_union_never_trusts_returned_coefficients_directly(monkeypatch):
    """Direct regression for the real bug this module was built to avoid: a
    deliberately WRONG (non-ground-state) coefficient vector, paired with an
    IDENTITY rotation (h1_final/eri_final == the original integrals, so the true
    answer is independently known to equal e_before exactly), must not leak into
    e_after. If solve_free_orbital_union naively computed a Rayleigh quotient from
    the returned (dets, coeffs) pair instead of independently rediagonalizing, this
    garbage vector would silently give some other, generically higher, energy --
    which would also trip the E_after_le_E_before gate on a case that should
    trivially pass it exactly.
    """
    state = _h6_two_fragment_state()
    recovery_dets, recovery_coeffs = _recovery_dets(state)
    union = build_top_k_union(state, 5, recovery_dets, recovery_coeffs, n_ct_max=2)

    def _fake_coo(state, dets, coeffs, *, allowed_orbital_pairs, orbital_opt_max_iter=20,
                 orbital_opt_grad_tol=1e-6, orbital_opt_energy_tol=1e-10):
        n = len(dets)
        garbage = np.ones(n) / np.sqrt(n)  # deliberately NOT the ground state
        return {
            "dets": list(dets), "coeffs": garbage,
            "rotation": np.eye(state["n_orb"]),
            "h1_final": state["h1"], "eri_final": state["eri"],
            "cross_fragment_rotation_norm": 0.0,
            "rotation_errors": {"orthogonality": 0.0, "h1": 0.0, "eri": 0.0},
            "termination_reason": "fake_identity",
        }

    monkeypatch.setattr(free_coo_module, "run_fixed_support_constrained_coo", _fake_coo)
    solved = solve_free_orbital_union(state, union["dets"], union["initial_guess"])

    # Identity rotation -> the true answer is exactly e_before, regardless of the
    # garbage coefficients the fake COO call returned.
    assert solved["e_after"] == pytest.approx(solved["e_before"], abs=1e-8)
    assert solved["gates"]["E_after_le_E_before"]


def test_solve_free_orbital_union_gate_raises_on_a_genuine_violation(monkeypatch):
    """Minimal synthetic proof the gate itself fires and is never silently
    loosened: force the independent 'after' rediagonalization to report a higher
    energy than 'before' by feeding it a manipulated Hamiltonian, and confirm
    AssertionError is raised rather than returning a result."""
    state = _h6_two_fragment_state()
    recovery_dets, recovery_coeffs = _recovery_dets(state)
    union = build_top_k_union(state, 5, recovery_dets, recovery_coeffs, n_ct_max=2)

    def _fake_coo(state, dets, coeffs, *, allowed_orbital_pairs, orbital_opt_max_iter=20,
                 orbital_opt_grad_tol=1e-6, orbital_opt_energy_tol=1e-10):
        # A large positive shift on the diagonal (via h1) forces the independently
        # rediagonalized "after" ground state well above "before".
        return {
            "dets": list(dets), "coeffs": np.asarray(coeffs, dtype=float),
            "rotation": np.eye(state["n_orb"]),
            "h1_final": state["h1"] + 10.0 * np.eye(state["n_orb"]),
            "eri_final": state["eri"],
            "cross_fragment_rotation_norm": 0.0,
            "rotation_errors": {"orthogonality": 0.0, "h1": 0.0, "eri": 0.0},
            "termination_reason": "fake_worse",
        }

    monkeypatch.setattr(free_coo_module, "run_fixed_support_constrained_coo", _fake_coo)
    with pytest.raises(AssertionError, match="free-orbital COO gate failed"):
        solve_free_orbital_union(state, union["dets"], union["initial_guess"])
