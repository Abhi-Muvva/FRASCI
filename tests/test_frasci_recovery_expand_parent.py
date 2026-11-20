"""frasci.recovery._expand_parent.compute_expanded_parent -- the Expanded-Parent
diagnostic promoted from the notebook-level probe into a first-class, optional
recovery.solve() stage.

Same H6/STO-3G two-fragment fixture as test_frasci_recovery_free_coo.py (3 x 2 = 6
Cartesian product terms -- small enough to check the full-K limit against an
independent dense oracle).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from frasci.recovery._expand_parent import compute_expanded_parent
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


def test_expand_parent_rejects_nonpositive_k():
    state = _h6_two_fragment_state()
    with pytest.raises(ValueError, match="expand_parent_k"):
        compute_expanded_parent(state, 0)


def test_expand_parent_small_k_matches_requested_and_is_a_genuine_subset():
    state = _h6_two_fragment_state()
    result = compute_expanded_parent(state, 2)

    assert result["requested_k"] == 2
    assert result["actual_n"] == 2
    assert result["target_met"] is True
    las_dets, _ = top_product_terms(state, 2)
    assert set(result["dets"]) == set(las_dets)


def test_expand_parent_full_k_matches_dense_oracle_and_caps_at_cartesian_product():
    state = _h6_two_fragment_state()
    full_product = len(state["frags"][0]["dets"]) * len(state["frags"][1]["dets"])
    assert full_product == 6

    # Request more than the full product: actual_n must cap at the achievable size,
    # exactly like achieved recovery N elsewhere in this package -- never padded,
    # never raised as an error.
    result = compute_expanded_parent(state, 100)
    assert result["requested_k"] == 100
    assert result["actual_n"] == full_product
    assert result["target_met"] is False

    # Independent correctness check: exactly diagonalize the same determinant set
    # with a dense oracle and confirm the energy matches.
    ham = selected_hamiltonian(state["h1"], state["eri"], result["dets"], n_orb=6)
    eigenvalues, _ = eigh(ham, subset_by_index=[0, 0])
    expected_energy = float(eigenvalues[0]) + state["e_const"]
    assert result["energy_ha"] == pytest.approx(expected_energy, abs=1e-8)

    expected_gain_mha = (state["h00"] - expected_energy) * 1000.0
    assert result["gain_mha"] == pytest.approx(expected_gain_mha, abs=1e-6)


def test_expand_parent_gain_sign_convention_positive_means_better_than_parent():
    """compute_expanded_parent must never silently flip this sign -- it is exactly
    the convention `free_coo_gain_mha` already uses elsewhere in this package
    (e_before - e_after, positive = improvement), reused here as h00 - energy."""
    state = _h6_two_fragment_state()
    result = compute_expanded_parent(state, 6)
    assert result["gain_mha"] == pytest.approx(
        (state["h00"] - result["energy_ha"]) * 1000.0, abs=1e-10)
