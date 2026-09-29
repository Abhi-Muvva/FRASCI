from __future__ import annotations

import importlib
import itertools

import numpy as np
import pytest
from pyscf import ao2mo

hybrid = importlib.import_module("frasci.recovery._solve")
from frasci.recovery._state import selected_hamiltonian


def _problem():
    rng = np.random.default_rng(91)
    n_orb = 4
    h1 = rng.standard_normal((n_orb, n_orb))
    h1 = np.ascontiguousarray(0.5 * (h1 + h1.T))
    eri_pair = rng.standard_normal((n_orb * (n_orb + 1) // 2,) * 2)
    eri = np.ascontiguousarray(ao2mo.restore(1, eri_pair, n_orb))
    eri = np.ascontiguousarray(0.5 * (eri + eri.transpose(2, 3, 0, 1)))
    dets = [
        (0b0011, 0b0011),
        (0b0101, 0b0011),
        (0b1001, 0b0011),
        (0b0011, 0b0101),
        (0b0101, 0b0101),
        (0b1001, 0b1001),
    ]
    h_all = selected_hamiltonian(h1, eri, dets, n_orb=n_orb)
    parent_coeffs = np.asarray([0.8, 0.6])
    parent = np.zeros(len(dets))
    parent[:2] = parent_coeffs
    h00 = float(parent @ h_all @ parent)
    h0d = {det: float(parent @ h_all[:, i]) for i, det in enumerate(dets)}
    state = {
        "n_orb": n_orb,
        "h1": h1,
        "eri": eri,
        "e_const": -0.37,
        "h00": h00 - 0.37,
    }
    return state, dets, h_all, parent_coeffs, h0d


def _eligible(dets, parent_coeffs, *, include_complete_parent=False):
    selected = dets[:5] if include_complete_parent else [dets[0], *dets[2:5]]
    overlaps = {dets[0]: float(parent_coeffs[0])}
    if include_complete_parent:
        overlaps[dets[1]] = float(parent_coeffs[1])
    return {
        det: {
            "coeff": 1.0,
            "s": overlaps.get(det, 0.0),
            "label": "same_sector_present" if det in overlaps else "ct1",
            "n_ct": 0 if det in overlaps else 1,
        }
        for det in selected
    }


def test_trimci_indexed_matvec_matches_dense_selected_hamiltonian():
    state, dets, h_all, _parent_coeffs, _h0d = _problem()
    operator = hybrid._MatrixFreeDeterminantHamiltonian(state, dets)
    vector = np.linspace(-0.7, 0.9, len(dets))

    got = operator.matvec(vector)
    expected = (h_all + np.eye(len(dets)) * state["e_const"]) @ vector

    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        operator.diagonal,
        np.diag(h_all) + state["e_const"], atol=1e-10, rtol=1e-10)


def test_indexed_matvec_matches_dense_on_larger_selected_space():
    rng = np.random.default_rng(104)
    n_orb = 8
    h1 = rng.standard_normal((n_orb, n_orb))
    h1 = np.ascontiguousarray(0.5 * (h1 + h1.T))
    pair_size = n_orb * (n_orb + 1) // 2
    eri_pair = rng.standard_normal((pair_size, pair_size))
    eri_pair = 0.5 * (eri_pair + eri_pair.T)
    eri = np.ascontiguousarray(ao2mo.restore(1, eri_pair, n_orb))
    strings = [sum(1 << i for i in occupied)
               for occupied in itertools.combinations(range(n_orb), 4)]
    all_dets = list(itertools.product(strings, strings))
    chosen = rng.choice(len(all_dets), size=96, replace=False)
    dets = [all_dets[i] for i in chosen]
    state = dict(
        n_orb=n_orb, h1=h1, eri=eri, e_const=-2.1, h00=-3.0)
    operator = hybrid._MatrixFreeDeterminantHamiltonian(state, dets)
    dense = selected_hamiltonian(h1, eri, dets, n_orb=n_orb)
    vector = rng.standard_normal(len(dets))

    np.testing.assert_allclose(
        operator.matvec(vector),
        (dense + np.eye(len(dets)) * state["e_const"]) @ vector,
        atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("include_complete_parent", [False, True])
def test_matrix_free_hybrid_matches_dense_with_same_sector_overlap(
        monkeypatch, include_complete_parent):
    state, dets, _h_all, parent_coeffs, h0d = _problem()
    eligible = _eligible(
        dets, parent_coeffs, include_complete_parent=include_complete_parent)
    monkeypatch.setattr(hybrid, "h0d", lambda det, _state: h0d[det] - 0.37 * eligible[det]["s"])

    dense = hybrid.solve_contracted_hybrid(state, eligible)
    matrix_free = hybrid.solve_contracted_hybrid_matrix_free(
        state, eligible, tol=1e-11, max_iter=500, max_subspace=40)

    assert matrix_free["e_tot"] == pytest.approx(dense["e_tot"], abs=2e-9)
    assert matrix_free["e_c"] == pytest.approx(dense["e_c"], abs=2e-9)
    assert matrix_free["overlap"]["c_dagger_S_c"] == pytest.approx(1.0, abs=1e-9)
    assert matrix_free["generalized_residual_norm"] < 1e-8
    assert matrix_free["solver_kind"] == "matrix_free"
    if include_complete_parent:
        assert matrix_free["overlap"]["n_overlap_directions_removed"] == 1
