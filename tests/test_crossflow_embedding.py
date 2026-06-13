import numpy as np
import pytest

from FRASCI.crossflow.embedding import (
    apply_delta_h,
    correlation_total_energy,
    dress_fragment_h1,
    mf_embedded_energy,
    mf_global_energy,
)


def test_apply_delta_h_diagonal_only():
    n = 4
    h1_base = np.ones((n, n))
    delta = np.array([0.1, 0.2, 0.3, 0.4])
    h1_out = apply_delta_h(h1_base, delta)
    assert h1_out.shape == (n, n)
    np.testing.assert_array_equal(h1_out[0, 1], 1.0)
    np.testing.assert_allclose(np.diag(h1_out), np.ones(n) + delta)


def test_apply_delta_h_does_not_mutate_input():
    n = 3
    h1_base = np.eye(n)
    original = h1_base.copy()
    delta = np.array([1.0, 2.0, 3.0])
    apply_delta_h(h1_base, delta)
    np.testing.assert_array_equal(h1_base, original)


def test_mf_embedded_uses_h1_solver_not_h1_mfa_base():
    n = 2
    eri = np.zeros((n, n, n, n))
    gamma = np.eye(n) * 0.5
    h1_mfa = np.diag([1.0, 2.0])
    delta = np.array([0.5, 0.25])
    h1_solver = apply_delta_h(h1_mfa, delta)
    e_mfa = mf_embedded_energy(h1_mfa, eri, gamma)
    e_solver = mf_embedded_energy(h1_solver, eri, gamma)
    assert abs(e_mfa - e_solver) > 1e-10


def test_correlation_total_energy_formula():
    E_mf = -10.0
    E_ci = [-3.0, -4.0, -2.0]
    E_emb = [-2.0, -3.0, -2.5]
    E_total, corr = correlation_total_energy(E_mf, E_ci, E_emb)
    assert abs(E_total - (-10.0 + (-1) + (-1) + 0.5)) < 1e-12
    assert len(corr) == 3


def test_mf_global_energy_agrees_with_embedded_sum():
    rng = np.random.default_rng(42)
    n = 4
    A = rng.random((n, n))
    h1 = (A + A.T) * 0.1
    B = rng.random((n, n, n, n))
    eri = (
        B
        + B.transpose(1, 0, 2, 3)
        + B.transpose(0, 1, 3, 2)
        + B.transpose(2, 3, 0, 1)
    ) * 0.01
    gamma = np.diag([1.0, 1.0, 0.5, 0.5])
    E_total, E_elec, F = mf_global_energy(h1, eri, gamma, 0.0)
    expected = 0.5 * float(np.einsum("pq,pq->", h1 + F, gamma))
    assert abs(E_elec - expected) < 1e-12
    assert abs(E_total - E_elec) < 1e-12


def test_dress_fragment_h1_zero_env_gamma():
    n = 4
    h1_full = np.diag([1.0, 2.0, 3.0, 4.0])
    eri = np.zeros((n, n, n, n))
    gamma = np.eye(n) * 0.5
    frag_orbs = [0, 1]
    h1_bare = h1_full[np.ix_(frag_orbs, frag_orbs)]
    h1_dressed = dress_fragment_h1(h1_bare, eri, frag_orbs, gamma, n)
    np.testing.assert_allclose(h1_dressed, h1_bare, atol=1e-12)


def test_correlation_total_energy_rejects_length_mismatch():
    with pytest.raises(ValueError):
        correlation_total_energy(-1.0, [-0.5], [-0.4, -0.3])
