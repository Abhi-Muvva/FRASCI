import numpy as np
import pytest

from FRASCI.crossflow.coupling import (
    apply_damped_delta_h,
    classify_fragment_spin_orbitals,
    compute_all_pair_coupling,
    compute_pt2_pair_coupling,
)


def _bits(orbs):
    return sum(1 << o for o in orbs)


def test_classify_alpha_beta_from_ref():
    alpha_bits = _bits([0, 1])
    beta_bits = _bits([1, 2])
    frag_orbs = [0, 1, 2]
    sc = classify_fragment_spin_orbitals(alpha_bits, beta_bits, frag_orbs)
    assert sc.occ_alpha == [0, 1]
    assert sc.virt_alpha == [2]
    assert sc.occ_beta == [1, 2]
    assert sc.virt_beta == [0]


def test_classify_open_shell_orbital():
    alpha_bits = _bits([0])
    beta_bits = _bits([])
    sc = classify_fragment_spin_orbitals(alpha_bits, beta_bits, [0, 1])
    assert 0 in sc.occ_alpha
    assert 0 in sc.virt_beta


def test_pt2_sign_convention():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 1.0
    eri[1, 2, 3, 0] = 0.0
    h1_I = np.diag([-1.0, 1.0])
    h1_J = np.diag([-1.0, 1.0])
    sc_I = classify_fragment_spin_orbitals(_bits([0]), _bits([0, 1]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2]), _bits([2, 3]), [2, 3])
    result = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    assert abs(result.E_pt2 - (-0.25)) < 1e-12


def test_pt2_energy_non_positive():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 0.7
    h1_I = np.diag([-1.0, 2.0])
    h1_J = np.diag([-1.0, 2.0])
    sc_I = classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2]), _bits([2]), [2, 3])
    result = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    assert result.E_pt2 <= 0.0


def test_same_spin_antisymmetrized():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 1.0
    eri[1, 2, 3, 0] = 0.5
    h1_I = np.diag([-1.0, 2.0])
    h1_J = np.diag([-1.0, 2.0])
    sc_I = classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2]), _bits([2]), [2, 3])
    result = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    expected = -(0.25 / 6 + 0.25 / 6 + 1.0 / 6 + 1.0 / 6)
    assert abs(result.E_pt2 - expected) < 1e-10


def test_opp_spin_direct_only():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 1.0
    eri[1, 2, 3, 0] = 99.0
    h1_I = np.diag([-1.0, 2.0])
    h1_J = np.diag([-1.0, 2.0])
    alpha_bits_I = _bits([0, 1])
    beta_bits_I = _bits([0])
    alpha_bits_J = _bits([2])
    beta_bits_J = _bits([2, 3])
    sc_I = classify_fragment_spin_orbitals(alpha_bits_I, beta_bits_I, [0, 1])
    sc_J = classify_fragment_spin_orbitals(alpha_bits_J, beta_bits_J, [2, 3])
    result = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    gap = (2.0 + 2.0) - (-1.0 + -1.0)
    expected = -(1.0 / gap)
    assert abs(result.E_pt2 - expected) < 1e-10


def test_zero_virt_pair_zero_energy():
    n = 4
    eri = np.random.default_rng(0).random((n, n, n, n))
    h1_I = np.diag([-1.0, -0.5])
    h1_J = np.diag([-1.0, -0.5])
    sc_I = classify_fragment_spin_orbitals(_bits([0, 1]), _bits([0, 1]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2, 3]), _bits([2, 3]), [2, 3])
    result = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    assert result.E_pt2 == 0.0
    assert result.n_terms == 0
    assert result.min_gap is None
    np.testing.assert_array_equal(result.delta_h_I, np.zeros(2))
    np.testing.assert_array_equal(result.delta_h_J, np.zeros(2))


def test_negative_gap_raises():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 1.0
    h1_I = np.diag([2.0, -1.0])
    h1_J = np.diag([2.0, -1.0])
    sc_I = classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2]), _bits([2]), [2, 3])
    with pytest.raises(ValueError, match="Non-positive gap"):
        compute_pt2_pair_coupling(
            eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
        )


def test_delta_h_virt_negative_occ_positive():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 1.0
    h1_I = np.diag([-1.0, 2.0])
    h1_J = np.diag([-1.0, 2.0])
    sc_I = classify_fragment_spin_orbitals(_bits([0]), _bits([]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2]), _bits([]), [2, 3])
    result = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    assert result.delta_h_I[1] < 0
    assert result.delta_h_J[1] < 0
    assert result.delta_h_I[0] > 0
    assert result.delta_h_J[0] > 0


def test_coupling_symmetric_swap():
    n = 4
    rng = np.random.default_rng(7)
    raw = rng.random((n, n, n, n)) * 0.1
    eri = np.zeros_like(raw)
    for p, q, r, s in np.ndindex(raw.shape):
        value = raw[p, q, r, s]
        for idx in {
            (p, q, r, s),
            (q, p, r, s),
            (p, q, s, r),
            (q, p, s, r),
            (r, s, p, q),
            (s, r, p, q),
            (r, s, q, p),
            (s, r, q, p),
        }:
            eri[idx] = value
    h1_I = np.diag([-1.0, 2.0])
    h1_J = np.diag([-0.5, 1.5])
    sc_I = classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1])
    sc_J = classify_fragment_spin_orbitals(_bits([2]), _bits([2]), [2, 3])
    r_IJ = compute_pt2_pair_coupling(
        eri, h1_I, h1_J, [0, 1], [2, 3], sc_I, sc_J, 0, 1
    )
    r_JI = compute_pt2_pair_coupling(
        eri, h1_J, h1_I, [2, 3], [0, 1], sc_J, sc_I, 1, 0
    )
    assert abs(r_IJ.E_pt2 - r_JI.E_pt2) < 1e-10


def test_all_pair_accumulates_delta_h():
    n = 6
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 0.5
    eri[1, 0, 5, 4] = 0.3
    h = {
        0: np.diag([-1.0, 2.0]),
        1: np.diag([-1.0, 2.0]),
        2: np.diag([-1.0, 2.0]),
    }
    frags = [[0, 1], [2, 3], [4, 5]]
    sc = [
        classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1]),
        classify_fragment_spin_orbitals(_bits([2]), _bits([2]), [2, 3]),
        classify_fragment_spin_orbitals(_bits([4]), _bits([4]), [4, 5]),
    ]
    result = compute_all_pair_coupling(eri, h, frags, sc, [(0, 1), (0, 2)])
    assert result.delta_h_by_frag[0].shape == (2,)
    assert result.delta_h_by_frag[1].shape == (2,)
    assert result.delta_h_by_frag[2].shape == (2,)
    assert np.any(result.delta_h_by_frag[0] != 0)


def test_zero_virt_fragment_unchanged_delta_h():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 0.5
    h = {0: np.diag([-1.0, 2.0]), 1: np.diag([-0.5, -0.3])}
    frags = [[0, 1], [2, 3]]
    sc = [
        classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1]),
        classify_fragment_spin_orbitals(_bits([2, 3]), _bits([2, 3]), [2, 3]),
    ]
    result = compute_all_pair_coupling(eri, h, frags, sc, [(0, 1)])
    np.testing.assert_array_equal(result.delta_h_by_frag[1], np.zeros(2))


def test_apply_damped_delta_h_first_call():
    delta_new = np.array([1.0, 2.0])
    result = apply_damped_delta_h(delta_new, None, 0.6)
    np.testing.assert_allclose(result, 0.6 * delta_new)


def test_apply_damped_delta_h_full_damping():
    delta_new = np.array([1.0, 2.0])
    result = apply_damped_delta_h(delta_new, None, 1.0)
    np.testing.assert_allclose(result, delta_new)


def test_apply_damped_delta_h_mixing():
    delta_new = np.array([1.0, 0.0])
    delta_prev = np.array([0.0, 1.0])
    result = apply_damped_delta_h(delta_new, delta_prev, 0.4)
    np.testing.assert_allclose(result, 0.4 * delta_new + 0.6 * delta_prev)
