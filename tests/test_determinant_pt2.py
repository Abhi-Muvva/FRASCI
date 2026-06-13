import numpy as np

from FRASCI.crossflow.determinant_pt2 import (
    analyze_determinant_energy_pt2,
    determinant_diagonal_energy,
    excite_one,
)


def test_determinant_energy_opposite_spin_coulomb():
    h1 = np.diag([1.0, 2.0])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 1, 1] = 0.3

    energy = determinant_diagonal_energy(
        h1,
        eri,
        alpha_bits=1 << 0,
        beta_bits=1 << 1,
    )

    assert energy == 3.3


def test_determinant_energy_same_spin_exchange():
    h1 = np.diag([1.0, 2.0])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 1, 1] = 0.4
    eri[1, 1, 0, 0] = 0.4
    eri[0, 1, 1, 0] = 0.1
    eri[1, 0, 0, 1] = 0.1

    energy = determinant_diagonal_energy(
        h1,
        eri,
        alpha_bits=(1 << 0) | (1 << 1),
        beta_bits=0,
    )

    assert np.isclose(energy, 3.3)


def test_excite_one_moves_occupation():
    bits = (1 << 0) | (1 << 2)

    out = excite_one(bits, 0, 1)

    assert out == ((1 << 1) | (1 << 2))


def test_determinant_pt2_matches_simple_positive_gap_case():
    h1 = np.diag([0.0, 1.0, 0.0, 3.0])
    eri = np.zeros((4, 4, 4, 4))
    eri[1, 0, 3, 2] = 0.2
    fragments = [[0, 1], [2, 3]]

    result = analyze_determinant_energy_pt2(
        h1,
        eri,
        ref_alpha_bits=1 << 0,
        ref_beta_bits=1 << 2,
        frag_orbs_list=fragments,
        coupled_pairs=[(0, 1)],
    )

    assert result["n_terms"] == 1
    assert result["n_negative_gap"] == 0
    assert np.isclose(result["E_pt2_cross"], -0.01)
    assert np.isclose(result["gap_summary"]["min"], 4.0)


def test_determinant_pt2_records_negative_gap_case():
    h1 = np.diag([1.0, 0.0, 1.0, 0.0])
    eri = np.zeros((4, 4, 4, 4))
    eri[1, 0, 3, 2] = 0.2
    fragments = [[0, 1], [2, 3]]

    result = analyze_determinant_energy_pt2(
        h1,
        eri,
        ref_alpha_bits=1 << 0,
        ref_beta_bits=1 << 2,
        frag_orbs_list=fragments,
        coupled_pairs=[(0, 1)],
    )

    assert result["n_terms"] == 1
    assert result["n_negative_gap"] == 1
    assert result["negative_gap_fraction"] == 1.0
    assert np.isclose(result["E_pt2_cross"], 0.02)
    assert result["worst_nonpositive_examples"][0]["gap"] == -2.0
