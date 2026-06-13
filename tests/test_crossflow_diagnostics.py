import numpy as np

from FRASCI.crossflow.coupling import classify_fragment_spin_orbitals
from FRASCI.crossflow.diagnostics import (
    analyze_denominator_model,
    summarize_gap_values,
)


def _bits(orbs):
    return sum(1 << orb for orb in orbs)


def test_summarize_gap_values_counts_signs():
    stats = summarize_gap_values([-1.0, 0.0, 2.0, 4.0])
    assert stats["n_terms"] == 4
    assert stats["n_negative"] == 1
    assert stats["n_zero"] == 1
    assert stats["n_positive"] == 2
    assert stats["negative_fraction"] == 0.25
    assert stats["min_gap"] == -1.0
    assert stats["max_gap"] == 4.0


def test_analyze_denominator_model_positive_only():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 0.5
    frag_orbs = [[0, 1], [2, 3]]
    spin_classes = [
        classify_fragment_spin_orbitals(_bits([0]), _bits([]), [0, 1]),
        classify_fragment_spin_orbitals(_bits([2]), _bits([]), [2, 3]),
    ]
    eps = {0: np.array([-1.0, 1.0]), 1: np.array([-1.0, 1.0])}
    result = analyze_denominator_model(eri, eps, frag_orbs, spin_classes, [(0, 1)])
    assert result["overall"]["n_terms"] == 1
    assert result["overall"]["n_negative"] == 0
    assert result["overall"]["min_gap"] == 4.0
    assert result["worst_nonpositive_examples"] == []


def test_analyze_denominator_model_records_worst_negative():
    n = 4
    eri = np.zeros((n, n, n, n))
    eri[1, 0, 3, 2] = 0.5
    frag_orbs = [[0, 1], [2, 3]]
    spin_classes = [
        classify_fragment_spin_orbitals(_bits([0]), _bits([]), [0, 1]),
        classify_fragment_spin_orbitals(_bits([2]), _bits([]), [2, 3]),
    ]
    eps = {0: np.array([1.0, -1.0]), 1: np.array([1.0, -1.0])}
    result = analyze_denominator_model(eri, eps, frag_orbs, spin_classes, [(0, 1)])
    assert result["overall"]["n_terms"] == 1
    assert result["overall"]["n_negative"] == 1
    example = result["worst_nonpositive_examples"][0]
    assert example["gap"] == -4.0
    assert example["pair"] == [0, 1]
    assert example["channel"] == "aa"
    assert example["i"] == 0
    assert example["a"] == 1
    assert example["j"] == 2
    assert example["b"] == 3


def test_analyze_denominator_model_pair_and_channel_breakdowns():
    n = 6
    eri = np.zeros((n, n, n, n))
    frag_orbs = [[0, 1], [2, 3], [4, 5]]
    spin_classes = [
        classify_fragment_spin_orbitals(_bits([0]), _bits([0]), [0, 1]),
        classify_fragment_spin_orbitals(_bits([2]), _bits([2]), [2, 3]),
        classify_fragment_spin_orbitals(_bits([4]), _bits([4]), [4, 5]),
    ]
    eps = {
        0: np.array([-1.0, 1.0]),
        1: np.array([-1.0, 1.0]),
        2: np.array([-1.0, 1.0]),
    }
    result = analyze_denominator_model(
        eri,
        eps,
        frag_orbs,
        spin_classes,
        [(0, 1), (0, 2)],
    )
    assert result["by_pair"]["0-1"]["n_terms"] == 4
    assert result["by_pair"]["0-2"]["n_terms"] == 4
    assert result["by_channel"]["aa"]["n_terms"] == 2
    assert result["by_channel"]["bb"]["n_terms"] == 2
    assert result["by_channel"]["ab"]["n_terms"] == 2
    assert result["by_channel"]["ba"]["n_terms"] == 2
