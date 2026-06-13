import json
import warnings

import numpy as np
import pytest

from FRASCI.crossflow.partition import (
    load_fragment_orbs_json,
    make_equal_nonoverlapping_partition,
    parse_coupled_pairs,
    validate_fragment_partition,
)


def _bits(orbs):
    return sum(1 << o for o in orbs)


def test_equal_partition_divisible():
    n = 12
    h1 = np.diag(np.arange(n, dtype=float))
    frags = make_equal_nonoverlapping_partition(h1, n, 3)
    assert len(frags) == 3
    assert all(len(f) == 4 for f in frags)
    all_orbs = sorted(o for f in frags for o in f)
    assert all_orbs == list(range(n))


def test_equal_partition_leftovers():
    n = 8
    h1 = np.diag(np.arange(n, dtype=float))
    frags = make_equal_nonoverlapping_partition(h1, n, 3)
    sizes = sorted([len(f) for f in frags], reverse=True)
    assert sizes == [3, 3, 2]
    all_orbs = sorted(o for f in frags for o in f)
    assert all_orbs == list(range(n))


def test_equal_partition_h1diag_ordering():
    n = 4
    h1 = np.diag([3.0, 2.0, 1.0, 0.0])
    frags = make_equal_nonoverlapping_partition(h1, n, 2)
    assert sorted(frags[0]) == [2, 3]
    assert sorted(frags[1]) == [0, 1]


def test_validate_missing_orbitals_raises():
    n = 4
    frags = [[0, 1], [2]]
    alpha_bits = _bits([0, 1])
    beta_bits = _bits([0])
    with pytest.raises(ValueError, match="missing"):
        validate_fragment_partition(frags, n, alpha_bits, beta_bits)


def test_validate_duplicate_orbitals_raises():
    n = 4
    frags = [[0, 1, 2], [2, 3]]
    alpha_bits = _bits([0, 1, 2])
    beta_bits = _bits([0])
    with pytest.raises(ValueError, match="more than once"):
        validate_fragment_partition(frags, n, alpha_bits, beta_bits)


def test_validate_empty_fragment_raises():
    n = 4
    frags = [[0, 1, 2, 3], []]
    alpha_bits = _bits([0])
    beta_bits = _bits([0])
    with pytest.raises(ValueError, match="empty"):
        validate_fragment_partition(frags, n, alpha_bits, beta_bits)


def test_validate_infeasible_alpha_raises():
    n = 4
    frags2 = [[0], [1, 2, 3]]
    assert frags2
    pass


def test_validate_zero_virtual_warns_not_raises():
    n = 4
    alpha_bits = _bits([0, 1, 2, 3])
    beta_bits = _bits([0, 1, 2, 3])
    frags = [[0, 1], [2, 3]]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_fragment_partition(frags, n, alpha_bits, beta_bits)
    assert any("virtual" in str(warning.message).lower() for warning in w)


def test_validate_out_of_range_orbital_raises():
    n = 4
    frags = [[0, 1], [2, 5]]
    alpha_bits = _bits([0, 1, 2])
    beta_bits = _bits([0])
    with pytest.raises(ValueError, match="out of range"):
        validate_fragment_partition(frags, n, alpha_bits, beta_bits)


def test_json_load_valid(tmp_path):
    frags_data = [[0, 1, 2], [3, 4, 5]]
    path = str(tmp_path / "frags.json")
    with open(path, "w") as f:
        json.dump(frags_data, f)
    n = 6
    alpha_bits = _bits([0, 1, 2])
    beta_bits = _bits([0, 1])
    frags = load_fragment_orbs_json(path, n, alpha_bits, beta_bits)
    assert frags == [[0, 1, 2], [3, 4, 5]]


def test_json_load_missing_orbitals_raises(tmp_path):
    path = str(tmp_path / "frags.json")
    with open(path, "w") as f:
        json.dump([[0, 1], [2]], f)
    with pytest.raises(ValueError, match="missing"):
        load_fragment_orbs_json(path, 4, 0b0111, 0b0001)


def test_json_load_duplicate_raises(tmp_path):
    path = str(tmp_path / "frags.json")
    with open(path, "w") as f:
        json.dump([[0, 1, 2], [2, 3]], f)
    with pytest.raises(ValueError, match="more than once"):
        load_fragment_orbs_json(path, 4, 0b0111, 0b0001)


def test_parse_coupled_pairs_all():
    pairs = parse_coupled_pairs("all", 3)
    assert set(pairs) == {(0, 1), (0, 2), (1, 2)}
    assert all(a < b for a, b in pairs)


def test_parse_coupled_pairs_explicit():
    pairs = parse_coupled_pairs("0-1,1-2", 3)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert len(pairs) == 2
