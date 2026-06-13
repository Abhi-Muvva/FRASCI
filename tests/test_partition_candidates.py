import json

import numpy as np

from FRASCI.crossflow.partition_candidates import (
    build_priority1_candidates,
    make_index_contiguous_partition,
    make_integral_graph_partition,
    make_strong_pair_partition,
    orbital_coupling_matrix,
    write_partition_json,
)
from FRASCI.crossflow.partition import load_fragment_orbs_json


def _bits(orbs):
    return sum(1 << o for o in orbs)


def _toy_integrals(n):
    h1 = np.diag(np.arange(n, dtype=float))
    eri = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            eri[p, p, q, q] = 0.01 * (p + q + 1)
            eri[p, q, q, p] = 0.02 * (p + q + 1)
    for p, q, value in [(0, 1, 5.0), (2, 3, 4.0), (4, 5, 3.0)]:
        h1[p, q] = h1[q, p] = value
        eri[p, p, q, q] = eri[q, q, p, p] = value
        eri[p, q, q, p] = eri[q, p, p, q] = value
    return h1, eri


def test_index_contiguous_partition_splits_evenly():
    assert make_index_contiguous_partition(6, 3) == [[0, 1], [2, 3], [4, 5]]


def test_orbital_coupling_matrix_is_symmetric_with_zero_diagonal():
    h1, eri = _toy_integrals(6)
    weights = orbital_coupling_matrix(h1, eri)
    assert weights.shape == (6, 6)
    assert np.allclose(weights, weights.T)
    assert np.allclose(np.diag(weights), 0.0)
    assert weights[0, 1] > weights[0, 2]


def test_integral_graph_partition_covers_orbitals_once():
    h1, eri = _toy_integrals(12)
    alpha_bits = _bits([0, 1, 2, 3, 4, 5])
    beta_bits = _bits([0, 1, 2, 3, 6, 7])
    frags = make_integral_graph_partition(h1, eri, alpha_bits, beta_bits, 3)
    assert len(frags) == 3
    assert sorted(o for frag in frags for o in frag) == list(range(12))
    assert all(len(frag) == 4 for frag in frags)


def test_strong_pair_partition_keeps_dominant_pairs():
    h1, eri = _toy_integrals(6)
    alpha_bits = _bits([0, 2, 4])
    beta_bits = _bits([0, 2, 4])
    frags = make_strong_pair_partition(h1, eri, alpha_bits, beta_bits, 3)
    assert any(set([0, 1]).issubset(set(frag)) for frag in frags)
    assert sorted(o for frag in frags for o in frag) == list(range(6))


def test_build_priority1_candidates_returns_valid_named_partitions():
    h1, eri = _toy_integrals(12)
    alpha_bits = _bits([0, 1, 2, 3, 4, 5])
    beta_bits = _bits([0, 1, 2, 3, 6, 7])
    candidates = build_priority1_candidates(h1, eri, alpha_bits, beta_bits, 3)
    assert [candidate.name for candidate in candidates] == [
        "index_contiguous",
        "integral_graph",
        "strong_pair",
    ]
    for candidate in candidates:
        assert sorted(o for frag in candidate.fragments for o in frag) == list(range(12))


def test_write_partition_json_round_trips(tmp_path):
    path = tmp_path / "partition.json"
    frags = [[2, 0], [3, 1]]
    write_partition_json(str(path), frags)
    assert json.loads(path.read_text()) == [[0, 2], [1, 3]]
    loaded = load_fragment_orbs_json(str(path), 4, _bits([0, 1]), _bits([0]))
    assert loaded == [[0, 2], [1, 3]]
