import json

import numpy as np

from FRASCI.lasscf.fragment_sweep import (
    build_fragmentation_candidates,
    default_size_patterns,
    split_order_by_sizes,
    write_candidate_files,
)


def _bits(orbs):
    return sum(1 << o for o in orbs)


def _toy_problem(n_orb=12):
    h1 = np.diag(np.linspace(-2.0, 2.0, n_orb))
    eri = np.zeros((n_orb, n_orb, n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            value = 0.02 * (p + q + 1)
            eri[p, p, q, q] = value
            eri[p, q, q, p] = 0.5 * value
    return {
        "h1": h1,
        "eri": eri,
        "n_orb": n_orb,
        "n_elec": 12,
        "n_alpha": 6,
        "n_beta": 6,
        "e_nuc": 0.0,
        "ref_alpha_bits": _bits(range(6)),
        "ref_beta_bits": _bits([0, 1, 2, 6, 7, 8]),
    }


def test_split_order_by_sizes_uses_all_orbitals_once():
    fragments = split_order_by_sizes(range(6), [1, 2, 3])
    assert fragments == [[0], [1, 2], [3, 4, 5]]


def test_default_size_patterns_are_valid_for_fe4s4_space():
    patterns = default_size_patterns(36)
    assert len(patterns) >= 18
    assert all(sum(pattern) == 36 for pattern in patterns)
    assert [12, 12, 12] in patterns
    assert [6, 12, 18] in patterns


def test_build_fragmentation_candidates_returns_valid_unique_partitions():
    candidates = build_fragmentation_candidates(_toy_problem(), target_count=10)
    assert len(candidates) == 10

    seen = set()
    for candidate in candidates:
        flat = sorted(orb for frag in candidate.fragments for orb in frag)
        assert flat == list(range(12))
        assert sum(candidate.size_pattern) == 12
        assert candidate.max_fragment_size == max(candidate.size_pattern)
        assert len(candidate.nelec_per_frag) == len(candidate.fragments)
        assert candidate.spin_sub == [
            abs(na - nb) + 1 for na, nb in candidate.nelec_per_frag
        ]
        canon = tuple(sorted(tuple(frag) for frag in candidate.fragments))
        assert canon not in seen
        seen.add(canon)


def test_write_candidate_files_creates_json_and_markdown(tmp_path):
    candidates = build_fragmentation_candidates(_toy_problem(), target_count=3)
    write_candidate_files(candidates, str(tmp_path))

    catalog = json.loads((tmp_path / "fragmentation_catalog.json").read_text())
    assert len(catalog) == 3
    assert (tmp_path / "fragmentation_catalog.md").exists()
    for candidate in candidates:
        path = tmp_path / "candidates" / f"{candidate.name}.json"
        assert json.loads(path.read_text()) == candidate.fragments
