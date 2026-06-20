"""
test_fragments_strong_pair.py
=============================
Tests for strong_pair_fragments() in FRASCI/lasscf/fragments.py.

The canonical source of truth for the strong_pair 12/12/12 orbital lists is:
    FRASCI/Outputs/mfa/
        outs_gs_strongpair_20260609_152510/gs_metadata.json

which is the most recent SC MFA Gauss-Seidel run and carries lists identical
to all five earlier runs (20260531_100318, 20260531_101138, 20260531_102033,
20260605_105313, 20260605_222218).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from FRASCI.lasscf.fragments import strong_pair_fragments

# Path to the canonical SC MFA GS metadata JSON (relative to project root).
_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
_CANONICAL_JSON = (
    _PROJECT_ROOT
    / "FRASCI"
    / "Outputs"
    / "mfa"
    / "outs_gs_strongpair_20260609_152510"
    / "gs_metadata.json"
)


def test_strong_pair_partition_is_valid():
    orbital_lists, nelec_per_frag, spin_sub = strong_pair_fragments()

    # Complete cover of range(36) with no overlaps.
    flat = sorted(o for frag in orbital_lists for o in frag)
    assert flat == list(range(36)), (
        f"Orbital lists do not cover range(36): {flat!r}"
    )

    # Fragment sizes must be exactly 12/12/12.
    assert [len(f) for f in orbital_lists] == [12, 12, 12], (
        f"Expected sizes [12,12,12], got {[len(f) for f in orbital_lists]}"
    )

    # Total electron count must equal (27, 27).
    total_a = sum(na for na, nb in nelec_per_frag)
    total_b = sum(nb for na, nb in nelec_per_frag)
    assert (total_a, total_b) == (27, 27), (
        f"Total electrons: got ({total_a},{total_b}), expected (27,27)"
    )

    # Must have exactly 3 spin_sub entries.
    assert len(spin_sub) == 3, (
        f"Expected 3 spin_sub entries, got {len(spin_sub)}"
    )

    # spin_sub must be consistent with |na - nb| per fragment (min multiplicity).
    for i, (na, nb) in enumerate(nelec_per_frag):
        expected_smult = abs(na - nb) + 1
        assert spin_sub[i] >= expected_smult, (
            f"spin_sub[{i}]={spin_sub[i]} is below minimum allowed by "
            f"|na-nb|={abs(na-nb)} (need smult>={expected_smult})"
        )


def test_strong_pair_matches_existing_mfa_partition():
    """The orbital lists must match what SC MFA uses for strong_pair so results
    stay comparable.  Source of truth: gs_metadata.json from the most recent
    SC MFA GS run (outs_gs_strongpair_20260609_152510)."""
    if not _CANONICAL_JSON.exists():
        pytest.skip(
            f"Canonical SC MFA GS metadata JSON not found: {_CANONICAL_JSON}"
        )

    with open(_CANONICAL_JSON) as fh:
        canonical_meta = json.load(fh)

    canonical_orbs = canonical_meta["fragment_orbs"]
    orbital_lists, _, _ = strong_pair_fragments()

    assert len(orbital_lists) == len(canonical_orbs), (
        f"Fragment count mismatch: got {len(orbital_lists)}, "
        f"canonical has {len(canonical_orbs)}"
    )

    for i, (actual, expected) in enumerate(zip(orbital_lists, canonical_orbs)):
        assert sorted(actual) == sorted(expected), (
            f"F{i} orbital list mismatch.\n"
            f"  actual   : {sorted(actual)}\n"
            f"  canonical: {sorted(expected)}"
        )
