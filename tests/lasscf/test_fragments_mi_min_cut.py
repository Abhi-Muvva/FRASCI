"""
test_fragments_mi_min_cut.py
============================
Tests for mi_min_cut_fragments() in FRASCI/lasscf/fragments.py.

The canonical source of truth for the mi_min_cut 6/12/18 orbital lists is:
    FRASCI/Outputs/mfa/
        outs_gs_d2_mi_min_cut_6_12_18_20260607_110509/
        fragments_mi_min_cut_6_12_18.json

which was written by GS_D2_MI_MinCut.ipynb and is identical to the lists
in both SC MFA canonical runs (timestamps 110509 and 110727).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from FRASCI.lasscf.fragments import mi_min_cut_fragments

# Path to the canonical SC MFA fragment JSON (relative to project root).
_PROJECT_ROOT = pathlib.Path(__file__).parents[3]
_CANONICAL_JSON = (
    _PROJECT_ROOT
    / "FRASCI"
    / "Outputs"
    / "mfa"
    / "outs_gs_d2_mi_min_cut_6_12_18_20260607_110509"
    / "fragments_mi_min_cut_6_12_18.json"
)


def test_mi_min_cut_partition_is_valid():
    orbital_lists, nelec_per_frag, spin_sub = mi_min_cut_fragments()

    # Complete cover of range(36) with no overlaps.
    flat = sorted(o for frag in orbital_lists for o in frag)
    assert flat == list(range(36)), (
        f"Orbital lists do not cover range(36): {flat!r}"
    )

    # Fragment sizes must be exactly 6/12/18.
    assert [len(f) for f in orbital_lists] == [6, 12, 18], (
        f"Expected sizes [6,12,18], got {[len(f) for f in orbital_lists]}"
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


def test_mi_min_cut_matches_existing_mfa_partition():
    """The orbital lists must match the SC MFA canonical JSON so that
    LASSCF+TrimCI and SC MFA results are directly comparable."""
    if not _CANONICAL_JSON.exists():
        pytest.skip(
            f"Canonical SC MFA fragment JSON not found: {_CANONICAL_JSON}"
        )

    with open(_CANONICAL_JSON) as fh:
        canonical = json.load(fh)

    orbital_lists, _, _ = mi_min_cut_fragments()

    assert len(orbital_lists) == len(canonical), (
        f"Fragment count mismatch: got {len(orbital_lists)}, "
        f"canonical has {len(canonical)}"
    )

    for i, (actual, expected) in enumerate(zip(orbital_lists, canonical)):
        assert sorted(actual) == sorted(expected), (
            f"F{i} orbital list mismatch.\n"
            f"  actual  : {sorted(actual)}\n"
            f"  canonical: {sorted(expected)}"
        )
