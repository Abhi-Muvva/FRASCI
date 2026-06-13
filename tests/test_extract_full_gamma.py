"""
tests/test_extract_full_gamma.py
=================================
Unit tests for mfa/extract_full_gamma.py — assemble_global_gamma_full.

All tests are pure-numpy (no TrimCI, no FCIDUMP) and run in < 1 s.
"""
import numpy as np
import pytest

from FRASCI.mfa.extract_full_gamma import assemble_global_gamma_full


# ---------------------------------------------------------------------------
# Test 21: output shape
# ---------------------------------------------------------------------------

def test_assemble_returns_correct_shape():
    """Output is always (n_orb, n_orb)."""
    rdm1 = np.array([[1.5, 0.1], [0.1, 0.5]])
    gamma = assemble_global_gamma_full([rdm1], [[0, 1]], n_orb=4)
    assert gamma.shape == (4, 4)


# ---------------------------------------------------------------------------
# Test 22: diagonal from single fragment equals fragment RDM diagonal
# ---------------------------------------------------------------------------

def test_assemble_diagonal_matches_fragment_rdm():
    """Covered diagonal entries equal the fragment 1-RDM diagonal."""
    rng = np.random.default_rng(42)
    raw = rng.random((3, 3))
    rdm1 = (raw + raw.T) / 2  # symmetric
    gamma = assemble_global_gamma_full([rdm1], [[1, 2, 4]], n_orb=6)
    assert np.allclose(np.diag(gamma)[[1, 2, 4]], np.diag(rdm1), atol=1e-14)
    # Non-fragment diagonal entries stay zero
    assert gamma[0, 0] == 0.0
    assert gamma[3, 3] == 0.0
    assert gamma[5, 5] == 0.0


# ---------------------------------------------------------------------------
# Test 23: assembled matrix is symmetric
# ---------------------------------------------------------------------------

def test_assemble_symmetric():
    """Assembled matrix is always symmetric."""
    rdm1_a = np.eye(3) * 1.5
    rdm1_a[0, 1] = rdm1_a[1, 0] = 0.3
    rdm1_b = np.eye(3) * 0.5
    rdm1_b[1, 2] = rdm1_b[2, 1] = 0.1
    gamma = assemble_global_gamma_full([rdm1_a, rdm1_b], [[0, 1, 2], [2, 3, 4]], n_orb=5)
    assert np.allclose(gamma, gamma.T, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 24: uncovered cross-fragment pairs remain zero
# ---------------------------------------------------------------------------

def test_assemble_uncovered_pairs_are_zero():
    """Pairs (r, s) sharing no fragment block → gamma[r, s] == 0."""
    rdm1_a = np.eye(2) * 1.8
    rdm1_b = np.eye(2) * 0.2
    # non-overlapping: frag0=[0,1], frag1=[2,3]
    gamma = assemble_global_gamma_full([rdm1_a, rdm1_b], [[0, 1], [2, 3]], n_orb=4)
    assert gamma[0, 2] == 0.0
    assert gamma[0, 3] == 0.0
    assert gamma[1, 2] == 0.0
    assert gamma[1, 3] == 0.0
    # Symmetric complement
    assert gamma[2, 0] == 0.0
    assert gamma[3, 1] == 0.0


# ---------------------------------------------------------------------------
# Test 25: overlapping orbital is averaged, not double-counted
# ---------------------------------------------------------------------------

def test_assemble_overlap_averages_not_sums():
    """
    Orbital shared by two fragments → diagonal entry is the average of the two
    fragment RDM diagonal values, not their sum.
    """
    # frag0=[0,1]: orbital 1 → rdm1_a[1,1] = 0.8
    # frag1=[1,2]: orbital 1 → rdm1_b[0,0] = 0.6
    rdm1_a = np.array([[1.0, 0.0], [0.0, 0.8]])
    rdm1_b = np.array([[0.6, 0.0], [0.0, 0.4]])
    gamma = assemble_global_gamma_full([rdm1_a, rdm1_b], [[0, 1], [1, 2]], n_orb=3)
    assert np.isclose(gamma[1, 1], (0.8 + 0.6) / 2, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 26: off-diagonal from single fragment equals fragment RDM off-diagonal
# ---------------------------------------------------------------------------

def test_assemble_offdiag_from_single_fragment():
    """
    Off-diagonal entry (r, s) covered by exactly one fragment equals
    the corresponding fragment RDM element.
    """
    rdm1 = np.array([[1.5, 0.3], [0.3, 0.5]])
    gamma = assemble_global_gamma_full([rdm1], [[2, 5]], n_orb=6)
    assert np.isclose(gamma[2, 5], 0.3, atol=1e-14)
    assert np.isclose(gamma[5, 2], 0.3, atol=1e-14)


# ---------------------------------------------------------------------------
# Test 27: single fragment covering all orbitals reproduces the full RDM
# ---------------------------------------------------------------------------

def test_assemble_full_coverage_reproduces_rdm():
    """When one fragment covers all orbitals, gamma_full == rdm1."""
    n = 5
    rng = np.random.default_rng(99)
    raw = rng.random((n, n))
    rdm1 = (raw + raw.T) / 2
    gamma = assemble_global_gamma_full([rdm1], [list(range(n))], n_orb=n)
    assert np.allclose(gamma, rdm1, atol=1e-14)
