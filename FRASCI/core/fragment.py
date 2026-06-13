"""
fragment.py
===========
Orbital fragmentation engine for FRASCI.

Provides three approaches:
  1. Sliding window  — simple, systematic, tunable W/S hyperparameters
  2. MI-based        — spectral clustering on orbital mutual information matrix
  3. Integral slicing — extract h1 and eri for a given orbital subset

All integral slicing uses CHEMIST notation: eri[p,q,r,s] = (pq|rs).
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Sliding-window fragmentation
# ---------------------------------------------------------------------------

def fragment_by_sliding_window(
    n_orb: int,
    orbital_order: np.ndarray,
    window_size: int,
    stride: int,
) -> list[list[int]]:
    """
    Create overlapping orbital fragments using a sliding window.

    Parameters
    ----------
    n_orb        : total number of orbitals (36 for Fe4S4)
    orbital_order: 1D int array — orbital indices sorted by desired criterion
                   (e.g. h1 diagonal energy, mutual information rank).
                   Must contain exactly n_orb distinct values in [0, n_orb).
    window_size  : number of orbitals per fragment (e.g. 15)
    stride       : step between consecutive window starts (e.g. 10)

    Returns
    -------
    fragments : list of lists, each containing the *original* orbital indices
                for one fragment, in ascending order.

    Notes
    -----
    - Last window is extended to the end of orbital_order so no orbitals
      are dropped.
    - With n_orb=36, window_size=15, stride=10:
        fragment 0 → orbital_order[0:15]
        fragment 1 → orbital_order[10:25]
        fragment 2 → orbital_order[20:36]   ← extended past 35
    """
    fragments = []
    start = 0
    while True:
        if (n_orb - start) <= window_size:  # last window — take everything remaining
            fragments.append(sorted(orbital_order[start:].tolist()))
            break
        else:
            # Check whether advancing by stride would leave a full window;
            # if not, this is the last window and we take everything remaining.
            if n_orb - (start + stride) < window_size:
                fragments.append(sorted(orbital_order[start:].tolist()))
                break
            fragments.append(sorted(orbital_order[start : start + window_size].tolist()))
            start += stride
    return fragments


# ---------------------------------------------------------------------------
# MI-based fragmentation
# ---------------------------------------------------------------------------

def fragment_by_mutual_information(
    n_orb: int,
    mi_matrix: np.ndarray,
    n_fragments: int,
    min_overlap: int = 2,
) -> list[list[int]]:
    """
    Create fragments by clustering orbitals with high mutual information.

    Uses spectral clustering on the MI matrix (scikit-learn required).

    Parameters
    ----------
    n_orb       : total number of orbitals
    mi_matrix   : (n_orb, n_orb) symmetric matrix — I(i,j) = orbital MI
    n_fragments : number of clusters / fragments
    min_overlap : minimum number of high-MI orbitals shared between adjacent
                  clusters to ensure coupling regions exist

    Returns
    -------
    fragments : list of lists of orbital indices (may overlap)
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Integral slicing
# ---------------------------------------------------------------------------

def extract_fragment_integrals(
    h1_full: np.ndarray,
    eri_full: np.ndarray,
    fragment_orbs: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice h1 and eri (chemist notation) to a fragment's orbital subset.

    Parameters
    ----------
    h1_full      : (n_orb, n_orb) one-body integral array
    eri_full     : (n_orb, n_orb, n_orb, n_orb) two-body array in
                   CHEMIST notation: eri[p,q,r,s] = (pq|rs)
    fragment_orbs: list of int — orbital indices of this fragment,
                   length n_frag

    Returns
    -------
    h1_frag  : (n_frag, n_frag) one-body integrals for the fragment
    eri_frag : (n_frag, n_frag, n_frag, n_frag) two-body integrals,
               still in chemist notation

    Notes
    -----
    h1_frag[i,j]       = h1_full[frag[i], frag[j]]
    eri_frag[i,j,k,l]  = eri_full[frag[i], frag[j], frag[k], frag[l]]
    """
    idx = np.array(fragment_orbs)
    h1_frag  = np.ascontiguousarray(h1_full[np.ix_(idx, idx)])
    eri_frag = np.ascontiguousarray(eri_full[np.ix_(idx, idx, idx, idx)])
    return h1_frag, eri_frag


def fragment_electron_count(
    ref_alpha_bits: int,
    ref_beta_bits: int,
    fragment_orbs: list[int],
) -> tuple[int, int]:
    """
    Count alpha and beta electrons from the reference determinant that fall
    in this fragment's orbital set.

    Parameters
    ----------
    ref_alpha_bits : integer bitstring — bit k set iff spatial orbital k is
                     alpha-occupied in the reference determinant
    ref_beta_bits  : integer bitstring — same for beta
    fragment_orbs  : list of orbital indices in this fragment

    Returns
    -------
    (n_alpha_frag, n_beta_frag) : electron counts for the fragment

    Notes
    -----
    For Fe4S4 reference determinant (from dets.npz row 0):
      alpha occupies: {0-20, 22-23, 30, 33-35}
      beta  occupies: {0-1, 4-5, 10, 12, 14-27, 29-35}
    These sets differ — the system is open-shell.
    """
    n_alpha = sum(1 for o in fragment_orbs if (ref_alpha_bits >> o) & 1)
    n_beta  = sum(1 for o in fragment_orbs if (ref_beta_bits  >> o) & 1)
    return n_alpha, n_beta


if __name__ == "__main__":
    import numpy as np

    # Test 1: sliding window produces exactly 3 fragments
    frags = fragment_by_sliding_window(36, np.arange(36), 15, 10)
    assert len(frags) == 3, f"got {len(frags)} fragments"
    assert frags[0] == list(range(0, 15))
    assert frags[1] == list(range(10, 25))
    assert frags[2] == list(range(20, 36))
    print("PASS: sliding window 36/15/10")

    # Test 2: ERI slicing correctness
    rng = np.random.default_rng(42)
    n = 20
    eri_big = rng.random((n, n, n, n))
    h1_big  = rng.random((n, n))
    frag = [2, 5, 11]
    h1_f, eri_f = extract_fragment_integrals(h1_big, eri_big, frag)
    assert h1_f.shape  == (3, 3)
    assert eri_f.shape == (3, 3, 3, 3)
    assert eri_f[0, 1, 2, 0] == eri_big[2, 5, 11, 2], "ERI slicing mismatch"
    assert h1_f[1, 2]         == h1_big[5, 11],         "h1 slicing mismatch"
    print("PASS: integral slicing")

    # Test 3: electron counts for fragment [0..14] using Fe4S4 reference
    # Alpha occ: 0-20, 22-23, 30, 33-35
    # Beta  occ: 0-1, 4-5, 10, 12, 14-27, 29-35
    alpha_occ = set(range(21)) | {22, 23, 30, 33, 34, 35}
    beta_occ  = {0, 1, 4, 5, 10, 12} | set(range(14, 28)) | set(range(29, 36))
    alpha_bits = sum(1 << o for o in alpha_occ)
    beta_bits  = sum(1 << o for o in beta_occ)
    frag0 = list(range(15))   # orbitals 0..14
    na, nb = fragment_electron_count(alpha_bits, beta_bits, frag0)
    assert na == 15, f"n_alpha for frag[0..14]: expected 15, got {na}"
    assert nb == 7,  f"n_beta  for frag[0..14]: expected 7,  got {nb}"
    print("PASS: electron counts")

    print("=== All fragment.py tests pass ===")
