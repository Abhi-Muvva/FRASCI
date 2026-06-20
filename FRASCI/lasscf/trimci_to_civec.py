"""
trimci_to_civec.py
==================
Decode TrimCI's sparse (dets, coeffs) selected-CI representation into a
dense PySCF FCI vector in PySCF cistring address order.

TrimCI Determinant Representation (confirmed by reading TrimCI source)
----------------------------------------------------------------------
For systems with norb <= 64 (our case: 12-orb fragments):
    det.alpha : Python int  — alpha-spin bitstring, bit i set <=> orbital i occupied
    det.beta  : Python int  — beta-spin bitstring,  bit i set <=> orbital i occupied

Orbital indexing is 0-based from the LSB.  For example, with 4 orbitals and
2 alpha electrons in orbitals 0 and 2:
    det.alpha = 0b0101 = 5

For norb > 64 (not used here):
    det.alpha : list of uint64  — each element covers 64 orbitals
    det.beta  : list of uint64

Source evidence:
    attentive_trimci.py:52  — dets_to_array: int(d.alpha), int(d.beta)
    TrimCI_runner/det_utils.py:56-75 — det_to_bitstring: loops i in range(64),
        checks alpha_bits & (1 << i) to recover orbital indices
    auto_selector.py:97-119 — norb <= 64 uses trimci_core.Determinant (single int),
        norb 65-128 uses Determinant128 (list of 2 uint64s)

PySCF cistring Convention
-------------------------
cistring.str2addr(norb, nelec, string) maps an integer bitstring (same bit
convention as TrimCI) to a row/column index in the dense FCI vector.
civec[addr_a, addr_b] = coefficient of the determinant |alpha_str, beta_str>.

The sign convention of TrimCI's eigenvector matches PySCF's, provided orbitals
are numbered identically (0-indexed).  The test test_matches_full_fci verifies
this: the absolute overlap |<civec_trimci | civec_pyscf>| must be 1.0 to 1e-6.
"""

from __future__ import annotations

import numpy as np
from pyscf.fci import cistring


def orbital_occ_to_bitmask(occ_list: list[int], norb: int) -> int:
    """
    Convert a list of occupied orbital indices (0-indexed) to an integer bitmask.

    Parameters
    ----------
    occ_list : list of int
        Occupied orbital indices, e.g. [0, 2] for orbitals 0 and 2.
    norb : int
        Total number of orbitals (used only for a bounds check).

    Returns
    -------
    bitmask : int
        Integer where bit i is set iff orbital i is in occ_list.
    """
    bitmask = 0
    for i in occ_list:
        if not (0 <= i < norb):
            raise ValueError(f"Orbital index {i} out of range [0, {norb})")
        bitmask |= (1 << i)
    return bitmask


def _extract_alpha_beta_bits(det, norb: int):
    """
    Extract (alpha_bits, beta_bits) as plain Python ints from a TrimCI determinant.

    Handles both the scalar (norb <= 64) and list-of-uint64 (norb > 64) cases.

    Returns
    -------
    alpha_bits, beta_bits : int, int
        Integer bitmasks with bit i set iff orbital i is occupied.
    """
    alpha_raw = det.alpha
    beta_raw = det.beta

    if isinstance(alpha_raw, (list, tuple)):
        # norb > 64: pack list of uint64 into a single Python int
        alpha_bits = 0
        beta_bits = 0
        for chunk_idx, (a_chunk, b_chunk) in enumerate(zip(alpha_raw, beta_raw)):
            shift = chunk_idx * 64
            alpha_bits |= int(a_chunk) << shift
            beta_bits |= int(b_chunk) << shift
    else:
        alpha_bits = int(alpha_raw)
        beta_bits = int(beta_raw)

    return alpha_bits, beta_bits


def trimci_to_pyscf_civec(
    dets,
    coeffs,
    norb: int,
    nelec: tuple,
) -> np.ndarray:
    """
    Pack TrimCI's sparse selected-CI representation into a dense PySCF FCI vector.

    Parameters
    ----------
    dets : list
        TrimCI Determinant objects, each with .alpha and .beta integer bitmasks
        (bit i set <=> orbital i occupied, 0-indexed from LSB).
    coeffs : list or array-like
        CI amplitudes parallel to dets.
    norb : int
        Number of spatial orbitals in the fragment.
    nelec : tuple[int, int]
        (n_alpha, n_beta) electron counts.

    Returns
    -------
    civec : ndarray, shape (na, nb), float64
        Dense FCI vector in PySCF cistring address order.
        civec[addr_a, addr_b] = CI coefficient of the corresponding determinant.
        Norm is preserved: np.sum(civec**2) == np.sum(np.array(coeffs)**2).
    """
    n_alpha, n_beta = int(nelec[0]), int(nelec[1])

    na = cistring.num_strings(norb, n_alpha)
    nb = cistring.num_strings(norb, n_beta)
    civec = np.zeros((na, nb), dtype=np.float64)

    coeffs_arr = np.asarray(coeffs, dtype=np.float64)

    for det, c in zip(dets, coeffs_arr):
        alpha_bits, beta_bits = _extract_alpha_beta_bits(det, norb)
        addr_a = cistring.str2addr(norb, n_alpha, alpha_bits)
        addr_b = cistring.str2addr(norb, n_beta, beta_bits)
        civec[addr_a, addr_b] = float(c)

    return civec


def pyscf_civec_to_trimci_dets(civec: np.ndarray, norb: int, nelec: tuple):
    """
    Round-trip helper: unpack a dense PySCF FCI vector into (alpha_bits_list,
    beta_bits_list, coeffs) for sanity checks in tests.

    Returns only entries with |c| > 0 (up to floating-point zero).

    Returns
    -------
    alpha_bits : list of int
    beta_bits  : list of int
    coeffs     : list of float
    """
    from pyscf.fci import cistring as cs

    n_alpha, n_beta = int(nelec[0]), int(nelec[1])
    alpha_strings = cs.make_strings(range(norb), n_alpha)
    beta_strings = cs.make_strings(range(norb), n_beta)

    alpha_bits_out = []
    beta_bits_out = []
    coeffs_out = []

    na, nb = civec.shape
    for ia in range(na):
        for ib in range(nb):
            c = civec[ia, ib]
            if c != 0.0:
                alpha_bits_out.append(int(alpha_strings[ia]))
                beta_bits_out.append(int(beta_strings[ib]))
                coeffs_out.append(float(c))

    return alpha_bits_out, beta_bits_out, coeffs_out
