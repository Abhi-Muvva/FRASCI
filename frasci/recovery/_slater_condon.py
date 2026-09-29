"""Auditable real-valued Slater--Condon edges for spin-separated bitstrings.

Convention used throughout this repository:

* ``h1[p,q]`` is a real symmetric spatial-orbital one-electron integral.
* ``eri[p,q,r,s]`` is the real chemists' integral ``(pq|rs)``.  Thus the
  determinant diagonal uses Coulomb ``eri[p,p,q,q]`` and exchange
  ``eri[p,q,q,p]``.
* A determinant is ``(alpha_bits, beta_bits)`` in PySCF's direct-spin basis.
  Alpha and beta excitation parities factor for opposite-spin doubles.

The formulas below are independent of PySCF's selected-space implementation.  PySCF's
``FCIpspace_h0tril`` remains the oracle in tests.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

Determinant = tuple[int, int]


def iter_excitation_batches(det: Determinant, n_orb: int, *, batch_size: int = 4096):
    """Yield deterministic rank-<=2 candidate batches, separated by class."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    alpha, beta = map(int, det)
    occ_a = [p for p in range(n_orb) if (alpha >> p) & 1]
    occ_b = [p for p in range(n_orb) if (beta >> p) & 1]
    vir_a = [p for p in range(n_orb) if not (alpha >> p) & 1]
    vir_b = [p for p in range(n_orb) if not (beta >> p) & 1]

    def emit(class_name, iterator):
        batch = []
        for candidate in iterator:
            batch.append(candidate)
            if len(batch) == batch_size:
                yield class_name, tuple(batch)
                batch.clear()
        if batch:
            yield class_name, tuple(batch)

    def singles(occ, vir, spin):
        for occupied in occ:
            for virtual in vir:
                if spin == "alpha":
                    yield alpha ^ (1 << occupied) ^ (1 << virtual), beta
                else:
                    yield alpha, beta ^ (1 << occupied) ^ (1 << virtual)

    def same_spin_doubles(occ, vir, spin):
        for ii in range(len(occ)):
            for jj in range(ii + 1, len(occ)):
                for aa in range(len(vir)):
                    for bb in range(aa + 1, len(vir)):
                        bits = ((1 << occ[ii]) ^ (1 << occ[jj])
                                ^ (1 << vir[aa]) ^ (1 << vir[bb]))
                        yield (alpha ^ bits, beta) if spin == "alpha" else (alpha, beta ^ bits)

    def opposite_spin_doubles():
        for occupied_a in occ_a:
            for virtual_a in vir_a:
                for occupied_b in occ_b:
                    for virtual_b in vir_b:
                        yield (alpha ^ (1 << occupied_a) ^ (1 << virtual_a),
                               beta ^ (1 << occupied_b) ^ (1 << virtual_b))

    yield from emit("alpha_single", singles(occ_a, vir_a, "alpha"))
    yield from emit("alpha_double", same_spin_doubles(occ_a, vir_a, "alpha"))
    yield from emit("beta_single", singles(occ_b, vir_b, "beta"))
    yield from emit("beta_double", same_spin_doubles(occ_b, vir_b, "beta"))
    yield from emit("opposite_double", opposite_spin_doubles())


def _set_bit_indices(masks: np.ndarray, n_bits: int, n_orb: int) -> np.ndarray:
    if len(masks) == 0:
        return np.empty((0, n_bits), dtype=np.int64)
    shifts = np.arange(n_orb, dtype=np.uint64)
    occupied = ((masks[:, None] >> shifts[None, :]) & np.uint64(1)).astype(bool)
    rows, columns = np.nonzero(occupied)
    if len(columns) != len(masks) * n_bits:
        raise ValueError("edge classification found an unexpected excitation rank")
    if not np.array_equal(rows, np.repeat(np.arange(len(masks)), n_bits)):
        raise AssertionError("bit-index extraction lost row ordering")
    return columns.reshape(len(masks), n_bits)


def _single_parity(source_bits: int, removed: np.ndarray, added: np.ndarray) -> np.ndarray:
    low = np.minimum(removed, added).astype(np.uint64)
    high = np.maximum(removed, added).astype(np.uint64)
    one = np.uint64(1)
    between = ((one << high) - (one << (low + one)))
    odd = np.bitwise_count(np.uint64(source_bits) & between) & np.uint8(1)
    return 1.0 - 2.0 * odd.astype(float)


def _same_spin_double_parity(source_bits: int, removed: np.ndarray,
                             added: np.ndarray) -> np.ndarray:
    """Parity of ``a+_a a+_b a_j a_i`` for sorted ``i<j`` and ``a<b``."""
    bits = np.full(len(removed), np.uint64(source_bits), dtype=np.uint64)
    parity = np.zeros(len(removed), dtype=np.uint8)
    # Operators act right-to-left: destroy i, destroy j, create b, create a.
    for orbitals, create in ((removed[:, 0], False), (removed[:, 1], False),
                             (added[:, 1], True), (added[:, 0], True)):
        shifts = orbitals.astype(np.uint64) + np.uint64(1)
        parity ^= (np.bitwise_count(bits >> shifts) & np.uint8(1))
        masks = np.left_shift(np.uint64(1), orbitals.astype(np.uint64))
        bits = (bits | masks) if create else (bits ^ masks)
    return 1.0 - 2.0 * parity.astype(float)


class RealSlaterCondon:
    """Vectorized exact matrix elements from one parent to determinant batches."""

    def __init__(self, h1: np.ndarray, eri: np.ndarray, *, e_const: float = 0.0):
        self.h1 = np.ascontiguousarray(h1, dtype=float)
        self.eri = np.ascontiguousarray(eri, dtype=float)
        if self.h1.ndim != 2 or self.h1.shape[0] != self.h1.shape[1]:
            raise ValueError("h1 must be square")
        self.n_orb = int(self.h1.shape[0])
        if self.eri.shape != (self.n_orb,) * 4:
            raise ValueError("eri must have shape (n_orb,)*4")
        if self.n_orb >= 64:
            raise NotImplementedError("bitstring edge kernel currently requires n_orb < 64")
        if not np.allclose(self.h1, self.h1.T, atol=1e-12):
            raise ValueError("real Slater--Condon kernel requires symmetric h1")
        self.e_const = float(e_const)
        self.max_batch_shape = (0, self.n_orb)

    def _diagonal(self, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        shifts = np.arange(self.n_orb, dtype=np.uint64)
        occ_a = ((alpha[:, None] >> shifts) & np.uint64(1)).astype(float)
        occ_b = ((beta[:, None] >> shifts) & np.uint64(1)).astype(float)
        density = occ_a + occ_b
        eri_j = np.einsum("ppqq->pq", self.eri)
        eri_k = np.einsum("pqqp->pq", self.eri)
        one = density @ np.diag(self.h1)
        coulomb = 0.5 * np.einsum("np,pq,nq->n", density, eri_j, density)
        exchange = -0.5 * (
            np.einsum("np,pq,nq->n", occ_a, eri_k, occ_a)
            + np.einsum("np,pq,nq->n", occ_b, eri_k, occ_b))
        return one + coulomb + exchange + self.e_const

    def elements(self, parent: Determinant, candidates: Sequence[Determinant], *,
                 return_classes: bool = False):
        """Return ``<candidate|H|parent>`` for a bounded candidate batch."""
        if not candidates:
            empty = np.empty(0, dtype=float)
            return (empty, np.empty(0, dtype=object)) if return_classes else empty
        parent_a, parent_b = map(int, parent)
        candidate_a = np.fromiter((int(det[0]) for det in candidates), dtype=np.uint64)
        candidate_b = np.fromiter((int(det[1]) for det in candidates), dtype=np.uint64)
        self.max_batch_shape = max(self.max_batch_shape, (len(candidates), self.n_orb))

        remove_a = np.uint64(parent_a) & ~candidate_a
        add_a = candidate_a & ~np.uint64(parent_a)
        remove_b = np.uint64(parent_b) & ~candidate_b
        add_b = candidate_b & ~np.uint64(parent_b)
        rank_a = np.bitwise_count(remove_a)
        rank_b = np.bitwise_count(remove_b)
        balanced = ((rank_a == np.bitwise_count(add_a))
                    & (rank_b == np.bitwise_count(add_b)))

        values = np.zeros(len(candidates), dtype=float)
        classes = np.full(len(candidates), "rank_gt2", dtype=object)
        diagonal = balanced & (rank_a == 0) & (rank_b == 0)
        if diagonal.any():
            values[diagonal] = self._diagonal(candidate_a[diagonal], candidate_b[diagonal])
            classes[diagonal] = "diagonal"

        alpha_single = balanced & (rank_a == 1) & (rank_b == 0)
        beta_single = balanced & (rank_a == 0) & (rank_b == 1)
        for mask, spin in ((alpha_single, "alpha"), (beta_single, "beta")):
            if not mask.any():
                continue
            rows = np.flatnonzero(mask)
            removed_mask = remove_a[mask] if spin == "alpha" else remove_b[mask]
            added_mask = add_a[mask] if spin == "alpha" else add_b[mask]
            removed = _set_bit_indices(removed_mask, 1, self.n_orb)[:, 0]
            added = _set_bit_indices(added_mask, 1, self.n_orb)[:, 0]
            same_bits = parent_a if spin == "alpha" else parent_b
            other_bits = parent_b if spin == "alpha" else parent_a
            same_occ = np.asarray([p for p in range(self.n_orb)
                                   if (same_bits >> p) & 1], dtype=np.int64)
            other_occ = np.asarray([p for p in range(self.n_orb)
                                    if (other_bits >> p) & 1], dtype=np.int64)
            value = self.h1[added, removed].copy()
            if len(same_occ):
                value += np.sum(
                    self.eri[added[:, None], removed[:, None], same_occ, same_occ]
                    - self.eri[added[:, None], same_occ, same_occ, removed[:, None]], axis=1)
            if len(other_occ):
                value += np.sum(
                    self.eri[added[:, None], removed[:, None], other_occ, other_occ], axis=1)
            values[rows] = _single_parity(same_bits, removed, added) * value
            classes[rows] = f"{spin}_single"

        alpha_double = balanced & (rank_a == 2) & (rank_b == 0)
        beta_double = balanced & (rank_a == 0) & (rank_b == 2)
        for mask, spin in ((alpha_double, "alpha"), (beta_double, "beta")):
            if not mask.any():
                continue
            rows = np.flatnonzero(mask)
            removed_mask = remove_a[mask] if spin == "alpha" else remove_b[mask]
            added_mask = add_a[mask] if spin == "alpha" else add_b[mask]
            removed = _set_bit_indices(removed_mask, 2, self.n_orb)
            added = _set_bit_indices(added_mask, 2, self.n_orb)
            i, j = removed[:, 0], removed[:, 1]
            a, b = added[:, 0], added[:, 1]
            antisym = self.eri[a, i, b, j] - self.eri[a, j, b, i]
            source_bits = parent_a if spin == "alpha" else parent_b
            values[rows] = _same_spin_double_parity(source_bits, removed, added) * antisym
            classes[rows] = f"{spin}_double"

        opposite = balanced & (rank_a == 1) & (rank_b == 1)
        if opposite.any():
            rows = np.flatnonzero(opposite)
            i = _set_bit_indices(remove_a[opposite], 1, self.n_orb)[:, 0]
            a = _set_bit_indices(add_a[opposite], 1, self.n_orb)[:, 0]
            j = _set_bit_indices(remove_b[opposite], 1, self.n_orb)[:, 0]
            b = _set_bit_indices(add_b[opposite], 1, self.n_orb)[:, 0]
            sign = (_single_parity(parent_a, i, a)
                    * _single_parity(parent_b, j, b))
            values[rows] = sign * self.eri[a, i, b, j]
            classes[rows] = "opposite_double"

        return (values, classes) if return_classes else values

    def element(self, left: Determinant, right: Determinant) -> float:
        return float(self.elements(right, [left])[0])
