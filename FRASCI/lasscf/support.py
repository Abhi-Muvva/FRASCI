"""Small shared utilities required by the LASSCF implementation."""

from __future__ import annotations

import warnings

import numpy as np


def fragment_electron_count(
    ref_alpha_bits: int,
    ref_beta_bits: int,
    fragment_orbs: list[int],
) -> tuple[int, int]:
    """Count alpha and beta reference electrons in an orbital fragment."""
    n_alpha = sum(1 for orb in fragment_orbs if (ref_alpha_bits >> orb) & 1)
    n_beta = sum(1 for orb in fragment_orbs if (ref_beta_bits >> orb) & 1)
    return n_alpha, n_beta


def load_ref_det(ref_dets_path: str, row: int = 0) -> tuple[int, int]:
    """Load alpha/beta determinant bitstrings from ``data/dets.npz``."""
    with np.load(ref_dets_path) as data:
        dets = data["dets"]
        return int(dets[row, 0]), int(dets[row, 1])


def make_nonoverlapping_partition(h1: np.ndarray, n_orb: int) -> list[list[int]]:
    """Build three equal fragments ordered by the diagonal of ``h1``."""
    if n_orb % 3:
        raise ValueError(f"n_orb={n_orb} is not divisible by 3")
    order = np.argsort(np.diag(h1), kind="stable")
    size = n_orb // 3
    return [
        sorted(order[0:size].tolist()),
        sorted(order[size : 2 * size].tolist()),
        sorted(order[2 * size : n_orb].tolist()),
    ]


def validate_fragment_partition(
    fragments: list[list[int]],
    n_orb: int,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> None:
    """Validate complete, non-overlapping fragment coverage and occupations."""
    if not fragments:
        raise ValueError("Partition must contain at least one fragment")

    seen: set[int] = set()
    for frag_idx, fragment in enumerate(fragments):
        if not fragment:
            raise ValueError(f"Fragment {frag_idx} is empty")
        for orb in fragment:
            if not 0 <= orb < n_orb:
                raise ValueError(
                    f"Fragment {frag_idx}: orbital {orb} out of range [0, {n_orb})"
                )
            if orb in seen:
                raise ValueError(
                    f"Fragment {frag_idx}: orbital {orb} appears more than once"
                )
            seen.add(orb)

    missing = sorted(set(range(n_orb)) - seen)
    if missing:
        raise ValueError(f"Partition missing orbitals: {missing[:10]}")

    for frag_idx, fragment in enumerate(fragments):
        n_alpha, n_beta = fragment_electron_count(
            ref_alpha_bits, ref_beta_bits, fragment
        )
        if n_alpha > len(fragment) or n_beta > len(fragment):
            raise ValueError(
                f"Fragment {frag_idx}: electron count exceeds orbital count"
            )
        has_alpha_virtual = any(
            not ((ref_alpha_bits >> orb) & 1) for orb in fragment
        )
        has_beta_virtual = any(
            not ((ref_beta_bits >> orb) & 1) for orb in fragment
        )
        if not has_alpha_virtual and not has_beta_virtual:
            warnings.warn(
                f"Fragment {frag_idx} has no virtual orbitals",
                stacklevel=2,
            )


def orbital_coupling_matrix(h1: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Build the integral-based orbital coupling graph used by fragment sweeps."""
    weights = np.abs(h1)
    weights = weights + np.abs(np.einsum("ppqq->pq", eri))
    weights = weights + np.abs(np.einsum("pqqp->pq", eri))
    weights = 0.5 * (weights + weights.T)
    weights[np.diag_indices(h1.shape[0])] = 0.0
    return weights
