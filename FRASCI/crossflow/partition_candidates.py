from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

from FRASCI.crossflow.partition import validate_fragment_partition


@dataclass(frozen=True)
class PartitionCandidate:
    name: str
    description: str
    fragments: list[list[int]]


def _split_order(order: list[int], n_fragments: int) -> list[list[int]]:
    base = len(order) // n_fragments
    remainder = len(order) % n_fragments
    fragments: list[list[int]] = []
    start = 0
    for frag_idx in range(n_fragments):
        size = base + (1 if frag_idx < remainder else 0)
        fragments.append(sorted(order[start : start + size]))
        start += size
    return fragments


def make_index_contiguous_partition(n_orb: int, n_fragments: int = 3) -> list[list[int]]:
    """Keep neighboring orbital indices together as a localization proxy."""
    return _split_order(list(range(n_orb)), n_fragments)


def _occupancy_class(ref_alpha_bits: int, ref_beta_bits: int, orb: int) -> str:
    alpha_occ = bool((ref_alpha_bits >> orb) & 1)
    beta_occ = bool((ref_beta_bits >> orb) & 1)
    if alpha_occ and beta_occ:
        return "docc"
    if alpha_occ:
        return "socc_alpha"
    if beta_occ:
        return "socc_beta"
    return "virt"


def _balance_by_occupancy(
    fragments: list[list[int]],
    h1_diag: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> list[list[int]]:
    """Light post-pass that avoids trivially closed fragments when swaps are easy."""
    target_size = len(fragments[0])
    if any(len(frag) != target_size for frag in fragments):
        return [sorted(frag) for frag in fragments]

    by_frag = [set(frag) for frag in fragments]

    def has_alpha_hole(frag: set[int]) -> bool:
        return any(not ((ref_alpha_bits >> orb) & 1) for orb in frag)

    def has_beta_hole(frag: set[int]) -> bool:
        return any(not ((ref_beta_bits >> orb) & 1) for orb in frag)

    def score_swap(closed_frag: set[int], donor_frag: set[int], out_orb: int, in_orb: int) -> float:
        return abs(float(h1_diag[out_orb] - h1_diag[in_orb]))

    for frag_idx, frag in enumerate(by_frag):
        needs_alpha = not has_alpha_hole(frag)
        needs_beta = not has_beta_hole(frag)
        if not (needs_alpha or needs_beta):
            continue

        best: tuple[float, int, int, int] | None = None
        for donor_idx, donor in enumerate(by_frag):
            if donor_idx == frag_idx:
                continue
            for in_orb in donor:
                supplies_alpha = needs_alpha and not ((ref_alpha_bits >> in_orb) & 1)
                supplies_beta = needs_beta and not ((ref_beta_bits >> in_orb) & 1)
                if not (supplies_alpha or supplies_beta):
                    continue
                for out_orb in frag:
                    trial_frag = (frag - {out_orb}) | {in_orb}
                    trial_donor = (donor - {in_orb}) | {out_orb}
                    if not has_alpha_hole(trial_donor) or not has_beta_hole(trial_donor):
                        continue
                    cost = score_swap(frag, donor, out_orb, in_orb)
                    if best is None or cost < best[0]:
                        best = (cost, donor_idx, out_orb, in_orb)
        if best is not None:
            _cost, donor_idx, out_orb, in_orb = best
            by_frag[frag_idx].remove(out_orb)
            by_frag[frag_idx].add(in_orb)
            by_frag[donor_idx].remove(in_orb)
            by_frag[donor_idx].add(out_orb)

    return [sorted(frag) for frag in by_frag]


def orbital_coupling_matrix(h1: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """
    Build a symmetric orbital-coupling graph from one- and two-electron integrals.

    This is not an atom assignment. It is a chemistry-motivated proxy: orbitals
    with large direct h1 mixing and large Coulomb/exchange couplings are kept close.
    """
    n_orb = h1.shape[0]
    weights = np.abs(h1)
    coulomb = np.abs(np.einsum("ppqq->pq", eri))
    exchange = np.abs(np.einsum("pqqp->pq", eri))
    weights = weights + coulomb + exchange
    weights = 0.5 * (weights + weights.T)
    weights[np.diag_indices(n_orb)] = 0.0
    return weights


def make_integral_graph_partition(
    h1: np.ndarray,
    eri: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
    n_fragments: int = 3,
) -> list[list[int]]:
    """
    Balanced greedy graph partition from integral couplings.

    Seeds are chosen by high graph strength with separation pressure. Remaining
    orbitals attach to the fragment with strongest accumulated coupling, while
    respecting equal fragment sizes.
    """
    n_orb = h1.shape[0]
    if n_orb % n_fragments != 0:
        raise ValueError("integral graph partition currently expects equal fragment sizes")

    weights = orbital_coupling_matrix(h1, eri)
    strength = weights.sum(axis=1)
    target_size = n_orb // n_fragments

    seeds = [int(np.argmax(strength))]
    while len(seeds) < n_fragments:
        best_orb = None
        best_score = None
        for orb in range(n_orb):
            if orb in seeds:
                continue
            separation = min(weights[orb, seed] for seed in seeds)
            score = float(strength[orb] - separation)
            if best_score is None or score > best_score:
                best_score = score
                best_orb = orb
        seeds.append(int(best_orb))

    fragments = [{seed} for seed in seeds]
    unassigned = set(range(n_orb)) - set(seeds)
    while unassigned:
        best: tuple[float, int, int] | None = None
        for orb in unassigned:
            for frag_idx, frag in enumerate(fragments):
                if len(frag) >= target_size:
                    continue
                coupling = float(sum(weights[orb, member] for member in frag))
                if best is None or coupling > best[0]:
                    best = (coupling, orb, frag_idx)
        if best is None:
            raise RuntimeError("No valid assignment found for integral graph partition")
        _coupling, orb, frag_idx = best
        fragments[frag_idx].add(orb)
        unassigned.remove(orb)

    return _balance_by_occupancy(
        [sorted(frag) for frag in fragments],
        np.diag(h1),
        ref_alpha_bits,
        ref_beta_bits,
    )


def make_strong_pair_partition(
    h1: np.ndarray,
    eri: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
    n_fragments: int = 3,
) -> list[list[int]]:
    """
    Keep strongest pair couplings together, then balance fragment sizes.

    This gives a second, intentionally different chemistry proxy from the
    integral graph partition.
    """
    n_orb = h1.shape[0]
    if n_orb % n_fragments != 0:
        raise ValueError("strong pair partition currently expects equal fragment sizes")

    weights = orbital_coupling_matrix(h1, eri)
    target_size = n_orb // n_fragments
    fragments = [set() for _ in range(n_fragments)]
    assigned: set[int] = set()

    pair_order: list[tuple[float, int, int]] = []
    for p in range(n_orb):
        for q in range(p + 1, n_orb):
            pair_order.append((float(weights[p, q]), p, q))
    pair_order.sort(reverse=True)

    for _w, p, q in pair_order:
        if p in assigned or q in assigned:
            continue
        frag_idx = min(range(n_fragments), key=lambda idx: len(fragments[idx]))
        if len(fragments[frag_idx]) <= target_size - 2:
            fragments[frag_idx].update({p, q})
            assigned.update({p, q})
        if len(assigned) == n_orb:
            break

    remaining = [orb for orb in range(n_orb) if orb not in assigned]
    for orb in remaining:
        frag_idx = min(
            (idx for idx in range(n_fragments) if len(fragments[idx]) < target_size),
            key=lambda idx: -sum(weights[orb, member] for member in fragments[idx]),
        )
        fragments[frag_idx].add(orb)

    return _balance_by_occupancy(
        [sorted(frag) for frag in fragments],
        np.diag(h1),
        ref_alpha_bits,
        ref_beta_bits,
    )


def write_partition_json(path: str, fragments: list[list[int]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([sorted(map(int, frag)) for frag in fragments], fh, indent=2)
        fh.write("\n")


def build_priority1_candidates(
    h1: np.ndarray,
    eri: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
    n_fragments: int = 3,
) -> list[PartitionCandidate]:
    builders: list[tuple[str, str, Callable[[], list[list[int]]]]] = [
        (
            "index_contiguous",
            "Neighboring orbital indices kept together; useful if orbitals are already localized/atom-sorted.",
            lambda: make_index_contiguous_partition(h1.shape[0], n_fragments),
        ),
        (
            "integral_graph",
            "Balanced graph partition using h1 plus Coulomb/exchange integral couplings.",
            lambda: make_integral_graph_partition(
                h1, eri, ref_alpha_bits, ref_beta_bits, n_fragments
            ),
        ),
        (
            "strong_pair",
            "Greedy partition that preserves strongest two-orbital integral-coupled pairs.",
            lambda: make_strong_pair_partition(
                h1, eri, ref_alpha_bits, ref_beta_bits, n_fragments
            ),
        ),
    ]

    candidates: list[PartitionCandidate] = []
    for name, description, builder in builders:
        fragments = builder()
        validate_fragment_partition(fragments, h1.shape[0], ref_alpha_bits, ref_beta_bits)
        candidates.append(PartitionCandidate(name, description, fragments))
    return candidates
