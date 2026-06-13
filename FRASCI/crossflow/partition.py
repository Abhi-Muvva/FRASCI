from __future__ import annotations

import json
import warnings
from typing import Optional

import numpy as np

from FRASCI.core.fragment import fragment_electron_count


def make_equal_nonoverlapping_partition(
    h1: np.ndarray,
    n_orb: int,
    n_fragments: int,
    strategy: str = "h1diag",
    ref_alpha_bits: Optional[int] = None,
    ref_beta_bits: Optional[int] = None,
) -> list[list[int]]:
    if n_fragments <= 0:
        raise ValueError("n_fragments must be positive")
    if n_fragments > n_orb:
        raise ValueError("n_fragments cannot exceed n_orb")

    h1_diag = np.diag(h1)
    if len(h1_diag) != n_orb:
        raise ValueError(f"h1 diagonal length {len(h1_diag)} does not match n_orb={n_orb}")

    if strategy == "h1diag":
        order = np.argsort(h1_diag, kind="stable")
    elif strategy == "balanced":
        if ref_alpha_bits is None or ref_beta_bits is None:
            raise ValueError("strategy='balanced' requires ref_alpha_bits and ref_beta_bits")
        classes: dict[str, list[int]] = {"docc": [], "socc_a": [], "socc_b": [], "virt": []}
        for orb in range(n_orb):
            alpha_occ = bool((ref_alpha_bits >> orb) & 1)
            beta_occ = bool((ref_beta_bits >> orb) & 1)
            if alpha_occ and beta_occ:
                classes["docc"].append(orb)
            elif alpha_occ:
                classes["socc_a"].append(orb)
            elif beta_occ:
                classes["socc_b"].append(orb)
            else:
                classes["virt"].append(orb)

        ordered: list[int] = []
        for cls in ("docc", "socc_a", "socc_b", "virt"):
            ordered.extend(sorted(classes[cls], key=lambda orb: (h1_diag[orb], orb)))
        order = np.array(ordered, dtype=int)
    else:
        raise ValueError(f"Unknown partition strategy: {strategy!r}")

    base = n_orb // n_fragments
    remainder = n_orb % n_fragments
    fragments: list[list[int]] = []
    start = 0
    for frag_idx in range(n_fragments):
        size = base + (1 if frag_idx < remainder else 0)
        fragments.append(sorted(order[start : start + size].tolist()))
        start += size
    return fragments


def validate_fragment_partition(
    fragments: list[list[int]],
    n_orb: int,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> None:
    if not fragments:
        raise ValueError("Partition must contain at least one fragment")

    seen: set[int] = set()
    for frag_idx, frag in enumerate(fragments):
        if len(frag) == 0:
            raise ValueError(f"Fragment {frag_idx} is empty")
        for orb in frag:
            if not (0 <= orb < n_orb):
                raise ValueError(f"Fragment {frag_idx}: orbital {orb} out of range [0, {n_orb})")
            if orb in seen:
                raise ValueError(f"Fragment {frag_idx}: orbital {orb} appears more than once")
            seen.add(orb)

    if len(seen) != n_orb:
        missing = sorted(set(range(n_orb)) - seen)
        raise ValueError(f"Partition missing orbitals: {missing[:10]}")

    for frag_idx, frag in enumerate(fragments):
        n_alpha, n_beta = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag)
        n_orb_frag = len(frag)
        if n_alpha > n_orb_frag:
            raise ValueError(f"Fragment {frag_idx}: n_alpha_I={n_alpha} > n_orb_I={n_orb_frag}")
        if n_beta > n_orb_frag:
            raise ValueError(f"Fragment {frag_idx}: n_beta_I={n_beta} > n_orb_I={n_orb_frag}")

        n_virt_alpha = sum(1 for orb in frag if not ((ref_alpha_bits >> orb) & 1))
        n_virt_beta = sum(1 for orb in frag if not ((ref_beta_bits >> orb) & 1))
        if n_virt_alpha == 0 and n_virt_beta == 0:
            warnings.warn(
                f"Fragment {frag_idx} has no virtual orbitals - contributes zero to PT2",
                stacklevel=2,
            )


def load_fragment_orbs_json(
    path: str,
    n_orb: int,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> list[list[int]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Fragment orbital JSON must be a list of fragments")

    fragments = [sorted(map(int, frag)) for frag in data]
    validate_fragment_partition(fragments, n_orb, ref_alpha_bits, ref_beta_bits)
    return fragments


def parse_coupled_pairs(spec: str, n_fragments: int) -> list[tuple[int, int]]:
    if spec == "all":
        return [(i, j) for i in range(n_fragments) for j in range(i + 1, n_fragments)]

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in spec.split(","):
        token = token.strip()
        parts = token.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair spec token: {token!r} (expected 'I-J')")
        a, b = int(parts[0]), int(parts[1])
        if a == b:
            raise ValueError(f"Pair ({a},{b}) must contain two distinct fragments")
        if a > b:
            a, b = b, a
        if not (0 <= a < n_fragments and 0 <= b < n_fragments):
            raise ValueError(f"Pair ({a},{b}) out of range for n_fragments={n_fragments}")
        if (a, b) not in seen:
            pairs.append((a, b))
            seen.add((a, b))
    return pairs
