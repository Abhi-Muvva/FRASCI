from __future__ import annotations

from typing import Optional

import numpy as np

from FRASCI.crossflow.coupling import classify_fragment_spin_orbitals
from FRASCI.crossflow.diagnostics import _channel_specs, _matrix_element
from FRASCI.crossflow.io import load_fcidump, load_or_derive_ref_det
from FRASCI.crossflow.partition import (
    load_fragment_orbs_json,
    make_equal_nonoverlapping_partition,
    parse_coupled_pairs,
    validate_fragment_partition,
)


def occupied_orbitals(bits: int, n_orb: int) -> list[int]:
    return [orb for orb in range(n_orb) if (bits >> orb) & 1]


def determinant_diagonal_energy(
    h1: np.ndarray,
    eri: np.ndarray,
    alpha_bits: int,
    beta_bits: int,
    *,
    E_nuc: float = 0.0,
) -> float:
    """Return <D|H|D> for a spin-separated spatial-orbital determinant."""
    n_orb = h1.shape[0]
    if h1.shape != (n_orb, n_orb):
        raise ValueError("h1 must be square")
    if eri.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("eri shape must be (n_orb, n_orb, n_orb, n_orb)")

    alpha_occ = occupied_orbitals(alpha_bits, n_orb)
    beta_occ = occupied_orbitals(beta_bits, n_orb)
    e = float(E_nuc)

    for p in alpha_occ:
        e += float(h1[p, p])
    for p in beta_occ:
        e += float(h1[p, p])

    for occ in (alpha_occ, beta_occ):
        for p in occ:
            for q in occ:
                e += 0.5 * (
                    float(eri[p, p, q, q]) - float(eri[p, q, q, p])
                )

    for p in alpha_occ:
        for q in beta_occ:
            e += float(eri[p, p, q, q])

    return e


def excite_one(bits: int, occ_orb: int, virt_orb: int) -> int:
    if not ((bits >> occ_orb) & 1):
        raise ValueError(f"Orbital {occ_orb} is not occupied")
    if (bits >> virt_orb) & 1:
        raise ValueError(f"Orbital {virt_orb} is already occupied")
    return (bits & ~(1 << occ_orb)) | (1 << virt_orb)


def _excited_bits_for_channel(
    channel: str,
    alpha_bits: int,
    beta_bits: int,
    i: int,
    a: int,
    j: int,
    b: int,
) -> tuple[int, int]:
    if channel == "aa":
        next_alpha = excite_one(excite_one(alpha_bits, i, a), j, b)
        return next_alpha, beta_bits
    if channel == "bb":
        next_beta = excite_one(excite_one(beta_bits, i, a), j, b)
        return alpha_bits, next_beta
    if channel == "ab":
        return excite_one(alpha_bits, i, a), excite_one(beta_bits, j, b)
    if channel == "ba":
        return excite_one(alpha_bits, j, b), excite_one(beta_bits, i, a)
    raise ValueError(f"Unknown spin channel: {channel!r}")


def _summarize_values(values: list[float]) -> dict:
    if not values:
        return {
            "n_terms": 0,
            "sum": 0.0,
            "min": None,
            "max": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "n_terms": int(arr.size),
        "sum": float(np.sum(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def analyze_determinant_energy_pt2(
    h1: np.ndarray,
    eri: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
    frag_orbs_list: list[list[int]],
    coupled_pairs: list[tuple[int, int]],
    *,
    E_nuc: float = 0.0,
    max_examples: int = 20,
    zero_gap_tol: float = 1e-12,
) -> dict:
    ref_energy = determinant_diagonal_energy(
        h1, eri, ref_alpha_bits, ref_beta_bits, E_nuc=E_nuc
    )
    spin_classes = [
        classify_fragment_spin_orbitals(ref_alpha_bits, ref_beta_bits, frag)
        for frag in frag_orbs_list
    ]

    gaps_all: list[float] = []
    contributions_all: list[float] = []
    abs_m_all: list[float] = []
    by_pair: dict[str, list[float]] = {}
    by_channel: dict[str, list[float]] = {channel: [] for channel in ("aa", "bb", "ab", "ba")}
    by_pair_channel: dict[str, list[float]] = {}
    nonpositive_examples: list[dict] = []
    tiny_gap_examples: list[dict] = []

    n_zero_gap = 0
    n_negative_gap = 0

    for frag_I, frag_J in coupled_pairs:
        pair_key = f"{frag_I}-{frag_J}"
        by_pair.setdefault(pair_key, [])

        for channel, occ_I, virt_I, occ_J, virt_J, same_spin in _channel_specs(
            spin_classes[frag_I], spin_classes[frag_J]
        ):
            pair_channel_key = f"{pair_key}:{channel}"
            by_pair_channel.setdefault(pair_channel_key, [])
            if not (occ_I and virt_I and occ_J and virt_J):
                continue

            for i in occ_I:
                for a in virt_I:
                    for j in occ_J:
                        for b in virt_J:
                            excited_alpha, excited_beta = _excited_bits_for_channel(
                                channel, ref_alpha_bits, ref_beta_bits, i, a, j, b
                            )
                            excited_energy = determinant_diagonal_energy(
                                h1, eri, excited_alpha, excited_beta, E_nuc=E_nuc
                            )
                            gap = excited_energy - ref_energy
                            matrix_element = _matrix_element(
                                eri, channel, same_spin, i, a, j, b
                            )
                            abs_m = abs(matrix_element)

                            gaps_all.append(gap)
                            abs_m_all.append(abs_m)
                            by_pair[pair_key].append(gap)
                            by_channel[channel].append(gap)
                            by_pair_channel[pair_channel_key].append(gap)

                            example = {
                                "gap": float(gap),
                                "pair": [frag_I, frag_J],
                                "channel": channel,
                                "i": int(i),
                                "a": int(a),
                                "j": int(j),
                                "b": int(b),
                                "M": float(matrix_element),
                                "abs_M": float(abs_m),
                                "excited_energy": float(excited_energy),
                            }
                            if gap < 0.0:
                                n_negative_gap += 1
                                nonpositive_examples.append(example)
                            if abs(gap) <= zero_gap_tol:
                                n_zero_gap += 1
                                tiny_gap_examples.append(example)
                                continue

                            contribution = -(matrix_element * matrix_element) / gap
                            contributions_all.append(float(contribution))

    nonpositive_examples.sort(key=lambda item: (item["gap"], -item["abs_M"]))
    tiny_gap_examples.sort(key=lambda item: (-item["abs_M"], abs(item["gap"])))
    n_terms = len(gaps_all)
    n_nonzero_terms = len(contributions_all)
    e_pt2_cross = float(np.sum(contributions_all)) if contributions_all else 0.0
    max_abs_m = max(abs_m_all) if abs_m_all else 0.0

    return {
        "method": "determinant_energy",
        "ref_determinant_energy": float(ref_energy),
        "E_pt2_cross": e_pt2_cross,
        "abs_E_pt2_cross": abs(e_pt2_cross),
        "n_terms": n_terms,
        "n_contributing_terms": n_nonzero_terms,
        "n_negative_gap": n_negative_gap,
        "n_zero_gap": n_zero_gap,
        "negative_gap_fraction": float(n_negative_gap / n_terms) if n_terms else 0.0,
        "zero_gap_fraction": float(n_zero_gap / n_terms) if n_terms else 0.0,
        "max_abs_M": float(max_abs_m),
        "gap_summary": _summarize_values(gaps_all),
        "contribution_summary": _summarize_values(contributions_all),
        "by_pair": {
            key: _summarize_values(values) for key, values in sorted(by_pair.items())
        },
        "by_channel": {
            key: _summarize_values(values) for key, values in sorted(by_channel.items())
        },
        "by_pair_channel": {
            key: _summarize_values(values)
            for key, values in sorted(by_pair_channel.items())
        },
        "worst_nonpositive_examples": nonpositive_examples[:max_examples],
        "largest_tiny_gap_examples": tiny_gap_examples[:max_examples],
    }


def run_determinant_energy_pt2_diagnostic(
    fcidump_path: str,
    *,
    ref_dets_path: Optional[str] = None,
    n_fragments: int = 3,
    partition_strategy: str = "h1diag",
    fragment_orbs_json: Optional[str] = None,
    coupled_pairs_spec: str = "all",
    max_examples: int = 20,
    zero_gap_tol: float = 1e-12,
) -> dict:
    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta = load_fcidump(fcidump_path)
    ref_alpha_bits, ref_beta_bits, ref_det_mode = load_or_derive_ref_det(
        ref_dets_path,
        h1,
        eri,
        n_alpha,
        n_beta,
        n_orb,
    )

    if fragment_orbs_json is None:
        fragments = make_equal_nonoverlapping_partition(
            h1,
            n_orb,
            n_fragments,
            strategy=partition_strategy,
            ref_alpha_bits=ref_alpha_bits,
            ref_beta_bits=ref_beta_bits,
        )
        validate_fragment_partition(fragments, n_orb, ref_alpha_bits, ref_beta_bits)
    else:
        fragments = load_fragment_orbs_json(
            fragment_orbs_json,
            n_orb,
            ref_alpha_bits,
            ref_beta_bits,
        )

    coupled_pairs = parse_coupled_pairs(coupled_pairs_spec, len(fragments))
    pt2_result = analyze_determinant_energy_pt2(
        h1,
        eri,
        ref_alpha_bits,
        ref_beta_bits,
        fragments,
        coupled_pairs,
        E_nuc=E_nuc,
        max_examples=max_examples,
        zero_gap_tol=zero_gap_tol,
    )

    return {
        "status": "SUCCESS",
        "n_orb": n_orb,
        "n_elec": n_elec,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "E_nuc": float(E_nuc),
        "ref_det_source_mode": ref_det_mode,
        "partition_strategy": partition_strategy,
        "fragment_orbs_json": fragment_orbs_json,
        "fragment_orbs": fragments,
        "coupled_pairs": coupled_pairs,
        "pt2_result": pt2_result,
    }
