#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from FRASCI.crossflow.determinant_pt2 import (
    run_determinant_energy_pt2_diagnostic,
)


def _clean_json_value(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {_clean_json_value(k): _clean_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json_value(v) for v in obj]
    return obj


def _fmt_optional(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_summary(result: dict, output_dir: str) -> None:
    pt2 = result["pt2_result"]
    gap = pt2["gap_summary"]
    contrib = pt2["contribution_summary"]
    lines = [
        "# Determinant-Energy PT2 Diagnostic",
        "",
        f"**Status:** {result['status']}",
        f"**Timestamp:** {result['timestamp']}",
        "",
        "## Setup",
        "",
        f"- n_orb: {result['n_orb']}",
        f"- n_alpha: {result['n_alpha']}",
        f"- n_beta: {result['n_beta']}",
        f"- ref_det_source_mode: {result['ref_det_source_mode']}",
        f"- partition_strategy: {result['partition_strategy']}",
        f"- fragment_orbs_json: {result.get('fragment_orbs_json')}",
        f"- coupled_pairs: {result['coupled_pairs']}",
        "",
        "## Overall",
        "",
        f"- ref_determinant_energy: {pt2['ref_determinant_energy']:.12f} Ha",
        f"- E_pt2_cross: {pt2['E_pt2_cross']:.12f} Ha",
        f"- |E_pt2_cross|: {pt2['abs_E_pt2_cross']:.12f} Ha",
        f"- n_terms: {pt2['n_terms']}",
        f"- n_contributing_terms: {pt2['n_contributing_terms']}",
        f"- n_negative_gap: {pt2['n_negative_gap']}",
        f"- negative_gap_fraction: {pt2['negative_gap_fraction']:.4f}",
        f"- n_zero_gap: {pt2['n_zero_gap']}",
        f"- max_abs_M: {pt2['max_abs_M']:.6f}",
        "",
        "## Gap Summary",
        "",
        "| Terms | Min | P05 | Median | P95 | Max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {gap['n_terms']} | {_fmt_optional(gap['min'])} | "
            f"{_fmt_optional(gap['p05'])} | {_fmt_optional(gap['median'])} | "
            f"{_fmt_optional(gap['p95'])} | {_fmt_optional(gap['max'])} |"
        ),
        "",
        "## Contribution Summary",
        "",
        "| Terms | Sum | Min | P05 | Median | P95 | Max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {contrib['n_terms']} | {contrib['sum']:.6f} | "
            f"{_fmt_optional(contrib['min'])} | {_fmt_optional(contrib['p05'])} | "
            f"{_fmt_optional(contrib['median'])} | {_fmt_optional(contrib['p95'])} | "
            f"{_fmt_optional(contrib['max'])} |"
        ),
        "",
        "## By Pair",
        "",
        "| Pair | Terms | Sum gap | Min gap | Median gap | Max gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, stats in pt2["by_pair"].items():
        lines.append(
            f"| {pair} | {stats['n_terms']} | {stats['sum']:.6f} | "
            f"{_fmt_optional(stats['min'])} | {_fmt_optional(stats['median'])} | "
            f"{_fmt_optional(stats['max'])} |"
        )

    lines += [
        "",
        "## By Channel",
        "",
        "| Channel | Terms | Sum gap | Min gap | Median gap | Max gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for channel, stats in pt2["by_channel"].items():
        lines.append(
            f"| {channel} | {stats['n_terms']} | {stats['sum']:.6f} | "
            f"{_fmt_optional(stats['min'])} | {_fmt_optional(stats['median'])} | "
            f"{_fmt_optional(stats['max'])} |"
        )

    lines += [
        "",
        "## Worst Non-Positive Gap Examples",
        "",
    ]
    examples = pt2["worst_nonpositive_examples"]
    if examples:
        lines += [
            "| Gap | Pair | Channel | i | a | j | b | M |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for ex in examples[:10]:
            lines.append(
                f"| {ex['gap']:.6f} | {ex['pair']} | {ex['channel']} | "
                f"{ex['i']} | {ex['a']} | {ex['j']} | {ex['b']} | {ex['M']:.6f} |"
            )
    else:
        lines.append("No non-positive determinant-energy gaps found.")

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Determinant-energy cross-fragment PT2 diagnostic")
    parser.add_argument("--fcidump", required=True)
    parser.add_argument("--ref-dets")
    parser.add_argument("--n-fragments", type=int, default=3)
    parser.add_argument("--partition-strategy", default="h1diag", choices=["h1diag", "balanced"])
    parser.add_argument("--fragment-orbs")
    parser.add_argument("--coupled-pairs", default="all")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--zero-gap-tol", type=float, default=1e-12)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    result = run_determinant_energy_pt2_diagnostic(
        args.fcidump,
        ref_dets_path=args.ref_dets,
        n_fragments=args.n_fragments,
        partition_strategy=args.partition_strategy,
        fragment_orbs_json=args.fragment_orbs,
        coupled_pairs_spec=args.coupled_pairs,
        max_examples=args.max_examples,
        zero_gap_tol=args.zero_gap_tol,
    )
    result = _clean_json_value(result)
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["fcidump_path"] = args.fcidump
    result["ref_dets_path"] = args.ref_dets

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    _write_summary(result, args.output_dir)

    pt2 = result["pt2_result"]
    print(f"[det-pt2] Output: {args.output_dir}")
    print(f"[det-pt2] E_pt2_cross={pt2['E_pt2_cross']:.12f} Ha")
    print(f"[det-pt2] |E_pt2_cross|={pt2['abs_E_pt2_cross']:.12f} Ha")
    print(
        "[det-pt2] terms={terms} negative_gap={neg} "
        "negative_fraction={frac:.4f}".format(
            terms=pt2["n_terms"],
            neg=pt2["n_negative_gap"],
            frac=pt2["negative_gap_fraction"],
        )
    )


if __name__ == "__main__":
    main()
