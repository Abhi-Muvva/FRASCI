#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from FRASCI.crossflow.diagnostics import run_denominator_diagnostic


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
    lines = [
        "# Crossflow Denominator Diagnostic",
        "",
        f"**Status:** {result['status']}",
        f"**Timestamp:** {result['timestamp']}",
        "",
        "## Setup",
        "",
        f"- n_orb: {result['n_orb']}",
        f"- n_alpha: {result['n_alpha']}",
        f"- n_beta: {result['n_beta']}",
        f"- E_mf_global: {result['E_mf_global']:.12f}",
        f"- gamma_load_mode: {result['gamma_load_mode']}",
        f"- ref_det_source_mode: {result['ref_det_source_mode']}",
        f"- partition_strategy: {result['partition_strategy']}",
        f"- coupled_pairs: {result['coupled_pairs']}",
        "",
        "## Model Summary",
        "",
        "| Model | Terms | Negative | Neg. frac | Min gap | Median gap | Max gap | Max abs M |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, model_result in result["model_results"].items():
        overall = model_result["overall"]
        lines.append(
            f"| {model} | {overall['n_terms']} | {overall['n_negative']} | "
            f"{overall['negative_fraction']:.4f} | {_fmt_optional(overall['min_gap'])} | "
            f"{_fmt_optional(overall['median_gap'])} | {_fmt_optional(overall['max_gap'])} | "
            f"{overall['max_abs_M']:.6f} |"
        )

    for model, model_result in result["model_results"].items():
        lines += [
            "",
            f"## {model}",
            "",
            "### By Pair",
            "",
            "| Pair | Terms | Negative | Neg. frac | Min gap | Median gap | Max gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for pair, stats in model_result["by_pair"].items():
            lines.append(
                f"| {pair} | {stats['n_terms']} | {stats['n_negative']} | "
                f"{stats['negative_fraction']:.4f} | {_fmt_optional(stats['min_gap'])} | "
                f"{_fmt_optional(stats['median_gap'])} | {_fmt_optional(stats['max_gap'])} |"
            )

        lines += [
            "",
            "### By Spin Channel",
            "",
            "| Channel | Terms | Negative | Neg. frac | Min gap | Median gap | Max gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for channel, stats in model_result["by_channel"].items():
            lines.append(
                f"| {channel} | {stats['n_terms']} | {stats['n_negative']} | "
                f"{stats['negative_fraction']:.4f} | {_fmt_optional(stats['min_gap'])} | "
                f"{_fmt_optional(stats['median_gap'])} | {_fmt_optional(stats['max_gap'])} |"
            )

        examples = model_result["worst_nonpositive_examples"]
        lines += [
            "",
            "### Worst Non-Positive Examples",
            "",
        ]
        if examples:
            lines += [
                "| gap | pair | channel | i | a | j | b | M |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for ex in examples[:10]:
                lines.append(
                    f"| {ex['gap']:.6f} | {ex['pair']} | {ex['channel']} | "
                    f"{ex['i']} | {ex['a']} | {ex['j']} | {ex['b']} | {ex['M']:.6f} |"
                )
        else:
            lines.append("No non-positive denominators found.")

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossflow PT2 denominator diagnostic")
    parser.add_argument("--fcidump", required=True)
    parser.add_argument("--gamma")
    parser.add_argument("--ref-dets")
    parser.add_argument("--n-fragments", type=int, default=3)
    parser.add_argument("--partition-strategy", default="h1diag", choices=["h1diag", "balanced"])
    parser.add_argument("--fragment-orbs")
    parser.add_argument("--coupled-pairs", default="all")
    parser.add_argument(
        "--denominator-models",
        default="h1_mfa,h1_bare,global_fock,embedded_fock",
        help="Comma-separated subset of h1_mfa,h1_bare,global_fock,embedded_fock",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    models = [item.strip() for item in args.denominator_models.split(",") if item.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    result = run_denominator_diagnostic(
        args.fcidump,
        gamma_path=args.gamma,
        ref_dets_path=args.ref_dets,
        n_fragments=args.n_fragments,
        partition_strategy=args.partition_strategy,
        fragment_orbs_json=args.fragment_orbs,
        coupled_pairs_spec=args.coupled_pairs,
        denominator_models=models,
        max_examples=args.max_examples,
    )
    result = _clean_json_value(result)
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["fcidump_path"] = args.fcidump
    result["gamma_path"] = args.gamma
    result["ref_dets_path"] = args.ref_dets

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    _write_summary(result, args.output_dir)

    print(f"[denom] Output: {args.output_dir}")
    for model, model_result in result["model_results"].items():
        overall = model_result["overall"]
        print(
            f"[denom] {model}: terms={overall['n_terms']} "
            f"negative={overall['n_negative']} "
            f"min_gap={overall['min_gap']} median={overall['median_gap']}"
        )


if __name__ == "__main__":
    main()
