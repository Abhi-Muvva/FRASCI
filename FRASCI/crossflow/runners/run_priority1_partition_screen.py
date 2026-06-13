#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from FRASCI.crossflow.diagnostics import run_denominator_diagnostic
from FRASCI.crossflow.io import load_fcidump
from FRASCI.crossflow.partition_candidates import (
    build_priority1_candidates,
    write_partition_json,
)


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DATA_DIR = os.path.join(_ROOT, "FRASCI", "data")
_DEFAULT_FCIDUMP = os.path.join(_DATA_DIR, "fcidump_cycle_6")
_DEFAULT_DETS = os.path.join(_DATA_DIR, "dets.npz")
_DEFAULT_GAMMA = os.path.join(
    _ROOT,
    "FRASCI",
    "Outputs",
    "mfa",
    "outs_extract_full_gamma_20260417_002006",
    "gamma_mixed_diag.npy",
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


def _fragment_electron_counts(fragments: list[list[int]], alpha_bits: int, beta_bits: int) -> list[dict]:
    rows = []
    for frag_idx, frag in enumerate(fragments):
        n_alpha = sum(1 for orb in frag if (alpha_bits >> orb) & 1)
        n_beta = sum(1 for orb in frag if (beta_bits >> orb) & 1)
        rows.append(
            {
                "fragment": frag_idx,
                "n_orb": len(frag),
                "n_alpha": n_alpha,
                "n_beta": n_beta,
                "n_alpha_holes": len(frag) - n_alpha,
                "n_beta_holes": len(frag) - n_beta,
            }
        )
    return rows


def _write_result(output_dir: str, result: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(_clean_json_value(result), fh, indent=2)


def _write_summary(output_dir: str, rows: list[dict]) -> None:
    lines = [
        "# Priority 1 Partition Screen",
        "",
        "| Candidate | Source | Terms | Negative | Negative fraction | Min gap | Median gap | Max gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        overall = row["h1_mfa_overall"]
        lines.append(
            f"| {row['name']} | {row['source']} | {overall['n_terms']} | "
            f"{overall['n_negative']} | {overall['negative_fraction']:.4f} | "
            f"{overall['min_gap']:.6f} | {overall['median_gap']:.6f} | "
            f"{overall['max_gap']:.6f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `h1diag` and `balanced` are built-in controls.",
        "- Other candidates are explicit fragment JSON files generated under `partitions/`.",
        "- These candidates are chemistry proxies because no atom/orbital ownership metadata is present in the dataset.",
    ]
    with open(os.path.join(output_dir, "screen_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Priority 1 partition denominator screen")
    parser.add_argument("--fcidump", default=_DEFAULT_FCIDUMP)
    parser.add_argument("--gamma", default=_DEFAULT_GAMMA)
    parser.add_argument("--ref-dets", default=_DEFAULT_DETS)
    parser.add_argument("--n-fragments", type=int, default=3)
    parser.add_argument("--denominator-models", default="h1_mfa,global_fock")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        _ROOT, "FRASCI", "Outputs", "crossflow", f"priority1_partition_screen_{timestamp}"
    )
    partitions_dir = os.path.join(output_dir, "partitions")
    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(partitions_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)

    h1, eri, _n_elec, _n_orb, _e_nuc, _n_alpha, _n_beta = load_fcidump(args.fcidump)
    dets = np.load(args.ref_dets)["dets"]
    ref_alpha_bits, ref_beta_bits = int(dets[0, 0]), int(dets[0, 1])
    models = [item.strip() for item in args.denominator_models.split(",") if item.strip()]

    rows: list[dict] = []

    for strategy in ("h1diag", "balanced"):
        result = run_denominator_diagnostic(
            args.fcidump,
            gamma_path=args.gamma,
            ref_dets_path=args.ref_dets,
            n_fragments=args.n_fragments,
            partition_strategy=strategy,
            denominator_models=models,
            max_examples=args.max_examples,
        )
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["candidate_name"] = strategy
        result["candidate_source"] = "built_in"
        result["fragment_electron_counts"] = _fragment_electron_counts(
            result["fragment_orbs"], ref_alpha_bits, ref_beta_bits
        )
        _write_result(os.path.join(diagnostics_dir, strategy), result)
        rows.append(
            {
                "name": strategy,
                "source": "built_in",
                "fragment_orbs_json": None,
                "fragment_orbs": result["fragment_orbs"],
                "fragment_electron_counts": result["fragment_electron_counts"],
                "h1_mfa_overall": result["model_results"]["h1_mfa"]["overall"],
            }
        )

    for candidate in build_priority1_candidates(
        h1, eri, ref_alpha_bits, ref_beta_bits, args.n_fragments
    ):
        json_path = os.path.join(partitions_dir, f"{candidate.name}.json")
        write_partition_json(json_path, candidate.fragments)
        result = run_denominator_diagnostic(
            args.fcidump,
            gamma_path=args.gamma,
            ref_dets_path=args.ref_dets,
            n_fragments=args.n_fragments,
            fragment_orbs_json=json_path,
            denominator_models=models,
            max_examples=args.max_examples,
        )
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["candidate_name"] = candidate.name
        result["candidate_source"] = "generated_json"
        result["candidate_description"] = candidate.description
        result["fragment_orbs_json"] = json_path
        result["fragment_electron_counts"] = _fragment_electron_counts(
            candidate.fragments, ref_alpha_bits, ref_beta_bits
        )
        _write_result(os.path.join(diagnostics_dir, candidate.name), result)
        rows.append(
            {
                "name": candidate.name,
                "source": "generated_json",
                "description": candidate.description,
                "fragment_orbs_json": json_path,
                "fragment_orbs": candidate.fragments,
                "fragment_electron_counts": result["fragment_electron_counts"],
                "h1_mfa_overall": result["model_results"]["h1_mfa"]["overall"],
            }
        )

    rows.sort(key=lambda row: row["h1_mfa_overall"]["negative_fraction"])
    with open(os.path.join(output_dir, "screen_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(_clean_json_value(rows), fh, indent=2)
    _write_summary(output_dir, rows)

    print(f"[priority1] Output: {output_dir}")
    for row in rows:
        overall = row["h1_mfa_overall"]
        print(
            f"[priority1] {row['name']}: negative_fraction={overall['negative_fraction']:.4f} "
            f"negative={overall['n_negative']}/{overall['n_terms']} "
            f"min_gap={overall['min_gap']:.6f}"
        )


if __name__ == "__main__":
    main()
