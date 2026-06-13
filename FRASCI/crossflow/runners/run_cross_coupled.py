#!/usr/bin/env python3
"""
CLI entry point for the Crossflow PT2 cross-fragment coupled solver.

Usage:
    python -m frasci.crossflow.runners.run_cross_coupled \
        --fcidump path/to/FCIDUMP \
        --output-dir path/to/output/

All system-specific values (--reference-energy, --brute-force-dets,
--gamma, --ref-dets) are optional CLI arguments and are never hardcoded.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from FRASCI.crossflow.solver import run_cross_coupled_solver


def _clean_json_value(obj: Any) -> Any:
    """Convert numpy-backed dataclass values to JSON-serializable objects."""
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


def _result_to_dict(result: Any) -> dict:
    return _clean_json_value(dataclasses.asdict(result))


def _write_summary(result: Any, output_dir: str) -> None:
    lines = [
        "# Crossflow Cross-Fragment Coupled Solver Summary",
        "",
        f"**Status:** {result.status}",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Energies",
        "| Quantity | Value (Ha) |",
        "| --- | --- |",
        f"| E_mf_global | {result.E_mf_global:.6f} |",
        f"| **E_total_baseline** | **{result.E_total_baseline:.6f}** |",
        f"| **E_total_final (re-solved)** | **{result.E_total_final:.6f}** |",
    ]
    if result.E_total_postprocessed_final_round is not None:
        lines.append(
            "| E_total_postprocessed (additive PT2) | "
            f"{result.E_total_postprocessed_final_round:.6f} |"
        )
    if result.reference_energy is not None:
        lines += [
            f"| Reference | {result.reference_energy:.6f} |",
            f"| Error vs reference | {result.error_vs_reference_final:+.4f} |",
        ]

    lines += [
        "",
        "## Determinants",
        "| | Baseline | Final |",
        "| --- | --- | --- |",
        f"| Total | {result.total_dets_baseline} | {result.total_dets_final} |",
    ]
    for i, (baseline, final) in enumerate(
        zip(result.fragment_n_dets_baseline, result.fragment_n_dets_final)
    ):
        lines.append(f"| Fragment {i} | {baseline} | {final} |")
    if result.brute_force_dets:
        lines.append(f"| Brute-force | {result.brute_force_dets} | - |")

    lines += [
        "",
        "## PT2 Coupling Diagnostics",
    ]
    if result.coupling_diagnostics_final:
        cd = result.coupling_diagnostics_final
        if "error" in cd:
            lines += [
                f"- failure_stage: {cd['failure_stage']}",
                f"- coupling_round: {cd['coupling_round']}",
                f"- n_pairs: {cd['n_pairs']}",
                f"- error: {cd['error']}",
            ]
        else:
            lines += [
                f"- n_terms_total: {cd['n_terms_total']}",
                f"- n_pairs: {cd['n_pairs']}",
                f"- min_gap_global: {cd['min_gap_global']}",
                f"- max_abs_M_global: {cd['max_abs_M_global']:.6f}",
                f"- max_abs_delta_h: {cd['max_abs_delta_h']:.6f}",
            ]

    lines += [
        "",
        "## Iteration history",
        "| iter | E_total | delta_E | E_pt2_cross | n_dets_total |",
        "| --- | --- | --- | --- | --- |",
    ]
    for iteration in result.iteration_history:
        delta_e = (
            f"{iteration.delta_E:+.6f}" if iteration.delta_E is not None else "-"
        )
        pt2 = (
            f"{iteration.E_pt2_cross:.6f}"
            if iteration.E_pt2_cross is not None
            else "-"
        )
        lines.append(
            f"| {iteration.iter} | {iteration.E_total:.6f} | {delta_e} | "
            f"{pt2} | {sum(iteration.fragment_n_dets)} |"
        )

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crossflow PT2 cross-fragment coupled solver"
    )
    parser.add_argument("--fcidump", required=True)
    parser.add_argument("--gamma")
    parser.add_argument("--ref-dets")
    parser.add_argument("--n-fragments", type=int, default=3)
    parser.add_argument(
        "--partition-strategy",
        default="h1diag",
        choices=["h1diag", "balanced"],
    )
    parser.add_argument("--fragment-orbs")
    parser.add_argument("--coupled-pairs", default="all")
    parser.add_argument("--max-iters", type=int, default=1)
    parser.add_argument("--energy-conv-threshold", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-energy", type=float)
    parser.add_argument("--brute-force-dets", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    trimci_config = {}
    if args.threshold is not None:
        trimci_config["threshold"] = args.threshold

    result = run_cross_coupled_solver(
        args.fcidump,
        gamma_path=args.gamma,
        ref_dets_path=args.ref_dets,
        n_fragments=args.n_fragments,
        partition_strategy=args.partition_strategy,
        fragment_orbs_json=args.fragment_orbs,
        coupled_pairs_spec=args.coupled_pairs,
        trimci_config=trimci_config or None,
        max_iters=args.max_iters,
        energy_conv_threshold=args.energy_conv_threshold,
        damping=args.damping,
        reference_energy=args.reference_energy,
        brute_force_dets=args.brute_force_dets,
    )

    out = _result_to_dict(result)
    out["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["fcidump_path"] = args.fcidump
    out["gamma_path"] = args.gamma
    out["ref_dets_path"] = args.ref_dets

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _write_summary(result, args.output_dir)

    print(f"\n[crossflow] Status:            {result.status}")
    print(f"[crossflow] E_total_baseline:  {result.E_total_baseline:.6f} Ha")
    print(f"[crossflow] E_total_final:     {result.E_total_final:.6f} Ha")
    if result.E_total_postprocessed_final_round is not None:
        print(
            "[crossflow] E_postprocessed:   "
            f"{result.E_total_postprocessed_final_round:.6f} Ha"
        )
    if result.reference_energy is not None:
        print(f"[crossflow] Error vs ref:      {result.error_vs_reference_final:+.4f} Ha")
    print(
        f"[crossflow] Total dets:        "
        f"{result.total_dets_baseline} -> {result.total_dets_final}"
    )
    print(f"[crossflow] Output:            {args.output_dir}")


if __name__ == "__main__":
    main()
