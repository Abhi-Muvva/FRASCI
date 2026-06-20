from __future__ import annotations

import argparse
import json
from datetime import datetime

from FRASCI.lasscf.fragment_sweep import run_lasscf_sweep


def _parse_thresholds(spec: str) -> list[float]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise argparse.ArgumentTypeError("threshold list cannot be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate many Fe4S4 active-space partitions and optionally run "
            "LASSCF+TrimCI on every candidate."
        )
    )
    parser.add_argument(
        "--fcidump",
        default="data/fcidump_cycle_6",
        help="Path to FCIDUMP Hamiltonian.",
    )
    parser.add_argument(
        "--dets",
        default="data/dets.npz",
        help="Reference determinant archive used for per-fragment electron counts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Sweep output directory. Default: "
            "Outputs/lasscf/fragment_sweep_<timestamp>"
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=24,
        help="Number of distinct partition candidates to generate (default: 24).",
    )
    parser.add_argument(
        "--max-fragment-size",
        type=int,
        default=None,
        help=(
            "Optional cap on largest fragment size. Useful for fast scans; "
            "omit to include 18/20/24-orbital stress tests."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run LASSCF+TrimCI. Without this flag only candidates and plots are generated.",
    )
    parser.add_argument(
        "--trimci-thresholds",
        type=_parse_thresholds,
        default=[0.06],
        help="Comma-separated TrimCI thresholds to run for each candidate (default: 0.06).",
    )
    parser.add_argument(
        "--trimci-max-dets",
        default="auto",
        help="TrimCI max_final_dets cap per fragment, or 'auto' (default).",
    )
    parser.add_argument(
        "--trimci-max-rounds",
        type=int,
        default=2,
        help="TrimCI max_rounds per fragment solve (default: 2).",
    )
    parser.add_argument(
        "--max-cycle-macro",
        type=int,
        default=20,
        help="Maximum LASSCF macro iterations per run (default: 20).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep on the first failed candidate instead of recording failure.json.",
    )
    args = parser.parse_args()

    trimci_max_dets = (
        "auto" if args.trimci_max_dets == "auto" else int(args.trimci_max_dets)
    )
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"Outputs/lasscf/fragment_sweep_{stamp}"

    summary = run_lasscf_sweep(
        fcidump_path=args.fcidump,
        dets_path=args.dets,
        output_dir=output_dir,
        target_count=args.target_count,
        max_fragment_size=args.max_fragment_size,
        execute=args.execute,
        trimci_thresholds=args.trimci_thresholds,
        max_cycle_macro=args.max_cycle_macro,
        trimci_max_dets=trimci_max_dets,
        trimci_max_rounds=args.trimci_max_rounds,
        stop_on_error=args.stop_on_error,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
