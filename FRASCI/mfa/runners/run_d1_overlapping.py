# frasci/mfa/runners/run_d1_overlapping.py
"""
Runner: MFA-TrimCI D1 (overlapping W=15 S=10) on Fe4S4.

Usage (from frasci/):
    source ../../qflowenv/bin/activate
    python -m mfa.runners.run_d1_overlapping
"""
import argparse, os, sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

_HERE         = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR     = os.path.join(_HERE, '..', '..', 'data')
_DEFAULT_FCIDUMP = os.path.normpath(os.path.join(_DATA_DIR, "fcidump_cycle_6"))
_DEFAULT_GAMMA   = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'Outputs', 'mfa',
    'outs_extract_full_gamma_20260417_002006', 'gamma_mixed_diag.npy'))
_DEFAULT_DETS    = os.path.normpath(os.path.join(_DATA_DIR, "dets.npz"))

TRIMCI_CONFIG = {
    "threshold": 0.06, "max_final_dets": "auto",
    "max_rounds": 2, "num_runs": 1,
    "pool_build_strategy": "heat_bath", "verbose": False,
}


def _coerce_max_dets(value: str):
    return "auto" if value == "auto" else int(value)


def main():
    parser = argparse.ArgumentParser(description="MFA-TrimCI D1 runner")
    parser.add_argument("--fcidump",       default=_DEFAULT_FCIDUMP)
    parser.add_argument("--gamma-path",    default=_DEFAULT_GAMMA)
    parser.add_argument("--ref-dets-path", default=_DEFAULT_DETS)
    parser.add_argument("--output-dir",    default=None)
    parser.add_argument("--trimci-threshold", type=float, default=0.06)
    parser.add_argument("--trimci-max-dets",  default="auto")
    args = parser.parse_args()

    config = {**TRIMCI_CONFIG,
              "threshold": args.trimci_threshold,
              "max_final_dets": _coerce_max_dets(args.trimci_max_dets)}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'Outputs', 'mfa',
        f"outs_d1_overlapping_{timestamp}",
    ))

    from FRASCI.mfa import run_mfa_d1
    results = run_mfa_d1(
        fcidump_path=args.fcidump,
        gamma_path=args.gamma_path,
        ref_dets_path=args.ref_dets_path,
        trimci_config=config,
        output_dir=output_dir,
    )
    print(f"\n[D1] total_dets={results['total_dets']}  "
          f"matches_phase_c={results['matches_phase_c_baseline']}  "
          f"delta_vs_phase_c={results['delta_dets_vs_phase_c']:+d}")


if __name__ == "__main__":
    main()
