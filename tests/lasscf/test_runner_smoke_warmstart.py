"""
test_runner_smoke_warmstart.py
==============================
Smoke test for the warm-start (--init-from) CLI surface on run_lasscf_trimci.

Verifies that:
  1. `run_lasscf_trimci --help` exits with code 0.
  2. `--init-from` appears in the help output (CLI arg is wired).

This is a surface test only — it does not actually run the Fe4S4 warm-start
(that is Phase 4.6 and requires the checkpoint from run_lasscf_csf.py).
"""

from __future__ import annotations

import os
import subprocess

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON = os.path.join(os.path.dirname(_PROJ_ROOT), "FRASCIenv", "bin", "python")


def test_smoke_warmstart_mode_passes():
    """
    Verify that --init-from CLI flag is accepted and appears in --help output.
    """
    result = subprocess.run(
        [
            _PYTHON,
            "-m",
            "FRASCI.lasscf.runners.run_lasscf_trimci",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=_PROJ_ROOT,
        timeout=30,
    )

    if result.stdout:
        print("--- stdout ---")
        print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)

    assert result.returncode == 0, (
        f"run_lasscf_trimci --help exited with code {result.returncode}.\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    assert "--init-from" in result.stdout, (
        f"'--init-from' not found in --help output.\n"
        f"stdout:\n{result.stdout}"
    )
