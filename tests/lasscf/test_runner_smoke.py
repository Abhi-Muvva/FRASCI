"""
test_runner_smoke.py
====================
Phase 4 smoke test: invoke run_lasscf_trimci --smoke-test end-to-end as a
subprocess and verify exit code 0 + "SMOKE TEST PASS" in stdout.

This is the gating test that confirms the full integration path
(H4 LASSCF+TrimCI → make_fcibox injection → kernel closure → e_tot vs FCI)
works without error before the Fe4S4 production run.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# Repository root is two levels above tests/lasscf; the shared environment is
# a sibling of the repository.
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON = os.path.join(os.path.dirname(_PROJ_ROOT), "FRASCIenv", "bin", "python")


def test_smoke_test_mode_passes(tmp_path):
    """
    Run `run_lasscf_trimci --smoke-test` end-to-end and check:
    - exit code 0
    - 'SMOKE TEST PASS' in stdout

    The smoke test internally asserts |las.e_tot - e_fci| <= 1e-6 Ha on H4/STO-3G.
    Exit code 1 means the energy tolerance was violated; any other non-zero
    exit code means a crash in the integration path.
    """
    result = subprocess.run(
        [
            _PYTHON,
            "-m",
            "FRASCI.lasscf.runners.run_lasscf_trimci",
            "--smoke-test",
        ],
        capture_output=True,
        text=True,
        cwd=_PROJ_ROOT,
        timeout=180,
    )

    # Print captured output for pytest -v / diagnostic purposes
    if result.stdout:
        print("--- stdout ---")
        print(result.stdout)
    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)

    assert result.returncode == 0, (
        f"run_lasscf_trimci --smoke-test exited with code {result.returncode}.\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    assert "SMOKE TEST PASS" in result.stdout, (
        f"'SMOKE TEST PASS' not found in stdout.\n"
        f"stdout:\n{result.stdout}"
    )
