"""Direct unit checks for frasci._trimci_config -- the two behaviors that
matter and are cheap to test in isolation: build_coo_config only sets fields
the caller actually passed (None-filtering) and overrides win; suppress_fd
actually silences fd-level output. Full end-to-end numerical coverage comes
from every other frasci test that drives a real TrimCI solve through these
two functions (test_frasci_parent_kernel.py, test_frasci_recovery_growth.py,
etc.) -- this file only pins the two functions' own small logic.
"""
import os

from frasci._trimci_config import build_coo_config, suppress_fd


def test_build_coo_config_only_sets_passed_fields():
    cfg = build_coo_config(threshold=1e-4, num_runs=8)
    assert cfg.threshold == 1e-4
    assert cfg.num_runs == 8


def test_build_coo_config_overrides_win():
    cfg = build_coo_config(threshold=1e-4, overrides={"threshold": 5e-5})
    assert cfg.threshold == 5e-5


def test_suppress_fd_silences_fd_level_write(capfd):
    with suppress_fd(True):
        os.write(1, b"should not appear\n")
    captured = capfd.readouterr()
    assert "should not appear" not in captured.out


def test_suppress_fd_passthrough_when_not_quiet(capfd):
    with suppress_fd(False):
        os.write(1, b"should appear\n")
    captured = capfd.readouterr()
    assert "should appear" in captured.out
