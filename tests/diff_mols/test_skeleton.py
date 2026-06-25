"""Smoke test: every module name in diff_mols/ is importable."""
import importlib
import pytest

EXPECTED = [
    "FRASCI.diff_mols",
    "FRASCI.diff_mols.config",
    "FRASCI.diff_mols.integrals_builder",
    "FRASCI.diff_mols.fragmentation",
    "FRASCI.diff_mols.run_writer",
    "FRASCI.diff_mols.methods",
    "FRASCI.diff_mols.lassis_states",
    "FRASCI.diff_mols.benchmark",
    "FRASCI.diff_mols.run",
    "FRASCI.diff_mols.report",
    "FRASCI.diff_mols.tuning",
]


@pytest.mark.parametrize("modname", EXPECTED)
def test_module_importable(modname):
    mod = importlib.import_module(modname)
    assert mod is not None
