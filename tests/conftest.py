"""Session-wide test infrastructure.

TrimCI's own ``Config.save_fcidumps`` defaults to ``True`` (confirmed:
``trimci.Config().save_fcidumps == True``), and with ``work_dir=None`` it writes
``trimci_orbopt_results/``, ``trimci_single_run_results/``, ``trimci_multi_run_results/``, and
``realtime_progress.out`` relative to the current working directory -- not something
frasci/frasci.recovery override. Running pytest from ``release/FRASCI`` therefore pollutes the
release root with real (if git-ignored) runtime output on every test session that exercises a
real TrimCI solve. Chdir the whole session into a throwaway directory before any test runs, so
those artifacts land there instead -- no test in this suite depends on the working directory
being ``release/FRASCI`` itself (imports resolve via ``sys.path``, not ``cwd``, and every test
that needs a real file uses its own ``tmp_path``/``tmp_path_factory``).
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _run_tests_outside_the_release_tree():
    original = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="frasci_test_run_")
    os.chdir(scratch)
    try:
        yield
    finally:
        os.chdir(original)
