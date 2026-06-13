# Core shared utilities
from FRASCI.core.results import FragmentedRunResult
from FRASCI.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from FRASCI.core.trimci_adapter import FragmentResult, solve_fragment_trimci, solve_fragment_exact
from FRASCI.core.analysis import determinant_summary, iteration_summary, convergence_summary

# Coupling-level solvers (optional in MFA-only environments)
try:
    from FRASCI.uncoupled.solver import run_fragmented_trimci
except ModuleNotFoundError:
    run_fragmented_trimci = None
from FRASCI.mfa.helpers import (
    compute_fragment_rdm1,
    dress_integrals_meanfield,
    assemble_global_rdm1_diag,
)
