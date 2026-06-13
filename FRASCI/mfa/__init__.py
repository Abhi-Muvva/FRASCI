# Phase D: MFA-FRASCI — mean-field active selected-CI embedding
from FRASCI.mfa.solver import run_mfa_d1

try:
    from FRASCI.mfa.solver import run_mfa_d2
except ImportError:
    run_mfa_d2 = None

try:
    from FRASCI.mfa.extract_full_gamma import assemble_global_gamma_full
except ImportError:
    assemble_global_gamma_full = None

try:
    from FRASCI.mfa.gamma_bootstrap import bootstrap_gamma
except ImportError:
    bootstrap_gamma = None

try:
    from FRASCI.mfa.gamma_experiment import (
        evaluate_gamma_candidate_d2,
        run_gamma_experiment,
    )
except ImportError:
    evaluate_gamma_candidate_d2 = None
    run_gamma_experiment = None

__all__ = [
    "run_mfa_d1",
    "run_mfa_d2",
    "assemble_global_gamma_full",
    "bootstrap_gamma",
    "evaluate_gamma_candidate_d2",
    "run_gamma_experiment",
]
