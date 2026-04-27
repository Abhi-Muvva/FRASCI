# Phase D: MFA-FRASCI — mean-field active selected-CI embedding
from FRASCI.mfa.solver import run_mfa_d1

try:
    from FRASCI.mfa.solver import run_mfa_d2
except ImportError:
    run_mfa_d2 = None

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
    run_gamma_experiment = None
    evaluate_gamma_candidate_d2 = None

__all__ = [
    "run_mfa_d1",
    "run_mfa_d2",
    "bootstrap_gamma",
    "run_gamma_experiment",
    "evaluate_gamma_candidate_d2",
]
