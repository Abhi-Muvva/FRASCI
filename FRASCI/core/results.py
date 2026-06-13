"""
core/results.py
===============
Shared result dataclass used by all coupling levels (uncoupled, meanfield, dmet).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class FragmentedRunResult:
    """Aggregated result from a fragmented TrimCI run."""
    fragment_energies: list[float]
    fragment_n_dets: list[int]
    fragment_orbs: list[list[int]]
    total_dets: int                     # sum of fragment_n_dets
    brute_force_dets: int = 10095       # Fe4S4 brute-force TrimCI reference
    iterations: int = 1                 # >1 for meanfield self-consistent loop
    # Meanfield / DMET telemetry (defaults preserve uncoupled compatibility)
    iteration_history: list = field(default_factory=list)
    converged: bool = False
    convergence_delta: float = float('inf')
    convergence_delta_rdm: float = float('inf')
