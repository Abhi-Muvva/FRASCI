from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class GammaResult:
    gamma: np.ndarray
    gamma_source_mode: str
    gamma_load_mode: str
    uhf_cache: Optional[Any]


@dataclass
class FragmentSpinOrbitalClasses:
    occ_alpha: list[int]
    virt_alpha: list[int]
    occ_beta: list[int]
    virt_beta: list[int]


@dataclass
class PairCouplingResult:
    frag_I: int
    frag_J: int
    E_pt2: float
    n_terms: int
    max_abs_M: float
    min_gap: Optional[float]
    delta_h_I: np.ndarray
    delta_h_J: np.ndarray


@dataclass
class AllPairCouplingResult:
    E_pt2_cross_total: float
    pair_results: list[PairCouplingResult]
    delta_h_by_frag: dict[int, np.ndarray]
    min_gap_global: Optional[float]
    max_abs_M_global: float
    max_abs_delta_h: float
    n_terms_total: int


@dataclass
class CrossflowIteration:
    iter: int
    E_total: float
    delta_E: Optional[float]
    fragment_n_dets: list[int]
    E_pt2_cross: Optional[float]
    E_total_postprocessed: Optional[float]
    max_abs_delta_h: Optional[float]
    min_gap: Optional[float]
    converged: bool


@dataclass
class CrossflowResult:
    status: str
    E_total_baseline: float
    E_total_final: float
    E_total_postprocessed_final_round: Optional[float]
    delta_E_final_vs_baseline: float
    E_pt2_cross_final: Optional[float]
    max_abs_delta_h_final: Optional[float]
    min_gap_final: Optional[float]
    fragment_n_dets_baseline: list[int]
    fragment_n_dets_final: list[int]
    total_dets_baseline: int
    total_dets_final: int
    reference_energy: Optional[float]
    error_vs_reference_final: Optional[float]
    brute_force_dets: Optional[int]
    det_fraction_vs_bruteforce: Optional[float]
    coupling_diagnostics_final: Optional[dict]
    iteration_history: list[CrossflowIteration]
    n_iters_completed: int
    converged: bool
    gamma_source_mode: str
    ref_det_source_mode: str
    gamma_load_mode: str
    partition_strategy: str
    fragment_orbs: list[list[int]]
    coupled_pairs: list[tuple[int, int]]
    E_mf_global: float
    n_orb: int
    n_alpha: int
    n_beta: int
    E_nuc: float
    trimci_config: dict
    max_iters: int
    damping: float
    energy_conv_threshold: float
