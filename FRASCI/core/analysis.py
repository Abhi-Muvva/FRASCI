"""
core/analysis.py
================
Post-processing and analysis for FRASCI results.
Used by both uncoupled and meanfield coupling levels.
"""
from __future__ import annotations
from typing import Optional
import numpy as np


BRUTE_FORCE_DETS = 10095        # TrimCI on full Fe4S4, 10K-determinant run
REFERENCE_ENERGY = -327.1920    # Ha, from Dr. Otten's group


def determinant_summary(result) -> dict:
    """
    Print and return a summary table comparing fragment determinant usage
    to the brute-force TrimCI reference.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("FRASCI — Determinant Summary")
    lines.append("=" * 60)

    for i, (orbs, n_dets, energy) in enumerate(
        zip(result.fragment_orbs, result.fragment_n_dets, result.fragment_energies)
    ):
        lines.append(
            f"  Fragment {i}: orbs [{orbs[0]:2d}..{orbs[-1]:2d}]"
            f"    n_dets={n_dets:6d}"
            f"    energy={energy:12.4f} Ha"
        )

    lines.append("-" * 60)
    total = result.total_dets
    ref   = result.brute_force_dets
    ratio = total / ref if ref > 0 else float("nan")
    savings = (1.0 - ratio) * 100.0

    lines.append(f"  Total dets     : {total:>8d}")
    lines.append(f"  Brute-force ref: {ref:>8d}  (E = {REFERENCE_ENERGY:.4f} Ha)")
    lines.append(f"  Ratio          : {ratio:>8.3f}x brute-force")
    lines.append(f"  Savings        : {savings:>+7.1f}%  (negative = MORE dets than brute-force)")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    return {
        "fragment_dets":    result.fragment_n_dets,
        "total_dets":       total,
        "brute_force_dets": ref,
        "ratio":            ratio,
        "savings_pct":      savings,
    }


def iteration_summary(result) -> list:
    """
    Print an iteration table and return the iteration history from a meanfield run.
    """
    history = list(getattr(result, 'iteration_history', []))
    if not history:
        print("No iteration history recorded.")
        return history

    print("=" * 78)
    print("Meanfield SCF — Iteration Summary")
    print("=" * 78)
    print(f"  {'Iter':>4}  {'max|ΔE| (Ha)':>14}  {'max|Δγ|':>12}  "
          f"{'total_dets':>10}  energies")
    print("-" * 78)
    for entry in history:
        it      = entry['iteration']
        dE      = entry['delta_E']
        drdm    = entry['delta_rdm']
        ndets   = sum(entry['n_dets'])
        energies_str = "[" + ", ".join(f"{e:.4f}" for e in entry['energies']) + "]"
        dE_str   = f"{dE:.3e}" if dE != float('inf') else "     inf"
        drdm_str = f"{drdm:.3e}" if drdm != float('inf') else "     inf"
        print(f"  {it:>4}  {dE_str:>14}  {drdm_str:>12}  {ndets:>10}  {energies_str}")
    print("=" * 78)
    return history


def convergence_summary(result) -> dict:
    """
    Print a convergence verdict and return a summary dict from a meanfield run.
    """
    converged   = getattr(result, 'converged', False)
    iterations  = getattr(result, 'iterations', 1)
    delta_E     = getattr(result, 'convergence_delta', float('inf'))
    delta_rdm   = getattr(result, 'convergence_delta_rdm', float('inf'))

    history = getattr(result, 'iteration_history', [])
    if not history:
        verdict = "No iterations recorded"
    elif converged:
        verdict = "SCF converged"
    else:
        verdict = "Hit max iterations without convergence"

    print("=" * 60)
    print("Meanfield SCF — Convergence Summary")
    print("=" * 60)
    print(f"  converged  : {converged}")
    print(f"  iterations : {iterations}")
    print(f"  final max|ΔE|  : {delta_E:.3e} Ha")
    print(f"  final max|Δγ|  : {delta_rdm:.3e}")
    print(f"  Verdict    : {verdict}")
    print("=" * 60)

    return {
        'converged':             converged,
        'iterations':            iterations,
        'convergence_delta':     delta_E,
        'convergence_delta_rdm': delta_rdm,
        'verdict':               verdict,
    }


def plot_det_comparison(result, save_path: Optional[str] = None):
    """Bar chart: per-fragment determinant count vs brute-force reference line."""
    raise NotImplementedError


def orbital_mi_analysis(
    dets_npz_path: str,
    n_orb: int = 36,
    n_alpha: int = 27,
    n_beta: int = 27,
) -> np.ndarray:
    """Compute orbital mutual information from the TrimCI reference wavefunction."""
    raise NotImplementedError


def plot_mi_heatmap(
    mi_matrix: np.ndarray,
    fragment_boundaries: Optional[list[list[int]]] = None,
    save_path: Optional[str] = None,
):
    """Heatmap of the orbital mutual information matrix."""
    raise NotImplementedError
