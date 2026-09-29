"""Milestone G (2026-09-22): final free (cross-fragment) orbital optimization on a
top-K-truncated ``P_LAS`` union ``R``, run unconditionally as the last step of
``frasci.recovery.solve()``.

Distinct from Milestone F's ``form_final_union``/``solve_union_davidson``
(``_union_solve.py``), which needs the full, exact ``P_LAS`` support -- intractable
past a few hundred thousand determinants (an exact fragment CAS can run past a
billion). This module truncates ``P_LAS`` to its top-K product terms by
``|coefficient|`` (``top_product_terms``, unchanged heap-select, no Cartesian
enumeration) instead, then lets orbitals rotate FREELY across fragment boundaries
(not fragment-block-restricted like ``frasci.recovery.constrained_coo_sweep``) on
that fixed support via ``run_fixed_support_constrained_coo``, and independently
rediagonalizes at the resulting rotated frame.

Two production lessons from the pilot that discovered this method, both load-bearing
here:

- ``run_fixed_support_constrained_coo``'s returned ``coeffs`` must NEVER be paired
  directly with its ``dets``/``h1_final``/``eri_final`` for an energy check -- doing
  so silently violates the variational bound (observed directly: an apparent energy
  INCREASE of 42 mHa on a case whose independently rediagonalized energy actually
  improved by 8.15 mHa). Always rediagonalize independently via
  ``frasci.parent._kernel._fixed_support_ground_state`` at the final frame, both
  before and after rotation -- never trust a returned coefficient array alone.
- The historical BFGS default used everywhere else in this project
  (``orbital_opt_max_iter=20``) is measurably under-converged for this purpose: on a
  2000-determinant fe4s4 union, the gain was still climbing more than 40% between 20
  and 100 iterations. This module keeps 20 as its own default for continuity, but
  callers doing a real production run should consider raising it.
"""
from __future__ import annotations

import itertools

import numpy as np

from frasci.parent._kernel import _fixed_support_ground_state
from frasci.recovery._native_stream import run_fixed_support_constrained_coo
from frasci.recovery._state import retain_candidates, top_product_terms

Determinant = tuple[int, int]


def build_top_k_union(state: dict, las_top_k: int, recovery_dets, recovery_coeffs, *,
                      n_ct_max: int) -> dict:
    """Top-``las_top_k`` ``P_LAS`` product terms union ``R`` (dispersion + allowed-CT
    only, same fixed rule as ``form_final_union``). Never enumerates the full
    Cartesian ``P_LAS`` -- ``top_product_terms`` heap-selects the top terms directly.

    Returns deduplicated determinants and a Davidson initial guess: the top-K LAS
    amplitude on those members, zero on new ``R`` members -- same convention as
    ``form_final_union``.
    """
    las_dets, las_coeffs = top_product_terms(state, int(las_top_k))
    eligible, classification = retain_candidates(
        recovery_dets, recovery_coeffs, state, n_ct_max=n_ct_max, include_same_sector=False)
    r_dets = list(eligible)

    combined: dict[Determinant, float] = {det: float(coeff)
                                          for det, coeff in zip(las_dets, las_coeffs)}
    n_overlap = 0
    for det in r_dets:
        if det in combined:
            # Should not happen: R is dispersion/CT by construction, the top-K LAS
            # terms are entirely same-sector. Not silently dropped or double-counted.
            n_overlap += 1
            continue
        combined[det] = 0.0

    dets = list(combined.keys())
    initial_guess = np.asarray([combined[d] for d in dets], dtype=float)
    return {
        "dets": dets, "initial_guess": initial_guess,
        "n_las": len(las_dets), "n_recovery": len(r_dets), "n_overlap": n_overlap,
        "n_work": len(dets), "classification": classification, "las_top_k": int(las_top_k),
    }


def solve_free_orbital_union(state: dict, dets, initial_guess: np.ndarray, *,
                             orbital_opt_max_iter: int = 20,
                             orbital_opt_grad_tol: float = 1e-6,
                             orbital_opt_energy_tol: float = 1e-10,
                             gate_tol: float = 1e-6) -> dict:
    """Solve the fixed-support ground state, free-rotate orbitals across every
    orbital pair (fragment boundaries included) via BFGS, and independently
    rediagonalize -- never trusting a native coefficient array directly, see module
    docstring. Raises ``AssertionError`` if the rotated energy exceeds the unrotated
    one beyond ``gate_tol``: a fixed-support orbital optimization must never raise
    the energy, and this is never loosened to hide a genuine violation.
    """
    n_orb = int(state["n_orb"])
    dets_arr = np.asarray([tuple(map(int, d)) for d in dets], dtype=np.int64)

    before = _fixed_support_ground_state(state["h1"], state["eri"], dets_arr,
                                         initial_guess, n_orb)
    e_before = float(before.energy) + float(state["e_const"])
    before_dets = [tuple(map(int, d)) for d in before.determinants]
    before_coeffs = np.asarray(before.coefficients, dtype=float)

    allowed_pairs = list(itertools.combinations(range(n_orb), 2))
    coo = run_fixed_support_constrained_coo(
        state, before_dets, before_coeffs, allowed_orbital_pairs=allowed_pairs,
        orbital_opt_max_iter=int(orbital_opt_max_iter),
        orbital_opt_grad_tol=float(orbital_opt_grad_tol),
        orbital_opt_energy_tol=float(orbital_opt_energy_tol))

    after_dets = np.asarray([tuple(map(int, d)) for d in coo["dets"]], dtype=np.int64)
    # CRITICAL (see module docstring): never trust coo["coeffs"] paired with
    # coo["dets"]/coo["h1_final"]/coo["eri_final"] for an energy check.
    # Independently rediagonalize at the rotated frame instead.
    after = _fixed_support_ground_state(
        coo["h1_final"], coo["eri_final"], after_dets,
        np.asarray(coo["coeffs"], dtype=float), n_orb)
    e_after = float(after.energy) + float(state["e_const"])

    gain_mha = (e_before - e_after) * 1000.0
    gate_passed = bool(e_after <= e_before + gate_tol)
    if not gate_passed:
        raise AssertionError(
            f"free-orbital COO gate failed: E_after ({e_after:.10f} Ha) exceeds "
            f"E_before ({e_before:.10f} Ha) by {(e_after - e_before) * 1000:.4f} mHa, "
            f"beyond tolerance {gate_tol:.0e} Ha -- a fixed-support orbital "
            "optimization must never raise the energy; do not loosen this tolerance "
            "to hide a genuine violation")

    return {
        "e_before": e_before, "e_after": e_after, "gain_mha": gain_mha,
        "n_dets": len(after_dets), "dets": [tuple(map(int, d)) for d in after_dets],
        "coefficients": np.asarray(after.coefficients, dtype=float),
        "h1_final": coo["h1_final"], "eri_final": coo["eri_final"],
        "cross_fragment_rotation_norm": coo["cross_fragment_rotation_norm"],
        "rotation_errors": coo["rotation_errors"],
        "orbital_opt_max_iter": int(orbital_opt_max_iter),
        "gates": {"E_after_le_E_before": gate_passed},
    }
