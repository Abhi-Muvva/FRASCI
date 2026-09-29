"""Fixed-parent-frame inter-fragment FRASCI recovery.

``solve()`` grows a native-filtered determinant space in the parent's existing
orbital frame and solves the variational ``{Phi_LAS, R}`` problem. Recovery does
not perform Phase-0 discovery, orbital optimization, doorway admission, or a
fragment rebuild. The exact union and top-K free-orbital COO final solves remain
available as independently gated outputs.
"""
from __future__ import annotations

import time
from math import prod
from pathlib import Path

import numpy as np

from frasci.recovery._config import RecoveryConfig, Result
from frasci.recovery._free_coo import build_top_k_union, solve_free_orbital_union
from frasci.recovery._las_reconstruct import certify_las_reconstruction
from frasci.recovery._native_stream import run_fixed_filtered_ladder
from frasci.recovery._solve import solve_recovery_point
from frasci.recovery._state import build_contracted_reference, top_product_terms
from frasci.recovery._union_solve import certify_final_union, form_final_union, solve_union_davidson

__all__ = ["solve", "RecoveryConfig", "Result"]


def solve(parent_dir: Path, bundle, config: RecoveryConfig) -> Result:
    """Run the validated fixed-frame recovery protocol."""
    started = time.time()
    state = build_contracted_reference(parent_dir, bundle)
    if not state["parent_converged"] and not config.allow_nonconverged_parent:
        raise ValueError(
            "frasci.recovery.solve() refuses to grow, generalized-solve, union-solve, or "
            "free-COO-solve on top of a parent recorded as converged=False -- Fe2S2 and "
            "Fe4S4 pilot runs on an unconverged parent produced multi-Ha 'recovery' "
            "energies that were artifacts of the non-convergence, not genuine physics. "
            "Pass RecoveryConfig(allow_nonconverged_parent=True) only for an explicitly "
            "labeled diagnostic -- every Result it produces will carry "
            "diagnostic_only=True and parent_converged=False.")
    seed_dets, seed_coeffs = top_product_terms(state, config.las_seed_dets)
    point = run_fixed_filtered_ladder(
        state, seed_dets, seed_coeffs,
        recovery_targets=(config.recovery_target,),
        n_ct_max=config.n_ct_max,
        threshold=config.threshold,
        growth_factor=config.growth_factor,
        max_rounds=config.max_rounds,
    )[0]

    solved = solve_recovery_point(
        state, point["dets"], point["coeffs"],
        n_ct_max=config.n_ct_max,
        solver_kind=config.solver_kind,
        dense_dimension_threshold=config.dense_dimension_threshold,
        max_memory_gb=config.max_memory_gb,
    )

    las_certificate = certify_las_reconstruction(
        state, tol=config.las_reconstruction_tol)
    union = form_final_union(
        state, point["dets"], point["coeffs"], n_ct_max=config.n_ct_max)
    union_solved = solve_union_davidson(
        state, union["dets"], union["initial_guess"])
    union_certificate = certify_final_union(
        union_solved, state["h00"], tol=config.union_energy_tol)

    free_coo_union = build_top_k_union(
        state, config.las_top_k, point["dets"], point["coeffs"],
        n_ct_max=config.n_ct_max)
    free_coo_solved = solve_free_orbital_union(
        state, free_coo_union["dets"], free_coo_union["initial_guess"],
        orbital_opt_max_iter=config.free_coo_orbital_opt_max_iter)

    fragment_support_sizes = [len(fragment["dets"]) for fragment in state["frags"]]
    las_certificate_summary = {
        "fragment_support_sizes": fragment_support_sizes,
        "cartesian_product_size": int(prod(fragment_support_sizes)),
        "n_unique_dets": las_certificate["n_dets"],
        "coefficient_norm": float(np.linalg.norm(las_certificate["coeffs"])),
        "e_reconstructed": las_certificate["e_reconstructed"],
        "h00": las_certificate["h00"],
        "abs_energy_difference": las_certificate["diff"],
        "tolerance": las_certificate["tol"],
        "passed": las_certificate["passed"],
    }

    return Result(
        hybrid_produced=True,
        energy=float(solved["e_tot"]),
        n_dets=int(solved["n_recovery"]),
        converged=bool(all(solved["gates"].values())),
        gates=solved["gates"],
        h00=float(state["h00"]),
        wall_s=time.time() - started,
        diagnostics={
            "protocol": "fixed_frame_native_filtered",
            "recovery_target": config.recovery_target,
            "target_met": point["target_met"],
            "n_protected": point["n_protected_original"],
            "n_recovery": point["n_recovery"],
            "n_work": point["n_work"],
            "classification": point["classification"],
            "filtered_composition": point["native_composition"],
            "filtered_telemetry": point["native_telemetry"],
            "termination_reason": point["termination_reason"],
            "frame_errors": point["frame_errors"],
            "solver_kind": solved["solver_kind"],
            "generalized_residual_norm": solved["generalized_residual_norm"],
            "recovery_mha": (float(state["h00"]) - float(solved["e_tot"])) * 1000.0,
            "protected_seed_dets": [tuple(map(int, det)) for det in seed_dets],
            "las_reconstruction_certificate": las_certificate_summary,
            "union_n_las": union["n_las"],
            "union_n_recovery": union["n_recovery"],
            "union_n_overlap": union["n_overlap"],
            "union_n_work": union["n_work"],
            "union_classification": union["classification"],
            "union_certificate": union_certificate,
            "union_n_candidates_tried": union_solved["n_candidates_tried"],
            "union_winner_start": union_solved["winner_start"],
            "union_start_diagnostics": union_solved["start_diagnostics"],
            "union_e_electronic_matrix_free": union_solved["e_electronic"],
            "union_n_iters": union_solved["n_iters"],
            "union_residual_norm_native": union_solved["residual_norm_native"],
            "free_coo_las_top_k": free_coo_union["las_top_k"],
            "free_coo_n_las": free_coo_union["n_las"],
            "free_coo_n_recovery": free_coo_union["n_recovery"],
            "free_coo_n_overlap": free_coo_union["n_overlap"],
            "free_coo_e_before": free_coo_solved["e_before"],
            "free_coo_orbital_opt_max_iter": free_coo_solved["orbital_opt_max_iter"],
            "free_coo_cross_fragment_rotation_norm": (
                free_coo_solved["cross_fragment_rotation_norm"]),
            "free_coo_rotation_errors": free_coo_solved["rotation_errors"],
            "free_coo_gates": free_coo_solved["gates"],
        },
        union_produced=True,
        union_energy=union_solved["e_final"],
        union_n_dets=len(union_solved["dets"]),
        union_dets=union_solved["dets"],
        union_coefficients=union_solved["coefficients"],
        union_residual_norm=union_solved["residual_norm"],
        free_coo_produced=True,
        free_coo_energy=free_coo_solved["e_after"],
        free_coo_n_dets=free_coo_solved["n_dets"],
        free_coo_gain_mha=free_coo_solved["gain_mha"],
        parent_converged=bool(state["parent_converged"]),
        diagnostic_only=bool(not state["parent_converged"]),
    )
