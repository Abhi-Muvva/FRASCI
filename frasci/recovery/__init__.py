"""Fixed-parent-frame inter-fragment FRASCI recovery.

``solve()`` is the public protocol: complete contracted Parent -> corrected A-prime
recovery at ONE user-selected ``RecoveryConfig.recovery_target`` -> two independent
optional refinements, an Expanded-Parent diagnostic (``expand_parent_k``) and a
post-recovery COO refinement (``run_post_recovery_coo``). It performs ONE direct
recovery calculation and never
makes an automatic stopping decision -- a sequence of targets, a plateau rule, and
campaign checkpointing belong to the separate ``frasci.recovery.benchmark_ladder``
driver (used for the project's own all-molecule recovery-limit study), not to this
module. Recovery does not perform Phase-0 discovery, doorway admission, or a fragment
rebuild; orbitals move only during the optional COO step, after growth and the
generalized solve, never during them. The exact union final solve remains available as
an independently gated output on every call.
"""
from __future__ import annotations

import time
from math import prod
from pathlib import Path

import numpy as np

from frasci.recovery._config import RecoveryConfig, Result
from frasci.recovery._expand_parent import compute_expanded_parent
from frasci.recovery._free_coo import build_top_k_union, solve_free_orbital_union
from frasci.recovery._growth import analyze_rung_spin, grow_recovery_target
from frasci.recovery._las_reconstruct import certify_las_reconstruction
from frasci.recovery._solve import solve_recovery_point
from frasci.recovery._spin import s_squared_expectation
from frasci.recovery._state import build_contracted_reference, top_product_terms
from frasci.recovery._union_solve import certify_final_union, form_final_union, solve_union_davidson

__all__ = ["solve", "RecoveryConfig", "Result"]


def solve(parent_dir: Path, bundle, config: RecoveryConfig) -> Result:
    """Run the validated public protocol: Parent + ONE corrected A-prime recovery
    calculation at ``config.recovery_target``, with an optional post-recovery COO
    step."""
    started = time.time()
    state = build_contracted_reference(parent_dir, bundle)
    parent_convergence_status = state.get("convergence_status")
    # A confirmed stationary_plateau (RDM-stable, extension-confirmed -- see
    # frasci.parent._convergence) is accepted like strict convergence: the whole point of
    # ParentConfig.tail_convergence is to certify "good enough to build on" short of the
    # strict conv_tol_grad threshold. A *candidate* plateau (not yet confirmed) or any other
    # non-strict status still needs the explicit opt-out below.
    parent_accepted = state["parent_converged"] or parent_convergence_status == "stationary_plateau"
    if not parent_accepted and not config.allow_nonconverged_parent:
        raise ValueError(
            "frasci.recovery.solve() refuses to grow, generalized-solve, union-solve, or "
            "free-COO-solve on top of a parent recorded as converged=False and not a "
            f"confirmed stationary_plateau (convergence_status={parent_convergence_status!r}) "
            "-- Fe2S2 and Fe4S4 pilot runs on an unconverged parent produced multi-Ha "
            "'recovery' energies that were artifacts of the non-convergence, not genuine "
            "physics. Pass RecoveryConfig(allow_nonconverged_parent=True) only for an "
            "explicitly labeled diagnostic -- every Result it produces will carry "
            "diagnostic_only=True and parent_converged=False.")
    if config.expand_parent_k is not None:
        expand_parent_result = compute_expanded_parent(state, config.expand_parent_k)
    else:
        expand_parent_result = None

    seed_dets, seed_coeffs = top_product_terms(state, config.las_seed_dets)
    n_alpha, n_beta = int(state["n_elec"][0]), int(state["n_elec"][1])

    growth = grow_recovery_target(state, seed_dets, seed_coeffs, config, parent_dir=parent_dir)
    point = growth["point"]

    solved = solve_recovery_point(
        state, point["dets"], point["coeffs"],
        n_ct_max=config.n_ct_max,
        solver_kind=config.solver_kind,
        dense_dimension_threshold=config.dense_dimension_threshold,
        max_memory_gb=config.max_memory_gb,
    )
    spin_info = analyze_rung_spin(point, state, n_alpha=n_alpha, n_beta=n_beta)

    las_certificate = certify_las_reconstruction(
        state, tol=config.las_reconstruction_tol)
    union = form_final_union(
        state, point["dets"], point["coeffs"], n_ct_max=config.n_ct_max)
    union_solved = solve_union_davidson(
        state, union["dets"], union["initial_guess"])
    union_certificate = certify_final_union(
        union_solved, state["h00"], tol=config.union_energy_tol)

    if config.run_post_recovery_coo:
        free_coo_union = build_top_k_union(
            state, config.las_top_k, point["dets"], point["coeffs"],
            n_ct_max=config.n_ct_max)
        free_coo_solved = solve_free_orbital_union(
            state, free_coo_union["dets"], free_coo_union["initial_guess"],
            orbital_opt_max_iter=config.free_coo_orbital_opt_max_iter)
        coo_diagnostics = {
            "produced": True,
            "las_top_k": free_coo_union["las_top_k"],
            "n_las": free_coo_union["n_las"],
            "n_recovery": free_coo_union["n_recovery"],
            "n_overlap": free_coo_union["n_overlap"],
            "matched_support_energy_before_ha": free_coo_solved["e_before"],
            "energy_after_ha": free_coo_solved["e_after"],
            "gain_mha": free_coo_solved["gain_mha"],
            "orbital_opt_max_iter": free_coo_solved["orbital_opt_max_iter"],
            "cross_fragment_rotation_norm": free_coo_solved["cross_fragment_rotation_norm"],
            "rotation_errors": free_coo_solved["rotation_errors"],
            "gates": free_coo_solved["gates"],
        }
    else:
        free_coo_solved = None
        coo_diagnostics = {
            "produced": False,
            "reason": (
                "run_post_recovery_coo=False (default): the public protocol is complete "
                "contracted Parent + corrected A-prime recovery only. COO is retained and "
                "available -- pass RecoveryConfig(run_post_recovery_coo=True) to run it "
                "once, on this recovery_target's own fixed support."),
        }

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

    # Spin reporting (see Result.spin_square docstring): discarding the "owned"-space
    # spin-exchange partners of the recovery support would NOT make it spin-closed --
    # a one-dimensional contracted Parent cannot span those owned partners, so this
    # release reports a diagnostic-only orbit analysis instead of claiming closure, and
    # (when computable) a PARTIAL S^2 of the explicit recovery block alone, explicitly
    # labeled non-certifying. Neither is ever promoted to Result.spin_square.
    explicit_block_spin_square_partial = None
    explicit_block_spin_square_note = (
        "no recovery determinants in this rung; nothing to evaluate")
    if solved["explicit_determinants"]:
        recovery_block_coeffs = np.asarray(solved["coefficients"][1:], dtype=float)
        block_norm = float(np.linalg.norm(recovery_block_coeffs))
        if block_norm > 0:
            try:
                explicit_block_spin_square_partial = s_squared_expectation(
                    solved["explicit_determinants"], recovery_block_coeffs / block_norm,
                    n_alpha=n_alpha, n_beta=n_beta)
                explicit_block_spin_square_note = (
                    "NON-CERTIFYING diagnostic: explicit recovery-determinant block "
                    "only, renormalized -- excludes the contracted Parent block and "
                    "Parent-determinant cross terms, so this is NOT a certification of "
                    "the complete hybrid wavefunction's spin state")
            except (ValueError, NotImplementedError) as error:
                explicit_block_spin_square_note = f"unavailable: {error}"
        else:
            explicit_block_spin_square_note = "recovery block has zero weight"

    return Result(
        hybrid_produced=True,
        energy=float(solved["e_tot"]),
        n_dets=int(solved["n_recovery"]),
        converged=bool(all(solved["gates"].values())),
        gates=solved["gates"],
        h00=float(state["h00"]),
        wall_s=time.time() - started,
        diagnostics={
            "protocol": "complete_contracted_parent_plus_corrected_a_prime_recovery",
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
            "growth": {
                "requested_n": config.recovery_target,
                "native_actual_n": point["n_recovery"],
                "target_met": point["target_met"],
                "termination_reason": point["termination_reason"],
                "checkpoint_status": growth["checkpoint_status"],
            },
            "spin": {
                **spin_info,
                "explicit_block_spin_square_partial": explicit_block_spin_square_partial,
                "explicit_block_spin_square_note": explicit_block_spin_square_note,
                "full_wavefunction_spin_square": None,
            },
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
            "coo": coo_diagnostics,
            "expand_parent": (
                {"produced": True, **expand_parent_result} if expand_parent_result else
                {"produced": False, "reason": (
                    "expand_parent_k=None (default): the public protocol is complete "
                    "contracted Parent + corrected A-prime recovery only. The Parent-"
                    "expansion diagnostic is retained and available -- pass "
                    "RecoveryConfig(expand_parent_k=<K>) to run it once, independent of "
                    "this recovery_target's own growth and solve.")}
            ),
        },
        union_produced=True,
        union_energy=union_solved["e_final"],
        union_n_dets=len(union_solved["dets"]),
        union_dets=union_solved["dets"],
        union_coefficients=union_solved["coefficients"],
        union_residual_norm=union_solved["residual_norm"],
        free_coo_produced=bool(config.run_post_recovery_coo),
        free_coo_energy=(free_coo_solved["e_after"] if free_coo_solved else None),
        free_coo_n_dets=(free_coo_solved["n_dets"] if free_coo_solved else None),
        free_coo_gain_mha=(free_coo_solved["gain_mha"] if free_coo_solved else None),
        expand_parent_produced=bool(expand_parent_result is not None),
        expand_parent_k_requested=(config.expand_parent_k if expand_parent_result else None),
        expand_parent_n_dets=(expand_parent_result["actual_n"] if expand_parent_result else None),
        expand_parent_energy=(expand_parent_result["energy_ha"] if expand_parent_result else None),
        expand_parent_gain_mha=(expand_parent_result["gain_mha"] if expand_parent_result else None),
        spin_square=None,
        parent_converged=bool(state["parent_converged"]),
        parent_convergence_status=parent_convergence_status,
        diagnostic_only=bool(not parent_accepted),
    )
