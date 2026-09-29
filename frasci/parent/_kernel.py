"""The phased LASSCF-COO fragment kernel and its large-fragment aufbau
init-guess patch (Layer A). Faithful, verbatim-preserving rewrite of
engine/lasscf/coo_kernel.py::make_phased_coo_trimci_kernel_for_fragment and
engine/lasscf/runners/run_lasscf_coo.py::_aufbau_init_rdm12s /
_patch_large_fragment_aufbau -- new module boundary, same numerics, per the
same "don't vary correctness-critical numerical code just to look more
rewritten" principle established in Stage 2 (see that plan's Self-Review).

build_coo_config/suppress_fd (frasci._trimci_config) and make_selected_rdm12s
(frasci._selected_ci_rdm) are now frasci-internal; trimci itself remains a direct
third-party call (plumbing, not rewritten) -- same Rewrite Scope Ruling as Stage 2.
"""
from __future__ import annotations

import math
import sys
import time
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from frasci._trimci_config import build_coo_config, suppress_fd
from frasci._selected_ci_rdm import make_selected_rdm12s

import trimci


def _nearest_orthogonal(matrix):
    """Return the polar factor nearest to ``matrix`` in Frobenius norm."""
    left, _singular_values, right_t = np.linalg.svd(np.asarray(matrix))
    return left @ right_t


def _fixed_support_ground_state(h1, eri, determinants, coefficients, n_orb: int):
    """Rediagonalize a fixed determinant support without running selection.

    The recovery rebuild uses this after its first fragment solve so small capped
    supports cannot jump between nearly degenerate determinant sets on successive
    RDM-JK iterations. Compact supports are diagonalized exactly; larger supports
    retain two Davidson starts as a ground-root safeguard. The returned energy and
    residual are independently checked by matrix-free multiplication rather than
    trusted from a native result scalar.
    """
    dets = np.asarray(determinants)
    if dets.ndim != 2 or dets.shape[1] != 2 or len(dets) == 0:
        raise ValueError(
            "fixed-support rediagonalization requires a nonempty (n_dets, 2) "
            f"determinant array, got {dets.shape}")
    if n_orb > 64:
        raise NotImplementedError(
            "fixed-support rebuild rediagonalization currently supports at most 64 "
            f"orbitals per fragment, got {n_orb}")

    alpha = np.ascontiguousarray(dets[:, 0], dtype=np.uint64)
    beta = np.ascontiguousarray(dets[:, 1], dtype=np.uint64)
    h1c = np.ascontiguousarray(h1, dtype=float)
    eric = np.ascontiguousarray(eri, dtype=float).ravel()
    previous = np.asarray(coefficients, dtype=float).reshape(-1).copy()
    if len(previous) != len(dets) or not np.all(np.isfinite(previous)):
        raise ValueError("fixed-support coefficients must be finite and match determinants")
    norm = float(np.linalg.norm(previous))
    if norm == 0.0:
        raise ValueError("fixed-support coefficients must have nonzero norm")
    previous /= norm

    fe = trimci.trimci_core.fast_expansion

    # For the deliberately compact rebuilds this option targets (p10 is ten
    # determinants per nontrivial fragment), construct the tiny matrix exactly.
    # This proves the lowest root and avoids accepting/rejecting an iterative
    # Davidson result based on two slightly different residual implementations.
    if len(dets) <= 256:
        columns = []
        for index in range(len(dets)):
            basis = np.zeros(len(dets), dtype=float)
            basis[index] = 1.0
            columns.append(
                fe.matvec_matfree(alpha, beta, basis, h1c, eric, int(n_orb)))
        hamiltonian = np.column_stack(columns)
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        coeffs = np.asarray(eigenvectors[:, 0], dtype=float)
        energy = float(eigenvalues[0])
        sigma = fe.matvec_matfree(alpha, beta, coeffs, h1c, eric, int(n_orb))
        residual = float(np.linalg.norm(sigma - energy * coeffs))
        if residual >= 1e-10:
            raise RuntimeError(
                f"fixed-support dense residual {residual:.3e} exceeds 1e-10")
        return SimpleNamespace(
            n_dets=len(dets), determinants=dets.copy(), coefficients=coeffs,
            energy=energy, energy_pt2=None, orbital_rotation=np.eye(n_orb),
            final_h1=h1c.copy(), final_eri=np.asarray(eri, dtype=float).copy(),
            energies=SimpleNamespace(history=[energy]),
            fixed_support_solver="dense_exact", fixed_support_davidson_iters=0,
            fixed_support_residual=residual,
        )

    def run(guess):
        params = fe.DavidsonParams()
        params.verbose = 0
        params.residual_tol = 1e-10
        # energy_tol=0.0, not a small-but-nonzero value: native Davidson stops on
        # EITHER residual<residual_tol OR |dE|<energy_tol, whichever fires first.
        # A nonzero energy_tol (even 1e-12) can let the energy trajectory plateau
        # and stop the iteration before the residual has actually tightened --
        # verified directly: on a real 300-det H12 fragment with delta_h_rel~1e-4,
        # energy_tol=1e-12 stopped at residual~3.6e-7 in 6 iterations, while
        # energy_tol=0.0 (residual-only stopping) reached residual~6.4e-11 in 16
        # iterations on the SAME problem. This function's own residual gate below
        # is independent of and unaffected by this -- it is not being loosened.
        params.energy_tol = 0.0
        result = fe.davidson_solve_matfree(
            alpha, beta, h1c, eric, int(n_orb), params, guess)
        if not result.converged:
            return None
        coeffs = np.asarray(result.eigenvectors, dtype=float).reshape(-1)
        coeffs /= np.linalg.norm(coeffs)
        sigma = fe.matvec_matfree(alpha, beta, coeffs, h1c, eric, int(n_orb))
        energy = float(coeffs @ sigma)
        residual = float(np.linalg.norm(sigma - energy * coeffs))
        return energy, residual, int(result.n_iters), coeffs

    candidates = [
        candidate for candidate in (
            # One-root guess: native contract is initial_guess.shape == (n_states, N);
            # for n_states=1 this is (1, N), not (N, 1). Empirically the two shapes
            # gave bit-identical results in direct testing against this same binding
            # (both share the same contiguous buffer for a single root, so no behavior
            # change was observed here) -- kept as (1, N) regardless, to match the
            # documented contract rather than rely on an accidental degenerate-shape
            # equivalence.
            run(np.ascontiguousarray(previous).reshape(1, -1)),
            run(np.empty((0, 0))),
        ) if candidate is not None
    ]
    if not candidates:
        raise RuntimeError(
            f"fixed-support Davidson failed for {len(dets)} determinants")
    energy, residual, n_iters, coeffs = min(candidates, key=lambda item: item[0])
    if residual >= 1e-8:
        raise RuntimeError(
            f"fixed-support Davidson residual {residual:.3e} exceeds 1e-8")

    return SimpleNamespace(
        n_dets=len(dets), determinants=dets.copy(), coefficients=coeffs,
        energy=energy, energy_pt2=None, orbital_rotation=np.eye(n_orb),
        final_h1=h1c.copy(), final_eri=np.asarray(eri, dtype=float).copy(),
        energies=SimpleNamespace(history=[energy]),
        fixed_support_solver="davidson_two_start",
        fixed_support_davidson_iters=n_iters,
        fixed_support_residual=residual,
    )


def _run_expansion_continuation(h1, eri, determinants, coefficients, n_orb: int, *,
                                phase_index: int, threshold: float, growth_factor: float,
                                max_n_dets: int, residual_tol: float = 1e-6,
                                orbital_optimization: bool = False,
                                orbital_opt_max_iter: int = 20):
    """Genuine direct Phase-1-only or Phase-2-only continuation from a cached
    determinant support, via TrimCI's lower-level
    ``trimci.trimci_core.fast_expansion.run_expansion_phased`` -- NOT
    ``trimci.ground_state_from_integrals`` (which always runs Phase 0's stochastic
    search first; see ``make_phased_kernel``'s docstring) and NOT a fresh Phase-0
    restart. This is the same native binding
    ``frasci.recovery._native_stream.py`` already drives for its own Phase-1/2
    continuation (there, sector-constrained; here, a single unfiltered fragment
    support), applied with a single ``ExpansionConfig`` so exactly one phase runs.

    Phase 1 may perform BFGS orbital optimization; Phase 2 must remain frozen.  The
    returned step rotation and optimized integrals are kept together, and the
    independent Rayleigh/residual check is evaluated in that returned frame.

    The returned energy is an independently recomputed Rayleigh quotient via
    ``matvec_matfree`` (never a native scalar taken on trust, matching this
    project's established distrust of stale native energy scalars), and the
    independently recomputed residual must clear ``residual_tol`` or this function
    raises -- a native ``result`` is never accepted on its own say-so alone.

    Returns a ``SimpleNamespace`` shaped like the fields of ``trimci.Result`` that
    ``make_phased_kernel``'s downstream RDM/back-rotation code actually reads:
    ``n_dets``, ``determinants`` ``(n_dets, 2)``, ``coefficients``, ``energy``,
    ``energy_pt2=None``, ``orbital_rotation=eye(n_orb)``, ``energies.history``.
    """
    if n_orb > 64:
        raise NotImplementedError(
            f"n_orb={n_orb} exceeds 64 -- run_expansion_phased's 64-bit alpha/beta "
            "binding covers orbitals 0-63 exactly; no wider Phase-1/2 expansion "
            "binding is wired into this adapter (frasci.recovery._native_stream.py's "
            "128-bit path exists but is not used here).")

    fe = trimci.trimci_core.fast_expansion
    dets_in = np.asarray(determinants)
    alpha = np.ascontiguousarray(dets_in[:, 0], dtype=np.uint64)
    beta = np.ascontiguousarray(dets_in[:, 1], dtype=np.uint64)
    coeffs_in = np.ascontiguousarray(coefficients, dtype=float)
    h1c = np.ascontiguousarray(h1, dtype=float)
    eric = np.ascontiguousarray(eri, dtype=float).ravel()

    cfg = fe.ExpansionConfig()
    cfg.verbose = 0
    cfg.orbital_optimization = bool(orbital_optimization)
    cfg.orbital_opt_max_iter = int(orbital_opt_max_iter)
    cfg.threshold = float(threshold)
    cfg.growth_factor = float(growth_factor)
    cfg.max_n_dets = int(max_n_dets)
    cfg.phase_index = int(phase_index)
    # Same residual-only-stopping fix as _fixed_support_ground_state's Davidson
    # calls, applied to this expansion's own internal per-round Davidson solves.
    cfg.davidson_params.energy_tol = 0.0

    result = fe.run_expansion_phased(
        alpha, beta, coeffs_in, h1c, eric, int(n_orb), [cfg])

    U = np.asarray(result.get_U_total(), dtype=float)
    h1_out = np.asarray(result.get_h1_optimized(), dtype=float)
    eri_out = np.asarray(result.get_eri_optimized(), dtype=float).reshape((n_orb,) * 4)
    if phase_index == 2 and not np.allclose(U, np.eye(n_orb), atol=1e-10):
        raise RuntimeError(
            "Phase 2 run_expansion_phased returned a non-identity orbital rotation "
            f"(max|U-I|={np.max(np.abs(U - np.eye(n_orb))):.3e}) despite "
            "orbital_optimization=False -- refusing to combine its coefficients "
            "with the caller's un-rotated integrals (a genuine frame mismatch)")
    h1_err = float(np.max(np.abs(h1_out - U.T @ h1c @ U)))
    eri_err = float(np.max(np.abs(eri_out - _rotate_eri(np.asarray(eri), U))))
    if h1_err > 1e-8 or eri_err > 1e-8:
        raise RuntimeError(
            "run_expansion_phased integral/rotation contract failed: "
            f"h1_err={h1_err:.3e}, eri_err={eri_err:.3e}")

    fa, fb = result.get_alpha_beta()
    fa = np.ascontiguousarray(fa, dtype=np.uint64)
    fb = np.ascontiguousarray(fb, dtype=np.uint64)
    new_coeffs = np.asarray(result.get_coefficients(), dtype=float)
    norm = float(np.linalg.norm(new_coeffs))
    new_coeffs = new_coeffs / norm

    out_dets = np.stack([fa.astype(np.int64), fb.astype(np.int64)], axis=1)
    # Expansion's own final vector can be looser than the parent residual gate
    # after BFGS (real H4 moderate-drift case: ~1e-5).  Selection/orbopt has
    # already done its job; certify the returned support by the same tight
    # dense/Davidson rediagonalization used for ordinary fixed-support calls.
    refined = _fixed_support_ground_state(
        h1_out, eri_out, out_dets, new_coeffs, n_orb)
    new_coeffs = np.asarray(refined.coefficients)
    energy = float(refined.energy)
    residual = float(refined.fixed_support_residual)
    if residual >= residual_tol:
        raise RuntimeError(
            f"phase-{phase_index} direct continuation residual {residual:.3e} "
            f"exceeds {residual_tol:.1e} (independently recomputed via "
            "matvec_matfree, not the native result taken on trust)")

    history = list(getattr(result, "energy_history", []) or []) or [energy]
    return SimpleNamespace(
        n_dets=len(out_dets), determinants=out_dets, coefficients=new_coeffs,
        energy=energy, energy_pt2=None, orbital_rotation=U,
        final_h1=h1_out, final_eri=eri_out,
        energies=SimpleNamespace(history=history),
        expansion_termination_reason=str(getattr(result, "termination_reason", None)),
        expansion_residual=residual,
        expansion_final_solver=refined.fixed_support_solver,
    )


_RDM_VALIDITY_FIELDS = (
    "rdm_all_finite",
    "rdm_trace_alpha_abs", "rdm_trace_beta_abs",
    "rdm_dm1a_hermiticity_max_abs", "rdm_dm1b_hermiticity_max_abs",
    "rdm_dm2aa_hermiticity_max_abs", "rdm_dm2aa_exchange_max_abs",
    "rdm_dm2ab_hermiticity_max_abs",
    "rdm_dm2bb_hermiticity_max_abs", "rdm_dm2bb_exchange_max_abs",
    "back_rotation_energy_mismatch_Ha", "back_rotation_tolerance_Ha",
    "back_rotation_passed", "res_energy_mismatch_Ha",
)


def _null_rdm_validity_fields() -> dict:
    return {key: None for key in _RDM_VALIDITY_FIELDS}


def _bitstring_entry(det, coeff: float, n_orb: int) -> list:
    alpha, beta = int(det[0]), int(det[1])
    occ_a = " ".join(str(i) for i in range(n_orb) if (alpha >> i) & 1)
    occ_b = " ".join(str(i) for i in range(n_orb) if (beta >> i) & 1)
    return [float(coeff), f"{occ_a},{occ_b}"]


def _finite_float_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _raw_rdm_validity(dm1a, dm1b, dm2aa, dm2ab, dm2bb, *, na, nb,
                      h1, h2, reference_energy, invariance_atol):
    tensors = (dm1a, dm1b, dm2aa, dm2ab, dm2bb)
    all_finite = all(bool(np.all(np.isfinite(x))) for x in tensors)

    def residual(lhs, rhs):
        return _finite_float_or_none(np.max(np.abs(lhs - rhs)))

    fields = {
        "rdm_all_finite": all_finite,
        "rdm_trace_alpha_abs": _finite_float_or_none(abs(np.trace(dm1a) - na)),
        "rdm_trace_beta_abs": _finite_float_or_none(abs(np.trace(dm1b) - nb)),
        "rdm_dm1a_hermiticity_max_abs": residual(dm1a, dm1a.T),
        "rdm_dm1b_hermiticity_max_abs": residual(dm1b, dm1b.T),
        "rdm_dm2aa_hermiticity_max_abs": residual(
            dm2aa, dm2aa.transpose(1, 0, 3, 2)),
        "rdm_dm2aa_exchange_max_abs": residual(
            dm2aa, dm2aa.transpose(2, 3, 0, 1)),
        "rdm_dm2ab_hermiticity_max_abs": residual(
            dm2ab, dm2ab.transpose(1, 0, 3, 2)),
        "rdm_dm2bb_hermiticity_max_abs": residual(
            dm2bb, dm2bb.transpose(1, 0, 3, 2)),
        "rdm_dm2bb_exchange_max_abs": residual(
            dm2bb, dm2bb.transpose(2, 3, 0, 1)),
    }
    dm2 = dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb
    e_check = (np.einsum("pq,pq->", h1, dm1a + dm1b)
               + 0.5 * np.einsum("pqrs,pqrs->", h2, dm2))
    mismatch = _finite_float_or_none(abs(e_check - reference_energy))
    tolerance = _finite_float_or_none(
        invariance_atol * max(1.0, abs(reference_energy)))
    passed = bool(all_finite and mismatch is not None and tolerance is not None
                  and mismatch <= tolerance)
    fields.update({
        "back_rotation_energy_mismatch_Ha": mismatch,
        "back_rotation_tolerance_Ha": tolerance,
        "back_rotation_passed": passed,
    })
    return fields, dm2, float(e_check)


def _back_rotate_dm1(dm1: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ dm1 @ U.T


def _back_rotate_dm2(dm2: np.ndarray, U: np.ndarray) -> np.ndarray:
    out = np.einsum("pa,abcd->pbcd", U, dm2, optimize=True)
    out = np.einsum("qb,pbcd->pqcd", U, out, optimize=True)
    out = np.einsum("rc,pqcd->pqrd", U, out, optimize=True)
    out = np.einsum("sd,pqrd->pqrs", U, out, optimize=True)
    return out


def _rotate_eri(eri: np.ndarray, U: np.ndarray) -> np.ndarray:
    out = np.einsum("pa,pqrs->aqrs", U, eri, optimize=True)
    out = np.einsum("qb,aqrs->abrs", U, out, optimize=True)
    out = np.einsum("rc,abrs->abcs", U, out, optimize=True)
    out = np.einsum("sd,abcs->abcd", U, out, optimize=True)
    return out


def aufbau_init_rdm12s(norb, nelec):
    """1- and 2-RDMs of the aufbau determinant (orbitals 0..na-1 alpha,
    0..nb-1 beta occupied), in pyscf make_rdm12s + mrh lasscf_rdm _ci2rdm
    convention. Built analytically -- no CSF transformer or dense CI vector
    is allocated, which is the point (a large near-half-filled sector
    overflows int32 on the CSF path). Bit-exact against pyscf, per the
    existing (untouched) tests/test_aufbau_init_rdm.py.

    Returns (dm1s, dm2): dm1s is (2, norb, norb); dm2 is the spin-summed
    (norb, norb, norb, norb) 2-RDM.
    """
    if isinstance(nelec, (list, tuple, np.ndarray)):
        na, nb = int(nelec[0]), int(nelec[1])
    else:
        na, nb = (int(nelec) + 1) // 2, int(nelec) // 2
    occa = np.zeros(norb); occa[:na] = 1.0
    occb = np.zeros(norb); occb[:nb] = 1.0
    dm1a, dm1b = np.diag(occa), np.diag(occb)
    dm1s = np.stack([dm1a, dm1b], axis=0)
    dm2aa = np.einsum('pq,rs->pqrs', dm1a, dm1a) - np.einsum('ps,rq->pqrs', dm1a, dm1a)
    dm2bb = np.einsum('pq,rs->pqrs', dm1b, dm1b) - np.einsum('ps,rq->pqrs', dm1b, dm1b)
    dm2ab = np.einsum('pq,rs->pqrs', dm1a, dm1b)
    dm2 = dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb
    return dm1s, dm2


def patch_large_fragment_aufbau(las, ncas_sub, nelec_per_frag, max_dense_bytes=None):
    """Swap mrh's CSF-based LASSCF init guess for the analytic aufbau RDM on
    any fragment whose dense CI vector would exceed max_dense_bytes. Two
    init-guess entry points reach the CSF machinery and both are redirected:
    the fcibox get_aufbau_guess (aufbau0) and the solver get_init_guess
    (aufbau1's per-fragment refinement, via the RDMSolver _get_init_guess
    hook). Small fragments keep mrh's original guess untouched.
    """
    if max_dense_bytes is None:
        from frasci._selected_ci_rdm import _DEFAULT_MAX_DENSE_BYTES
        max_dense_bytes = _DEFAULT_MAX_DENSE_BYTES

    def _solver_init_guess(norb_, nelec, nroots, h1s, h2):
        return aufbau_init_rdm12s(norb_, nelec)

    for i, (norb, nelec) in enumerate(zip(ncas_sub, nelec_per_frag)):
        na, nb = int(nelec[0]), int(nelec[1])
        dense_bytes = math.comb(norb, na) * math.comb(norb, nb) * 8
        if dense_bytes <= max_dense_bytes:
            continue
        box = las.fciboxes[i]

        def _aufbau_guess(norb_, nelec0, orbsym=None, _box=box):
            return [aufbau_init_rdm12s(norb_, _box._get_nelec(s, nelec0))
                    for s in _box.fcisolvers]

        box.get_aufbau_guess = _aufbau_guess
        for solver in box.fcisolvers:
            solver._get_init_guess = _solver_init_guess
        print(f"[frasci.parent] F{i}: CSF aufbau init-guess infeasible "
              f"({norb}o {na}a{nb}b, dense civec ~{dense_bytes / 1024 ** 3:.1f} GiB) "
              f"-> analytic single-determinant RDM seed")


def make_phased_kernel(
    *,
    threshold: float = 0.01,
    discovery_num_runs: int = 64,
    refine_num_runs: int = 1,
    orbopt: bool = True,
    orbopt_cycles: int = 10,
    save_fcidumps: bool = False,
    max_dets_phase0: int = 100,
    max_dets_phase1: int = 2000,
    growth_factor_phase1: float = 1.1,
    enable_phase2: bool = False,
    max_dets_phase2: int = 1_000_000,
    growth_factor_phase2: float = 2.0,
    pt2: bool = False,
    seed: int = 0,
    rediscover_tol: float = 1e-2,
    warm_start_kappa: bool = True,
    warm_start_dets: bool = False,
    freeze_support_after_discovery: bool = False,
    adaptive_phase_policy: bool = False,
    adaptive_fixed_support_tol: float = 1e-4,
    adaptive_phase0_tol: float = 1e-2,
    adaptive_phase2_after_plateau: bool = False,
    phase2_plateau_delta_e_tol: float = 1e-6,
    phase2_plateau_delta_rdm_tol: float = 1e-5,
    frame_provider=None,
    frame_leakage_tol: float = 5e-2,
    frame_strict: bool = True,
    dm_mixing_alpha: float = 1.0,
    invariance_atol: float = 1e-4,
    rdm_backend: str = "auto",
    log_callback=None,
    quiet: bool = True,
    config_overrides: Optional[dict] = None,
    phase1_state_dir: Optional[str] = None,
    fragment_idx: Optional[int] = None,
    failure_capture_dir: Optional[str] = None,
) -> Callable:
    """Return a phased COO-TrimCI fragment kernel satisfying mrh's
    lasscf_rdm.make_fcibox kernel contract: kernel(norb, nelec, h0, h1s, h2)
    -> (etot, dm1s, dm2).

    Runs TrimCI's 3-phase ground_state_from_integrals per fragment. The
    expensive stochastic Phase 0 (discovery_num_runs restarts) is amortized
    across LASSCF macro-iterations: it runs on the first call and thereafter
    only when the fragment's embedding h1 has drifted past rediscover_tol
    (relative Frobenius) since the last discovery; calls in between warm-start
    from the cached COO rotation and run cheap refinement (refine_num_runs).

    The wavefunction/RDMs come back in the COO-optimized basis and are
    back-rotated to the kernel's input basis by R = U_prev @ U_step. An
    energy-invariance check in the input basis guards this: it recomputes the
    electronic energy from the back-rotated RDMs and requires it to match a
    NATIVE-FRAME reference (the same RDMs' Rayleigh quotient against
    h1_pre/eri_pre rotated by U_step alone) within invariance_atol (relative).
    It deliberately does NOT compare against res.energy: res.energy can be
    stale relative to the (determinants, coefficients, orbital_rotation)
    TrimCI returns for that SAME Result (root-caused in the original codebase
    2026-08-19 via capture-and-replay of two independent failures -- four
    independent ways of computing the wavefunction's own energy agreed with
    each other to 1e-13 and disagreed with res.energy by the identical gap
    every time). The mismatch against res.energy is still recorded
    (non-fatal, res_energy_mismatch_Ha) for visibility.

    ``warm_start_dets`` (default off, preserves prior behavior for existing
    callers) seeds each call's Phase-0 discovery with the PREVIOUS call's own
    converged determinant/coefficient wavefunction, via the same
    ``phase0_overrides.initial_dets_dict`` mechanism ``frasci.recovery._growth``
    already uses to warm-start Phase-1 growth from a saved Phase-0. Without it,
    every call implicitly restarts its search from nothing even though the
    embedding integrals only drift slightly between outer RDM-JK/macro
    iterations -- letting TrimCI's threshold-based selection independently
    re-roll which determinants pass the cut each time, rather than refining a
    stable answer. Used alone, this remains restricted to ``orbopt=False``.
    Under ``adaptive_phase_policy`` it may also be used with ``orbopt=True``,
    but only with the external ``frame_provider`` certificate described below.

    ``freeze_support_after_discovery`` goes one step further for frozen-orbital
    RDM-JK rebuilds: after the first normal TrimCI call, later calls keep those
    determinant identities exactly and rediagonalize only their coefficients in
    the updated embedding. This removes discontinuous support replacement from
    the density fixed-point map. It requires the same fixed-basis conditions as
    ``warm_start_dets`` and is off by default for every existing parent caller.

    ``adaptive_phase_policy`` is a second, mutually-exclusive alternative to
    ``freeze_support_after_discovery`` -- the PRODUCTION episode-aware policy
    (corrected 2026-09-14; supersedes an earlier version of this dispatcher
    that chose between three actions from the CONSECUTIVE relative drift of h1
    since the immediately previous call, ``delta_h_rel``. That drift-triggered
    design was verified WRONG on a real Fe4S4 run: 165 spurious Phase0+Phase1
    calls, trapped at raw fragment-RDM changes of 20.563/26.151 -- drift
    between RDM-JK iterations of an otherwise-converging embedding routinely
    exceeded ``adaptive_phase0_tol``, so nearly every call re-triggered
    discovery instead of refining a stable support). The unit of dispatch is
    now the EXTERNAL ORBITAL-FRAME EPISODE, not the per-call drift magnitude:

    - No cached support yet (``state["dets_prev"] is None`` -- the fragment's
      true first call, or immediately after an unsafe frame transition
      atomically cleared the episode, see below) -> run the normal TrimCI
      phased solver, seeded from any previous support if one exists
      (Phase 0 restarts + Phase 1, same as the legacy discover/refine path;
      logged ``phase_policy_action="phase0_phase1"``). Caches determinants,
      coefficients, the cumulative TrimCI rotation, RDMs, and (with
      ``orbopt=True``) the external frame identifier.
    - Otherwise (a cached support exists for this episode -- whether this is
      the Nth call at an unchanged external frame, or the call right after a
      REUSABLE external frame transition transported the cache, see below)
      -> rediagonalize the cached determinant support in place
      (``_fixed_support_ground_state``, exactly as
      ``freeze_support_after_discovery`` does; logged
      ``phase_policy_action="fixed_support"`` or
      ``"post_phase2_fixed_support"``, see below). ``delta_h_rel`` is computed
      and logged every call as telemetry ONLY -- it never gates dispatch here,
      however large it is; that was the superseded behavior. This is also why
      there is no "genuine Phase-1-only" tier reachable from ordinary drift
      any more: ordinary inner embedding drift must never call
      ``_run_expansion_continuation`` on its own initiative. That helper (via
      ``trimci.trimci_core.fast_expansion.run_expansion_phased`` -- never
      ``ground_state_from_integrals``, never a fresh Phase-0 restart) is kept
      and still used directly by the Phase-2 promotion action below; it
      remains available for a future explicitly-justified caller, just not
      wired into this per-call dispatch.

    With ``orbopt=True``, every cached call requires a frame packet published by
    the parent orchestration layer. A REUSABLE external transition supplies
    ``U_align`` and the kernel forms ``U_base = U_align @ U_cached`` before
    rotating the current integrals and measuring drift; the cached determinant
    support is kept and the call above still lands in ``fixed_support`` --
    Phase 0 never reruns merely because the embedding Hamiltonian changed
    across a legitimate frame transition. Phase-1 BFGS (when it does run, from
    a genuine ``phase0_phase1`` call) gives ``U_new = U_base @ U_step``. An
    UNSAFE transition atomically drops the support, coefficients, rotations,
    drift/RDM history, and Phase-2 state and forces exactly one fresh
    Phase-0+1 call to open a new episode; every later call in that episode is
    again ``fixed_support``.

    ``adaptive_fixed_support_tol``/``adaptive_phase0_tol`` are retained for
    backward-compatible construction (existing callers/config records) but no
    longer govern dispatch for same-episode calls -- see above. They are
    validated (``adaptive_phase0_tol >= adaptive_fixed_support_tol``) but
    otherwise inert under ``adaptive_phase_policy``.

    ``adaptive_phase2_after_plateau`` (only meaningful with
    ``adaptive_phase_policy``) fires exactly one direct Phase-2-only
    continuation (via ``_run_expansion_continuation``/``run_expansion_phased``,
    never Phase 0 or Phase 1 rerun) when the caller sets
    ``state["force_phase2_next"] = True`` -- logged
    ``phase_policy_action="phase2_promote"``. This is deliberately NOT an
    autonomous trigger: an inner RDM-JK loop reaching its own plateau (small
    ``last_delta_e``/``last_delta_rdm_mixed``) must NOT promote by itself --
    only the parent's post-LASSCF-convergence finalization path
    (``frasci.parent.solve``) may request it, once outer orbital optimization
    has itself converged. A filled/closed fragment (``ci_dim <= 1``, nothing
    to promote into) treats a pending request as already satisfied instead of
    running Phase-2 as expensive no-op work: it clears
    ``force_phase2_next``/marks ``phase2_promoted_ever``/
    ``phase2_episode_complete`` without calling the continuation solver.
    Promotion is a PROMOTION, not a final answer: the enlarged support becomes
    the new persistent ``state["dets_prev"]``/``["coeffs_prev"]`` baseline
    (never shrunk back to ``max_dets_phase1`` by any later ``fixed_support``
    call -- it caps its target at ``max(max_dets_phase1_or_2,
    current_support_size)``), and the kernel enters a "post-Phase-2" tracking
    window (``state["post_phase2_active"]``): the next calls are labeled
    ``phase_policy_action="post_phase2_fixed_support"`` (mechanically
    identical to ``"fixed_support"``, distinguished only so the caller can
    tell a reconverging promoted support apart from an ordinary pre-promotion
    span). Promotion is declared converged (``post_phase2_active`` clears,
    later calls revert to the plain ``"fixed_support"`` label) only after 3
    CONSECUTIVE calls whose own ``delta_e_ha``/``delta_rdm_norm`` both clear
    the plateau tolerances -- logged ``post_phase2_converged_now=True`` on the
    call where the streak completes. A genuinely new episode
    (``phase_policy_action="phase0_phase1"``) clears all Phase-2/post-Phase-2
    tracking unconditionally: a future finalization request after it may
    promote again from scratch.
    """
    if (warm_start_dets or freeze_support_after_discovery) and not adaptive_phase_policy and (
            orbopt or warm_start_kappa):
        raise ValueError(
            "determinant warm-start/support freezing requires orbopt=False and "
            "warm_start_kappa=False -- determinant identities are only reusable when "
            "the orbital basis is guaranteed unchanged between calls")
    if freeze_support_after_discovery and not warm_start_dets:
        raise ValueError(
            "freeze_support_after_discovery requires warm_start_dets=True")
    if adaptive_phase_policy and freeze_support_after_discovery:
        raise ValueError(
            "adaptive_phase_policy and freeze_support_after_discovery are mutually "
            "exclusive phase-policy modes")
    if adaptive_phase_policy and not warm_start_dets:
        raise ValueError("adaptive_phase_policy requires warm_start_dets=True")
    if adaptive_phase_policy and orbopt and frame_provider is None:
        raise ValueError(
            "adaptive orbopt=True requires a frame_provider; cached determinants "
            "cannot be reused without an external LASSCF frame certificate")
    if adaptive_phase_policy and orbopt and not warm_start_kappa:
        raise ValueError(
            "adaptive orbopt=True requires warm_start_kappa=True so the certified "
            "cached TrimCI frame is retained")
    if frame_leakage_tol < 0:
        raise ValueError("frame_leakage_tol must be non-negative")

    state = {"U_cached": None, "h1_ref": None,
             "dm1s_prev": None, "dm2_prev": None, "h2_ref_mix": None,
             "dets_prev": None, "coeffs_prev": None,
             "e_ret_prev": None, "support_prev": None,
             "h1_prev": None, "dm1s_prev_raw": None, "dm2_prev_raw": None,
             "last_delta_e": None, "last_delta_rdm_mixed": None,
             "phase2_done_this_plateau": False,
             "post_phase2_active": False, "post_phase2_stable_streak": 0,
             "phase2_promoted_ever": False,
             "phase2_episode_complete": False,
             "force_phase2_next": False,
             "h1_prev_common": None, "h1_ref_common": None,
             "external_frame_id": None,
             "frame_leakage": None, "frame_reusable": None,
             "frame_invalidations": 0}

    invalidation_keys = (
        "U_cached", "h1_ref", "dm1s_prev", "dm2_prev", "h2_ref_mix",
        "dets_prev", "coeffs_prev", "e_ret_prev", "support_prev",
        "h1_prev", "dm1s_prev_raw", "dm2_prev_raw", "last_delta_e",
        "last_delta_rdm_mixed", "h1_prev_common", "h1_ref_common",
    )

    def _invalidate_cached_episode():
        """Drop every object whose meaning depends on the old orbital frame."""
        for key in invalidation_keys:
            state[key] = None
        state["phase2_done_this_plateau"] = False
        state["post_phase2_active"] = False
        state["post_phase2_stable_streak"] = 0
        state["phase2_promoted_ever"] = False
        state["phase2_episode_complete"] = False
        state["force_phase2_next"] = False
        state["external_frame_id"] = None
        state["frame_invalidations"] += 1

    def _transport_rdm_history(U_align):
        """Put previous external-frame density history in the current frame."""
        for key in ("dm1s_prev", "dm1s_prev_raw"):
            if state[key] is not None:
                state[key] = np.stack(
                    [_back_rotate_dm1(dm, U_align) for dm in state[key]], axis=0)
        for key in ("dm2_prev", "dm2_prev_raw", "h2_ref_mix"):
            if state[key] is not None:
                state[key] = _back_rotate_dm2(state[key], U_align)

    def _cfg(num_runs, *, seed_entries=None, seed_override=None):
        overrides = {"orbopt_phase1": orbopt, **(config_overrides or {})}
        if seed_entries is not None:
            overrides = {**overrides,
                        "phase0_overrides": {
                            **(overrides.get("phase0_overrides") or {}),
                            "initial_dets_dict": {"bitstring": seed_entries},
                        }}
        return build_coo_config(
            threshold=threshold, num_runs=num_runs, orbopt=orbopt,
            orbopt_cycles=orbopt_cycles, max_dets_phase0=max_dets_phase0,
            save_fcidumps=save_fcidumps,
            enable_phase1=True, max_dets_phase1=max_dets_phase1,
            growth_factor_phase1=growth_factor_phase1,
            # Adaptive Phase 2 is a separate late support-promotion action;
            # never also run it inside the cold/large-drift Phase-0+1 call.
            enable_phase2=(enable_phase2 and not adaptive_phase2_after_plateau),
            max_dets_phase2=max_dets_phase2,
            growth_factor_phase2=growth_factor_phase2, pt2=pt2,
            seed=(seed if seed_override is None else seed_override),
            overrides=overrides,
        )

    def kernel(norb, nelec, h0, h1s, h2):
        t_start = time.perf_counter()

        if isinstance(nelec, (int, np.integer)):
            n = int(nelec)
            na, nb = (n + 1) // 2, n // 2
        else:
            na, nb = int(nelec[0]), int(nelec[1])

        h1 = 0.5 * (h1s[0] + h1s[1])

        if state["h1_ref"] is None:
            discover, drift = True, float("inf")
        else:
            ref_norm = np.linalg.norm(state["h1_ref"]) or 1.0
            drift = float(np.linalg.norm(h1 - state["h1_ref"]) / ref_norm)
            discover = drift > rediscover_tol

        frame_packet = None
        frame_transition = None
        frame_invalidated = False
        U_align = np.eye(norb)
        if adaptive_phase_policy and orbopt:
            frame_packet = frame_provider()
            if frame_packet is None:
                if frame_strict or state["dets_prev"] is not None:
                    raise RuntimeError(
                        "adaptive orbopt=True kernel call has no current frame packet")
            else:
                current_id = frame_packet["frame_id_curr"]
                cached_id = state["external_frame_id"]
                if state["dets_prev"] is None or cached_id is None:
                    pass
                elif cached_id == current_id:
                    pass
                elif cached_id == frame_packet["frame_id_prev"]:
                    frame_transition = frame_packet["transport"]
                    state["frame_leakage"] = float(frame_transition.leakage)
                    packet_reusable = bool(
                        frame_transition.reusable
                        and frame_transition.leakage <= frame_leakage_tol)
                    state["frame_reusable"] = packet_reusable
                    if packet_reusable:
                        U_align = np.asarray(frame_transition.usable_transform)
                        if U_align.shape != (norb, norb):
                            raise RuntimeError(
                                "frame provider returned an alignment with shape "
                                f"{U_align.shape}, expected {(norb, norb)}")
                        align_err = float(np.max(np.abs(
                            U_align.T @ U_align - np.eye(norb))))
                        if align_err > 1e-8:
                            raise RuntimeError(
                                "frame provider returned an invalid orthogonal alignment: "
                                f"shape={U_align.shape}, orthogonality_err={align_err:.3e}")
                        _transport_rdm_history(U_align)
                    else:
                        _invalidate_cached_episode()
                        frame_invalidated = True
                elif frame_strict:
                    raise RuntimeError(
                        "stale/out-of-order external frame packet: cached frame "
                        f"{cached_id!r}, packet transition "
                        f"{frame_packet['frame_id_prev']!r}->{current_id!r}")
                else:
                    _invalidate_cached_episode()
                    frame_invalidated = True

        if (warm_start_kappa and state["U_cached"] is not None
                and state["U_cached"].shape == (norb, norb)):
            U_prev = U_align @ state["U_cached"]
            h1_pre = U_prev.T @ h1 @ U_prev
            eri_pre = _rotate_eri(h2, U_prev)
        else:
            U_prev = np.eye(norb)
            h1_pre, eri_pre = h1, h2

        # Adaptive drift is only meaningful after both Hamiltonians have been
        # represented in the certified cached determinant frame.  Legacy mode
        # retains its input-frame diagnostics byte-for-byte.
        if adaptive_phase_policy:
            if state["h1_prev_common"] is None:
                delta_h_rel = None
            else:
                prev_norm = max(float(np.linalg.norm(state["h1_prev_common"])), 1e-15)
                delta_h_rel = float(
                    np.linalg.norm(h1_pre - state["h1_prev_common"]) / prev_norm)
            if state["h1_ref_common"] is None:
                drift = float("inf")
            else:
                ref_norm = max(float(np.linalg.norm(state["h1_ref_common"])), 1e-15)
                drift = float(np.linalg.norm(h1_pre - state["h1_ref_common"]) / ref_norm)
            discover = bool(state["dets_prev"] is None or drift > rediscover_tol)
        elif state["h1_prev"] is None:
            delta_h_rel = None
        else:
            prev_norm = max(float(np.linalg.norm(state["h1_prev"])), 1e-15)
            delta_h_rel = float(np.linalg.norm(h1 - state["h1_prev"]) / prev_norm)

        # ponytail: never launch more stochastic restarts than the fragment's
        # CI space has states. Closed-shell fragments collapse to a single
        # determinant (ci_dim == 1); firing discovery_num_runs (e.g. 64)
        # random restarts at a one-state space is wasted work and breaks the
        # discovery sampler. Clamp per fragment so a global num_runs is safe.
        ci_dim = math.comb(norb, na) * math.comb(norb, nb)

        # Adaptive dispatch (opt-in): decide the action from episode/support-cache
        # state (never from delta_h_rel), fully replacing the legacy discover/refine
        # choice below for this call -- the two dispatchers are never layered. See
        # make_phased_kernel's docstring for the full episode-based rationale.
        adaptive_action = None
        adaptive_requested_runs = None
        phase2_plateau_now = False
        post_phase2_active_now = bool(state["post_phase2_active"])
        current_support_size = (len(state["dets_prev"])
                                if state["dets_prev"] is not None else 0)
        if adaptive_phase_policy:
            # Episode-based dispatch (production, 2026-09-14): the ONLY trigger
            # for phase0_phase1 is "no cached support for this episode" --
            # either the true first call, or immediately after
            # _invalidate_cached_episode() dropped dets_prev on an unsafe
            # external frame transition (see above). delta_h_rel is recorded
            # as telemetry only and never gates dispatch -- see this
            # function's docstring for why (the superseded drift-triggered
            # version made 165 spurious Phase0+Phase1 calls on a real trapped
            # Fe4S4 run). A reusable external frame transition keeps
            # dets_prev, so it lands here too: "no rerun of Phase 0 merely
            # because the embedding Hamiltonian changed."
            large_drift = state["dets_prev"] is None
            if large_drift:
                adaptive_action = "phase0_phase1"
                adaptive_requested_runs = discovery_num_runs
            else:
                if (adaptive_phase2_after_plateau and state["force_phase2_next"]
                        and ci_dim <= 1):
                    # Filled/closed fragment: nothing to promote into. Satisfy
                    # the finalizer's request as a no-op rather than running
                    # Phase-2 continuation on a one-determinant support.
                    state["force_phase2_next"] = False
                    state["phase2_promoted_ever"] = True
                    state["phase2_episode_complete"] = True
                # Phase 2 fires ONLY when the parent's post-convergence
                # finalizer has set force_phase2_next -- never autonomously
                # from this inner RDM-JK loop's own plateau (see docstring).
                phase2_plateau_now = bool(
                    adaptive_phase2_after_plateau and state["force_phase2_next"])
                if phase2_plateau_now:
                    adaptive_action = "phase2_promote"
                else:
                    adaptive_action = ("post_phase2_fixed_support"
                                       if post_phase2_active_now else "fixed_support")
                adaptive_requested_runs = 0
            discover = adaptive_action == "phase0_phase1"

        requested_runs = int(adaptive_requested_runs if adaptive_phase_policy
                              else (discovery_num_runs if discover else refine_num_runs))
        effective_runs = max(1, min(requested_runs, ci_dim))
        seed_entries = None
        if warm_start_dets and state["dets_prev"] is not None:
            seed_entries = [_bitstring_entry(d, c, norb)
                            for d, c in zip(state["dets_prev"], state["coeffs_prev"])]
        if adaptive_phase_policy:
            fixed_support = adaptive_action in ("fixed_support", "post_phase2_fixed_support")
        else:
            fixed_support = bool(
                freeze_support_after_discovery and state["dets_prev"] is not None)
        if fixed_support:
            res = _fixed_support_ground_state(
                h1_pre, eri_pre, state["dets_prev"], state["coeffs_prev"], norb)
        elif adaptive_phase_policy and adaptive_action == "phase2_promote":
            # Direct Phase-2-only continuation from the cached Phase-1 support --
            # never reruns Phase 0 or Phase 1. The result becomes the new persistent
            # support baseline (state["dets_prev"]/["coeffs_prev"] below, same as
            # every other action) -- it is never shrunk back to max_dets_phase1.
            with suppress_fd(quiet):
                res = _run_expansion_continuation(
                    h1_pre, eri_pre, state["dets_prev"], state["coeffs_prev"], norb,
                    phase_index=2, threshold=threshold, growth_factor=growth_factor_phase2,
                    max_n_dets=max(max_dets_phase2, current_support_size),
                    orbital_optimization=False,
                    orbital_opt_max_iter=orbopt_cycles)
        else:
            # Escape-rule seed schedule: the parent-level controller (frasci.
            # parent's safeguard/escape installer) derives a distinct episode
            # seed per rescue attempt and publishes it on the frame packet's
            # "episode_seed" field, separately from TrimCI's own per-restart
            # seed derivation WITHIN this discovery call. Falling back to the
            # kernel's own constructor-time `seed` (attempt 1, or when no
            # packet exists / adaptive_phase_policy=False) makes this a no-op
            # for every run that never hits the escape rule.
            episode_seed = (frame_packet.get("episode_seed")
                           if frame_packet is not None else None)
            cfg = _cfg(effective_runs, seed_entries=seed_entries,
                      seed_override=episode_seed)
            with suppress_fd(quiet):
                res = trimci.ground_state_from_integrals(
                    h1_pre, eri_pre, na + nb, ms2=na - nb, e_nuc=0.0, config=cfg,
                )

        if res.n_dets == 0 or not np.isfinite(res.energy):
            mode = "discovery" if discover else "refine"
            raise RuntimeError(
                f"phased COO {mode} returned invalid result "
                f"(norb={norb}, nelec=({na},{nb})): n_dets={res.n_dets}, "
                f"energy={res.energy}"
            )

        # Repeated BFGS rotations are individually orthogonal only to their
        # numerical tolerance.  Long legacy runs can compose O(100) such
        # matrices and accumulate enough roundoff to leave O(n), even though
        # no physical nonunitary transformation was intended.  Project every
        # cumulative rotation back to the nearest orthogonal matrix (polar
        # factor via SVD) before caching or back-rotating RDMs.
        U_step_raw = np.asarray(res.orbital_rotation)
        R_raw = U_prev @ U_step_raw
        R = _nearest_orthogonal(R_raw)
        # Express the snapped cumulative frame as the effective step from the
        # already-orthogonal U_prev frame.  All common-frame Hamiltonian
        # bookkeeping below must use this same transform, or a pure external
        # frame rotation would appear as an artificial ~1e-10 drift.
        U_step = U_prev.T @ R
        rotation_raw_err = float(np.max(np.abs(
            R_raw.T @ R_raw - np.eye(norb))))
        rotation_snap_norm = float(np.linalg.norm(R - R_raw))
        rotation_err = float(np.max(np.abs(R.T @ R - np.eye(norb))))
        if rotation_err > 1e-8:
            raise RuntimeError(
                "cached-frame rotation composition lost orthogonality: "
                f"max|R.T R-I|={rotation_err:.3e}")

        rdm_diagnostics = {}
        common_extras = {
            "phased_mode": ("fixed_support" if fixed_support
                            else ("discover" if discover else "refine")),
            "phased_drift": drift,
            "phased_num_runs_requested": 0 if fixed_support else requested_runs,
            "phased_num_runs_effective": 0 if fixed_support else effective_runs,
            "fixed_support": fixed_support,
            "fixed_support_solver": getattr(res, "fixed_support_solver", None),
            "fixed_support_davidson_iters": getattr(
                res, "fixed_support_davidson_iters", None),
            "fixed_support_residual": getattr(res, "fixed_support_residual", None),
            "phase_policy_action": adaptive_action,
            "phase_policy_actual_eigensolver": (
                None if not adaptive_phase_policy
                else "dense_exact" if (fixed_support and getattr(
                    res, "fixed_support_solver", None) == "dense_exact")
                else "davidson" if fixed_support
                else "trimci_expansion_phased" if adaptive_action == "phase2_promote"
                else "trimci_phased"),
            "phase2_plateau_triggered": phase2_plateau_now,
            "post_phase2_active": post_phase2_active_now,
            "post_phase2_stable_streak": state["post_phase2_stable_streak"],
            "phase2_promoted_ever": state["phase2_promoted_ever"],
            "phase2_episode_complete": state["phase2_episode_complete"],
            "external_frame_id": (None if frame_packet is None
                                  else frame_packet["frame_id_curr"]),
            "frame_changed": bool(frame_transition is not None),
            "frame_reusable": state["frame_reusable"],
            "frame_leakage": state["frame_leakage"],
            "frame_invalidated": frame_invalidated,
            "frame_invalidations": state["frame_invalidations"],
            "rotation_raw_orthogonality_err": rotation_raw_err,
            "rotation_orthogonalization_correction_norm": rotation_snap_norm,
            "rotation_orthogonality_err": rotation_err,
            "expansion_residual": getattr(res, "expansion_residual", None),
            "expansion_termination_reason": getattr(
                res, "expansion_termination_reason", None),
        }
        try:
            (dm1a_r, dm1b_r), (dm2aa_r, dm2ab_r, dm2bb_r) = make_selected_rdm12s(
                res.determinants, res.coefficients, norb, (na, nb),
                backend=rdm_backend, diagnostics=rdm_diagnostics,
            )
        except Exception:
            if callable(log_callback):
                extras = {
                    "kernel_status": "rdm_failed",
                    **{f"rdm_{key}": value
                       for key, value in rdm_diagnostics.items()},
                    **common_extras,
                    **_null_rdm_validity_fields(),
                }
                try:
                    log_callback(
                        norb, (na, nb), int(res.n_dets), float(res.energy),
                        time.perf_counter() - t_start, extras)
                except Exception as callback_exc:
                    print(
                        "[frasci.parent] secondary callback failure while recording "
                        f"rdm_failed: {callback_exc}", file=sys.stderr)
            raise
        dm1a = _back_rotate_dm1(dm1a_r, R)
        dm1b = _back_rotate_dm1(dm1b_r, R)
        dm1s = np.stack([dm1a, dm1b], axis=0)
        dm2aa = _back_rotate_dm2(dm2aa_r, R)
        dm2ab = _back_rotate_dm2(dm2ab_r, R)
        dm2bb = _back_rotate_dm2(dm2bb_r, R)

        h1_native = U_step.T @ h1_pre @ U_step
        eri_native = _rotate_eri(eri_pre, U_step)
        dm2_native = dm2aa_r + dm2ab_r + dm2ab_r.transpose(2, 3, 0, 1) + dm2bb_r
        e_native = float(
            np.einsum("pq,pq->", h1_native, dm1a_r + dm1b_r)
            + 0.5 * np.einsum("pqrs,pqrs->", eri_native, dm2_native))

        validity, dm2, e_check = _raw_rdm_validity(
            dm1a, dm1b, dm2aa, dm2ab, dm2bb, na=na, nb=nb,
            h1=h1, h2=h2, reference_energy=e_native,
            invariance_atol=invariance_atol)
        validity["res_energy_mismatch_Ha"] = _finite_float_or_none(
            abs(float(res.energy) - e_check))

        if not validity["back_rotation_passed"]:
            mismatch = validity["back_rotation_energy_mismatch_Ha"]
            if callable(log_callback):
                extras = {
                    "kernel_status": "back_rotation_failed",
                    **{f"rdm_{key}": value
                       for key, value in rdm_diagnostics.items()},
                    **common_extras,
                    **validity,
                }
                try:
                    log_callback(
                        norb, (na, nb), int(res.n_dets), float(res.energy),
                        time.perf_counter() - t_start, extras)
                except Exception as callback_exc:
                    print(
                        "[frasci.parent] secondary callback failure while recording "
                        f"back_rotation_failed: {callback_exc}", file=sys.stderr)
            if failure_capture_dir is not None:
                import os
                os.makedirs(failure_capture_dir, exist_ok=True)
                cap_path = os.path.join(
                    failure_capture_dir,
                    f"failure_fragment_{fragment_idx}_{time.time():.6f}.npz")
                np.savez_compressed(
                    cap_path,
                    norb=np.asarray(norb), na=np.asarray(na), nb=np.asarray(nb),
                    h0=np.asarray(h0), h1=np.asarray(h1), h2=np.asarray(h2),
                    U_prev=np.asarray(U_prev), h1_pre=np.asarray(h1_pre),
                    eri_pre=np.asarray(eri_pre), discover=np.asarray(discover),
                    drift=np.asarray(drift), R=np.asarray(R),
                    res_energy=np.asarray(float(res.energy)),
                    res_energy_pt2=np.asarray(
                        float(res.energy_pt2) if res.energy_pt2 is not None else np.nan),
                    res_orbital_rotation=np.asarray(res.orbital_rotation),
                    res_final_h1=np.asarray(res.final_h1),
                    res_final_eri=np.asarray(res.final_eri),
                    res_determinants=np.asarray(res.determinants),
                    res_coefficients=np.asarray(res.coefficients),
                    dm1a_rotated=dm1a_r, dm1b_rotated=dm1b_r,
                    dm2aa_rotated=dm2aa_r, dm2ab_rotated=dm2ab_r, dm2bb_rotated=dm2bb_r,
                    dm1a_backrotated=dm1a, dm1b_backrotated=dm1b,
                    dm2aa_backrotated=dm2aa, dm2ab_backrotated=dm2ab, dm2bb_backrotated=dm2bb,
                    e_check=np.asarray(e_check), e_native=np.asarray(e_native),
                    h1_native=h1_native, eri_native=eri_native,
                    mismatch=np.asarray(mismatch if mismatch is not None else np.nan),
                    tolerance=np.asarray(validity["back_rotation_tolerance_Ha"]),
                )
                print(f"[frasci.parent] failure capture written -> {cap_path}",
                      file=sys.stderr)
            mismatch_text = "nonfinite" if mismatch is None else f"{mismatch:.2e}"
            raise RuntimeError(
                f"phased COO energy-invariance check failed: "
                f"E_from_RDM(input)={e_check:.10f} vs native-frame E={e_native:.10f} "
                f"(|d|={mismatch_text}; res.energy={res.energy:.10f} for context, "
                f"not the comparison target) -> rotation/composition bug in "
                f"back-rotation (norb={norb}, nelec=({na},{nb}))"
            )

        state["U_cached"] = R.copy()
        if adaptive_phase_policy:
            if discover or state["h1_ref_common"] is None:
                state["h1_ref_common"] = h1_native.copy()
            else:
                state["h1_ref_common"] = (
                    U_step.T @ state["h1_ref_common"] @ U_step)
            state["h1_prev_common"] = h1_native.copy()
            if frame_packet is not None:
                state["external_frame_id"] = frame_packet["frame_id_curr"]
        if discover:
            state["h1_ref"] = h1.copy()
        if warm_start_dets:
            state["dets_prev"] = np.asarray(res.determinants)
            state["coeffs_prev"] = np.asarray(res.coefficients)

        if phase1_state_dir is not None:
            import os
            if fragment_idx is None:
                raise ValueError("phase1_state_dir requires fragment_idx")
            os.makedirs(phase1_state_dir, exist_ok=True)
            state_path = os.path.join(
                phase1_state_dir, f"fragment_{int(fragment_idx)}.npz")
            tmp_path = state_path + ".tmp.npz"
            np.savez_compressed(
                tmp_path,
                determinants=np.asarray(res.determinants),
                coefficients=np.asarray(res.coefficients),
                orbital_rotation=R,
                h1_input=np.asarray(h1),
                n_orb=np.asarray(norb),
                n_alpha=np.asarray(na),
                n_beta=np.asarray(nb),
                energy=np.asarray(float(res.energy)),
            )
            os.replace(tmp_path, state_path)

        dm1s_raw, dm2_raw = dm1s.copy(), dm2.copy()

        e_ret = float(e_check)
        if (dm_mixing_alpha < 1.0 and not discover
                and state["dm1s_prev"] is not None
                and state["h2_ref_mix"] is not None
                and np.shape(h2) == np.shape(state["h2_ref_mix"])
                and np.allclose(h2, state["h2_ref_mix"], atol=1e-10)):
            a = float(dm_mixing_alpha)
            dm1s = (1.0 - a) * state["dm1s_prev"] + a * dm1s
            dm2 = (1.0 - a) * state["dm2_prev"] + a * dm2
            dm1_tot = dm1s[0] + dm1s[1]
            e_ret = float(np.einsum("pq,pq->", h1, dm1_tot)
                          + 0.5 * np.einsum("pqrs,pqrs->", h2, dm2))

        # Cycle-to-cycle stability diagnostics (report sec 7-9's finding that energy
        # alone can look converged while the density/support are still unsettled --
        # not just a warm_start_dets by-product, always tracked). None on the first
        # call for a fragment (nothing to diff against yet).
        delta_rdm_norm = None
        if (state["dm1s_prev"] is not None and state["dm2_prev"] is not None
                and np.shape(state["dm1s_prev"]) == np.shape(dm1s)
                and np.shape(state["dm2_prev"]) == np.shape(dm2)):
            delta_rdm_norm = float(np.linalg.norm(dm1s - state["dm1s_prev"])
                                   + np.linalg.norm(dm2 - state["dm2_prev"]))
        delta_rdm_norm_raw = None
        if (state["dm1s_prev_raw"] is not None and state["dm2_prev_raw"] is not None
                and np.shape(state["dm1s_prev_raw"]) == np.shape(dm1s_raw)
                and np.shape(state["dm2_prev_raw"]) == np.shape(dm2_raw)):
            delta_rdm_norm_raw = float(np.linalg.norm(dm1s_raw - state["dm1s_prev_raw"])
                                       + np.linalg.norm(dm2_raw - state["dm2_prev_raw"]))
        delta_e = (abs(e_ret - state["e_ret_prev"])
                  if state["e_ret_prev"] is not None else None)

        # Did the Phase-2 promotion actually change the RDM it fed back into the
        # outer RDM-JK loop? delta_rdm_norm above (this call vs the call that
        # plateaued) already answers this directly -- no separate mechanism.
        phase2_changed_rdm = None
        post_phase2_converged_now = False
        if adaptive_phase_policy:
            # Debounce on THIS call's own deltas (what next call will see as
            # last_delta_e/last_delta_rdm_mixed), not the dispatch-time flag computed
            # from the call before -- else the "done" guard would clear itself again
            # on the very next call even while the plateau is still ongoing, and
            # Phase 2 would repromote every other call instead of once per episode.
            still_plateaued = bool(
                delta_e is not None and delta_rdm_norm is not None
                and delta_e <= phase2_plateau_delta_e_tol
                and delta_rdm_norm <= phase2_plateau_delta_rdm_tol)
            if adaptive_action == "phase2_promote":
                state["force_phase2_next"] = False
                phase2_changed_rdm = bool(
                    delta_rdm_norm is not None
                    and delta_rdm_norm > phase2_plateau_delta_rdm_tol)
                state["phase2_done_this_plateau"] = True
                state["post_phase2_active"] = True
                state["post_phase2_stable_streak"] = 0
                state["phase2_promoted_ever"] = True
            elif adaptive_action == "phase0_phase1":
                # Genuinely large drift (the only way phase0_phase1 fires once
                # adaptive_phase_policy is on -- see the dispatch above): a fresh
                # episode. Clears any in-progress plateau/post-Phase-2 tracking so a
                # future plateau after rediscovery may promote again from scratch.
                state["phase2_done_this_plateau"] = False
                state["post_phase2_active"] = False
                state["post_phase2_stable_streak"] = 0
                state["phase2_episode_complete"] = False
            elif not still_plateaued:
                state["phase2_done_this_plateau"] = False
                if state["post_phase2_active"]:
                    state["post_phase2_stable_streak"] = 0
            elif state["post_phase2_active"]:
                # Small/moderate drift AND still plateaued AND already in the
                # post-Phase-2 window: one more consecutive stable call.
                state["post_phase2_stable_streak"] += 1
                if state["post_phase2_stable_streak"] >= 3:
                    post_phase2_converged_now = True
                    state["post_phase2_active"] = False
                    state["post_phase2_stable_streak"] = 0
                    state["phase2_episode_complete"] = True
            state["last_delta_e"] = delta_e
            state["last_delta_rdm_mixed"] = delta_rdm_norm

        support_now = {(int(a), int(b)) for a, b in res.determinants}
        # Bitstring identities are comparable only within one determinant
        # orbital frame.  A genuine Phase-0/1 BFGS step changes that frame.
        support_prev = (None if not np.allclose(
            np.asarray(res.orbital_rotation), np.eye(norb), atol=1e-10)
                        else state["support_prev"])
        support_retained = support_added = support_dropped = support_jaccard = None
        if support_prev is not None:
            support_retained = len(support_now & support_prev)
            support_added = len(support_now - support_prev)
            support_dropped = len(support_prev - support_now)
            union_size = len(support_now | support_prev)
            support_jaccard = (support_retained / union_size) if union_size else None

        state["dm1s_prev"] = dm1s.copy()
        state["dm2_prev"] = dm2.copy()
        state["dm1s_prev_raw"] = dm1s_raw.copy()
        state["dm2_prev_raw"] = dm2_raw.copy()
        state["h2_ref_mix"] = np.asarray(h2).copy()
        state["e_ret_prev"] = e_ret
        state["support_prev"] = support_now
        state["h1_prev"] = h1.copy()

        etot = float(h0) + e_ret

        if callable(log_callback):
            wall = time.perf_counter() - t_start
            hist = list(getattr(res.energies, "history", []) or [])
            dE_coo = ((float(hist[-1]) - float(hist[0])) * 1000.0
                      if len(hist) >= 2 else 0.0)
            extras = {
                "kernel_status": "ok",
                **{f"rdm_{key}": value
                   for key, value in rdm_diagnostics.items()},
                **common_extras,
                **validity,
                "n_coo_cycles_done": 0,
                "U_offdiag_norm": float(np.linalg.norm(R - np.eye(norb))),
                "dE_from_coo_mHa": dE_coo,
                "n_dets_seed": int(res.n_dets),
                "e_pt2": (float(res.energy_pt2)
                          if res.energy_pt2 is not None else None),
                "phase2_changed_rdm": phase2_changed_rdm,
                "post_phase2_converged_now": post_phase2_converged_now,
                "post_phase2_stable_streak_after": state["post_phase2_stable_streak"],
                "delta_h_rel": delta_h_rel,
                "delta_rdm_norm": delta_rdm_norm,
                "delta_rdm_norm_raw": delta_rdm_norm_raw,
                "delta_e_ha": delta_e,
                "support_retained": support_retained,
                "support_added": support_added,
                "support_dropped": support_dropped,
                "support_jaccard": support_jaccard,
            }
            log_callback(norb, (na, nb), int(res.n_dets), float(res.energy),
                         wall, extras)

        return etot, dm1s, dm2

    # Exposed so frasci.parent._parallel can carry a fragment's warm-start cache
    # between processes (workers are stateless; the parent owns the state).
    # Read/written only between calls -- never while kernel() is running.
    kernel.state = state
    return kernel
