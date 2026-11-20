"""Milestone F: form W_final = P_LAS union R and solve the ordinary (uncontracted)
Hermitian Davidson eigenproblem in that space -- not the generalized contracted
solve in ``_solve.py``. Every coefficient, including original LAS amplitudes, is
independently reoptimized.

Uses TrimCI's own native matrix-free Davidson (``fast_expansion.davidson_solve_matfree``,
verified directly against a dense FCI reference to 2.7e-12 Ha before use here) rather than
reimplementing Davidson in Python, per the design doc's call for a matrix-free production
operator. The energy/residual recompute below (never trust the native scalars alone) is
ALSO matrix-free, via ``fast_expansion.matvec_matfree`` (verified against a dense matvec to
3.6e-15 before use) -- a dense N_union x N_union Hamiltonian is never built in this
production path. ``_state.selected_hamiltonian`` (PySCF-backed, dense, O(N^2) memory) is
used ONLY in this module's own small-system tests as an independent ground-state oracle,
never in `solve_union_davidson`/`certify_final_union` themselves.

Depends on ``frasci.recovery._las_reconstruct``'s reconstruction certificate having
already passed on ``state`` -- ``solve()`` runs that as a mandatory gate before ever
calling into this module. This module does not re-certify the LAS reconstruction itself.
"""
from __future__ import annotations

import numpy as np
import trimci.trimci_core as tc

from frasci.recovery._las_reconstruct import reconstruct_las_vector
from frasci.recovery._state import retain_candidates

fe = tc.fast_expansion
Determinant = tuple[int, int]


def form_final_union(state: dict, recovery_dets: list[Determinant], recovery_coeffs, *,
                     n_ct_max: int) -> dict:
    """P_LAS (the full certified Cartesian product) union R (dispersion + allowed-CT
    only). ``include_same_sector`` is ALWAYS False: R's scientific definition
    excludes same-sector determinants by fixed rule. P_K is already inside P_LAS,
    so it is never appended a second time.

    Returns deduplicated determinants and a Davidson initial guess: the certified LAS
    amplitude on P_LAS members, zero on new R members (per the design doc).
    """
    las_dets, las_coeffs = reconstruct_las_vector(state)
    eligible, classification = retain_candidates(
        recovery_dets, recovery_coeffs, state, n_ct_max=n_ct_max,
        include_same_sector=False)
    r_dets = list(eligible)

    combined: dict[Determinant, float] = {det: float(coeff)
                                          for det, coeff in zip(las_dets, las_coeffs)}
    n_overlap = 0
    for det in r_dets:
        if det in combined:
            # Should not happen: R is dispersion/CT by construction, P_LAS is
            # entirely same-sector. Not silently dropped or double-counted --
            # surfaced in the returned diagnostics for the caller to notice.
            n_overlap += 1
            continue
        combined[det] = 0.0

    dets = list(combined.keys())
    initial_guess = np.asarray([combined[d] for d in dets], dtype=float)
    return {
        "dets": dets, "initial_guess": initial_guess,
        "n_las": len(las_dets), "n_recovery": len(r_dets), "n_overlap": n_overlap,
        "n_work": len(dets), "classification": classification,
    }


def solve_union_davidson(state: dict, dets: list[Determinant], initial_guess: np.ndarray,
                         *, davidson_params=None,
                         residual_tol: float = 1e-6) -> dict:
    """Ordinary Hermitian Davidson diagonalization in the exact determinant space
    ``dets`` -- no truncation, no contracted reference, no additional determinant cap,
    and no dense N_union x N_union allocation anywhere in this function.

    A candidate is never accepted on the native ``res.converged`` flag alone: the
    independently recomputed matrix-free residual (below) must also clear
    ``residual_tol``, or that candidate is rejected outright, same as
    ``frasci.parent._kernel._fixed_support_ground_state``'s gate.
    """
    n_orb = int(state["n_orb"])
    if n_orb > 64:
        raise NotImplementedError(
            f"n_orb={n_orb} exceeds 64 -- alpha/beta strings are native uint64_t, which "
            "covers orbitals 0-63 exactly (n_orb=64 itself is fully supported: verified "
            "directly against matvec_matfree/davidson_solve_matfree, not assumed). No "
            "wider matrix-free binding (uint64_t pair, 128-bit, ...) exists for this "
            "path in this TrimCI build.")
    alpha = np.array([int(d[0]) for d in dets], dtype=np.uint64)
    beta = np.array([int(d[1]) for d in dets], dtype=np.uint64)
    h1c = np.ascontiguousarray(state["h1"])
    eric = np.ascontiguousarray(state["eri"]).ravel()

    def _run(guess, start_name: str) -> dict | None:
        if davidson_params is not None:
            params = davidson_params
        else:
            # Residual-only stopping by default: native Davidson stops on EITHER
            # residual<residual_tol OR |dE|<energy_tol, whichever fires first, and the
            # default energy_tol=1e-7 can fire first, well before the residual this
            # caller actually requires (verified directly: on the real alfe2_11o10e
            # 103-det union space this produced a converged=True result whose
            # independently recomputed residual failed the residual_tol gate below).
            params = fe.DavidsonParams()
            params.energy_tol = 0.0
        res = fe.davidson_solve_matfree(alpha, beta, h1c, eric, n_orb, params, guess)
        if not res.converged:
            return None
        coeffs = np.asarray(res.eigenvectors, dtype=float).reshape(-1)
        coeffs = coeffs / np.linalg.norm(coeffs)
        # Never trust the native eigenvalue/residual scalars alone -- independently
        # recompute the Rayleigh quotient and residual from the returned determinants
        # and coefficients (sigma = H c via matvec_matfree, E = c.sigma/c.c,
        # ||r|| = ||sigma - E c||), matching this project's established distrust of
        # stale native energy scalars. This independent recompute is also the ONLY
        # reason the wrong-root bug this module's tests regress against was ever
        # caught in the first place, not a formality -- and it stays matrix-free:
        # no dense Hamiltonian is built here.
        sigma = fe.matvec_matfree(alpha, beta, coeffs, h1c, eric, n_orb)
        e_electronic = float(coeffs @ sigma) / float(coeffs @ coeffs)
        residual_norm = float(np.linalg.norm(sigma - e_electronic * coeffs))
        if residual_norm >= residual_tol:
            # Never accepted on res.converged alone: the native flag can be True
            # (its own internal residual/energy-change criteria satisfied) while
            # this independent matrix-free recompute is still well above the bar
            # this caller actually requires. Do not loosen residual_tol to hide this.
            return None
        return {
            "coefficients": coeffs, "e_electronic": e_electronic,
            "n_iters": int(res.n_iters), "residual_norm_native": float(res.residual_norm),
            "residual_norm": residual_norm, "start": start_name,
        }

    # A custom initial guess is not guaranteed to converge to the TRUE ground state:
    # verified directly (not assumed) that davidson_solve_matfree can report
    # converged=True with a near-zero residual for a genuine but non-lowest
    # eigenpair when the supplied guess is nearly orthogonal to the true ground
    # state (a real determinant space where the LAS-weighted guess has zero weight
    # on the new recovery determinants, and the true minimum turned out to live
    # almost entirely on one of them). Running the mandated LAS-weighted guess AND
    # the native default guess, then keeping the independently-recomputed lower
    # energy, is a GROUND-ROOT SAFEGUARD against that specific known failure mode --
    # not a mathematical proof that no lower root exists. On this project's small
    # test systems that proof comes separately, from the dense selected_hamiltonian
    # + eigh comparison in this module's own tests (never from the production path
    # itself, which has no dense oracle to compare against for a real, large
    # system). Mirrors Milestone D's Phase-0 ensemble winner-selection pattern.
    # One-root guess: native contract is initial_guess.shape == (n_states, N), i.e.
    # (1, N) for a single root -- not (N, 1).
    guess_2d = np.ascontiguousarray(initial_guess, dtype=float).reshape(1, -1)
    candidates = [
        c for c in (
            _run(guess_2d, "las_weighted"),
            _run(np.empty((0, 0)), "native_default"),
        ) if c is not None
    ]
    if not candidates:
        raise RuntimeError(
            f"final union Davidson did not converge to residual<{residual_tol:.1e} "
            f"(guess or default; native res.converged=True alone is not accepted) "
            f"over {len(dets)} determinants")
    winner = min(candidates, key=lambda c: c["e_electronic"])

    e_electronic = winner["e_electronic"]
    e_final = e_electronic + float(state["e_const"])
    return {
        "dets": dets, "coefficients": winner["coefficients"], "e_final": e_final,
        "e_electronic": e_electronic, "converged": True,
        "n_iters": winner["n_iters"],
        "residual_norm_native": winner["residual_norm_native"],
        "residual_norm": winner["residual_norm"],
        "n_candidates_tried": len(candidates),
        "winner_start": winner["start"],
        "start_diagnostics": [
            {
                "start": candidate["start"],
                "e_electronic": candidate["e_electronic"],
                "e_total": candidate["e_electronic"] + float(state["e_const"]),
                "n_iters": candidate["n_iters"],
                "residual_norm_native": candidate["residual_norm_native"],
                "residual_norm": candidate["residual_norm"],
            }
            for candidate in candidates
        ],
    }


def certify_final_union(solved: dict, h00: float, *, tol: float = 1e-6) -> dict:
    """E_final <= H00 within ``tol`` -- the certified LAS vector lies inside this
    space, so this variational bound must hold. Raises on violation; never loosened
    to hide a genuine bound violation.
    """
    e_final = solved["e_final"]
    margin = float(h00) - e_final
    if margin < -tol:
        raise AssertionError(
            f"final union Davidson energy ({e_final:.12f} Ha) exceeds H00 "
            f"({float(h00):.12f} Ha) by {-margin:.3e} Ha, beyond tolerance {tol:.0e} Ha "
            "-- the certified LAS vector lies inside this space so this bound must "
            "hold; do not loosen this tolerance to hide a genuine violation")
    return {"e_final": e_final, "h00": float(h00), "margin": margin, "tol": tol,
           "passed": True}
