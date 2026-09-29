"""Milestone E: reconstruct the full uncontracted Cartesian-product Phi_LAS in a
state dict's own orbital frame, and certify its Rayleigh energy against that
state's H00.

This is the same explicit-uncontraction + Rayleigh-vs-H00 approach validated in
``docs/reports/fe4s4_p10_uncontracted_union_pilot_2026-09-10.py`` (Stage 3, a real
Fe4S4/p10 system, 10x10x1=100-term product, passed at 1e-8 Ha) -- promoted here to
tested production code rather than one-off diagnostic script, per
docs/methods/filter-aware-phased-trimci-interfragment-recovery.md Milestone E.

Scope: reconstruction and certification ONLY. Does not touch ``frasci.recovery.solve``,
does not form any P_LAS union R, does not run a Davidson solve -- those are later
milestones (F), gated on this certificate passing.
"""
from __future__ import annotations

import itertools

import numpy as np

from frasci.recovery._state import _embed_bits, selected_hamiltonian

Determinant = tuple[int, int]


def reconstruct_las_vector(state: dict) -> tuple[list[Determinant], np.ndarray]:
    """The complete Cartesian product of every fragment's local determinant support,
    embedded into the global orbital convention and combined into one normalized
    vector -- never a top-K truncation (contrast ``_state.top_product_terms``).

    Deterministic given ``state["frags"]``'s own ordering: the accumulated coefficient
    for each global determinant is a sum over whichever local-index combinations
    produce it (duplicates only arise from redundant entries within one fragment's own
    support, since disjoint orbital blocks make the global bitstring a bijective
    function of the per-fragment local determinant tuple), independent of the order
    those combinations are visited in.
    """
    frags = state["frags"]
    blocks = state["blocks"]
    per_frag_sizes = [len(frag["dets"]) for frag in frags]

    combined: dict[Determinant, float] = {}
    index_ranges = [range(n) for n in per_frag_sizes]
    for idx_tuple in itertools.product(*index_ranges):
        alpha = beta = 0
        coeff = 1.0
        for j, idx in enumerate(idx_tuple):
            la, lb = int(frags[j]["dets"][idx][0]), int(frags[j]["dets"][idx][1])
            coeff *= float(frags[j]["coeffs"][idx])
            alpha |= _embed_bits(la, blocks[j])
            beta |= _embed_bits(lb, blocks[j])
        if coeff == 0.0:
            continue
        det = (alpha, beta)
        combined[det] = combined.get(det, 0.0) + coeff

    dets = list(combined.keys())
    if not dets:
        raise ValueError("Cartesian-product LAS reconstruction produced zero determinants")
    coeffs = np.asarray([combined[d] for d in dets], dtype=float)
    norm = float(np.linalg.norm(coeffs))
    if not np.isfinite(norm) or norm <= 0.0:
        raise AssertionError(f"reconstructed LAS vector has invalid norm {norm}")
    coeffs = coeffs / norm
    return dets, coeffs


def las_reconstruction_energy(state: dict, dets: list[Determinant],
                              coeffs: np.ndarray, *, dense_limit: int = 5000) -> float:
    """E_LAS,reconstructed = <Phi_LAS|H|Phi_LAS>/<Phi_LAS|Phi_LAS>, using ``state``'s
    own integrals and scalar constant exactly once. ``coeffs`` need not already be
    normalized -- the ``/ (c @ c)`` division makes this robust regardless."""
    c = np.asarray(coeffs, dtype=float)
    n_orb = int(state["n_orb"])
    if len(dets) <= dense_limit:
        ham = selected_hamiltonian(state["h1"], state["eri"], dets, n_orb=n_orb)
        electronic = float(c @ ham @ c) / float(c @ c)
    else:
        if n_orb > 64:
            raise NotImplementedError(
                "matrix-free LAS reconstruction energy currently supports n_orb <= 64")
        import trimci.trimci_core as tc

        alpha = np.ascontiguousarray([int(det[0]) for det in dets], dtype=np.uint64)
        beta = np.ascontiguousarray([int(det[1]) for det in dets], dtype=np.uint64)
        sigma = tc.fast_expansion.matvec_matfree(
            alpha, beta, np.ascontiguousarray(c),
            np.ascontiguousarray(state["h1"]),
            np.ascontiguousarray(state["eri"]).ravel(), n_orb)
        electronic = float(c @ np.asarray(sigma, dtype=float)) / float(c @ c)
    return electronic + float(state["e_const"])


def certify_las_reconstruction(state: dict, *, tol: float = 1e-8) -> dict:
    """Reconstruct Phi_LAS in ``state``'s frame and require its Rayleigh energy to
    reproduce ``state["h00"]`` within ``tol`` -- the hard correctness gate this
    milestone exists for. Raises ``AssertionError`` on failure; never widens ``tol``
    or substitutes a stored scalar to paper over a genuine mismatch (see the design
    doc's explicit prohibition). Returns a diagnostics dict on success.
    """
    dets, coeffs = reconstruct_las_vector(state)
    e_reconstructed = las_reconstruction_energy(state, dets, coeffs)
    h00 = float(state["h00"])
    diff = abs(e_reconstructed - h00)
    result = {
        "n_dets": len(dets),
        "dets": dets,
        "coeffs": coeffs,
        "e_reconstructed": e_reconstructed,
        "h00": h00,
        "diff": diff,
        "tol": tol,
        "passed": diff <= tol,
    }
    if diff > tol:
        raise AssertionError(
            f"LAS reconstruction Rayleigh-energy certificate FAILED: "
            f"|E_LAS_reconstructed ({e_reconstructed:.12f}) - H00 ({h00:.12f})| = "
            f"{diff:.3e} Ha exceeds tolerance {tol:.0e} Ha over {len(dets)} determinants "
            "-- do not trust this frame's embedding/sign convention; do not add recovery "
            "determinants or form a Davidson union until this is fixed")
    return result
