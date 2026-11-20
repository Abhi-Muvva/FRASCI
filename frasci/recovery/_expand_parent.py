"""Expanded-Parent: relax the contracted Parent's rigid tensor-product contraction by
exactly diagonalizing its own top-K explicit product terms -- no recovery determinants,
no orbital rotation. Isolates how much energy is available just from NOT forcing every
fragment's product term into one single factorized amplitude.

Promoted from a notebook-level probe (``09_expanding_parent_ladder``) to a first-class,
optional ``frasci.recovery.solve()`` stage. The probe composed two existing primitives
directly -- ``top_product_terms`` (heap-select, no Cartesian enumeration) feeding
``frasci.parent._kernel._fixed_support_ground_state`` (exact fixed-support
diagonalization) -- unchanged here; this module only wires that composition in as a
reusable, independently tested function with the same result-reporting conventions as
``_free_coo.py``.

This is a diagnostic on the Parent alone, not a recovery step: it never touches the
recovery growth, the generalized solve, or either union, and its result does not feed
back into them. Run it to calibrate ``RecoveryConfig.las_top_k`` before a COO run, or to
report the Parent's own contraction-relaxation gap next to the recovery result -- not to
replace either.
"""
from __future__ import annotations

import numpy as np

from frasci.parent._kernel import _fixed_support_ground_state
from frasci.recovery._state import top_product_terms


def compute_expanded_parent(state: dict, k: int, *, max_subspace: int = 200) -> dict:
    """Exactly diagonalize the top-``k`` explicit LAS product terms (by
    ``|product of local coefficients|``, via ``top_product_terms`` -- never the full
    Cartesian product) as independent, freely-variable determinants.

    ``k`` is a request, not a guarantee: if the fragment supports admit fewer than
    ``k`` distinct product terms, ``actual_n`` (and the returned support) is the
    smaller achieved count, exactly as ``target_met``/achieved-``N`` are reported
    elsewhere in this package -- never silently padded or treated as an error.

    ``max_subspace=200`` matches ``_free_coo.py``'s free-orbital union rediagonalization
    (root-caused there against a captured Davidson failure at ~80,000 determinants);
    kept here for the same reason, since this diagonalization can reach comparable
    sizes at large ``k``.
    """
    if k < 1:
        raise ValueError(f"expand_parent_k must be >= 1, got {k}")
    dets, coeffs = top_product_terms(state, int(k))
    actual_n = len(dets)
    dets_arr = np.asarray([tuple(map(int, d)) for d in dets], dtype=np.int64)
    n_orb = int(state["n_orb"])

    result = _fixed_support_ground_state(
        state["h1"], state["eri"], dets_arr, coeffs, n_orb, max_subspace=max_subspace)
    energy = float(result.energy) + float(state["e_const"])
    h00 = float(state["h00"])
    gain_mha = (h00 - energy) * 1000.0

    return {
        "requested_k": int(k),
        "actual_n": int(actual_n),
        "target_met": bool(actual_n >= k),
        "energy_ha": energy,
        "h00": h00,
        "gain_mha": gain_mha,
        "dets": [tuple(map(int, d)) for d in result.determinants],
        "coefficients": np.asarray(result.coefficients, dtype=float),
        "residual_norm": float(result.fixed_support_residual),
        "solver_kind": str(result.fixed_support_solver),
    }
