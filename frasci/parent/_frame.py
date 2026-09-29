"""Orbital-frame certificate used by the cached-frame adaptive parent policy.

Closes the second of the two gaps
``docs/reports/frasci_orbopt_frame_h8_contract_2026-09-14.md`` identified: the
The fragment-kernel interface (``RDMSolver.kernel(norb, nelec, h0, h1s,
h2)``, confirmed by reading ``mrh/my_pyscf/mcscf/lasscf_rdm.py:380``) has no
access to ``mo_coeff`` or the AO overlap at all, so it cannot know whether the
orbital frame moved between two calls, nor by how much. This module is the
smallest self-contained computation a caller who DOES have that data (an
parent orchestration layer wrapping ``las.get_hop``, the same hook
``run_experiment6`` in the contract report used) can run to decide whether a
cached determinant support is safe to reuse.

Four explicit inputs, always supplied by the caller (never inferred from h1
alone, per the contract report's Q8/Q9 guidance):
  - ``C_prev``, ``C_curr``: ``(n_AO, n_frag)`` fragment MO coefficient blocks
    at two frame generations (both must be S-orthonormal on their own -- see
    ``_assert_orthonormal``, a real trust-boundary check on caller input).
  - ``S_AO``: the ``(n_AO, n_AO)`` overlap in whatever basis ``C_prev``/
    ``C_curr`` are expressed (a real molecule's Gaussian AO overlap, or a
    mock-SCF's ``get_ovlp()`` -- see the contract report's own bug-and-fix
    note: using the WRONG one is a real, previously-hit mistake, not a
    hypothetical).
  - ``frame_id_prev``, ``frame_id_curr``: opaque, comparable identifiers
    (e.g. a macro-iteration index). ``frame_id_prev == frame_id_curr`` is a
    fast path that skips all computation and returns the trivial exact
    result -- it also avoids reporting spurious nonzero leakage from pure
    floating-point noise when the caller passes bit-identical matrices for
    what is genuinely the same frame (true within one RDM-JK inner loop).

Outputs: the raw overlap ``Q``, its singular values (== cos of the principal
angles between the two S-orthonormal subspaces -- exact for this case, not an
approximation, because both blocks are already S-orthonormal columns of a
larger orthonormal `mo_coeff`), the principal angles themselves, orthogonality
error, subspace leakage (asymmetric: how much of the OLD subspace's column
space lies outside the NEW subspace), the nearest ORTHOGONAL matrix to ``Q``
(solving the orthogonal Procrustes problem via its SVD -- the "usable frame
transformation" IF leakage is small enough that treating this genuinely
non-orthogonal empirical overlap as a pure rotation is a good approximation),
and ``reusable`` (``leakage <= leakage_tol``).

``reusable=False`` is the caller's discard signal: cached determinant support
(and any coefficients/RDMs computed in the previous frame) MUST NOT be
transported through ``usable_transform`` in that case -- the contract report's
own negative control shows this fails silently (residual and self-consistency
checks pass; the energy is wrong by 15-22 mHa on H8, 1.8 mHa on H12) precisely
when a stale frame is combined with fresh integrals as though nothing moved.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class FrameTransport(NamedTuple):
    same_frame: bool
    Q: np.ndarray
    singular_values: np.ndarray
    principal_angles_rad: np.ndarray
    orthogonality_err: float
    leakage: float
    usable_transform: np.ndarray
    reusable: bool


def _assert_orthonormal(C: np.ndarray, S_AO: np.ndarray, *, label: str, atol: float = 1e-8) -> None:
    n = C.shape[1]
    err = float(np.max(np.abs(C.T @ S_AO @ C - np.eye(n))))
    if err > atol:
        raise ValueError(
            f"{label} is not S-orthonormal (max|C.T S C - I|={err:.3e} > {atol:.0e}) -- "
            "compute_frame_transport's principal-angle formula (singular values of "
            "C_curr.T @ S_AO @ C_prev) is only exact for S-orthonormal fragment blocks "
            "(true when they are column subsets of a larger orthonormal mo_coeff, which "
            "is the only case this module is designed for); passing a non-orthonormal "
            "block would silently give a wrong leakage/reusable verdict")


def compute_frame_transport(
    C_prev: np.ndarray,
    C_curr: np.ndarray,
    S_AO: np.ndarray,
    *,
    frame_id_prev: Any,
    frame_id_curr: Any,
    leakage_tol: float,
) -> FrameTransport:
    """The smallest explicit frame-data path: overlap, principal angles/
    leakage, and the usable (orthogonalized) transform, from exactly the four
    inputs a future parent-kernel orchestration layer would need to supply
    per fragment per call. See the module docstring for the full contract.

    Raises ``ValueError`` if ``C_prev``/``C_curr`` are not S-orthonormal (a
    trust-boundary check on the caller's own data, not a numerical nicety) or
    if their shapes disagree (a fragment cannot change orbital COUNT between
    calls without this being a different fragment entirely).
    """
    if C_prev.shape != C_curr.shape:
        raise ValueError(
            f"C_prev.shape={C_prev.shape} != C_curr.shape={C_curr.shape} -- a fragment's "
            "orbital count cannot change between calls; this is not a frame-transport "
            "question if it does")
    n_frag = C_prev.shape[1]

    if frame_id_prev == frame_id_curr:
        return FrameTransport(
            same_frame=True, Q=np.eye(n_frag), singular_values=np.ones(n_frag),
            principal_angles_rad=np.zeros(n_frag), orthogonality_err=0.0, leakage=0.0,
            usable_transform=np.eye(n_frag), reusable=True)

    _assert_orthonormal(C_prev, S_AO, label="C_prev")
    _assert_orthonormal(C_curr, S_AO, label="C_curr")

    Q = C_curr.T @ S_AO @ C_prev
    U, sv, Vt = np.linalg.svd(Q)
    principal_angles = np.arccos(np.clip(sv, -1.0, 1.0))
    orthogonality_err = float(np.max(np.abs(Q.T @ Q - np.eye(n_frag))))

    # Leakage: how much of the OLD subspace's column space lies outside the
    # NEW subspace, in the S metric (an S-orthogonal projector onto C_curr's
    # column space, applied to C_prev). Asymmetric on purpose -- the question
    # is "is the cached (OLD-frame) support still representable in the NEW
    # frame," not a symmetric distance between the two subspaces.
    P_curr = C_curr @ np.linalg.solve(C_curr.T @ S_AO @ C_curr, C_curr.T @ S_AO)
    leakage = float(np.linalg.norm(C_prev - P_curr @ C_prev) / np.linalg.norm(C_prev))

    # Usable transform: the nearest ORTHOGONAL matrix to Q (orthogonal
    # Procrustes solution, U @ Vt from Q's own SVD) -- Q itself need not be
    # exactly orthogonal once leakage is nonzero, but a genuine orbital
    # rotation used to transport cached data (rotate_eri/back_rotate_dm2 etc.,
    # as frasci.parent._kernel already does with a KNOWN-orthogonal U) must
    # be. Meaningful only when reusable=True -- returned unconditionally for
    # diagnostics, but the caller must not use it when reusable=False.
    usable_transform = U @ Vt

    return FrameTransport(
        same_frame=False, Q=Q, singular_values=sv, principal_angles_rad=principal_angles,
        orthogonality_err=orthogonality_err, leakage=leakage,
        usable_transform=usable_transform, reusable=bool(leakage <= leakage_tol))
