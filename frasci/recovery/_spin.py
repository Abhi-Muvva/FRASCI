"""Spin-group closure and S^2 utilities for fixed-parent-frame recovery.

Pure math only: every function here takes plain determinant tuples (and, for
``s_squared_expectation``, their coefficients) -- never a ``frasci.recovery`` state
dict, a ``Parent``, or an mrh/LASSIS object.

Ported from the research tree's ``frasci/recovery/_selectors.py`` (spin-closure
section only; the CIPSI/ASCI/2x2 determinant-selector scoring functions in that
module are research-only and are not part of this release).

Important scope note: ``s_squared_expectation`` computes ``<Psi|S^2|Psi>`` for an
explicit determinant expansion.  The complete FRASCI hybrid wavefunction is
``c0 |Phi_LAS> + sum_D d_D |D>`` where ``|Phi_LAS>`` stays factorized (never expanded).
Calling this function on only the explicit recovery determinants -- without the
contracted parent block and the parent-determinant cross terms -- is NOT the full
hybrid wavefunction's spin and must never be reported as ``Result.spin_square``.
Any such partial number must be named ``explicit_block_spin_square_partial`` by its
caller, exactly as ``frasci.recovery`` does.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

Determinant = tuple[int, int]


def spin_exchange_partners(det: Determinant, *, n_alpha: int, n_beta: int) -> tuple[Determinant, ...]:
    """Every determinant sharing ``det``'s spatial configuration (same doubly- and
    singly-occupied orbitals), with the singly-occupied orbitals reassigned to alpha/beta
    in every way consistent with the fixed (n_alpha, n_beta) sector -- the full spin-
    exchange orbit needed to close that configuration under S^2 (Chilkuri & Neese)."""
    alpha, beta = int(det[0]), int(det[1])
    if alpha.bit_count() != n_alpha or beta.bit_count() != n_beta:
        raise ValueError(
            f"det {det} has ({alpha.bit_count()},{beta.bit_count()}) electrons, "
            f"expected ({n_alpha},{n_beta})")
    doubly = alpha & beta
    singles_mask = alpha ^ beta
    single_orbs = [p for p in range(singles_mask.bit_length()) if (singles_mask >> p) & 1]
    n_single_alpha = n_alpha - doubly.bit_count()
    if not (0 <= n_single_alpha <= len(single_orbs)):
        raise ValueError(f"det {det} has an inconsistent occupation for its own sector")
    partners = set()
    for combo in itertools.combinations(single_orbs, n_single_alpha):
        combo_mask = 0
        for p in combo:
            combo_mask |= 1 << p
        new_alpha = doubly | combo_mask
        new_beta = doubly | (singles_mask ^ combo_mask)
        partners.add((new_alpha, new_beta))
    return tuple(sorted(partners))


def close_spin_complete(dets: Sequence[Determinant], *, n_alpha: int,
                        n_beta: int) -> tuple[tuple[Determinant, ...], int]:
    """Union of ``spin_exchange_partners`` over every determinant in ``dets``.

    Returns ``(closed_dets, overhead)`` where ``overhead`` is how many EXTRA
    determinants closure added beyond the original (deduplicated) set -- callers must
    report this, never silently exceed a requested cap."""
    original = {tuple(map(int, d)) for d in dets}
    closed: set[Determinant] = set()
    for det in original:
        closed.update(spin_exchange_partners(det, n_alpha=n_alpha, n_beta=n_beta))
    closed_sorted = tuple(sorted(closed))
    return closed_sorted, len(closed_sorted) - len(original)


def spatial_occupation_orbit_count(dets: Sequence[Determinant]) -> int:
    """Number of distinct spatial-occupation orbits (doubly/singly occupied masks)
    represented in ``dets`` -- the unit that spin closure keeps indivisible."""
    orbits = {(int(alpha) & int(beta), int(alpha) ^ int(beta)) for alpha, beta in dets}
    return len(orbits)


def s_squared_diagonal(det: Determinant, *, n_alpha: int, n_beta: int) -> float:
    """<D|S^2|D> = Sz(Sz+1) + N_beta - n_doubly_occupied, for same-spatial-orbital
    (non-UHF-broken) alpha/beta strings."""
    alpha, beta = int(det[0]), int(det[1])
    sz = 0.5 * (n_alpha - n_beta)
    n_double = (alpha & beta).bit_count()
    return sz * (sz + 1.0) + n_beta - n_double


def s_squared_expectation(dets: Sequence[Determinant], coeffs, *, n_alpha: int,
                          n_beta: int, max_open_shell_orbitals: int = 20) -> float:
    """<Psi|S^2|Psi> for Psi = sum_i coeffs[i]|dets[i]>, grouped by spatial configuration
    (S^2 cannot connect determinants from different configurations -- Ŝ₋Ŝ₊'s off-diagonal
    action only ever reassigns the spin of two already-singly-occupied orbitals, touching
    no other orbital).

    See the module docstring: this is an EXPLICIT-DETERMINANT-BASIS quantity only. It is
    not the complete FRASCI hybrid wavefunction's spin unless ``dets``/``coeffs`` already
    represent that complete wavefunction in an explicit orthonormal basis (e.g. the exact
    union solve's ``union_dets``/``union_coefficients``, which DOES expand the full
    parent + recovery space). Called on the recovery determinants alone, the result must
    be reported as ``explicit_block_spin_square_partial``, never as the full spin.

    The off-diagonal sign within one configuration is NOT a universal -1 (verified
    against ``pyscf.fci.spin_op`` during development: a naive constant-sign guess
    mismatched pyscf on a real 3-member orbit -- the mixed alpha/beta spin-exchange
    operator does not factor into independent per-channel Slater-Condon parities the way
    an ordinary double excitation does). Rather than re-deriving that fermionic sign by
    hand, each configuration's off-diagonal block is computed exactly via a LOCAL
    embedding: pyscf's own ``contract_ss`` restricted to just that configuration's own
    open-shell (singly occupied) orbitals, remapped to compact local indices --
    spectator doubly-occupied/vacant orbitals never enter S^2's off-diagonal action, so
    this reproduces the full-system element exactly (verified numerically against the
    non-embedded full-system matrix). Bounded by ``max_open_shell_orbitals`` since the
    local dense array is O(2^k).
    """
    from pyscf.fci import cistring, spin_op

    dets = [tuple(map(int, d)) for d in dets]
    coeffs = np.asarray(coeffs, dtype=float)
    for alpha, beta in dets:
        if alpha.bit_count() != n_alpha or beta.bit_count() != n_beta:
            raise ValueError(f"det ({alpha},{beta}) does not match sector ({n_alpha},{n_beta})")

    total = sum(coeffs[i] * coeffs[i] * s_squared_diagonal(det, n_alpha=n_alpha, n_beta=n_beta)
               for i, det in enumerate(dets))

    by_configuration: dict[tuple[int, int], list[int]] = {}
    for i, (alpha, beta) in enumerate(dets):
        by_configuration.setdefault((alpha | beta, alpha & beta), []).append(i)

    for (union, doubly), indices in by_configuration.items():
        if len(indices) < 2:
            continue
        singles_mask = union ^ doubly
        open_orbs = [p for p in range(singles_mask.bit_length()) if (singles_mask >> p) & 1]
        k = len(open_orbs)
        if k == 0:
            continue
        if k > max_open_shell_orbitals:
            raise NotImplementedError(
                f"configuration has {k} open-shell orbitals; the local S^2 embedding is "
                f"bounded at max_open_shell_orbitals={max_open_shell_orbitals}")
        local_pos = {orb: p for p, orb in enumerate(open_orbs)}
        local_alpha = []
        for i in indices:
            alpha_i = dets[i][0]
            la = 0
            for orb in open_orbs:
                if (alpha_i >> orb) & 1:
                    la |= 1 << local_pos[orb]
            local_alpha.append(la)
        local_na = local_alpha[0].bit_count()
        local_nb = k - local_na
        full_mask = (1 << k) - 1
        lca = list(map(int, cistring.make_strings(range(k), local_na)))
        lcb = list(map(int, cistring.make_strings(range(k), local_nb)))
        local_positions = [(lca.index(la), lcb.index(full_mask ^ la)) for la in local_alpha]

        for a_idx, i in enumerate(indices):
            ia, ib = local_positions[a_idx]
            basis = np.zeros((len(lca), len(lcb)))
            basis[ia, ib] = 1.0
            acted = spin_op.contract_ss(basis, k, (local_na, local_nb))
            for b_idx in range(a_idx + 1, len(indices)):
                j = indices[b_idx]
                ja, jb = local_positions[b_idx]
                element = acted[ja, jb]
                if element:
                    total += 2.0 * coeffs[i] * coeffs[j] * element
    return float(total)
