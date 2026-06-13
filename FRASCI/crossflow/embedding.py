from __future__ import annotations

import numpy as np


def dress_fragment_h1(
    h1_bare_frag: np.ndarray,
    eri_full: np.ndarray,
    frag_orbs: list[int],
    gamma: np.ndarray,
    n_orb: int,
) -> np.ndarray:
    """Return the fragment one-electron Hamiltonian dressed by its environment."""
    fa = np.array(frag_orbs, dtype=np.intp)
    if h1_bare_frag.shape != (len(fa), len(fa)):
        raise ValueError("h1_bare_frag shape must match frag_orbs")
    if gamma.shape != (n_orb, n_orb):
        raise ValueError("gamma shape must be (n_orb, n_orb)")
    if eri_full.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("eri_full shape must be (n_orb, n_orb, n_orb, n_orb)")

    all_orbs = np.arange(n_orb)
    gamma_ext = np.array(gamma, copy=True)
    gamma_ext[np.ix_(fa, fa)] = 0.0

    J = np.einsum(
        "rs,pqrs->pq",
        gamma_ext,
        eri_full[np.ix_(fa, fa, all_orbs, all_orbs)],
    )
    K = 0.5 * np.einsum(
        "rs,pqrs->pq",
        gamma_ext,
        eri_full.transpose(0, 3, 2, 1)[np.ix_(fa, fa, all_orbs, all_orbs)],
    )
    return np.array(h1_bare_frag, copy=True) + J - K


def apply_delta_h(h1_mfa_dressed: np.ndarray, delta_h_diag: np.ndarray) -> np.ndarray:
    """Apply a diagonal self-energy correction without mutating the input."""
    if h1_mfa_dressed.ndim != 2 or h1_mfa_dressed.shape[0] != h1_mfa_dressed.shape[1]:
        raise ValueError("h1_mfa_dressed must be a square matrix")
    if delta_h_diag.shape != (h1_mfa_dressed.shape[0],):
        raise ValueError("delta_h_diag length must match h1_mfa_dressed")
    return np.array(h1_mfa_dressed, copy=True) + np.diag(delta_h_diag)


def mf_global_energy(
    h1: np.ndarray,
    eri: np.ndarray,
    gamma: np.ndarray,
    E_nuc: float,
) -> tuple[float, float, np.ndarray]:
    """Compute global mean-field total/electronic energies and Fock matrix."""
    J = np.einsum("rs,pqrs->pq", gamma, eri)
    K = 0.5 * np.einsum("rs,psrq->pq", gamma, eri)
    F = h1 + J - K
    E_elec = 0.5 * float(np.einsum("pq,pq->", h1 + F, gamma))
    return float(E_nuc) + E_elec, E_elec, F


def mf_embedded_energy(
    h1_solver: np.ndarray,
    eri_frag: np.ndarray,
    gamma_frag: np.ndarray,
) -> float:
    """Compute the mean-field energy for the Hamiltonian sent to a fragment solver."""
    J = np.einsum("rs,pqrs->pq", gamma_frag, eri_frag)
    K = 0.5 * np.einsum("rs,psrq->pq", gamma_frag, eri_frag)
    F_emb = h1_solver + J - K
    return 0.5 * float(np.einsum("pq,pq->", h1_solver + F_emb, gamma_frag))


def correlation_total_energy(
    E_mf_global: float,
    E_trimci_list: list[float],
    E_mf_emb_list: list[float],
) -> tuple[float, list[float]]:
    """Add fragment correlation corrections to the global mean-field energy."""
    E_corr = [
        float(e_ci - e_mf)
        for e_ci, e_mf in zip(E_trimci_list, E_mf_emb_list, strict=True)
    ]
    return float(E_mf_global) + sum(E_corr), E_corr
