"""
energy.py — MFA-TrimCI energy functions (Phase D).

All functions use chemist-notation ERIs (pq|rs).
gamma is spin-summed: gamma[p,p] in [0, 2].
"""
from __future__ import annotations
import numpy as np


def build_fock(h1: np.ndarray, eri: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Build the Fock matrix from h1, ERI, and spin-summed 1-RDM gamma.

    F[p,q] = h1[p,q]
           + sum_{r,s} gamma[r,s] * eri[p,q,r,s]          (J term)
           - 0.5 * sum_{r,s} gamma[r,s] * eri[p,s,r,q]    (K term)

    Parameters
    ----------
    h1 : (n, n) ndarray — one-electron integrals
    eri : (n, n, n, n) ndarray — two-electron integrals in chemist notation (pq|rs)
    gamma : (n, n) ndarray — spin-summed 1-RDM

    Returns
    -------
    F : (n, n) ndarray — Fock matrix
    """
    J = np.einsum("rs,pqrs->pq", gamma, eri)
    K = 0.5 * np.einsum("rs,psrq->pq", gamma, eri)
    return h1 + J - K


def mf_global_energy(
    h1: np.ndarray,
    eri: np.ndarray,
    gamma_mixed: np.ndarray,
    E_nuc: float,
) -> tuple[float, float, np.ndarray]:
    """Compute the global mean-field (HF/DFT) total energy.

    Parameters
    ----------
    h1 : (n, n) ndarray — one-electron integrals
    eri : (n, n, n, n) ndarray — two-electron integrals in chemist notation
    gamma_mixed : (n, n) ndarray — spin-summed global 1-RDM
    E_nuc : float — nuclear repulsion energy (from FCIDUMP)

    Returns
    -------
    E_mf_global : float — total mean-field energy (electronic + nuclear)
    E_mf_global_elec : float — electronic mean-field energy only
    F_full : (n, n) ndarray — full Fock matrix
    """
    F_full = build_fock(h1, eri, gamma_mixed)
    E_mf_global_elec = 0.5 * float(np.einsum("pq,pq->", h1 + F_full, gamma_mixed))
    return E_nuc + E_mf_global_elec, E_mf_global_elec, F_full


def mf_rowpartition_energy(
    h1: np.ndarray,
    F_full: np.ndarray,
    gamma_mixed: np.ndarray,
    frag_orbs: list[int],
    n_orb: int,
) -> float:
    """Row-partition diagnostic energy for fragment I.

    For non-overlapping partitions covering all orbitals,
    sum_I mf_rowpartition_energy(...) == E_mf_global_elec  (electronic only).

    NOT used in E_total — diagnostic/decomposition only.

    Parameters
    ----------
    h1 : (n, n) ndarray — one-electron integrals
    F_full : (n, n) ndarray — full Fock matrix
    gamma_mixed : (n, n) ndarray — spin-summed global 1-RDM
    frag_orbs : list of int — orbital indices for fragment I
    n_orb : int — total number of orbitals

    Returns
    -------
    float — fragment row-partition energy contribution
    """
    rows = np.array(frag_orbs)
    cols = np.arange(n_orb)
    return 0.5 * float(np.einsum(
        "pq,pq->",
        (h1 + F_full)[np.ix_(rows, cols)],
        gamma_mixed[np.ix_(rows, cols)],
    ))


def mf_embedded_energy(
    h1_dressed: np.ndarray,
    eri_frag: np.ndarray,
    gamma_frag: np.ndarray,
) -> float:
    """Mean-field energy of fragment I on its embedded Hamiltonian.

    Apples-to-apples with E_TrimCI_I — both use the same embedded Hamiltonian
    (h1_dressed, eri_frag).

    Parameters
    ----------
    h1_dressed : (m, m) ndarray — dressed one-electron integrals for fragment
    eri_frag : (m, m, m, m) ndarray — fragment two-electron integrals
    gamma_frag : (m, m) ndarray — fragment spin-summed 1-RDM

    Returns
    -------
    float — mean-field energy on the embedded Hamiltonian
    """
    F_emb = build_fock(h1_dressed, eri_frag, gamma_frag)
    return 0.5 * float(np.einsum("pq,pq->", h1_dressed + F_emb, gamma_frag))


def correlation_total_energy(
    E_mf_global: float,
    E_trimci_list: list[float],
    E_mf_emb_list: list[float],
) -> tuple[float, list[float]]:
    """Compute total correlation-corrected energy and per-fragment corrections.

    E_total = E_mf_global + sum_I (E_TrimCI_I - E_mf_emb_I)

    E_nuc is already inside E_mf_global — do NOT add it again.

    Parameters
    ----------
    E_mf_global : float — global MF total energy (includes E_nuc)
    E_trimci_list : list of float — TrimCI energies per fragment
    E_mf_emb_list : list of float — MF embedded energies per fragment

    Returns
    -------
    E_total : float — correlation-corrected total energy
    E_corr_list : list of float — per-fragment correlation corrections
    """
    E_corr_list = [float(e_ci - e_mf) for e_ci, e_mf in zip(E_trimci_list, E_mf_emb_list)]
    return E_mf_global + sum(E_corr_list), E_corr_list
