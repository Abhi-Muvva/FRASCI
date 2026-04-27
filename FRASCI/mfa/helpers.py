from __future__ import annotations

import numpy as np


def compute_fragment_rdm1(dets: list, coeffs: list, n_orb_frag: int) -> np.ndarray:
    from trimci.trimci_core import compute_1rdm
    gamma_flat = compute_1rdm(dets, list(coeffs), n_orb_frag)
    return np.asarray(gamma_flat, dtype=np.float64).reshape(n_orb_frag, n_orb_frag)


def dress_integrals_meanfield(
    h1_frag: np.ndarray,
    eri_full: np.ndarray,
    fragment_orbs: list[int],
    external_rdm1_diag: np.ndarray,
    external_orbs: list[int],
) -> np.ndarray:
    """
    Fock mean-field correction from external orbitals to h1_frag.
    h1_eff[p,q] += Σ_{r ∈ external} γ_r · (eri[p,q,r,r] − 0.5·eri[p,r,r,q])
    γ_r is spin-summed ∈ [0,2]; uses J − ½K form (not 2J−K).
    """
    fa = np.asarray(fragment_orbs, dtype=np.intp)
    ea = np.asarray(external_orbs, dtype=np.intp)
    gamma_r = np.asarray(external_rdm1_diag, dtype=np.float64)
    assert gamma_r.shape == (ea.shape[0],), \
        f"gamma length {gamma_r.shape} != external_orbs length {ea.shape}"

    J_block = eri_full[np.ix_(fa, fa, ea, ea)]
    J_diag  = np.diagonal(J_block, axis1=2, axis2=3)
    J_term  = np.einsum('pqr,r->pq', J_diag, gamma_r)

    K_block = eri_full[np.ix_(fa, ea, ea, fa)]
    K_diag  = np.diagonal(K_block, axis1=1, axis2=2)
    K_term  = 0.5 * np.einsum('pqr,r->pq', K_diag, gamma_r)

    h1_eff = h1_frag + J_term - K_term
    assert np.allclose(h1_eff, h1_eff.T, atol=1e-12), \
        "h1_eff lost symmetry — indexing bug in dress_integrals_meanfield"
    return h1_eff


def assemble_global_rdm1_diag(
    fragment_rdm1s: list[np.ndarray],
    fragment_orbs_list: list[list[int]],
    n_orb: int,
    n_elec: int,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> np.ndarray:
    """
    Build global γ diagonal by averaging fragment RDMs over overlapping orbitals,
    then renormalizing to conserve total electron count.
    """
    total = np.zeros(n_orb, dtype=np.float64)
    count = np.zeros(n_orb, dtype=np.int64)
    for gamma_F, frag_orbs in zip(fragment_rdm1s, fragment_orbs_list):
        for local_idx, full_idx in enumerate(frag_orbs):
            total[full_idx] += float(gamma_F[local_idx, local_idx])
            count[full_idx] += 1

    global_diag = np.zeros(n_orb, dtype=np.float64)
    for r in range(n_orb):
        if count[r] > 0:
            global_diag[r] = total[r] / count[r]
        else:
            global_diag[r] = int((ref_alpha_bits >> r) & 1) + int((ref_beta_bits >> r) & 1)

    gamma_sum = global_diag.sum()
    if gamma_sum > 0:
        global_diag *= n_elec / gamma_sum
    global_diag = np.clip(global_diag, 0.0, 2.0)
    return global_diag
