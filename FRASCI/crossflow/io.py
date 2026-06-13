from __future__ import annotations

from typing import Any, Optional
import warnings

import numpy as np

from FRASCI.crossflow.types import GammaResult


def load_fcidump(path: str) -> tuple[np.ndarray, np.ndarray, int, int, float, int, int]:
    import trimci

    h1, eri, n_elec, n_orb, e_nuc, n_alpha, n_beta, _psym = trimci.read_fcidump(
        path
    )
    return h1, eri, n_elec, n_orb, e_nuc, n_alpha, n_beta


def _aufbau_gamma(
    h1: np.ndarray,
    n_alpha: int,
    n_beta: int,
    n_orb: int,
) -> np.ndarray:
    order = np.argsort(np.diag(h1))
    occ = np.zeros(n_orb)
    for i in range(n_alpha):
        occ[order[i]] += 1.0
    for i in range(n_beta):
        occ[order[i]] += 1.0
    return np.diag(occ)


def _aufbau_bits(h1: np.ndarray, n_alpha: int, n_beta: int) -> tuple[int, int]:
    order = np.argsort(np.diag(h1))
    alpha_bits = sum(1 << int(order[i]) for i in range(n_alpha))
    beta_bits = sum(1 << int(order[i]) for i in range(n_beta))
    return alpha_bits, beta_bits


def load_or_compute_gamma(
    gamma_path: Optional[str],
    h1: np.ndarray,
    eri: np.ndarray,
    n_alpha: int,
    n_beta: int,
    n_orb: int,
) -> GammaResult:
    if gamma_path is not None:
        raw = np.load(gamma_path)
        if raw.shape == (n_orb,):
            return GammaResult(
                gamma=np.diag(raw),
                gamma_source_mode="provided_file",
                gamma_load_mode="diagonal_vector_promoted_to_matrix",
                uhf_cache=None,
            )
        if raw.shape == (n_orb, n_orb):
            return GammaResult(
                gamma=raw,
                gamma_source_mode="provided_file",
                gamma_load_mode="full_matrix",
                uhf_cache=None,
            )
        raise ValueError(f"gamma shape {raw.shape} incompatible with n_orb={n_orb}")

    try:
        from pyscf import gto, scf

        mol = gto.M()
        mol.nelectron = n_alpha + n_beta
        mol.spin = n_alpha - n_beta
        mol.verbose = 0

        mf = scf.UHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(n_orb)
        mf._eri = eri.ravel()
        mf.kernel()

        if mf.converged:
            gamma_a = np.einsum(
                "pi,i,qi->pq", mf.mo_coeff[0], mf.mo_occ[0], mf.mo_coeff[0]
            )
            gamma_b = np.einsum(
                "pi,i,qi->pq", mf.mo_coeff[1], mf.mo_occ[1], mf.mo_coeff[1]
            )
            return GammaResult(
                gamma=gamma_a + gamma_b,
                gamma_source_mode="computed_uhf",
                gamma_load_mode="n/a",
                uhf_cache=mf,
            )
    except Exception:
        pass

    warnings.warn(
        "[crossflow] UHF failed or PySCF unavailable - using Aufbau gamma. "
        "This is approximate and not self-consistent mean-field.",
        stacklevel=2,
    )
    return GammaResult(
        gamma=_aufbau_gamma(h1, n_alpha, n_beta, n_orb),
        gamma_source_mode="computed_aufbau",
        gamma_load_mode="n/a",
        uhf_cache=None,
    )


def load_or_derive_ref_det(
    ref_dets_path: Optional[str],
    h1: np.ndarray,
    eri: np.ndarray,
    n_alpha: int,
    n_beta: int,
    n_orb: int,
    uhf_cache: Optional[Any] = None,
) -> tuple[int, int, str]:
    if ref_dets_path is not None:
        data = np.load(ref_dets_path)
        dets = data["dets"]
        return int(dets[0, 0]), int(dets[0, 1]), "dets_npz"

    warnings.warn(
        "[crossflow] No ref_dets provided - using Aufbau reference bitstrings "
        "(FCIDUMP orbital basis, lowest orbitals by h1 diagonal). Approximate.",
        stacklevel=2,
    )
    alpha_bits, beta_bits = _aufbau_bits(h1, n_alpha, n_beta)
    return alpha_bits, beta_bits, "computed_aufbau"
