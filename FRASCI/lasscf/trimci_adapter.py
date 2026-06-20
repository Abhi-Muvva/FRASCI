"""
trimci_adapter.py
=================
Clean interface for calling TrimCI on fragment integrals.

TrimCI expects integrals in CHEMIST notation: eri[p,q,r,s] = (pq|rs).
The `read_fcidump` in TrimCI_runner/io_utils.py already returns chemist-
notation arrays, so no conversion is needed when reading FCIDUMP files.

When integrals come from fragment.py (extract_fragment_integrals), they
are also in chemist notation by construction.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimci
from pyscf.fci import direct_spin1
from pyscf.tools.fcidump import from_integrals


@dataclass
class FragmentResult:
    """Result from a single fragment TrimCI solve."""
    energy: float
    n_dets: int
    dets: list       # TrimCI Determinant objects
    coeffs: list     # CI coefficients (parallel to dets)
    n_orb_frag: int
    n_alpha_frag: int
    n_beta_frag: int


DEFAULT_CONFIG = {
    "threshold": 0.06,
    "max_final_dets": "auto",
    "max_rounds": 2,
    "num_runs": 1,
    "pool_build_strategy": "heat_bath",
    "verbose": False,
}


def solve_fragment_trimci(
    h1_frag: np.ndarray,
    eri_frag: np.ndarray,
    n_alpha_frag: int,
    n_beta_frag: int,
    n_orb_frag: int,
    config: Optional[dict] = None,
) -> FragmentResult:
    """
    Run TrimCI on a single fragment.

    Parameters
    ----------
    h1_frag      : (n_orb_frag, n_orb_frag) one-body integrals, chemist notation
    eri_frag     : (n_orb_frag,)*4 two-body integrals, chemist notation (pq|rs)
    n_alpha_frag : alpha electrons in this fragment
    n_beta_frag  : beta electrons in this fragment
    n_orb_frag   : orbitals in this fragment
    config       : optional TrimCI config overrides, e.g.
                   {'ndets': 2000, 'goal': 'balanced'}
                   Defaults to TrimCI auto mode.

    Returns
    -------
    FragmentResult with energy, n_dets, dets, coeffs

    Notes
    -----
    Uses trimci.run_full or trimci.TrimCI_runner.run_auto.run_auto depending
    on whether an FCIDUMP file exists. For in-memory fragment integrals we
    write a temporary FCIDUMP and call run_auto, or call the lower-level
    screening/trim API directly.
    """
    effective_config = {**DEFAULT_CONFIG, **(config or {})}

    # Write temp FCIDUMP
    fd, tmp_path = tempfile.mkstemp(suffix=".fcidump")
    os.close(fd)
    try:
        from_integrals(
            tmp_path,
            h1_frag,
            eri_frag,
            nmo=n_orb_frag,
            nelec=n_alpha_frag + n_beta_frag,
            nuc=0.0,
            ms=n_alpha_frag - n_beta_frag,
        )
        energy, dets, coeffs_list, details, run_args = trimci.run_full(
            fcidump_path=tmp_path,
            config_dict=effective_config,
        )
    finally:
        os.unlink(tmp_path)

    return FragmentResult(
        energy=float(energy),
        n_dets=len(dets),
        dets=list(dets),
        coeffs=list(coeffs_list),
        n_orb_frag=n_orb_frag,
        n_alpha_frag=n_alpha_frag,
        n_beta_frag=n_beta_frag,
    )


def solve_fragment_exact(
    h1_frag: np.ndarray,
    eri_frag: np.ndarray,
    n_alpha_frag: int,
    n_beta_frag: int,
    n_orb_frag: int,
) -> FragmentResult:
    """
    Exact diagonalization fallback for small fragments (n_orb_frag <= ~14).

    Uses PySCF FCI (direct_spin1). Provides a reference to validate
    TrimCI results on small fragments before scaling to Fe4S4 sizes.

    Parameters
    ----------
    (same as solve_fragment_trimci, minus config)

    Returns
    -------
    FragmentResult with energy and dets=[], coeffs=[] (FCI gives a dense
    vector, not a sparse determinant list).
    """
    if n_orb_frag > 14:
        raise ValueError(f"exact solver only supported for n_orb <= 14, got {n_orb_frag}")

    fci_solver = direct_spin1.FCISolver()
    e, civec = fci_solver.kernel(
        h1_frag, eri_frag, n_orb_frag, (n_alpha_frag, n_beta_frag),
        ecore=0.0,
    )
    civec_flat = np.asarray(civec).ravel()

    return FragmentResult(
        energy=float(e),
        n_dets=civec_flat.size,
        dets=[],         # FCI gives dense vector, not sparse dets
        coeffs=civec_flat.tolist(),
        n_orb_frag=n_orb_frag,
        n_alpha_frag=n_alpha_frag,
        n_beta_frag=n_beta_frag,
    )


if __name__ == "__main__":
    import numpy as np

    # 4-orbital toy system, 1 alpha + 1 beta electron
    rng = np.random.default_rng(0)
    n = 4
    A = rng.random((n, n))
    h1 = (A + A.T) * 0.1 - 0.5 * np.eye(n)   # negative diagonal to ensure bound state
    # Build chemist-symmetric ERI
    eri = rng.random((n, n, n, n)) * 0.05
    eri = eri + eri.transpose(1,0,2,3)
    eri = eri + eri.transpose(0,1,3,2)
    eri = eri + eri.transpose(2,3,0,1)

    print("Testing solve_fragment_exact...")
    r_exact = solve_fragment_exact(h1, eri, 1, 1, n)
    print(f"  exact energy = {r_exact.energy:.6f} Ha, n_dets = {r_exact.n_dets}")

    print("Testing solve_fragment_trimci...")
    r_trimci = solve_fragment_trimci(h1, eri, 1, 1, n)
    print(f"  TrimCI energy = {r_trimci.energy:.6f} Ha, n_dets = {r_trimci.n_dets}")

    diff = abs(r_exact.energy - r_trimci.energy)
    print(f"  Energy diff = {diff:.2e} Ha")
    assert diff < 1e-3, f"Energies differ by {diff:.4e} — too large"
    print("PASS: adapter self-test")
