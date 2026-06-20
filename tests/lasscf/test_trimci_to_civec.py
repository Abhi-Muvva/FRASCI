"""
test_trimci_to_civec.py
=======================
Integration tests for FRASCI/lasscf/trimci_to_civec.py.

Runs actual TrimCI on H4/STO-3G (generated on the fly via PySCF).
No large fixtures are committed.

Test suite
----------
1. test_norm_preserved           — norm of decoded civec == norm of TrimCI coeffs
2. test_energy_matches_trimci    — E(PySCF h1/h2, decoded civec) == TrimCI energy (1e-8 Ha)
3. test_matches_full_fci         — |<civec_trimci | civec_pyscf>| == 1.0 (1e-6)
4. test_h1diag_fragment_shape    — civec shape correct for 12-orb mock fragment
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure project root is importable
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from math import comb

from pyscf import gto, scf, tools
from pyscf.fci import direct_spin1, cistring

from FRASCI.lasscf.trimci_to_civec import trimci_to_pyscf_civec
from FRASCI.lasscf.trimci_adapter import solve_fragment_trimci


# ---------------------------------------------------------------------------
# Shared H4/STO-3G fixture helpers
# ---------------------------------------------------------------------------

def _build_h4_integrals():
    """
    Build H4/STO-3G h1 and h2 (chemist convention, 4D restored) in-memory.
    Returns (h1, h2, norb, nelec) where nelec = (2, 2).
    """
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    norb = mol.nao_nr()
    nelec = (mol.nelectron // 2, mol.nelectron // 2)

    # Get integrals in MO basis (MO = AO for H4/STO-3G HF MOs)
    from pyscf import ao2mo
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False)
    h2 = eri.reshape(norb, norb, norb, norb)

    return h1, h2, norb, nelec


def _run_trimci_h4(h1, h2, norb, nelec, threshold=1e-8):
    """
    Run TrimCI on H4 with very small threshold so essentially all dets are kept.
    Returns FragmentResult.
    """
    n_alpha, n_beta = nelec
    # Use a very small threshold to approach full FCI
    config = {
        "threshold": threshold,
        "max_final_dets": "auto",
        "max_rounds": 3,
        "num_runs": 1,
        "verbose": False,
    }
    result = solve_fragment_trimci(
        h1_frag=h1,
        eri_frag=h2,
        n_alpha_frag=n_alpha,
        n_beta_frag=n_beta,
        n_orb_frag=norb,
        config=config,
    )
    return result


# ---------------------------------------------------------------------------
# Test 1: Norm preserved
# ---------------------------------------------------------------------------

def test_norm_preserved():
    """
    Decoded civec norm must equal TrimCI coeffs norm regardless of sign convention.
    """
    h1, h2, norb, nelec = _build_h4_integrals()
    result = _run_trimci_h4(h1, h2, norb, nelec)

    civec = trimci_to_pyscf_civec(result.dets, result.coeffs, norb, nelec)

    norm_trimci = float(np.sum(np.array(result.coeffs, dtype=np.float64) ** 2))
    norm_civec = float(np.sum(civec ** 2))

    assert abs(norm_civec - norm_trimci) < 1e-12, (
        f"Norm mismatch: civec norm={norm_civec:.15e}, "
        f"TrimCI coeffs norm={norm_trimci:.15e}"
    )


# ---------------------------------------------------------------------------
# Test 2: Energy matches TrimCI (the most important correctness test)
# ---------------------------------------------------------------------------

def test_energy_matches_trimci():
    """
    Energy computed via PySCF direct_spin1.energy on decoded civec must match
    TrimCI's returned electronic energy to 1e-8 Ha.

    This certifies the bitstring decoder places each coefficient in the correct
    (addr_a, addr_b) slot of the FCI vector.
    """
    h1, h2, norb, nelec = _build_h4_integrals()
    result = _run_trimci_h4(h1, h2, norb, nelec)

    civec = trimci_to_pyscf_civec(result.dets, result.coeffs, norb, nelec)

    # Normalize civec to unit norm before energy evaluation
    norm = np.sqrt(np.sum(civec ** 2))
    civec_norm = civec / norm

    e_pyscf = direct_spin1.energy(h1, h2, civec_norm, norb, nelec)

    assert abs(e_pyscf - result.energy) < 1e-8, (
        f"Energy mismatch: pyscf_energy={e_pyscf:.10f}, "
        f"trimci_energy={result.energy:.10f}, "
        f"diff={abs(e_pyscf - result.energy):.2e} Ha"
    )


# ---------------------------------------------------------------------------
# Test 3: Cardinal test — matches full PySCF FCI to global phase
# ---------------------------------------------------------------------------

def test_matches_full_fci():
    """
    With threshold -> 0, TrimCI approaches full FCI.  The decoded civec must
    have |overlap| with PySCF FCI civec = 1.0 to 1e-6 (global phase allowed).

    This catches any per-determinant sign or index-ordering errors.
    """
    h1, h2, norb, nelec = _build_h4_integrals()
    result = _run_trimci_h4(h1, h2, norb, nelec, threshold=1e-8)

    civec_trimci = trimci_to_pyscf_civec(result.dets, result.coeffs, norb, nelec)

    # Normalize
    norm_t = np.sqrt(np.sum(civec_trimci ** 2))
    civec_trimci_norm = civec_trimci / norm_t

    # Full PySCF FCI reference
    e_fci, civec_pyscf = direct_spin1.kernel(h1, h2, norb, nelec)
    # direct_spin1.kernel returns normalized civec
    norm_p = np.sqrt(np.sum(civec_pyscf ** 2))
    civec_pyscf_norm = civec_pyscf / norm_p

    overlap = abs(float(np.dot(civec_trimci_norm.ravel(), civec_pyscf_norm.ravel())))

    assert overlap > 1.0 - 1e-6, (
        f"TrimCI civec does not match PySCF FCI: |overlap|={overlap:.8f} "
        f"(expected >= {1.0 - 1e-6:.6f}). "
        f"TrimCI n_dets={result.n_dets}, civec shape={civec_trimci.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4: Shape correctness for 12-orb mock fragment
# ---------------------------------------------------------------------------

def _sym_h1(n, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    h1 = (A + A.T) * 0.5 - 1.0 * np.eye(n)  # negative diagonal for bound state
    return h1


def _sym_eri(n, seed=43):
    """Build a fully chemist-symmetric ERI (pq|rs)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n, n, n)) * 0.05
    A = A + A.transpose(1, 0, 2, 3)
    A = A + A.transpose(0, 1, 3, 2)
    A = A + A.transpose(2, 3, 0, 1)
    return A / 4.0


@pytest.mark.parametrize("nelec", [(6, 6), (5, 7), (7, 5)])
def test_h1diag_fragment_shape(nelec):
    """
    Run TrimCI on a 12-orbital mock fragment and confirm the decoded civec
    has the correct shape (C(12, na), C(12, nb)) with no crashes.
    """
    norb = 12
    n_alpha, n_beta = nelec
    h1 = _sym_h1(norb)
    h2 = _sym_eri(norb)

    config = {
        "threshold": 0.06,
        "max_final_dets": 200,
        "max_rounds": 2,
        "num_runs": 1,
        "verbose": False,
    }
    result = solve_fragment_trimci(
        h1_frag=h1,
        eri_frag=h2,
        n_alpha_frag=n_alpha,
        n_beta_frag=n_beta,
        n_orb_frag=norb,
        config=config,
    )

    civec = trimci_to_pyscf_civec(result.dets, result.coeffs, norb, nelec)

    expected_na = comb(norb, n_alpha)
    expected_nb = comb(norb, n_beta)

    assert civec.shape == (expected_na, expected_nb), (
        f"civec shape {civec.shape} != expected ({expected_na}, {expected_nb}) "
        f"for norb={norb}, nelec={nelec}"
    )
    assert civec.dtype == np.float64

    # Also confirm at least one non-zero entry
    assert np.any(civec != 0.0), "civec is all zeros — no dets were decoded"

    # Confirm norm is reasonable (TrimCI should return near-unit-norm coeffs)
    norm = np.sqrt(np.sum(civec ** 2))
    assert norm > 0.1, f"civec norm too small: {norm}"
