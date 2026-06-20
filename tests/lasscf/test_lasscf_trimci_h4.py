"""
test_lasscf_trimci_h4.py
========================
Phase 3c validation: TrimCI kernel closure in LASSCF on H4/STO-3G.

H4 geometry: linear chain, H-H spacing 1.4 Bohr, 4 spatial orbitals, (2,2) electrons.
All integrals generated on the fly via PySCF — no committed fixtures.

Test suite
----------
1. test_h4_single_fragment_trimci_kernel_matches_full_fci
       LASSCF+TrimCI (1 fragment = full space) vs PySCF full FCI.
       Assert |e_tot - e_fci| < 1e-6 Ha.

2. test_h4_single_fragment_trimci_matches_csf_solver
       LASSCF+TrimCI vs LASSCF+default-csf-solver on same 1-fragment system.
       Assert |e_trimci - e_csf| < 1e-6 Ha.

3. test_kernel_signature_contract
       Unit test: call kernel closure directly on H2/STO-3G integrals.
       Assert shapes, finite etot, 1-RDM trace = N, 2-RDM finite + partial symmetry.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pytest

# Ensure project root is importable
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from pyscf import gto, scf, ao2mo
from pyscf.fci import direct_spin1
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox

from FRASCI.lasscf.trimci_kernel import make_trimci_kernel_for_fragment


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_h4_mol_mf():
    """
    Build H4/STO-3G: linear chain with 1.4 Bohr spacing.
    Returns (mol, mf) with mf.kernel() already called.
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
    return mol, mf


def _h4_full_fci_energy(mol, mf):
    """
    Compute full FCI energy for H4 in MO basis (includes nuclear repulsion).
    Uses ecore=mol.energy_nuc() to match LASSCF total energy convention.
    """
    norb = mol.nao_nr()
    na = mol.nelectron // 2
    nb = mol.nelectron // 2
    nelec = (na, nb)
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False)
    h2 = eri.reshape(norb, norb, norb, norb)
    e_fci, _ = direct_spin1.kernel(h1, h2, norb, nelec, ecore=mol.energy_nuc())
    return e_fci


def _build_h4_las_trimci(mol, mf, threshold=1e-8):
    """
    Build LASSCF (lasscf_rdm.LASSCF) with TrimCI kernel for H4, 1 fragment.
    Injects TrimCI into fciboxes[0].
    """
    las = LASSCF(mf, ncas_sub=(4,), nelecas_sub=[(2, 2)], spin_sub=(1,))
    las.verbose = 0
    las.fciboxes[0] = make_fcibox(
        mol,
        kernel=make_trimci_kernel_for_fragment(threshold=threshold),
        spin=0,
        smult=1,
    )
    return las


def _build_h4_las_csf(mol, mf):
    """
    Build LASSCF (lasscf_rdm.LASSCF) with default csf_solver for H4, 1 fragment.
    No fciboxes injection — uses mrh built-in.
    """
    las = LASSCF(mf, ncas_sub=(4,), nelecas_sub=[(2, 2)], spin_sub=(1,))
    las.verbose = 0
    return las


# ---------------------------------------------------------------------------
# Test 1: LASSCF+TrimCI vs full PySCF FCI
# ---------------------------------------------------------------------------

def test_h4_single_fragment_trimci_kernel_matches_full_fci():
    """
    LASSCF + TrimCI (1 fragment, threshold=1e-8) must reproduce full FCI
    energy for H4/STO-3G to within 1e-6 Ha.

    1-fragment LASSCF with the full active space is identical to full FCI
    (no inter-fragment approximation).  Any deviation indicates a bug in
    the kernel closure (h1 averaging, dm2 spin-sum, etot formula, etc.).
    """
    mol, mf = _build_h4_mol_mf()

    # Ground truth: full PySCF FCI
    e_fci = _h4_full_fci_energy(mol, mf)

    # LASSCF+TrimCI: single fragment covering all 4 orbitals
    las = _build_h4_las_trimci(mol, mf, threshold=1e-8)
    mo = mf.mo_coeff  # 1-fragment: no localization needed, whole space = one block
    las.kernel(mo)

    e_lasscf = float(las.e_tot)
    delta = abs(e_lasscf - e_fci)

    assert np.isfinite(e_lasscf), f"LASSCF+TrimCI e_tot is not finite: {e_lasscf}"
    assert delta < 1e-6, (
        f"LASSCF+TrimCI energy does not match full FCI:\n"
        f"  E_FCI          = {e_fci:.10f} Ha\n"
        f"  E_LASSCF+TrimCI = {e_lasscf:.10f} Ha\n"
        f"  |delta|         = {delta:.2e} Ha  (threshold: 1e-6)"
    )


# ---------------------------------------------------------------------------
# Test 2: LASSCF+TrimCI vs LASSCF+csf_solver
# ---------------------------------------------------------------------------

def test_h4_single_fragment_trimci_matches_csf_solver():
    """
    LASSCF+TrimCI and LASSCF+default-csf-solver must agree to 1e-6 Ha on H4.

    Both use the same 1-fragment setup and the same MO guess.  The only
    difference is the CI solver.  Disagreement would indicate a bug in the
    TrimCI kernel's energy formula, RDM construction, or spin-sum recipe.
    """
    mol, mf = _build_h4_mol_mf()
    mo = mf.mo_coeff

    # Run A: csf_solver (mrh default)
    las_csf = _build_h4_las_csf(mol, mf)
    las_csf.kernel(mo)
    e_csf = float(las_csf.e_tot)

    # Run B: TrimCI kernel
    las_trimci = _build_h4_las_trimci(mol, mf, threshold=1e-8)
    las_trimci.kernel(mo)
    e_trimci = float(las_trimci.e_tot)

    delta = abs(e_trimci - e_csf)

    assert np.isfinite(e_csf), f"LASSCF+csf e_tot is not finite: {e_csf}"
    assert np.isfinite(e_trimci), f"LASSCF+TrimCI e_tot is not finite: {e_trimci}"
    assert delta < 1e-6, (
        f"LASSCF+TrimCI and LASSCF+csf energies disagree:\n"
        f"  E_csf    = {e_csf:.10f} Ha\n"
        f"  E_TrimCI = {e_trimci:.10f} Ha\n"
        f"  |delta|  = {delta:.2e} Ha  (threshold: 1e-6)"
    )


# ---------------------------------------------------------------------------
# Test 3: Kernel signature contract (unit test, no LASSCF orchestration)
# ---------------------------------------------------------------------------

def test_kernel_signature_contract():
    """
    Call the kernel closure directly on H2/STO-3G to verify:
    - Returned shapes: dm1s.shape == (2, norb, norb), dm2.shape == (norb,)*4
    - etot is finite
    - 1-RDM trace = na + nb  (to 1e-10)
    - 2-RDM is finite
    - 2-RDM has the correct partial symmetry:
        dm2[p,q,r,s] == dm2[r,s,p,q]  (from Mulliken hermitian symmetry of the
        spin-summed form; exact for real wavefunctions)

    Uses H2/STO-3G for speed: norb=2, nelec=(1,1).
    """
    # Build H2/STO-3G
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    norb = mol.nao_nr()  # 2
    na = mol.nelectron // 2  # 1
    nb = mol.nelectron // 2  # 1
    nelec = (na, nb)
    h0 = mol.energy_nuc()

    # h1s: spin-resolved (2, norb, norb)
    h1_mo = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h1s = np.stack([h1_mo, h1_mo], axis=0)  # closed-shell: alpha == beta

    # h2: 4D Mulliken (norb,)*4  — as mrh would provide (already restored)
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False)
    h2 = eri.reshape(norb, norb, norb, norb)

    # Build kernel closure (near-exact threshold for a 2-orbital system)
    kernel = make_trimci_kernel_for_fragment(threshold=1e-10)

    # Call kernel directly (mimicking mrh's RDMSolver.kernel dispatch,
    # but note: mrh already calls ao2mo.restore(1,...) so h2 is already 4D here)
    etot, dm1s, dm2 = kernel(norb, nelec, h0, h1s, h2)

    # --- Shape checks ---
    assert dm1s.shape == (2, norb, norb), (
        f"dm1s.shape={dm1s.shape}, expected (2, {norb}, {norb})"
    )
    assert dm2.shape == (norb,) * 4, (
        f"dm2.shape={dm2.shape}, expected ({norb}, {norb}, {norb}, {norb})"
    )

    # --- Finiteness ---
    assert np.isfinite(etot), f"etot is not finite: {etot}"
    assert np.all(np.isfinite(dm1s)), "dm1s contains non-finite values"
    assert np.all(np.isfinite(dm2)), "dm2 contains non-finite values"

    # --- 1-RDM trace = total electrons ---
    trace_dm1s = float(np.trace(dm1s[0]) + np.trace(dm1s[1]))  # tr(dm1a) + tr(dm1b)
    expected_trace = float(na + nb)
    assert abs(trace_dm1s - expected_trace) < 1e-10, (
        f"1-RDM trace = {trace_dm1s:.14f}, expected {expected_trace:.14f} "
        f"(diff={abs(trace_dm1s - expected_trace):.2e})"
    )

    # --- 2-RDM partial symmetry: dm2[p,q,r,s] == dm2[r,s,p,q] ---
    # This is the Hermitian symmetry of the spin-summed 2-RDM.
    # For real wavefunctions it holds exactly.
    dm2_T = dm2.transpose(2, 3, 0, 1)
    assert np.allclose(dm2, dm2_T, atol=1e-10), (
        f"dm2 fails Hermitian symmetry dm2[p,q,r,s] != dm2[r,s,p,q]; "
        f"max deviation = {np.max(np.abs(dm2 - dm2_T)):.2e}"
    )

    # --- etot reasonableness: H2 FCI ~ -1.117 Ha ---
    assert etot < 0.0, f"H2 ground state energy should be negative, got {etot:.6f}"
    assert etot > -5.0, f"H2 energy unreasonably large in magnitude: {etot:.6f}"
