"""
Tests for frasci.mfa.energy — Phase D MFA-TrimCI energy functions.
Written TDD-style: tests defined before implementation.
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sym_eri(n, seed=0):
    """Build a fully chemist-symmetric toy ERI (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n, n, n)) * 0.1
    A = A + A.transpose(1, 0, 2, 3)
    A = A + A.transpose(0, 1, 3, 2)
    A = A + A.transpose(2, 3, 0, 1)
    return A / 4.0


def _sym_h1(n, seed=1):
    A = np.random.default_rng(seed).random((n, n))
    return (A + A.T) * 0.5


# ---------------------------------------------------------------------------
# Tests 1–3: build_fock
# ---------------------------------------------------------------------------

def test_fock_shape_and_symmetry():
    from FRASCI.mfa.energy import build_fock
    n = 4
    h1  = _sym_h1(n)
    eri = _sym_eri(n)
    gamma = np.diag([1.2, 0.8, 1.5, 0.5])
    F = build_fock(h1, eri, gamma)
    assert F.shape == (n, n)
    assert np.allclose(F, F.T, atol=1e-12), "Fock matrix not symmetric"


def test_fock_zero_eri():
    from FRASCI.mfa.energy import build_fock
    n = 4
    h1    = _sym_h1(n)
    eri   = np.zeros((n, n, n, n))
    gamma = np.eye(n)
    F = build_fock(h1, eri, gamma)
    assert np.allclose(F, h1, atol=1e-14), "F != h1 when eri=0"


def test_fock_diagonal_gamma_known_result():
    from FRASCI.mfa.energy import build_fock
    n = 2
    h1 = np.zeros((n, n))
    eri = np.zeros((n, n, n, n))
    eri[0, 0, 0, 0] = 1.0
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 2.0
    eri[0, 1, 1, 0] = eri[1, 0, 0, 1] = 3.0
    gamma = np.diag([1.0, 0.5])
    F = build_fock(h1, eri, gamma)
    # F[0,0]:
    #   J[0,0] = gamma[0,0]*eri[0,0,0,0] + gamma[1,1]*eri[0,0,1,1] = 1.0*1.0 + 0.5*2.0 = 2.0
    #   K[0,0] = 0.5*(gamma[0,0]*eri[0,0,0,0] + gamma[1,1]*eri[0,1,1,0])
    #          = 0.5*(1.0*1.0 + 0.5*3.0) = 0.5*(1.0+1.5) = 1.25
    # F[0,0] = 2.0 - 1.25 = 0.75
    assert np.isclose(F[0, 0], 0.75), f"F[0,0]={F[0,0]}, expected 0.75"


# ---------------------------------------------------------------------------
# Tests 4–6: mf_global_energy, mf_rowpartition_energy
# ---------------------------------------------------------------------------

def test_mf_global_known_result():
    from FRASCI.mfa.energy import mf_global_energy
    h1 = np.diag([1.0, 2.0])
    eri = np.zeros((2, 2, 2, 2))
    gamma = np.diag([2.0, 0.0])
    # F = h1 (eri=0). (h1+F)·gamma: 2*1*2 + 2*2*0 = 4. E_elec=0.5*4=2.0.
    E_global, E_elec, F = mf_global_energy(h1, eri, gamma, E_nuc=0.5)
    assert np.isclose(E_elec, 2.0), f"E_elec={E_elec}"
    assert np.isclose(E_global, 2.5), f"E_global={E_global}"
    assert F.shape == (2, 2)


def test_enuc_shifts_global_not_corrections():
    from FRASCI.mfa.energy import mf_global_energy, correlation_total_energy
    h1 = np.diag([1.0, 2.0])
    eri = np.zeros((2, 2, 2, 2))
    gamma = np.diag([1.0, 1.0])
    E_global_0, E_elec_0, _ = mf_global_energy(h1, eri, gamma, E_nuc=0.0)
    E_global_1, E_elec_1, _ = mf_global_energy(h1, eri, gamma, E_nuc=1.0)
    assert np.isclose(E_elec_0, E_elec_1), "E_elec must not change with E_nuc"
    assert np.isclose(E_global_1 - E_global_0, 1.0), "E_global must shift by E_nuc"
    # Corrections (fragment-local) must be unaffected by E_nuc
    E_total_0, corr_0 = correlation_total_energy(E_global_0, [0.5, 0.3], [0.5, 0.3])
    E_total_1, corr_1 = correlation_total_energy(E_global_1, [0.5, 0.3], [0.5, 0.3])
    assert corr_0 == corr_1, "Corrections must not change with E_nuc"
    assert np.isclose(E_total_1 - E_total_0, 1.0), "E_total shifts by E_nuc only"


def test_row_partition_sums_to_global_elec():
    from FRASCI.mfa.energy import mf_global_energy, mf_rowpartition_energy
    n = 4
    h1  = _sym_h1(n, seed=7)
    eri = _sym_eri(n, seed=7)
    gamma = _sym_h1(n, seed=8)
    gamma = (gamma + gamma.T) / 2
    E_global, E_elec, F_full = mf_global_energy(h1, eri, gamma, E_nuc=0.0)
    e0 = mf_rowpartition_energy(h1, F_full, gamma, [0, 1], n)
    e1 = mf_rowpartition_energy(h1, F_full, gamma, [2, 3], n)
    # Must sum to E_elec (electronic only), NOT E_global (which includes E_nuc)
    assert np.isclose(e0 + e1, E_elec, atol=1e-12), \
        f"Row partition {e0+e1:.10f} != E_elec {E_elec:.10f}"


# ---------------------------------------------------------------------------
# Tests 7–10: mf_embedded_energy, correlation_total_energy
# ---------------------------------------------------------------------------

def test_embedded_fock_shape_and_symmetry():
    from FRASCI.mfa.energy import build_fock
    n = 3
    h1_d  = _sym_h1(n, seed=10)
    eri_f = _sym_eri(n, seed=10)
    gf    = _sym_h1(n, seed=11)
    F_emb = build_fock(h1_d, eri_f, gf)
    assert F_emb.shape == (n, n)
    assert np.allclose(F_emb, F_emb.T, atol=1e-12), "F_emb not symmetric"


def test_embedded_mf_changes_with_dressing():
    from FRASCI.mfa.energy import mf_embedded_energy
    n = 3
    eri_f    = _sym_eri(n, seed=12)
    gamma_f  = np.diag([1.0, 1.0, 0.5])
    h1_bare  = _sym_h1(n, seed=12)
    h1_dress = h1_bare + 0.3 * np.ones((n, n))
    E_bare   = mf_embedded_energy(h1_bare,  eri_f, gamma_f)
    E_dress  = mf_embedded_energy(h1_dress, eri_f, gamma_f)
    assert not np.isclose(E_bare, E_dress), "Dressing must change E_mf_emb"


def test_zero_correlation_gives_mf_global():
    from FRASCI.mfa.energy import correlation_total_energy
    E_mf_global = -327.0
    E_mf_emb    = [-100.0, -120.0, -107.0]
    E_total, E_corr = correlation_total_energy(E_mf_global, list(E_mf_emb), E_mf_emb)
    assert np.isclose(E_total, E_mf_global), f"E_total={E_total}, expected {E_mf_global}"
    assert all(np.isclose(c, 0.0) for c in E_corr), f"E_corr={E_corr}"


def test_negative_correlation_lowers_E_total():
    from FRASCI.mfa.energy import correlation_total_energy
    E_mf_global = -327.0
    E_mf_emb    = [-100.0, -120.0, -107.0]
    E_trimci    = [-101.0, -121.5, -108.0]
    E_total, E_corr = correlation_total_energy(E_mf_global, E_trimci, E_mf_emb)
    assert E_total < E_mf_global, f"E_total={E_total} must be < {E_mf_global}"
    assert all(c < 0 for c in E_corr), f"All corrections must be negative: {E_corr}"
