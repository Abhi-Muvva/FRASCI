"""
tests/test_mfa_solver.py
========================
Unit tests for frasci/mfa/solver.py helper functions (Phase D, Task 2).
"""
import os
from pathlib import Path
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sym_eri(n, seed=0):
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
# Test 11: diagonal gamma_mixed matches Phase B dress_integrals_meanfield
# ---------------------------------------------------------------------------

def test_dressing_diagonal_gamma_matches_phase_b():
    """With diagonal gamma_mixed, dress_fragment_h1_mfa == dress_integrals_meanfield."""
    from FRASCI.mfa.solver import dress_fragment_h1_mfa
    from FRASCI.mfa.helpers import dress_integrals_meanfield
    n_orb, n_frag = 6, 3
    frag_orbs = [0, 2, 4]
    ext_orbs  = sorted(set(range(n_orb)) - set(frag_orbs))
    eri_full  = _sym_eri(n_orb, seed=20)
    h1_bare   = _sym_h1(n_frag, seed=21)
    diag_vals = np.array([1.5, 0.3, 1.8, 0.7, 1.2, 0.5])
    gamma_mixed = np.diag(diag_vals)

    result_mfa = dress_fragment_h1_mfa(h1_bare, eri_full, frag_orbs, gamma_mixed, n_orb)

    ext_diag = diag_vals[ext_orbs]
    result_phase_b = dress_integrals_meanfield(h1_bare, eri_full, frag_orbs, ext_diag, ext_orbs)

    assert np.allclose(result_mfa, result_phase_b, atol=1e-12), \
        f"MFA dressing != Phase B for diagonal gamma\nmax diff: {np.max(np.abs(result_mfa - result_phase_b)):.2e}"


# ---------------------------------------------------------------------------
# Test 12: off-diagonal gamma gives a different result than diagonal-only
# ---------------------------------------------------------------------------

def test_dressing_offdiagonal_gamma_differs_from_diagonal_only():
    from FRASCI.mfa.solver import dress_fragment_h1_mfa
    n_orb, n_frag = 6, 3
    frag_orbs = [0, 2, 4]
    eri_full  = _sym_eri(n_orb, seed=22)
    h1_bare   = _sym_h1(n_frag, seed=23)
    gamma_diag = np.diag([1.5, 0.3, 1.8, 0.7, 1.2, 0.5])
    gamma_offdiag = gamma_diag.copy()
    gamma_offdiag[1, 3] = gamma_offdiag[3, 1] = 0.4
    gamma_offdiag[0, 1] = gamma_offdiag[1, 0] = 0.2

    result_diag    = dress_fragment_h1_mfa(h1_bare, eri_full, frag_orbs, gamma_diag,    n_orb)
    result_offdiag = dress_fragment_h1_mfa(h1_bare, eri_full, frag_orbs, gamma_offdiag, n_orb)

    assert not np.allclose(result_diag, result_offdiag, atol=1e-10), \
        "Off-diagonal upgrade must change the dressing"


# ---------------------------------------------------------------------------
# Test 13: internal-only gamma → v_ext = 0 (dressed h1 == bare h1)
# ---------------------------------------------------------------------------

def test_dressing_internal_gamma_zero_contribution():
    from FRASCI.mfa.solver import dress_fragment_h1_mfa
    n_orb, n_frag = 6, 3
    frag_orbs = [0, 2, 4]
    eri_full  = _sym_eri(n_orb, seed=24)
    h1_bare   = _sym_h1(n_frag, seed=25)
    fa = np.array(frag_orbs)
    gamma_internal = np.zeros((n_orb, n_orb))
    gamma_internal[np.ix_(fa, fa)] = 1.0

    result = dress_fragment_h1_mfa(h1_bare, eri_full, frag_orbs, gamma_internal, n_orb)
    assert np.allclose(result, h1_bare, atol=1e-12), \
        "Internal-only gamma must give zero dressing"


# ---------------------------------------------------------------------------
# Test 14: external-only gamma → v_ext ≠ 0
# ---------------------------------------------------------------------------

def test_dressing_external_gamma_nonzero_contribution():
    from FRASCI.mfa.solver import dress_fragment_h1_mfa
    n_orb, n_frag = 6, 3
    frag_orbs = [0, 2, 4]
    ext_orbs  = [1, 3, 5]
    eri_full  = _sym_eri(n_orb, seed=26)
    h1_bare   = _sym_h1(n_frag, seed=27)
    ea = np.array(ext_orbs)
    gamma_external = np.zeros((n_orb, n_orb))
    gamma_external[np.ix_(ea, ea)] = 1.0

    result = dress_fragment_h1_mfa(h1_bare, eri_full, frag_orbs, gamma_external, n_orb)
    assert not np.allclose(result, h1_bare, atol=1e-12), \
        "External-only gamma must produce nonzero dressing"


# ---------------------------------------------------------------------------
# Test 19: cross-block gamma → v_ext ≠ 0 (key physics test)
# ---------------------------------------------------------------------------

def test_dressing_crossblock_gamma_nonzero_contribution():
    """Cross-block (one index in frag, one outside) → v_ext ≠ 0.
    Validates 'not both in frag' mask. env-env-only mask would miss these."""
    from FRASCI.mfa.solver import dress_fragment_h1_mfa
    n_orb, n_frag = 6, 3
    frag_orbs = [0, 2, 4]
    eri_full  = _sym_eri(n_orb, seed=28)
    h1_bare   = _sym_h1(n_frag, seed=29)
    gamma_cross = np.zeros((n_orb, n_orb))
    gamma_cross[0, 1] = gamma_cross[1, 0] = 0.5   # frag-ext
    gamma_cross[2, 3] = gamma_cross[3, 2] = 0.4   # frag-ext
    gamma_cross[4, 5] = gamma_cross[5, 4] = 0.3   # frag-ext

    result = dress_fragment_h1_mfa(h1_bare, eri_full, frag_orbs, gamma_cross, n_orb)
    assert not np.allclose(result, h1_bare, atol=1e-12), \
        "Cross-block gamma must produce nonzero dressing — check 'not both in frag' mask"


# ---------------------------------------------------------------------------
# Test 15: overlapping partition matches Phase C (uses core only)
# ---------------------------------------------------------------------------

def test_overlapping_partition_matches_phase_c():
    from FRASCI.core.fragment import fragment_by_sliding_window
    frags = fragment_by_sliding_window(36, np.arange(36), 15, 10)
    assert len(frags) == 3
    assert frags[0] == list(range(0, 15))
    assert frags[1] == list(range(10, 25))
    assert frags[2] == list(range(20, 36))


# ---------------------------------------------------------------------------
# Test 16: non-overlapping partition covers all 36 orbitals with no duplicates
# ---------------------------------------------------------------------------

def test_nonoverlapping_covers_all_36_orbs():
    from FRASCI.mfa.solver import make_nonoverlapping_partition
    h1 = np.diag(np.random.default_rng(30).random(36))
    frags = make_nonoverlapping_partition(h1, 36)
    assert len(frags) == 3
    all_orbs = sorted(o for f in frags for o in f)
    assert all_orbs == list(range(36))
    flat = [o for f in frags for o in f]
    assert len(flat) == len(set(flat)), "Duplicate orbitals"


# ---------------------------------------------------------------------------
# Test 20: balanced non-overlapping partition avoids closed fragments
# ---------------------------------------------------------------------------

def test_balanced_nonoverlapping_partition_has_alpha_and_beta_holes():
    from FRASCI.mfa.solver import make_balanced_nonoverlapping_partition
    from FRASCI.core.fragment import fragment_electron_count

    n_orb = 12
    h1 = np.diag(np.arange(n_orb, dtype=float))
    docc = {0, 1, 2, 3, 4, 5}
    socc_alpha = {6, 7}
    socc_beta = {8, 9}
    virt = {10, 11}
    alpha_bits = sum(1 << o for o in docc | socc_alpha)
    beta_bits = sum(1 << o for o in docc | socc_beta)

    frags = make_balanced_nonoverlapping_partition(
        h1, alpha_bits, beta_bits, n_orb, n_fragments=3
    )

    assert len(frags) == 3
    assert sorted(o for frag in frags for o in frag) == list(range(n_orb))
    for frag in frags:
        n_alpha, n_beta = fragment_electron_count(alpha_bits, beta_bits, frag)
        assert n_alpha < len(frag), f"closed alpha fragment: frag={frag}, n_alpha={n_alpha}"
        assert n_beta < len(frag), f"closed beta fragment: frag={frag}, n_beta={n_beta}"
        assert n_alpha > 0, f"empty alpha fragment: frag={frag}"
        assert n_beta > 0, f"empty beta fragment: frag={frag}"


# ---------------------------------------------------------------------------
# Test 17: electron counts sum to 27+27 (integration test, Fe4S4 data)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_electron_counts_sum_to_27_27():
    import os
    from FRASCI.mfa.solver import load_ref_det, make_nonoverlapping_partition
    from FRASCI.core.fragment import fragment_electron_count
    import trimci

    FCIDUMP  = str(Path(__file__).parent.parent / "data" / "fcidump_cycle_6")
    DETS_NPZ = str(Path(__file__).parent.parent / "data" / "dets.npz")
    if not os.path.exists(FCIDUMP):
        pytest.skip("Fe4S4 data not available")

    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(FCIDUMP)
    alpha_bits, beta_bits = load_ref_det(DETS_NPZ, row=0)
    frags = make_nonoverlapping_partition(h1, n_orb)

    total_alpha = sum(fragment_electron_count(alpha_bits, beta_bits, f)[0] for f in frags)
    total_beta  = sum(fragment_electron_count(alpha_bits, beta_bits, f)[1] for f in frags)
    assert total_alpha == 27, f"sum(n_alpha)={total_alpha}"
    assert total_beta  == 27, f"sum(n_beta)={total_beta}"


# ---------------------------------------------------------------------------
# Test 18: gamma path validation (FileNotFoundError and ValueError)
# ---------------------------------------------------------------------------

def test_gamma_path_validation(tmp_path):
    from FRASCI.mfa.solver import load_gamma_mixed
    missing = str(tmp_path / "nonexistent.npy")
    with pytest.raises(FileNotFoundError, match="gamma_mixed not found"):
        load_gamma_mixed(missing, 36)

    wrong = tmp_path / "wrong_shape.npy"
    np.save(str(wrong), np.zeros((35, 35)))
    with pytest.raises(ValueError, match="gamma_mixed shape"):
        load_gamma_mixed(str(wrong), 36)

    diag_vec = tmp_path / "diag_vec.npy"
    np.save(str(diag_vec), np.arange(36.0))
    gamma = load_gamma_mixed(str(diag_vec), 36, allow_diagonal_vector=True)
    assert gamma.shape == (36, 36)
    assert np.allclose(np.diag(gamma), np.arange(36.0))
