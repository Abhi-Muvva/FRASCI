"""Native gates for ExpansionConfig.allowed_orbital_pairs (fragment-block-constrained COO).

Covers spec gates 1 (empty mask == unrestricted), 2 (two-block mask -> exact zero
cross-block U), and 9 (both 64-bit and 128-bit ExpansionConfig bindings), plus the
validation contract (out-of-range/duplicate pairs raise ValueError). Gates 3-8 (the
Python-side fixed-support/covariant-transform pipeline) live in
tests/test_frasci_recovery_constrained_coo.py.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimci.trimci_core as tc

from frasci.recovery._state import selected_hamiltonian

fe = tc.fast_expansion


def _h6_integrals():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {1.0 * i}" for i in range(6)),
               basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    return h1, eri


# Two 3-orbital fragment blocks, 3 alpha / 3 beta total (1 electron pair short of
# half-filling per block, so the intra-block 1-RDM is asymmetric and BFGS has a
# genuine nonzero gradient to act on -- a degenerate/symmetric fixture would make
# gate 2 vacuous (U stays identity everywhere, cross-block AND intra-block).
BLOCKS = [[0, 1, 2], [3, 4, 5]]
ALLOWED_PAIRS = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
N_ORB = 6


def _seed_dets_coeffs():
    # Two determinants that superpose two DIFFERENT alpha-occupation patterns within
    # fragment 0's block (orbitals {0,1} vs {0,2}), fragment 1 held fixed (alpha orbital
    # 3, beta orbitals 4,5). A single closed/empty-shell determinant gives an exactly
    # zero intra-block orbital gradient (RDM proportional to identity) -- verified
    # empirically to be vacuous for gate 2; this superposition creates genuine
    # off-diagonal 1-RDM structure inside fragment 0's block instead.
    dets = [(0b001011, 0b111001), (0b001101, 0b111001)]
    coeffs = np.array([0.9, 0.436])
    coeffs /= np.linalg.norm(coeffs)
    return dets, coeffs


def _config(cls, *, allowed_pairs):
    cfg = cls()
    cfg.verbose = 0
    cfg.evaluate_only = True
    cfg.orbital_optimization = True
    cfg.orbital_opt_max_iter = 15
    cfg.allowed_orbital_pairs = allowed_pairs
    return cfg


def _run_64(dets, coeffs, h1, eri, cfg):
    alpha = np.array([d[0] for d in dets], dtype=np.uint64)
    beta = np.array([d[1] for d in dets], dtype=np.uint64)
    return fe.run_expansion(alpha, beta, np.asarray(coeffs, dtype=float),
                            np.ascontiguousarray(h1), np.ascontiguousarray(eri).ravel(),
                            N_ORB, cfg)


def _run_128(dets, coeffs, h1, eri, cfg):
    alpha = np.array([[d[0], 0] for d in dets], dtype=np.uint64)
    beta = np.array([[d[1], 0] for d in dets], dtype=np.uint64)
    return fe.run_expansion_128(alpha, beta, np.asarray(coeffs, dtype=float),
                                np.ascontiguousarray(h1), np.ascontiguousarray(eri).ravel(),
                                N_ORB, cfg)


def _rayleigh(h1, eri, dets, coeffs):
    ham = selected_hamiltonian(h1, eri, dets, n_orb=N_ORB)
    c = np.asarray(coeffs, dtype=float)
    return float(c @ ham @ c) / float(c @ c)


@pytest.mark.parametrize("cls,run", [
    (fe.ExpansionConfig, _run_64),
    (fe.ExpansionConfig128, _run_128),
])
def test_gate1_empty_mask_matches_unrestricted(cls, run):
    h1, eri = _h6_integrals()
    dets, coeffs = _seed_dets_coeffs()

    cfg_default = _config(cls, allowed_pairs=[])
    assert cfg_default.allowed_orbital_pairs == []
    res_default = run(dets, coeffs, h1, eri, cfg_default)

    all_pairs = [(i, j) for i in range(N_ORB) for j in range(i + 1, N_ORB)]
    cfg_explicit = _config(cls, allowed_pairs=all_pairs)
    res_explicit = run(dets, coeffs, h1, eri, cfg_explicit)

    U_default = res_default.get_U_total()
    U_explicit = res_explicit.get_U_total()
    assert np.max(np.abs(U_default - U_explicit)) < 1e-12
    assert abs(res_default.energy_var - res_explicit.energy_var) < 1e-12


@pytest.mark.parametrize("cls,run", [
    (fe.ExpansionConfig, _run_64),
    (fe.ExpansionConfig128, _run_128),
])
def test_gate2_block_mask_gives_zero_cross_block_rotation(cls, run):
    h1, eri = _h6_integrals()
    dets, coeffs = _seed_dets_coeffs()
    cfg = _config(cls, allowed_pairs=ALLOWED_PAIRS)
    res = run(dets, coeffs, h1, eri, cfg)
    U = res.get_U_total()

    block0, block1 = BLOCKS
    cross = U[np.ix_(block0, block1)]
    assert np.max(np.abs(cross)) < 1e-12

    # The rotation must be genuinely nontrivial INSIDE at least one block -- otherwise
    # this test would pass vacuously even if allowed_orbital_pairs were silently ignored.
    identity = np.eye(N_ORB)
    intra0 = U[np.ix_(block0, block0)] - identity[np.ix_(block0, block0)]
    intra1 = U[np.ix_(block1, block1)] - identity[np.ix_(block1, block1)]
    assert max(np.max(np.abs(intra0)), np.max(np.abs(intra1))) > 1e-4

    # And the returned wavefunction's own Rayleigh quotient (recomputed independently,
    # never trusting res.energy_var alone per this repo's established caution) must not
    # have regressed versus the un-rotated starting point.
    alpha_out, beta_out = res.get_alpha_beta()
    if cls is fe.ExpansionConfig:
        out_dets = [(int(a), int(b)) for a, b in zip(alpha_out, beta_out)]
    else:
        out_dets = [(int(a[0]), int(b[0])) for a, b in zip(alpha_out, beta_out)]
    e_rotated = _rayleigh(res.get_h1_optimized(), res.get_eri_optimized(),
                         out_dets, res.get_coefficients())
    e_start = _rayleigh(h1, eri, dets, coeffs)
    assert e_rotated <= e_start + 1e-10


@pytest.mark.parametrize("cls", [fe.ExpansionConfig, fe.ExpansionConfig128])
@pytest.mark.parametrize("bad_pairs", [
    [(2, 1)],           # p >= q
    [(0, 0)],            # p == q
    [(-1, 2)],            # p < 0
    [(0, N_ORB)],          # q >= n_orb
    [(0, 1), (0, 1)],       # duplicate
])
def test_invalid_allowed_pairs_raise_value_error(cls, bad_pairs):
    h1, eri = _h6_integrals()
    dets, coeffs = _seed_dets_coeffs()
    cfg = _config(cls, allowed_pairs=bad_pairs)
    run = _run_64 if cls is fe.ExpansionConfig else _run_128
    with pytest.raises(ValueError):
        run(dets, coeffs, h1, eri, cfg)
