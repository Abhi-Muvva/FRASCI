"""Port of engine/lasscf/selected_ci_rdm.py's demo() self-check gates as
pytest tests -- the existing, already-validated correctness net for
frasci._selected_ci_rdm. Same test cases, same tolerances, no new coverage
invented (deliberately: re-deriving test cases for correctness-critical
numerical code is riskier than reusing ones already proven to catch bugs)."""
import numpy as np
import pytest

from frasci._selected_ci_rdm import estimate_dense_civec_bytes, make_selected_rdm12s

_TOL = 1e-9


def _deterministic_wf(norb, nelec, n_keep, seed):
    na, nb = int(nelec[0]), int(nelec[1])
    rng = np.random.default_rng(seed)

    def _mask(occ):
        m = 0
        for i in occ:
            m |= (1 << int(i))
        return m

    seen, rows = set(), []
    while len(rows) < n_keep:
        a = _mask(rng.choice(norb, na, replace=False))
        b = _mask(rng.choice(norb, nb, replace=False))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((a, b))
    return np.asarray(rows, dtype=np.int64), rng.standard_normal(n_keep)


def _spin_sum(d2):
    aa, ab, bb = d2
    return aa + ab + ab.transpose(2, 3, 0, 1) + bb


@pytest.mark.parametrize("norb,nelec", [(8, (3, 3)), (8, (4, 2)), (10, (3, 2))])
def test_gate1_dense_equals_sparse(norb, nelec):
    dets2, coeffs = _deterministic_wf(norb, nelec, 40, seed=1)
    (Da, Db), D2 = make_selected_rdm12s(dets2, coeffs, norb, nelec, backend="dense")
    (Sa, Sb), S2 = make_selected_rdm12s(dets2, coeffs, norb, nelec, backend="sparse")
    for X, Y in [(Da, Sa), (Db, Sb), *zip(D2, S2)]:
        assert np.max(np.abs(np.asarray(X) - np.asarray(Y))) < _TOL


def test_gate7_non_cartesian_rectangle():
    dets2, coeffs = _deterministic_wf(10, (3, 2), 40, seed=2)
    n_ua = np.unique(dets2[:, 0]).size
    n_ub = np.unique(dets2[:, 1]).size
    assert n_ua * n_ub > dets2.shape[0]
    (_, _), D2 = make_selected_rdm12s(dets2, coeffs, 10, (3, 2), backend="dense")
    (_, _), S2 = make_selected_rdm12s(dets2, coeffs, 10, (3, 2), backend="sparse")
    assert np.max(np.abs(np.asarray(D2[1]) - np.asarray(S2[1]))) < _TOL


def test_gate2_24_orbital_dense_equals_sparse():
    dets2, coeffs = _deterministic_wf(24, (21, 21), 60, seed=3)
    (Da, Db), D2 = make_selected_rdm12s(dets2, coeffs, 24, (21, 21), backend="dense")
    (Sa, Sb), S2 = make_selected_rdm12s(dets2, coeffs, 24, (21, 21), backend="sparse")
    for X, Y in [(Da, Sa), (Db, Sb), *zip(D2, S2)]:
        assert np.max(np.abs(np.asarray(X) - np.asarray(Y))) < _TOL


def test_gate3and4_traces_and_symmetries():
    dets2, coeffs = _deterministic_wf(12, (6, 6), 80, seed=4)
    (Sa, Sb), (Saa, Sab, Sbb) = make_selected_rdm12s(
        dets2, coeffs, 12, (6, 6), backend="sparse")
    assert abs(np.trace(Sa) - 6) < _TOL and abs(np.trace(Sb) - 6) < _TOL
    assert np.allclose(Sa, Sa.T) and np.allclose(Sb, Sb.T)
    assert np.allclose(Saa, Saa.transpose(1, 0, 3, 2))
    assert np.allclose(Saa, Saa.transpose(2, 3, 0, 1))
    assert np.allclose(Sab, Sab.transpose(1, 0, 3, 2))


def test_gate8_determinant_order_invariance():
    dets2, coeffs = _deterministic_wf(10, (4, 3), 50, seed=5)
    (Aa, _), _ = make_selected_rdm12s(dets2, coeffs, 10, (4, 3), backend="sparse")
    perm = np.random.default_rng(0).permutation(dets2.shape[0])
    (Ba, _), _ = make_selected_rdm12s(dets2[perm], coeffs[perm], 10, (4, 3), backend="sparse")
    assert np.allclose(Aa, Ba)


def test_gate5_energy_consistency_h6_sto3g():
    from pyscf import ao2mo, gto, scf
    from pyscf.fci import cistring, direct_spin1

    mol = gto.M(atom="; ".join(f"H 0 0 {1.6 * i}" for i in range(6)),
                basis="sto-3g", unit="Bohr", verbose=0)
    norb = mol.nao_nr()
    na = nb = mol.nelectron // 2
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape([norb] * 4)
    e_elec, civ = direct_spin1.kernel(h1, h2, norb, (na, nb), ecore=0.0)
    stra = np.asarray(cistring.make_strings(range(norb), na))
    strb = np.asarray(cistring.make_strings(range(norb), nb))
    ia, ib = np.nonzero(np.abs(civ) > 0)
    dets2 = np.stack([stra[ia], strb[ib]], axis=1).astype(np.int64)
    coeffs = civ[ia, ib]
    (s1a, s1b), s2 = make_selected_rdm12s(dets2, coeffs, norb, (na, nb), backend="sparse")
    e_rdm = (np.einsum("pq,pq->", h1, s1a + s1b)
             + 0.5 * np.einsum("pqrs,pqrs->", h2, _spin_sum(s2)))
    assert abs(e_rdm - e_elec) < 1e-8


def test_gate9_enormous_sector_never_allocates_dense():
    dets2, coeffs = _deterministic_wf(22, (11, 11), 16, seed=6)
    assert estimate_dense_civec_bytes(22, (11, 11)) > 10 ** 12
    (s1a, s1b), _ = make_selected_rdm12s(dets2, coeffs, 22, (11, 11), backend="sparse")
    assert abs(np.trace(s1a) - 11) < _TOL and abs(np.trace(s1b) - 11) < _TOL
    assert np.allclose(s1a, s1a.T)
    perm = np.random.default_rng(1).permutation(dets2.shape[0])
    (p1a, _), _ = make_selected_rdm12s(dets2[perm], coeffs[perm], 22, (11, 11), backend="sparse")
    assert np.allclose(s1a, p1a)
