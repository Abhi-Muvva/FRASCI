"""Independent oracle checks for the vectorized real Slater--Condon kernel."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from pyscf import ao2mo, gto

from frasci.recovery._slater_condon import RealSlaterCondon, iter_excitation_batches
from frasci.recovery._state import selected_hamiltonian


def _determinants(n_orb, n_alpha, n_beta):
    alpha = [sum(1 << p for p in occupied)
             for occupied in itertools.combinations(range(n_orb), n_alpha)]
    beta = [sum(1 << p for p in occupied)
            for occupied in itertools.combinations(range(n_orb), n_beta)]
    return [(a, b) for a in alpha for b in beta]


def _physical_integrals(atom, basis, spin):
    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    overlap = mol.intor_symmetric("int1e_ovlp")
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    orbitals = eigenvectors / np.sqrt(eigenvalues)[None, :]
    h1 = orbitals.T @ mol.intor_symmetric("int1e_kin") @ orbitals
    h1 += orbitals.T @ mol.intor_symmetric("int1e_nuc") @ orbitals
    eri = ao2mo.incore.full(mol.intor("int2e"), orbitals, compact=False)
    eri = eri.reshape((mol.nao_nr(),) * 4)
    n_alpha = (mol.nelectron + spin) // 2
    n_beta = mol.nelectron - n_alpha
    return h1, eri, (n_alpha, n_beta)


def _symmetry_correct_integrals(n_orb, seed, *, h1=True, eri=True):
    rng = np.random.default_rng(seed)
    one = rng.standard_normal((n_orb, n_orb)) if h1 else np.zeros((n_orb, n_orb))
    one = 0.5 * (one + one.T)
    if eri:
        factors = rng.standard_normal((n_orb + 2, n_orb, n_orb))
        factors = 0.5 * (factors + factors.transpose(0, 2, 1))
        two = np.einsum("Lpq,Lrs->pqrs", factors, factors, optimize=True)
    else:
        two = np.zeros((n_orb,) * 4)
    return one, two


def _exhaustive_error(h1, eri, sector, *, e_const=0.0):
    dets = _determinants(len(h1), *sector)
    oracle = selected_hamiltonian(h1, eri, dets, n_orb=len(h1))
    oracle += np.eye(len(dets)) * e_const
    kernel = RealSlaterCondon(h1, eri, e_const=e_const)
    built = np.column_stack([kernel.elements(parent, dets) for parent in dets])
    return float(np.max(np.abs(built - oracle))), built, oracle, kernel


@pytest.mark.parametrize(
    "atom,basis,spin",
    [
        ("H 0 0 0; H 0 0 0.74", "sto-3g", 0),
        ("H 0 0 0; H 0 0 1.0; H 0 0 2.1", "sto-3g", 1),
        ("Li 0 0 0; H 0 0 1.6", "sto-3g", 0),
    ],
)
def test_exhaustive_physical_fci_spaces_match_pyscf(atom, basis, spin):
    h1, eri, sector = _physical_integrals(atom, basis, spin)
    error, built, oracle, _ = _exhaustive_error(h1, eri, sector)
    assert error <= 1e-12
    assert np.max(np.abs(built - built.T)) <= 1e-12
    assert np.max(np.abs(oracle - oracle.T)) <= 1e-12


@pytest.mark.parametrize("n_orb,sector,seed", [(4, (2, 1), 11), (5, (3, 1), 12),
                                                (5, (2, 2), 13), (6, (3, 2), 14)])
def test_exhaustive_symmetry_correct_spaces_cover_all_classes(n_orb, sector, seed):
    h1, eri = _symmetry_correct_integrals(n_orb, seed)
    error, built, _, kernel = _exhaustive_error(h1, eri, sector, e_const=-0.37)
    assert error <= 1e-12
    assert np.max(np.abs(built - built.T)) <= 1e-12

    dets = _determinants(n_orb, *sector)
    classes = set()
    for parent in dets:
        _, labels = kernel.elements(parent, dets, return_classes=True)
        classes.update(labels.tolist())
    expected = {"diagonal", "alpha_single", "beta_single", "opposite_double", "rank_gt2"}
    if sector[0] >= 2 and n_orb - sector[0] >= 2:
        expected.add("alpha_double")
    if sector[1] >= 2 and n_orb - sector[1] >= 2:
        expected.add("beta_double")
    assert expected <= classes


@pytest.mark.parametrize("one_body,two_body", [(True, False), (False, True)])
def test_one_and_two_body_isolates(one_body, two_body):
    h1, eri = _symmetry_correct_integrals(5, 21 + two_body, h1=one_body, eri=two_body)
    error, _, _, _ = _exhaustive_error(h1, eri, (2, 2))
    assert error <= 1e-12


def test_core_constant_changes_only_diagonal_and_absorb_double_count_regression():
    h1, eri = _symmetry_correct_integrals(4, 31)
    dets = _determinants(4, 2, 1)
    plain = RealSlaterCondon(h1, eri)
    shifted = RealSlaterCondon(h1, eri, e_const=-7.25)
    parent = dets[0]
    difference = shifted.elements(parent, dets) - plain.elements(parent, dets)
    assert difference[dets.index(parent)] == pytest.approx(-7.25, abs=1e-13)
    assert np.max(np.abs(np.delete(difference, dets.index(parent)))) == 0.0

    # Regression for the abandoned absorb/contract convention: an h1-absorbed ERI
    # cannot also be interpreted as this repo's raw chemists' ERI with h1 retained.
    from pyscf.fci import direct_spin1
    absorbed = direct_spin1.absorb_h1e(h1, eri, 4, (2, 1), fac=0.5)
    absorbed_full = ao2mo.restore(1, absorbed, 4)
    wrong = RealSlaterCondon(h1, absorbed_full).element(parent, parent)
    oracle = selected_hamiltonian(h1, eri, [parent], n_orb=4)[0, 0]
    assert abs(wrong - oracle) > 1e-3
    assert plain.element(parent, parent) == pytest.approx(oracle, abs=1e-12)


def test_edge_orbitals_and_randomized_pair_hermiticity():
    h1, eri = _symmetry_correct_integrals(7, 41)
    dets = _determinants(7, 3, 2)
    kernel = RealSlaterCondon(h1, eri)
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, len(dets), size=(500, 2))
    # Force edge-orbital excitations involving orbitals 0 and n_orb-1.
    pairs[:2] = [[0, len(dets) - 1], [len(dets) - 1, 0]]
    for left_index, right_index in pairs:
        left, right = dets[int(left_index)], dets[int(right_index)]
        got = kernel.element(left, right)
        reverse = kernel.element(right, left)
        oracle_dets = [left] if left == right else [left, right]
        oracle = selected_hamiltonian(h1, eri, oracle_dets, n_orb=7)
        oracle_value = oracle[0, 0] if left == right else oracle[0, 1]
        assert got == pytest.approx(reverse, abs=1e-12)
        assert got == pytest.approx(oracle_value, abs=1e-12)


def test_all_parity_signs_occur_and_match_oracle():
    h1, eri = _symmetry_correct_integrals(6, 51)
    dets = _determinants(6, 3, 2)
    kernel = RealSlaterCondon(h1, eri)
    oracle = selected_hamiltonian(h1, eri, dets, n_orb=6)
    signs_by_class = {}
    for column, parent in enumerate(dets):
        values, classes = kernel.elements(parent, dets, return_classes=True)
        assert np.max(np.abs(values - oracle[:, column])) <= 1e-12
        for value, class_name in zip(values, classes):
            if class_name not in {"diagonal", "rank_gt2"} and abs(value) > 1e-10:
                signs_by_class.setdefault(class_name, set()).add(int(np.sign(value)))
    for class_name in ("alpha_single", "beta_single", "alpha_double",
                       "beta_double", "opposite_double"):
        assert signs_by_class[class_name] == {-1, 1}


def test_36_orbital_vectorized_elements_match_oracle_sample():
    """RealSlaterCondon.elements() at a realistic scale (36 orbitals): a sampled
    cross-section of every excitation class produced by iter_excitation_batches must
    match the PySCF selected-space oracle exactly.

    (The live development suite additionally benchmarks frasci.recovery._bridge's
    heat-bath candidate generation on this same fixture -- that module is excluded from
    this release, so only the RealSlaterCondon correctness check below is kept.)
    """
    n_orb = 36
    occupied = (1 << 27) - 1
    parent = (occupied, occupied)
    h1, eri = _symmetry_correct_integrals(n_orb, 61)
    kernel = RealSlaterCondon(h1, eri)

    samples = []
    for _, batch in iter_excitation_batches(parent, n_orb, batch_size=4096):
        samples.extend(batch[:3])
    samples = list(dict.fromkeys(samples))

    oracle_dets = [parent] + samples
    oracle = selected_hamiltonian(h1, eri, oracle_dets, n_orb=n_orb)
    sample_values = kernel.elements(parent, samples)
    max_error = float(np.max(np.abs(sample_values - oracle[1:, 0])))

    assert max_error <= 1e-12
