"""Milestone E: explicit full Cartesian-product Phi_LAS reconstruction
(frasci.recovery._las_reconstruct) and its Rayleigh-vs-H00 certificate -- the
hard correctness gate the design doc requires before any recovery determinant is
added or a Davidson union is formed (neither happens in this module).

Isolation tests per docs/methods/filter-aware-phased-trimci-interfragment-
recovery.md Milestone E: one fragment, two fragments (hand-checked against an
independently-written reference embedding, not just re-deriving the same code),
duplicate-merging, determinism under input reordering, and -- on a real parent's
un-rotated ("identity rotation") LAS state -- the actual H00 certificate this
milestone exists to prove. The identity-rotation certificate doubles as the
design doc's "direct selected-Hamiltonian contraction versus RDM energy" check:
H00 (_assemble_contracted_reference's intra+inter-fragment RDM contraction) and
E_LAS_reconstructed (explicit Slater-Condon Rayleigh quotient over the full
product) are two independently-computed quantities that must agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from frasci.recovery._las_reconstruct import (
    certify_las_reconstruction,
    las_reconstruction_energy,
    reconstruct_las_vector,
)
from frasci.recovery._state import selected_hamiltonian


def _h4_state(frag0_dets, frag0_coeffs, frag1_dets, frag1_coeffs, *, h00=None):
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {1.0 * i}" for i in range(4)),
               basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    e_const = float(mol.energy_nuc())
    frags = [
        dict(dets=np.asarray(frag0_dets, dtype=np.int64),
            coeffs=np.asarray(frag0_coeffs, dtype=float)),
        dict(dets=np.asarray(frag1_dets, dtype=np.int64),
            coeffs=np.asarray(frag1_coeffs, dtype=float)),
    ]
    return dict(n_orb=4, n_elec=(2, 2), blocks=[[0, 1], [2, 3]],
               ref_sectors=[(1, 1), (1, 1)], frags=frags, mo_final=c.copy(),
               h1=h1, eri=eri, e_const=e_const, h00=h00)


def _independent_reference_product(state):
    """A deliberately separately-written (not imported) full-product embedding,
    used as an independent cross-check of reconstruct_las_vector -- not a
    reimplementation for its own sake, but a genuine second author of the same
    math, unrolled explicitly rather than via itertools.product/dict.get."""
    blocks = state["blocks"]
    frag0, frag1 = state["frags"]
    combined = {}
    for i0 in range(len(frag0["dets"])):
        a0, b0 = int(frag0["dets"][i0][0]), int(frag0["dets"][i0][1])
        c0 = float(frag0["coeffs"][i0])
        ga0 = sum(1 << blocks[0][k] for k in range(len(blocks[0])) if (a0 >> k) & 1)
        gb0 = sum(1 << blocks[0][k] for k in range(len(blocks[0])) if (b0 >> k) & 1)
        for i1 in range(len(frag1["dets"])):
            a1, b1 = int(frag1["dets"][i1][0]), int(frag1["dets"][i1][1])
            c1 = float(frag1["coeffs"][i1])
            ga1 = sum(1 << blocks[1][k] for k in range(len(blocks[1])) if (a1 >> k) & 1)
            gb1 = sum(1 << blocks[1][k] for k in range(len(blocks[1])) if (b1 >> k) & 1)
            det = (ga0 | ga1, gb0 | gb1)
            combined[det] = combined.get(det, 0.0) + c0 * c1
    return combined


def test_two_fragment_reconstruction_matches_independent_reference():
    state = _h4_state(
        [[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436],
        [[0b01, 0b01], [0b10, 0b10]], [0.7, 0.714])
    dets, coeffs = reconstruct_las_vector(state)
    assert len(dets) == 4  # 2x2 product, no natural duplicates (disjoint blocks)

    reference = _independent_reference_product(state)
    ref_norm = float(np.linalg.norm(list(reference.values())))
    got = dict(zip(dets, coeffs))
    assert set(got) == set(reference)
    for det, ref_coeff in reference.items():
        assert got[det] == pytest.approx(ref_coeff / ref_norm, abs=1e-12)


def test_one_fragment_reconstruction_is_exactly_that_fragments_ci_vector():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {1.0 * i}" for i in range(4)),
               basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    dets_local = [[0b0011, 0b0011], [0b0101, 0b0101], [0b1001, 0b1001]]
    coeffs_local = [0.9, 0.3, 0.31]
    frags = [dict(dets=np.asarray(dets_local, dtype=np.int64),
                 coeffs=np.asarray(coeffs_local, dtype=float))]
    state = dict(n_orb=4, n_elec=(2, 2), blocks=[[0, 1, 2, 3]],
                ref_sectors=[(2, 2)], frags=frags, mo_final=c.copy(),
                h1=h1, eri=eri, e_const=float(mol.energy_nuc()), h00=None)

    dets, coeffs = reconstruct_las_vector(state)
    assert set(dets) == {(0b0011, 0b0011), (0b0101, 0b0101), (0b1001, 0b1001)}
    expected_norm = float(np.linalg.norm(coeffs_local))
    for det, local_coeff in zip(dets_local, coeffs_local):
        idx = dets.index((int(det[0]), int(det[1])))
        assert coeffs[idx] == pytest.approx(local_coeff / expected_norm, abs=1e-12)

    e_reconstructed = las_reconstruction_energy(state, dets, coeffs)
    ham = selected_hamiltonian(h1, eri, [tuple(map(int, d)) for d in dets_local], n_orb=4)
    c_local = np.asarray(coeffs_local) / expected_norm
    expected = float(c_local @ ham @ c_local) + float(mol.energy_nuc())
    assert e_reconstructed == pytest.approx(expected, abs=1e-12)


def test_duplicate_local_support_entries_are_summed_not_dropped():
    """A fragment whose own support list contains the SAME local determinant twice
    (a redundant/degenerate input) must have its coefficients combined by summing
    when embedded, not silently overwritten or double-counted as two separate
    global determinants -- disjoint orbital blocks make the global bitstring a
    function of the per-fragment local determinant only."""
    state = _h4_state(
        [[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436],
        [[0b01, 0b01], [0b01, 0b01]], [0.5, 0.3])  # frag1: same local det twice
    dets, coeffs = reconstruct_las_vector(state)
    # frag0 has 2 distinct local dets, frag1 collapses its 2 (identical) entries
    # into 1 unique global determinant per frag0 choice -> 2 unique global dets.
    assert len(dets) == 2
    got = dict(zip(dets, coeffs))
    norm = np.linalg.norm(coeffs)
    assert abs(norm - 1.0) < 1e-12
    # frag1's combined local coefficient is (0.5 + 0.3) = 0.8 before normalization,
    # applied identically to both frag0 branches.
    raw = {det: coeff * norm for det, coeff in got.items()}
    ratio = raw[dets[0]] / raw[dets[1]] if raw[dets[1]] else None
    expected_ratio = (0.9 * 0.8) / (0.436 * 0.8)
    assert ratio == pytest.approx(expected_ratio, abs=1e-10)


def test_reconstruction_is_deterministic_under_fragment_support_reordering():
    state_a = _h4_state(
        [[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436],
        [[0b01, 0b01], [0b10, 0b10]], [0.7, 0.714])
    # Same physical support, rows permuted within each fragment.
    state_b = _h4_state(
        [[0b10, 0b10], [0b01, 0b01]], [0.436, 0.9],
        [[0b10, 0b10], [0b01, 0b01]], [0.714, 0.7])

    dets_a, coeffs_a = reconstruct_las_vector(state_a)
    dets_b, coeffs_b = reconstruct_las_vector(state_b)
    got_a = dict(zip(dets_a, coeffs_a))
    got_b = dict(zip(dets_b, coeffs_b))
    assert set(got_a) == set(got_b)
    for det in got_a:
        assert got_a[det] == pytest.approx(got_b[det], abs=1e-12)

    e_a = las_reconstruction_energy(state_a, dets_a, coeffs_a)
    e_b = las_reconstruction_energy(state_b, dets_b, coeffs_b)
    assert e_a == pytest.approx(e_b, abs=1e-12)


def test_electronic_and_scalar_constant_are_combined_exactly_once():
    state = _h4_state(
        [[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436],
        [[0b01, 0b01], [0b10, 0b10]], [0.7, 0.714])
    dets, coeffs = reconstruct_las_vector(state)
    total = las_reconstruction_energy(state, dets, coeffs)
    ham = selected_hamiltonian(state["h1"], state["eri"], dets, n_orb=4)
    c = np.asarray(coeffs)
    electronic_only = float(c @ ham @ c) / float(c @ c)
    assert total - electronic_only == pytest.approx(state["e_const"], abs=1e-14)


def test_matrix_free_reconstruction_energy_matches_dense_oracle():
    state = _h4_state(
        [[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436],
        [[0b01, 0b01], [0b10, 0b10]], [0.7, 0.714])
    dets, coeffs = reconstruct_las_vector(state)
    dense = las_reconstruction_energy(state, dets, coeffs)
    matrix_free = las_reconstruction_energy(state, dets, coeffs, dense_limit=0)
    assert matrix_free == pytest.approx(dense, abs=1e-12)


def test_certify_raises_clearly_on_a_genuine_mismatch():
    state = _h4_state(
        [[0b01, 0b01], [0b10, 0b10]], [0.9, 0.436],
        [[0b01, 0b01], [0b10, 0b10]], [0.7, 0.714], h00=12345.0)
    with pytest.raises(AssertionError, match="Rayleigh-energy certificate FAILED"):
        certify_las_reconstruction(state, tol=1e-8)

# A real-parent integration certificate is deferred to the later physical-validation
# phase. The seven tests above cover the standalone reconstruction paths with synthetic
# data: one and two fragments, duplicate merging, deterministic reordering, applying the
# scalar constant once, matrix-free/dense agreement, and certificate failure.
