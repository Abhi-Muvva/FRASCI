"""Fixed-frame native-filtered recovery growth tests."""
from __future__ import annotations

import numpy as np
import pytest
import trimci.trimci_core as tc

from frasci.recovery._native_stream import build_sector_constraint, run_fixed_filtered_ladder


def _h4_recovery_state():
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="; ".join(f"H 0 0 {1.0 * i}" for i in range(4)),
                basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])
    blocks = [[0, 1], [2, 3]]
    frag0_dets = np.array([[0b01, 0b01], [0b10, 0b10]], dtype=np.int64)
    frag0_coeffs = np.array([0.9, 0.436], dtype=float)
    frag0_coeffs /= np.linalg.norm(frag0_coeffs)
    frag1_dets = np.array([[0b01, 0b01]], dtype=np.int64)
    frag1_coeffs = np.array([1.0])
    frags = [
        dict(dets=frag0_dets, coeffs=frag0_coeffs,
             lookup=dict(zip(map(tuple, frag0_dets.tolist()), frag0_coeffs.tolist()))),
        dict(dets=frag1_dets, coeffs=frag1_coeffs,
             lookup=dict(zip(map(tuple, frag1_dets.tolist()), frag1_coeffs.tolist()))),
    ]
    return dict(n_orb=4, n_elec=(2, 2), blocks=blocks, orbital_lists=blocks,
                ref_sectors=[(1, 1), (1, 1)], frags=frags, mo_final=c.copy(),
                h1=h1, eri=eri, e_const=float(mol.energy_nuc()), h00=0.0)


PK_DET = (0b0101, 0b0101)
LAS_PRESENT_NOT_PROTECTED = (0b0110, 0b0110)
LAS_ABSENT = (0b0101, 0b0110)
DISPERSION = (0b0011, 0b1100)
CT1 = (0b0011, 0b0101)


def test_constraint_has_no_doorway_admission():
    state = _h4_recovery_state()
    constraint, suffix = build_sector_constraint(state, [PK_DET], n_ct_max=2)
    assert suffix == "64"
    dets = [tc.Determinant(*det) for det in (
        PK_DET, LAS_PRESENT_NOT_PROTECTED, LAS_ABSENT, DISPERSION, CT1)]
    labels, admitted = tc.screen_dets_64(constraint, dets)
    assert labels == [
        "protected", "las_present", "las_absent", "dispersion", "ct_allowed"]
    assert admitted == [True, False, False, True, True]


def test_fixed_filtered_ladder_is_nested_and_never_changes_parent_frame():
    state = _h4_recovery_state()
    ladder = run_fixed_filtered_ladder(
        state, [PK_DET], [1.0], recovery_targets=(1, 2, 3),
        n_ct_max=2, threshold=1e-6, growth_factor=2.0, max_rounds=8)
    assert [point["target_r"] for point in ladder] == [1, 2, 3]
    assert [point["n_recovery"] for point in ladder] == [1, 2, 3]
    assert all(point["target_met"] for point in ladder)
    assert all(max(point["frame_errors"].values()) == 0.0 for point in ladder)
    for left, right in zip(ladder, ladder[1:]):
        assert set(left["dets"]) <= set(right["dets"])
        assert set(left["eligible"]) <= set(right["eligible"])
    assert all(PK_DET in point["dets"] for point in ladder)
    assert all(LAS_PRESENT_NOT_PROTECTED not in point["dets"] for point in ladder)
    assert all(LAS_ABSENT not in point["dets"] for point in ladder)


def test_fixed_filtered_ladder_matrix_free_energy_matches_dense():
    from frasci.recovery._solve import (
        solve_contracted_hybrid, solve_contracted_hybrid_matrix_free)

    state = _h4_recovery_state()
    point = run_fixed_filtered_ladder(
        state, [PK_DET], [1.0], recovery_targets=(3,),
        n_ct_max=2, threshold=1e-6, growth_factor=2.0, max_rounds=8)[0]
    dense = solve_contracted_hybrid(state, point["eligible"])
    matrix_free = solve_contracted_hybrid_matrix_free(
        state, point["eligible"], tol=1e-11, max_iter=500)
    assert matrix_free["e_tot"] == pytest.approx(dense["e_tot"], abs=2e-9)
    assert matrix_free["generalized_residual_norm"] < 1e-8


def test_fixed_filtered_ladder_rejects_widths_above_128():
    state = dict(
        n_orb=129, n_elec=(1, 0), blocks=[[0]], ref_sectors=[(1, 0)],
        frags=[dict(dets=np.array([[1, 0]], dtype=np.int64), coeffs=np.array([1.0]))],
        mo_final=None, h1=None, eri=None, e_const=0.0, h00=0.0)
    with pytest.raises(NotImplementedError, match="192-bit"):
        run_fixed_filtered_ladder(
            state, [(1, 0)], [1.0], recovery_targets=(1,), n_ct_max=2)
