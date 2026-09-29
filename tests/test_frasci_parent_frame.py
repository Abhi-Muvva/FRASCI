"""frasci.parent._frame.compute_frame_transport -- the smallest explicit
orbital-frame data path a future cached-frame adaptive orbopt=True policy
needs (design/prototype, not yet wired into make_phased_kernel's dispatch).

Synthetic cases establish the mathematical contract directly (identity frame,
pure internal rotation, a genuine subspace move, and the non-orthonormal-input
trust-boundary check). The real-LASSCF case reuses this repo's own
established fact (from docs/reports/frasci_orbopt_frame_h8_contract_2026-09-14.md's
Experiment 6): an H8 LASSCF run's FIRST macro transition has real (~16%)
subspace leakage while a LATER, more-converged transition does not -- proving
this module's reusable/discard decision actually distinguishes the two on
real data, not just synthetic examples built to be easy.
"""
from __future__ import annotations

import numpy as np
import pytest

from frasci.parent._frame import compute_frame_transport


def _random_orthonormal_columns(n_ao, n_frag, seed):
    rng = np.random.default_rng(seed)
    full, _ = np.linalg.qr(rng.standard_normal((n_ao, n_ao)))
    return full[:, :n_frag], full


def test_same_frame_id_short_circuits_to_trivial_identity_result():
    C_prev, _ = _random_orthonormal_columns(8, 4, seed=1)
    S = np.eye(8)
    r = compute_frame_transport(C_prev, C_prev, S, frame_id_prev=3, frame_id_curr=3,
                                leakage_tol=0.01)
    assert r.same_frame is True
    assert r.leakage == 0.0
    assert r.reusable is True
    assert np.array_equal(r.Q, np.eye(4))
    assert np.array_equal(r.usable_transform, np.eye(4))


def test_pure_internal_rotation_has_zero_leakage_and_is_reusable():
    """A fragment subspace that only rotates INTERNALLY (same column space,
    different basis within it) must show ~zero leakage and an exactly
    orthogonal Q -- and compute_frame_transport's own usable_transform must
    equal that Q (both should recover R.T, since Q = C_curr.T S C_prev =
    (C_prev R).T S C_prev = R.T (C_prev.T S C_prev) = R.T for S-orthonormal
    C_prev)."""
    C_prev, _ = _random_orthonormal_columns(8, 4, seed=2)
    rng = np.random.default_rng(99)
    R = np.linalg.qr(rng.standard_normal((4, 4)))[0]
    C_curr = C_prev @ R
    S = np.eye(8)

    r = compute_frame_transport(C_prev, C_curr, S, frame_id_prev=0, frame_id_curr=1,
                                leakage_tol=0.01)
    assert r.same_frame is False
    assert r.leakage == pytest.approx(0.0, abs=1e-12)
    assert r.reusable is True
    assert r.orthogonality_err == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(r.Q - R.T)) < 1e-12
    assert np.max(np.abs(r.usable_transform - r.Q)) < 1e-12
    assert np.allclose(r.singular_values, 1.0, atol=1e-12)
    # arccos is ill-conditioned right at its domain boundary (derivative
    # diverges as x->1): a singular value at 1.0-1e-12 gives an angle error
    # of order sqrt(1e-12)~1e-6, not 1e-12 -- a property of arccos, not a bug.
    assert np.allclose(r.principal_angles_rad, 0.0, atol=1e-6)


def test_genuine_subspace_move_has_large_leakage_and_is_not_reusable():
    """A fragment whose orbital WINDOW shifts (not just an internal rotation
    of the same column space) must show real, large leakage and fail the
    reusable check at a normal tolerance."""
    C_prev, full = _random_orthonormal_columns(8, 4, seed=3)
    C_curr = full[:, 1:5]  # shifted window: shares only 3 of 4 original columns

    r = compute_frame_transport(C_prev, C_curr, np.eye(8), frame_id_prev=0, frame_id_curr=1,
                                leakage_tol=0.01)
    assert r.leakage > 0.1
    assert r.reusable is False


def test_leakage_tol_boundary_is_a_hard_cutoff():
    C_prev, full = _random_orthonormal_columns(8, 4, seed=3)
    C_curr = full[:, 1:5]
    r = compute_frame_transport(C_prev, C_curr, np.eye(8), frame_id_prev=0, frame_id_curr=1,
                                leakage_tol=1.0)
    assert r.reusable is True  # tolerance set above the actual leakage
    r2 = compute_frame_transport(C_prev, C_curr, np.eye(8), frame_id_prev=0, frame_id_curr=1,
                                 leakage_tol=1e-9)
    assert r2.reusable is False  # tolerance set (far) below the actual leakage


def test_non_orthonormal_input_is_rejected():
    C_prev, _ = _random_orthonormal_columns(8, 4, seed=4)
    C_curr, _ = _random_orthonormal_columns(8, 4, seed=5)
    with pytest.raises(ValueError, match="not S-orthonormal"):
        compute_frame_transport(C_prev * 1.1, C_curr, np.eye(8), frame_id_prev=0,
                                frame_id_curr=1, leakage_tol=0.01)


def test_mismatched_fragment_orbital_count_is_rejected():
    C_prev, _ = _random_orthonormal_columns(8, 4, seed=6)
    C_curr, _ = _random_orthonormal_columns(8, 3, seed=7)
    with pytest.raises(ValueError, match="orbital count cannot change"):
        compute_frame_transport(C_prev, C_curr, np.eye(8), frame_id_prev=0, frame_id_curr=1,
                                leakage_tol=0.01)


def _capture_h8_lasscf_mo_coeff(max_cycle_macro=4):
    """Minimal, test-local reproduction of
    docs/reports/frasci_orbopt_frame_h8_contract_2026-09-14.py's
    run_experiment6() capture mechanism -- duplicated (not imported from a
    dated docs/reports script) so this test file stays self-contained, same
    reasoning this repo's other tests build their own small fixtures inline
    rather than importing from report scripts."""
    from pyscf import gto, lo, scf
    from pyscf.tools import fcidump
    from frasci._mock_scf import build_mock_scf_from_fcidump
    from frasci.parent._kernel import make_phased_kernel, patch_large_fragment_aufbau
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox
    import tempfile
    import os

    atoms = "; ".join(f"H 0 0 {1.8 * i}" for i in range(8))
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    n_orb = mol.nao_nr()
    loc = lo.Boys(mol, mf.mo_coeff).kernel()
    centers = np.einsum("pi,pq,qi->i", loc, mol.intor("int1e_r")[2], loc)
    mo_ordered = loc[:, np.argsort(centers)]

    orbital_lists = [list(range(4)), list(range(4, 8))]
    nelec_per_frag = [(2, 2), (2, 2)]
    spin_sub = [1, 1]
    ncas_sub = tuple(len(f) for f in orbital_lists)

    with tempfile.TemporaryDirectory(prefix="test_frasci_parent_frame_") as work_dir:
        fcidump_path = os.path.join(work_dir, "h8.fcidump")
        fcidump.from_mo(mol, fcidump_path, mo_ordered)
        mf2 = build_mock_scf_from_fcidump(fcidump_path)
        S_AO = mf2.get_ovlp()

        las = LASSCF(mf2, ncas_sub, nelec_per_frag, spin_sub=spin_sub)
        las.max_cycle_macro = max_cycle_macro
        las.conv_tol_grad = 1e-4
        las.verbose = 0
        for i in range(len(orbital_lists)):
            kernel_fn = make_phased_kernel(
                fragment_idx=i, threshold=0.01, max_dets_phase0=6, max_dets_phase1=12,
                growth_factor_phase1=1.1, enable_phase2=False, max_dets_phase2=24,
                pt2=False, seed=1, invariance_atol=1e-4, discovery_num_runs=4,
                refine_num_runs=1, rediscover_tol=1e-2, orbopt=True, orbopt_cycles=3,
                warm_start_kappa=True)
            las.fciboxes[i] = make_fcibox(
                mf2.mol, kernel=kernel_fn, spin=spin_sub[i] - 1, smult=spin_sub[i])
        patch_large_fragment_aufbau(las, ncas_sub, nelec_per_frag)

        captures = []
        real_get_hop = las.get_hop

        def wrapped_get_hop(*args, **kwargs):
            mo_before = np.array(kwargs.get("mo_coeff"), copy=True)
            H_op = real_get_hop(*args, **kwargs)
            real_update_mo_eri = H_op.update_mo_eri

            def wrapped_update_mo_eri(x, h2eff_sub):
                mo_after, h2eff_new = real_update_mo_eri(x, h2eff_sub)
                captures.append({"mo_before": mo_before, "mo_after": np.array(mo_after, copy=True)})
                return mo_after, h2eff_new

            H_op.update_mo_eri = wrapped_update_mo_eri
            return H_op

        las.get_hop = wrapped_get_hop
        las.kernel(np.eye(n_orb))

    return captures, S_AO, orbital_lists


def test_reusable_decision_matches_real_lasscf_macro_convergence():
    """Real H8 LASSCF data, not synthetic: the FIRST macro transition has
    genuine (~16%) subspace leakage (established in
    docs/reports/frasci_orbopt_frame_h8_contract_2026-09-14.md's Experiment 6)
    and must be flagged not-reusable; a LATER, more-converged transition must
    be flagged reusable, at the SAME tolerance."""
    captures, S_AO, orbital_lists = _capture_h8_lasscf_mo_coeff(max_cycle_macro=4)
    assert len(captures) >= 3, "need at least 3 macro transitions to see convergence"

    leakage_tol = 0.05
    frag_orbs = orbital_lists[0]

    first = captures[0]
    C_prev = first["mo_before"][:, frag_orbs]
    C_curr = first["mo_after"][:, frag_orbs]
    r_first = compute_frame_transport(
        C_prev, C_curr, S_AO, frame_id_prev=0, frame_id_curr=1, leakage_tol=leakage_tol)
    assert r_first.leakage > leakage_tol
    assert r_first.reusable is False

    last = captures[-1]
    C_prev_last = last["mo_before"][:, frag_orbs]
    C_curr_last = last["mo_after"][:, frag_orbs]
    r_last = compute_frame_transport(
        C_prev_last, C_curr_last, S_AO, frame_id_prev=len(captures) - 1,
        frame_id_curr=len(captures), leakage_tol=leakage_tol)
    assert r_last.leakage < leakage_tol
    assert r_last.reusable is True
    assert r_last.leakage < r_first.leakage / 10  # orders-of-magnitude improvement
