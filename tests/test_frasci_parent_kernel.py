"""Correctness check: frasci.parent._kernel must reproduce
engine.lasscf.coo_kernel / engine.lasscf.runners.run_lasscf_coo's aufbau
patch and phased kernel exactly, on both direct numerics and a real
LASSCF+TrimCI smoke run."""
import numpy as np
import pytest

from frasci.parent._kernel import (
    _fixed_support_ground_state,
    _nearest_orthogonal,
    aufbau_init_rdm12s,
    make_phased_kernel,
)
from frasci.recovery._state import selected_hamiltonian


def test_nearest_orthogonal_prevents_long_rotation_composition_drift():
    theta = 0.013
    exact_step = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    # Model the tiny nonorthogonality of a numerically optimized BFGS step.
    numerical_step = exact_step * (1.0 + 5e-11)
    raw = np.eye(2)
    snapped = np.eye(2)
    for _ in range(250):
        raw = raw @ numerical_step
        snapped = _nearest_orthogonal(snapped @ numerical_step)

    raw_err = np.max(np.abs(raw.T @ raw - np.eye(2)))
    snapped_err = np.max(np.abs(snapped.T @ snapped - np.eye(2)))
    assert raw_err > 1e-8  # reproduces the AlFe2 accumulation scale
    assert snapped_err < 1e-14


@pytest.mark.parametrize("norb,na,nb", [(4, 2, 2), (6, 3, 3), (6, 3, 2), (8, 5, 3), (10, 6, 6), (12, 6, 6)])
def test_aufbau_init_rdm12s_matches_pyscf(norb, na, nb):
    """Same reference computation as the existing, untouched
    tests/test_aufbau_init_rdm.py::test_analytic_matches_pyscf -- verifies
    frasci's copy against pyscf directly, not against the engine copy (both
    should independently match pyscf; that's a stronger check than comparing
    two copies of the same rewritten formula to each other)."""
    from pyscf.fci import direct_spin1, cistring

    addr_a = cistring.str2addr(norb, na, (1 << na) - 1)
    addr_b = cistring.str2addr(norb, nb, (1 << nb) - 1)
    ci = np.zeros((cistring.num_strings(norb, na), cistring.num_strings(norb, nb)))
    ci[addr_a, addr_b] = 1.0
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = direct_spin1.make_rdm12s(ci, norb, (na, nb))
    ref_dm1s = np.stack([dm1a, dm1b], axis=0)
    ref_dm2 = dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb

    dm1s, dm2 = aufbau_init_rdm12s(norb, (na, nb))
    assert np.abs(dm1s - ref_dm1s).max() < 1e-12
    assert np.abs(dm2 - ref_dm2).max() < 1e-12


def test_phased_kernel_one_fragment_reaches_fci():
    """1 fragment == full active space -> LASSCF+phased-COO must reach FCI
    exactly, same assertion as coo_kernel.py's own __main__ smoke test."""
    from pyscf import gto, scf, ao2mo
    from pyscf.fci import direct_spin1 as ds1
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox

    mol = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
               basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    norb = mol.nao_nr()
    na = nb = mol.nelectron // 2
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)
    e_fci, _ = ds1.kernel(h1, h2, norb, (na, nb), ecore=mol.energy_nuc())

    las = LASSCF(mf, ncas_sub=(norb,), nelecas_sub=[(na, nb)], spin_sub=(1,))
    las.verbose = 0
    kernel_fn = make_phased_kernel(
        threshold=1e-8, discovery_num_runs=4, refine_num_runs=1,
        max_dets_phase0=50, max_dets_phase1=200, seed=1,
    )
    las.fciboxes[0] = make_fcibox(mol, kernel=kernel_fn, spin=0, smult=1)
    las.kernel(mf.mo_coeff)
    assert abs(las.e_tot - e_fci) < 1e-6


def test_fixed_support_ground_state_keeps_identities_and_finds_lowest_root():
    """The rebuild-only rediagonalizer changes coefficients, never support."""
    n_orb = 4
    h1 = np.diag([-1.2, -0.8, 0.1, 0.4])
    h1[1, 2] = h1[2, 1] = 0.15
    eri = np.zeros((n_orb,) * 4)
    dets = np.asarray([
        (0b0011, 0b0011),
        (0b0101, 0b0011),
        (0b1001, 0b0011),
    ], dtype=np.uint64)
    previous = np.asarray([0.0, 0.0, 1.0])

    result = _fixed_support_ground_state(h1, eri, dets, previous, n_orb)
    dense = selected_hamiltonian(
        h1, eri, [tuple(map(int, d)) for d in dets], n_orb=n_orb)

    assert np.array_equal(result.determinants, dets)
    assert result.energy == pytest.approx(np.linalg.eigvalsh(dense)[0], abs=1e-10)
    assert result.fixed_support_solver == "dense_exact"
    assert result.fixed_support_residual < 1e-10
    assert not np.allclose(np.abs(result.coefficients), np.abs(previous))


def test_phased_kernel_freezes_support_after_first_discovery(monkeypatch):
    """A second RDM-JK call must bypass TrimCI selection and keep call-1 support."""
    from pyscf import ao2mo, gto, scf
    import trimci

    mol = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
                basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    n_orb = mol.nao_nr()
    na = nb = mol.nelectron // 2
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n_orb,) * 4)
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=1, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, enable_phase2=True,
        max_dets_phase2=6, orbopt=False, warm_start_kappa=False,
        warm_start_dets=True, freeze_support_after_discovery=True,
        rediscover_tol=float("inf"), dm_mixing_alpha=1.0, seed=3,
        log_callback=lambda *args: seen.append(args[-1]),
    )
    args = (n_orb, (na, nb), 0.0, np.stack([h1, h1]), eri)
    kernel(*args)
    first_support = np.asarray(kernel.state["dets_prev"]).copy()

    def fail_if_selection_runs_again(*_args, **_kwargs):
        raise AssertionError("TrimCI selection ran after support was frozen")

    monkeypatch.setattr(trimci, "ground_state_from_integrals", fail_if_selection_runs_again)
    h1_next = h1.copy()
    h1_next[0, 0] += 1e-4
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_next, h1_next]), eri)

    assert np.array_equal(kernel.state["dets_prev"], first_support)
    assert seen[0]["phased_mode"] == "discover"
    assert seen[1]["phased_mode"] == "fixed_support"
    assert seen[1]["phased_num_runs_effective"] == 0
    assert seen[1]["fixed_support_solver"] == "dense_exact"
    assert seen[1]["fixed_support_residual"] < 1e-10


def _h4_chain_integrals():
    """Shared toy fragment integrals for the phase-policy diagnostics below --
    same H4/STO-3G chain as the existing freeze-support test above."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
                basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    n_orb = mol.nao_nr()
    na = nb = mol.nelectron // 2
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n_orb,) * 4)
    return n_orb, na, nb, h1, eri


def test_first_call_delta_metrics_are_null():
    """Nothing to diff against yet on a fragment's very first call."""
    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=1, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False, seed=1,
        log_callback=lambda *args: seen.append(args[-1]))
    kernel(n_orb, (na, nb), 0.0, np.stack([h1, h1]), eri)
    assert seen[0]["delta_h_rel"] is None
    assert seen[0]["delta_e_ha"] is None
    assert seen[0]["delta_rdm_norm"] is None
    assert seen[0]["delta_rdm_norm_raw"] is None
    assert seen[0]["support_jaccard"] is None


def test_mixed_and_raw_rdm_telemetry_remain_distinct_under_mixing():
    """Required test 10: with dm_mixing_alpha=0.5, the mixed delta_rdm_norm
    (post-damping, what actually feeds back into the outer RDM-JK loop) and
    the raw delta_rdm_norm_raw (pre-damping, this call's own undamped RDM)
    must be tracked as genuinely separate quantities -- not aliased to the
    same number -- while the embedding is still changing enough to matter."""
    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False,
        warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, dm_mixing_alpha=0.5, seed=1,
        log_callback=lambda *args: seen.append(args[-1]))
    h1_norm = float(np.linalg.norm(h1))

    kernel(n_orb, (na, nb), 0.0, np.stack([h1, h1]), eri)
    h1_1 = h1.copy()
    h1_1[0, 0] += 1e-2 * h1_norm
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_1, h1_1]), eri)
    h1_2 = h1_1.copy()
    h1_2[0, 0] += 1e-2 * h1_norm
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_2, h1_2]), eri)

    for row in seen[1:]:
        assert row["delta_rdm_norm"] is not None
        assert row["delta_rdm_norm_raw"] is not None
    # Damping (alpha=0.5) blends this call's raw RDM with the PREVIOUS call's
    # already-mixed RDM, so the mixed step is smaller than the raw step it was
    # built from -- a real numerical distinction, not merely two different keys.
    assert seen[-1]["delta_rdm_norm"] != pytest.approx(seen[-1]["delta_rdm_norm_raw"])
    assert seen[-1]["delta_rdm_norm"] < seen[-1]["delta_rdm_norm_raw"]


def test_support_jaccard_identical_and_disjoint(monkeypatch):
    """Same support back-to-back -> J=1; a fully disjoint support -> J=0."""
    from types import SimpleNamespace

    import trimci

    n_orb, na, nb, h1, eri = _h4_chain_integrals()

    def make_result(dets, coeffs):
        coeffs = np.asarray(coeffs, dtype=float)
        coeffs /= np.linalg.norm(coeffs)
        return SimpleNamespace(
            n_dets=len(dets), determinants=np.asarray(dets, dtype=np.uint64),
            coefficients=coeffs, energy=-1.0, energy_pt2=None,
            orbital_rotation=np.eye(n_orb),
            energies=SimpleNamespace(history=[-1.0]))

    dets_a = [(0b0011, 0b0011), (0b0101, 0b0011)]
    dets_b = [(0b0110, 0b0011), (0b1001, 0b0011)]
    results = iter([
        make_result(dets_a, [0.9, 0.436]),
        make_result(dets_a, [0.436, 0.9]),
        make_result(dets_b, [0.9, 0.436]),
    ])
    monkeypatch.setattr(
        trimci, "ground_state_from_integrals",
        lambda *a, **k: next(results))

    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, orbopt=False, seed=1,
        log_callback=lambda *args: seen.append(args[-1]))
    h1s = np.stack([h1, h1])
    kernel(n_orb, (na, nb), 0.0, h1s, eri)
    kernel(n_orb, (na, nb), 0.0, h1s, eri)
    kernel(n_orb, (na, nb), 0.0, h1s, eri)

    assert seen[0]["support_jaccard"] is None
    assert seen[1]["support_jaccard"] == pytest.approx(1.0)
    assert seen[2]["support_jaccard"] == pytest.approx(0.0)


def test_delta_h_rel_uses_immediately_previous_hamiltonian():
    """delta_h_rel must track the LAST call's h1, unlike phased_drift (which
    tracks the last DISCOVERY's h1 and is silent between two refine calls)."""
    n_orb, na, nb, h1_a, eri = _h4_chain_integrals()
    rediscover_tol = 1e-2
    h1_norm = float(np.linalg.norm(h1_a))
    step = 0.006 * h1_norm

    # h1_b: one small step from h1_a (relative drift ~0.006 < rediscover_tol=0.01
    # -- legacy "refine", h1_ref stays at h1_a). h1_c: ANOTHER identical small
    # step from h1_b -- delta_h_rel(c vs b) is again ~0.006 (small), but the
    # ACCUMULATED drift from h1_a is now ~0.012 > rediscover_tol, so legacy
    # phased_drift (always relative to h1_ref=h1_a, since it never updated at
    # call 2) reports a large drift and triggers a fresh "discover" -- exactly
    # the two metrics diverging the user's spec says they must.
    h1_b = h1_a.copy()
    h1_b[0, 0] += step
    h1_c = h1_b.copy()
    h1_c[0, 0] += step

    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=1, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False, seed=1,
        rediscover_tol=rediscover_tol,
        log_callback=lambda *args: seen.append(args[-1]))
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_a, h1_a]), eri)
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_b, h1_b]), eri)
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_c, h1_c]), eri)

    assert seen[1]["phased_mode"] == "refine"
    assert seen[2]["phased_drift"] > rediscover_tol
    assert seen[2]["phased_mode"] == "discover"
    assert seen[2]["delta_h_rel"] < rediscover_tol


def test_adaptive_dispatch_only_discovers_on_first_call_then_stays_fixed_support():
    """Production episode-aware dispatch (required tests 1-2): the very first
    call is phase0_phase1; every later call in the SAME episode is
    fixed_support regardless of delta_h_rel magnitude -- including a
    deliberately enormous embedding change, which the superseded
    drift-triggered dispatcher would have treated as a rediscovery trigger."""
    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False,
        warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, seed=1,
        log_callback=lambda *args: seen.append(args[-1]))
    h1_norm = float(np.linalg.norm(h1))

    def bumped(base, rel):
        out = base.copy()
        out[0, 0] += rel * h1_norm
        return out

    kernel(n_orb, (na, nb), 0.0, np.stack([h1, h1]), eri)                        # call 0: first
    h1_1 = bumped(h1, 1e-6)
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_1, h1_1]), eri)                    # call 1: tiny
    h1_2 = bumped(h1_1, 5e-3)
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_2, h1_2]), eri)                    # call 2: moderate
    h1_3 = bumped(h1_2, 50.0)  # delta_h_rel ~ 50 -- would have forced rediscovery
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_3, h1_3]), eri)                    # call 3: huge

    assert seen[0]["phase_policy_action"] == "phase0_phase1"
    assert seen[0]["phase_policy_actual_eigensolver"] == "trimci_phased"
    assert seen[0]["delta_h_rel"] is None

    for i in (1, 2, 3):
        assert seen[i]["phase_policy_action"] == "fixed_support", i
        assert seen[i]["phase_policy_actual_eigensolver"] == "dense_exact", i
        assert seen[i]["support_jaccard"] == pytest.approx(1.0), i
        assert seen[i]["fixed_support_residual"] < 1e-8, i
    # call 3's delta_h_rel is the huge one -- recorded as telemetry, not acted on.
    assert seen[3]["delta_h_rel"] > 1.0


def test_adaptive_dispatch_ten_same_frame_calls_yield_exactly_one_discovery():
    """Required test 3: ten same-frame calls -> exactly one phase0_phase1
    (the first), the other nine all fixed_support with support_jaccard==1."""
    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False,
        warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, seed=1,
        log_callback=lambda *args: seen.append(args[-1]))
    h1s = np.stack([h1, h1])
    for _ in range(10):
        kernel(n_orb, (na, nb), 0.0, h1s, eri)

    actions = [row["phase_policy_action"] for row in seen]
    assert actions.count("phase0_phase1") == 1
    assert actions[0] == "phase0_phase1"
    assert actions[1:] == ["fixed_support"] * 9
    for row in seen[1:]:
        assert row["support_jaccard"] == pytest.approx(1.0)
        assert row["fixed_support_residual"] < 1e-8


def test_adaptive_fixed_support_never_invokes_trimci_selection(monkeypatch):
    """A same-episode call under adaptive_phase_policy must rediagonalize the
    existing support, never call TrimCI selection, and keep determinant
    identities -- even when the embedding change is large (the dispatch no
    longer looks at drift magnitude at all, see make_phased_kernel's docstring)."""
    import trimci

    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False,
        warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, seed=1)
    kernel(n_orb, (na, nb), 0.0, np.stack([h1, h1]), eri)
    first_support = np.asarray(kernel.state["dets_prev"]).copy()

    def fail_if_selection_runs_again(*_args, **_kwargs):
        raise AssertionError("TrimCI selection ran under a same-episode adaptive call")

    monkeypatch.setattr(trimci, "ground_state_from_integrals", fail_if_selection_runs_again)
    h1_next = h1.copy()
    h1_next[0, 0] += 10.0 * max(abs(h1[0, 0]), 1.0)  # deliberately huge
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_next, h1_next]), eri)

    assert np.array_equal(kernel.state["dets_prev"], first_support)


def test_inner_plateau_alone_never_triggers_phase2_promotion():
    """Required test 11: an inner RDM-JK loop reaching its own plateau (many
    consecutive identical-h1 fixed_support calls, both delta_e and
    delta_rdm well under the plateau tolerances) must NEVER promote to Phase 2
    on its own -- only the parent's post-convergence finalizer may, by setting
    state["force_phase2_next"] (see test_adaptive_kernel_phase2_promotion_
    persists_and_reconverges below for that path). This directly guards
    against the superseded autonomous-plateau trigger."""
    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=2, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, max_dets_phase2=6,
        orbopt=False, warm_start_kappa=False, warm_start_dets=True,
        adaptive_phase_policy=True, adaptive_phase2_after_plateau=True,
        phase2_plateau_delta_e_tol=1e-3, phase2_plateau_delta_rdm_tol=1e-2,
        seed=1, log_callback=lambda *args: seen.append(args[-1]))

    h1s = np.stack([h1, h1])
    for _ in range(12):
        kernel(n_orb, (na, nb), 0.0, h1s, eri)

    actions = [row["phase_policy_action"] for row in seen]
    assert "phase2_promote" not in actions
    assert actions[0] == "phase0_phase1"
    assert actions[1:] == ["fixed_support"] * 11
    # The plateau genuinely formed (both deltas settle far under the
    # tolerances) -- this is a real plateau being correctly ignored, not an
    # absence of one.
    assert seen[-1]["delta_e_ha"] is not None and seen[-1]["delta_e_ha"] <= 1e-3
    assert (seen[-1]["delta_rdm_norm"] is not None
            and seen[-1]["delta_rdm_norm"] <= 1e-2)


def test_legacy_behavior_unchanged_when_adaptive_policy_disabled():
    """New extras keys must be inert/None and phased_mode must be exactly as
    before when adaptive_phase_policy is left at its default False."""
    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    seen = []
    kernel = make_phased_kernel(
        threshold=1e-6, discovery_num_runs=1, refine_num_runs=1,
        max_dets_phase0=6, max_dets_phase1=6, orbopt=False, seed=1,
        log_callback=lambda *args: seen.append(args[-1]))
    h1s = np.stack([h1, h1])
    kernel(n_orb, (na, nb), 0.0, h1s, eri)
    h1_next = h1.copy()
    h1_next[0, 0] += 1e-3
    kernel(n_orb, (na, nb), 0.0, np.stack([h1_next, h1_next]), eri)

    for row in seen:
        assert row["phased_mode"] in ("discover", "refine")
        assert row["phase_policy_action"] is None
        assert row["phase_policy_actual_eigensolver"] is None
        assert row["phase2_plateau_triggered"] is False
        assert row["phase2_changed_rdm"] is None
        assert row["post_phase2_active"] is False
        assert row["post_phase2_stable_streak"] == 0
        assert row["phase2_promoted_ever"] is False
        assert row["expansion_residual"] is None
        assert row["expansion_termination_reason"] is None
        assert row["post_phase2_converged_now"] is False
        assert row["post_phase2_stable_streak_after"] == 0


@pytest.fixture(scope="module")
def h12_fixed_frame_support():
    """Real H12/STO-3G 8-orbital (4a4b) fragment support, grown by an actual
    TrimCI Phase-0+Phase-1 call with Phase-1 orbital optimization explicitly
    DISABLED (``cfg.orbopt_phase1 = False``) -- comfortably exceeds the
    dense-exact cutoff (256 dets: lands at 380).

    Corrected 2026-09-14: an earlier version of this fixture left
    ``orbopt_phase1`` at its ``Config`` default of ``True``, so Phase 1
    rotated orbitals internally; ``res.orbital_rotation`` had off-diagonal
    elements up to ~15% (verified directly, not assumed), yet that earlier
    test still combined ``res.coefficients`` (living in the ROTATED frame)
    with the ORIGINAL (un-rotated) ``h1``/``eri`` -- a genuine frame mismatch,
    not a real Davidson limitation. This fixture verifies (not assumes)
    ``orbital_rotation`` is the identity before returning, so every caller of
    this fixture is guaranteed a self-consistent (h1, eri, dets, coefficients)
    frame throughout.
    """
    from pyscf import ao2mo, gto, scf
    import trimci

    atoms = "; ".join(f"H 0 0 {1.8 * i}" for i in range(12))
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    assert mf.converged

    frag_orbs = list(range(2, 10))
    n_orb = 8
    mo_frag = mf.mo_coeff[:, frag_orbs]
    h1 = mo_frag.T @ mf.get_hcore() @ mo_frag
    eri = ao2mo.full(mol, mo_frag, compact=False).reshape((n_orb,) * 4)

    cfg = trimci.Config(threshold=1e-6, num_runs=4, orbopt=False,
                         max_dets_phase0=8, enable_phase1=True, max_dets_phase1=380,
                         enable_phase2=False, seed=7, verbosity=0)
    cfg.orbopt_phase1 = False
    res = trimci.ground_state_from_integrals(h1, eri, 8, ms2=0, e_nuc=0.0, config=cfg)

    U = np.asarray(res.orbital_rotation)
    assert np.allclose(U, np.eye(n_orb), atol=1e-10), (
        f"expected orbopt_phase1=False to keep orbital_rotation == identity, got "
        f"max|U-I|={np.max(np.abs(U - np.eye(n_orb))):.3e} -- this fixture's "
        "(h1, eri, dets, coefficients) frame consistency depends on this")
    dets = np.asarray(res.determinants)
    assert len(dets) > 256, (
        f"got {len(dets)} dets, need >256 to exercise the Davidson branch -- "
        "tune threshold/num_runs/max_dets_phase1/the fragment window above")
    return n_orb, h1, eri, dets, np.asarray(res.coefficients)


def test_fixed_support_ground_state_davidson_guess_shape_is_one_by_n(
        monkeypatch, h12_fixed_frame_support):
    """Davidson guess-shape regression: the native contract is
    initial_guess.shape == (n_states, N); for one root that is (1, N), never
    (N, 1). Intercept the native call and assert this directly on the real
    >256-det H12 support (the Davidson branch), not a mocked stand-in."""
    import trimci

    n_orb, h1, eri, dets, coeffs = h12_fixed_frame_support
    fe = trimci.trimci_core.fast_expansion
    real_davidson = fe.davidson_solve_matfree
    seen_shapes = []

    def spy(alpha, beta, h1c, eric, norb_, params, guess):
        arr = np.asarray(guess)
        if arr.size > 0:
            seen_shapes.append(arr.shape)
        return real_davidson(alpha, beta, h1c, eric, norb_, params, guess)

    monkeypatch.setattr(fe, "davidson_solve_matfree", spy)

    dense = selected_hamiltonian(h1, eri, [tuple(map(int, d)) for d in dets], n_orb=n_orb)
    exact_gs = np.linalg.eigh(dense)[1][:, 0]
    _fixed_support_ground_state(h1, eri, dets, exact_gs, n_orb)

    assert seen_shapes, "davidson_solve_matfree was never called -- dense_exact branch?"
    for shape in seen_shapes:
        assert shape[0] == 1, (
            f"one-root guess shape {shape} has shape[0]={shape[0]} != 1 -- "
            "native contract is (n_states, N), not (N, n_states)")


def test_fixed_support_ground_state_rejects_energy_only_convergence(
        monkeypatch, h12_fixed_frame_support):
    """Davidson stopping regression: native Davidson can report
    converged=True (its own residual/energy-change criteria satisfied) for a
    result whose INDEPENDENTLY recomputed residual is nowhere near this
    function's 1e-8 gate -- exactly the failure mode energy_tol=1e-12 caused
    on the real binding before the fix (a nonzero energy_tol lets the energy
    trajectory plateau and stop the iteration before the residual has
    actually tightened; see this function's own energy_tol=0.0 comment).
    Reproduce that shape directly (not by trying to defeat the now-hardcoded
    energy_tol=0.0, which the fix made uncontrollable from the outside on
    purpose): mock the native call to report converged=True for a
    deliberately wrong eigenvector (weight on a single determinant, on a
    380-determinant space), and verify the independent matvec-based residual
    check rejects it -- never accepted on res.converged alone."""
    import trimci

    n_orb, h1, eri, dets, coeffs = h12_fixed_frame_support
    fe = trimci.trimci_core.fast_expansion

    class _FakeConverged:
        def __init__(self, eigenvectors):
            self.converged = True
            self.eigenvectors = eigenvectors
            self.n_iters = 1

    bad_vec = np.zeros(len(dets))
    bad_vec[0] = 1.0

    def fake_davidson(alpha, beta, h1c, eric, norb_, params, guess):
        return _FakeConverged(bad_vec.copy())

    monkeypatch.setattr(fe, "davidson_solve_matfree", fake_davidson)
    with pytest.raises(RuntimeError, match="residual"):
        _fixed_support_ground_state(h1, eri, dets, coeffs, n_orb)


def test_fixed_support_ground_state_h12_corrected_frame_and_drift_cases(
        h12_fixed_frame_support):
    """Corrected replacement for the earlier (superseded) H12 Davidson check
    -- see docs/reports/frasci_phase_policy_h8_ab_2026-09-14.md's Corrections
    section. The earlier version fed a rotated-frame res.coefficients
    together with un-rotated integrals (a genuine frame mismatch, not a real
    Davidson limitation) and used the pre-fix energy_tol=1e-12. This version
    uses the frame-verified h12_fixed_frame_support fixture and this
    function's corrected energy_tol=0.0, at three drift magnitudes: zero,
    ~1e-4 relative (this experiment's adaptive_fixed_support_tol), and ~1e-2
    relative (its adaptive_phase0_tol) -- chaining each case's own converged
    output into the next, same as a real RDM-JK sequence would."""
    n_orb, h1, eri, dets, _native_coeffs = h12_fixed_frame_support
    dense = selected_hamiltonian(h1, eri, [tuple(map(int, d)) for d in dets], n_orb=n_orb)
    exact_gs = np.linalg.eigh(dense)[1][:, 0]

    h1_norm = float(np.linalg.norm(h1))
    cases = [
        ("zero_drift", 0.0),
        ("delta_h_rel~1e-4", 1e-4 * h1_norm),
        ("delta_h_rel~1e-2", 1e-2 * h1_norm),
    ]

    previous_dets, previous_coeffs, h1_current = dets, exact_gs, h1
    for label, bump in cases:
        h1_current = h1_current.copy()
        h1_current[0, 0] += bump
        result = _fixed_support_ground_state(
            h1_current, eri, previous_dets, previous_coeffs, n_orb)
        assert result.fixed_support_solver == "davidson_two_start", label
        assert result.fixed_support_residual < 1e-8, label
        assert np.array_equal(np.asarray(result.determinants), dets), label

        dense_case = selected_hamiltonian(
            h1_current, eri, [tuple(map(int, d)) for d in dets], n_orb=n_orb)
        e_exact_case = float(np.linalg.eigh(dense_case)[0][0])
        assert result.energy == pytest.approx(e_exact_case, abs=1e-7), label

        previous_dets, previous_coeffs = result.determinants, result.coefficients


def test_adaptive_kernel_phase2_promotion_persists_and_reconverges():
    """Required tests 12-13: Phase 2 fires ONLY when the caller (standing in
    for the parent's post-LASSCF-convergence finalizer) sets
    state["force_phase2_next"] -- never autonomously from a natural plateau
    (constant h1 across many calls, which alone must produce zero
    "phase2_promote" actions -- required test 11, reinforced here in the
    exact scenario Phase-2 persistence is checked in). Exactly one direct
    Phase-2 promotion per request (never rerunning Phase 0/Phase 1); the
    enlarged support persists as the new baseline (never shrinks back to
    max_dets_phase1); support_jaccard == 1.0 through the post-Phase-2
    fixed-support span; post-Phase-2 convergence (the "three-stable-call
    contract") is declared after exactly 3 consecutive stable calls, not
    before and not again while still active. (A large embedding change alone
    does NOT reset an episode without an actual external-frame transition --
    see test_two_episodes_each_get_their_own_forced_promotion in
    test_frasci_parent_cached_frame.py for that path, via the real
    frame_provider/invalidation mechanism, not a bare h1 jump.)"""
    import trimci

    n_orb, na, nb, h1, eri = _h4_chain_integrals()
    real_run_expansion_phased = trimci.trimci_core.fast_expansion.run_expansion_phased
    real_ground_state = trimci.ground_state_from_integrals
    expansion_calls = []
    ground_state_calls = []

    def spy_expansion(alpha, beta, coefficients, h1c, eric, norb_, phase_configs):
        expansion_calls.append([cfg.phase_index for cfg in phase_configs])
        return real_run_expansion_phased(
            alpha, beta, coefficients, h1c, eric, norb_, phase_configs)

    def spy_ground_state(*args, **kwargs):
        ground_state_calls.append(True)
        return real_ground_state(*args, **kwargs)

    seen = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(trimci.trimci_core.fast_expansion, "run_expansion_phased", spy_expansion)
        mp.setattr(trimci, "ground_state_from_integrals", spy_ground_state)

        kernel = make_phased_kernel(
            threshold=1e-6, discovery_num_runs=4, refine_num_runs=1,
            max_dets_phase0=6, max_dets_phase1=6, max_dets_phase2=12,
            orbopt=False, warm_start_kappa=False, warm_start_dets=True,
            adaptive_phase_policy=True, adaptive_phase2_after_plateau=True,
            phase2_plateau_delta_e_tol=1e-3, phase2_plateau_delta_rdm_tol=1e-2,
            seed=3, log_callback=lambda *args: seen.append(args[-1]))

        h1s = np.stack([h1, h1])

        # Cold start + a genuine constant-h1 plateau, which must NOT promote
        # on its own.
        for _ in range(5):
            kernel(n_orb, (na, nb), 0.0, h1s, eri)
        assert "phase2_promote" not in [row["phase_policy_action"] for row in seen]

        # Only now (standing in for the parent's post-convergence finalizer)
        # request promotion explicitly.
        kernel.state["force_phase2_next"] = True
        kernel(n_orb, (na, nb), 0.0, h1s, eri)
        for _ in range(4):
            kernel(n_orb, (na, nb), 0.0, h1s, eri)

    actions = [row["phase_policy_action"] for row in seen]
    n_dets = [row["n_dets_seed"] for row in seen]

    # Exactly one promotion for the one explicit request, and it never reruns
    # Phase 0/1.
    assert actions.count("phase2_promote") == 1
    promote_idx = actions.index("phase2_promote")
    assert [c for c in expansion_calls if c == [2]] == [[2]]
    # Only the genuine cold-start call ever reaches the public
    # ground_state_from_integrals path -- promotion never does.
    assert len(ground_state_calls) == 1

    pre_promote_n_dets = n_dets[promote_idx - 1]
    promoted_n_dets = n_dets[promote_idx]
    assert promoted_n_dets > pre_promote_n_dets

    # The support never shrinks back to the Phase-1 cap after promotion, and
    # support_jaccard == 1.0 throughout the post-Phase-2 fixed-support span.
    for i in range(promote_idx + 1, len(actions)):
        assert actions[i] in ("post_phase2_fixed_support", "fixed_support")
        assert n_dets[i] >= promoted_n_dets
        if seen[i]["support_jaccard"] is not None:
            assert seen[i]["support_jaccard"] == pytest.approx(1.0)

    # Post-Phase-2 convergence is declared exactly once, after the 3rd
    # consecutive stable call following promotion (not sooner, not again).
    converged_indices = [i for i, row in enumerate(seen) if row["post_phase2_converged_now"]]
    assert converged_indices == [promote_idx + 3]
    assert seen[promote_idx]["post_phase2_active"] is False  # going into the promote call
    assert seen[promote_idx + 1]["post_phase2_active"] is True  # active right after
    assert seen[promote_idx + 3]["post_phase2_active"] is True  # still active going into the 3rd stable call
    assert seen[promote_idx + 4]["post_phase2_active"] is False  # cleared after the streak completed
    assert actions[promote_idx + 4] == "fixed_support"  # reverted from post_phase2_*
