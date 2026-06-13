import numpy as np
import pytest
from pyscf.tools.fcidump import from_integrals

from FRASCI.crossflow.solver import run_cross_coupled_solver


def _make_toy_fcidump(tmp_path, n=6, n_alpha=3, n_beta=2):
    rng = np.random.default_rng(42)
    h1 = np.diag(np.linspace(-1.0, 1.0, n))
    B = rng.random((n, n, n, n)) * 0.05
    eri = (
        B
        + B.transpose(1, 0, 2, 3)
        + B.transpose(0, 1, 3, 2)
        + B.transpose(2, 3, 0, 1)
    )
    path = str(tmp_path / "toy.fcidump")
    from_integrals(
        path,
        h1,
        eri,
        nmo=n,
        nelec=n_alpha + n_beta,
        nuc=0.0,
        ms=n_alpha - n_beta,
    )
    return path, h1, eri, n_alpha, n_beta


def _make_toy_gamma(h1, n_alpha, n_beta):
    n = h1.shape[0]
    occ = np.zeros(n)
    order = np.argsort(np.diag(h1))
    for i in range(n_alpha):
        occ[order[i]] += 1.0
    for i in range(n_beta):
        occ[order[i]] += 1.0
    return np.diag(occ)


def _run_toy_solver(tmp_path, *, n=6, n_alpha=2, n_beta=1, **kwargs):
    fcidump, h1, _eri, na, nb = _make_toy_fcidump(
        tmp_path,
        n=n,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )
    gamma_path = str(tmp_path / "gamma.npy")
    np.save(gamma_path, _make_toy_gamma(h1, na, nb))
    result = run_cross_coupled_solver(
        fcidump,
        gamma_path=gamma_path,
        n_fragments=2,
        max_iters=1,
        **kwargs,
    )
    return result


def test_max_iters_1_produces_two_history_entries(tmp_path):
    result = _run_toy_solver(tmp_path)

    assert len(result.iteration_history) == 2
    assert result.iteration_history[0].iter == 0
    assert result.iteration_history[1].iter == 1


def test_iter_0_no_pt2_fields(tmp_path):
    result = _run_toy_solver(tmp_path)
    it0 = result.iteration_history[0]

    assert it0.E_pt2_cross is None
    assert it0.delta_E is None
    assert it0.E_total_postprocessed is None
    assert it0.max_abs_delta_h is None
    assert it0.min_gap is None


def test_iter_1_has_pt2_fields(tmp_path):
    result = _run_toy_solver(tmp_path)
    it1 = result.iteration_history[1]

    assert it1.E_pt2_cross is not None
    assert isinstance(it1.E_pt2_cross, float)
    assert it1.delta_E is not None


def test_status_success_one_shot(tmp_path):
    result = _run_toy_solver(tmp_path)

    assert result.status == "SUCCESS_ONE_SHOT"


def test_e_total_postprocessed_definition(tmp_path):
    result = _run_toy_solver(tmp_path)
    it0, it1 = result.iteration_history[0], result.iteration_history[1]

    expected_pp = it0.E_total + it1.E_pt2_cross
    assert abs(it1.E_total_postprocessed - expected_pp) < 1e-12


def test_e_total_final_is_re_solved(tmp_path):
    result = _run_toy_solver(tmp_path)
    it1 = result.iteration_history[1]

    assert abs(result.E_total_final - it1.E_total) < 1e-12


def test_e_mf_global_recorded(tmp_path):
    result = _run_toy_solver(tmp_path)

    assert isinstance(result.E_mf_global, float)


def test_electron_count_partition_coverage(tmp_path):
    n, na, nb = 6, 3, 2
    result = _run_toy_solver(tmp_path, n=n, n_alpha=na, n_beta=nb)

    assert result.n_alpha == na
    assert result.n_beta == nb
    assert result.n_orb == n


def test_invalid_max_iters_raises():
    with pytest.raises(ValueError, match="max_iters"):
        run_cross_coupled_solver("dummy.fcidump", max_iters=0)


def test_invalid_damping_raises():
    with pytest.raises(ValueError, match="damping"):
        run_cross_coupled_solver("dummy.fcidump", damping=0.0)
