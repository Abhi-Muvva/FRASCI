import numpy as np
import pytest

from FRASCI.crossflow.io import load_or_compute_gamma, load_or_derive_ref_det


def test_gamma_load_full_matrix(tmp_path):
    n = 6
    gamma = np.eye(n) * 0.5
    path = str(tmp_path / "gamma.npy")
    np.save(path, gamma)
    h1 = np.diag(np.arange(n, dtype=float))
    eri = np.zeros((n, n, n, n))

    result = load_or_compute_gamma(path, h1, eri, 3, 3, n)

    assert result.gamma_source_mode == "provided_file"
    assert result.gamma_load_mode == "full_matrix"
    assert result.gamma.shape == (n, n)
    np.testing.assert_array_equal(result.gamma, gamma)
    assert result.uhf_cache is None


def test_gamma_load_diagonal_vector_promoted(tmp_path):
    n = 6
    diag_vec = np.ones(n) * 0.8
    path = str(tmp_path / "gamma.npy")
    np.save(path, diag_vec)
    h1 = np.diag(np.arange(n, dtype=float))
    eri = np.zeros((n, n, n, n))

    result = load_or_compute_gamma(path, h1, eri, 3, 3, n)

    assert result.gamma_source_mode == "provided_file"
    assert result.gamma_load_mode == "diagonal_vector_promoted_to_matrix"
    assert result.gamma.shape == (n, n)
    np.testing.assert_array_equal(result.gamma, np.diag(diag_vec))


def test_gamma_shape_mismatch_raises(tmp_path):
    n = 6
    path = str(tmp_path / "gamma.npy")
    np.save(path, np.ones((4, 4)))
    h1 = np.diag(np.arange(n, dtype=float))
    eri = np.zeros((n, n, n, n))

    with pytest.raises(ValueError, match="gamma shape"):
        load_or_compute_gamma(path, h1, eri, 3, 3, n)


def test_gamma_aufbau_fallback_no_pyscf(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pyscf":
            raise ImportError("pyscf not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    n = 4
    h1 = np.diag([-0.5, -0.3, 0.1, 0.4])
    eri = np.zeros((n, n, n, n))

    result = load_or_compute_gamma(None, h1, eri, 1, 1, n)

    assert result.gamma_source_mode == "computed_aufbau"
    assert result.gamma_load_mode == "n/a"
    assert result.gamma.shape == (n, n)
    assert abs(np.trace(result.gamma) - 2.0) < 1e-10


def test_ref_det_from_dets_npz(tmp_path):
    path = str(tmp_path / "dets.npz")
    dets = np.array([[0b0011, 0b0001], [0b1100, 0b1000]], dtype=np.uint64)
    np.savez(path, dets=dets)
    n = 4
    h1 = np.diag(np.arange(n, dtype=float))
    eri = np.zeros((n, n, n, n))

    alpha, beta, mode = load_or_derive_ref_det(path, h1, eri, 2, 1, n)

    assert alpha == 0b0011
    assert beta == 0b0001
    assert mode == "dets_npz"


def test_ref_det_aufbau_fallback(tmp_path):
    n = 4
    h1 = np.diag([-0.5, -0.3, 0.1, 0.4])
    eri = np.zeros((n, n, n, n))

    alpha, beta, mode = load_or_derive_ref_det(None, h1, eri, 2, 1, n)

    assert mode == "computed_aufbau"
    assert bin(alpha).count("1") == 2
    assert bin(beta).count("1") == 1


def test_aufbau_electron_count_correct(tmp_path):
    n = 6
    h1 = np.diag([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    eri = np.zeros((n, n, n, n))

    alpha, beta, mode = load_or_derive_ref_det(None, h1, eri, 2, 1, n)

    assert mode == "computed_aufbau"
    assert bin(alpha).count("1") == 2
    assert bin(beta).count("1") == 1
    assert (alpha & 1) == 1
    assert (alpha >> 1) & 1
    assert (beta & 1) == 1
