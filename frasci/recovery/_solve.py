"""Generalized {Phi_LAS, D_1...D_N} eigenproblem: H/S assembly, canonical
orthogonalization, and the five variational gates. See
docs/prompts/frasci_library_PROMPT.md sec 2.5 Steps 4-6 and sec 4.1 for the invariants
these gates enforce -- they are FATAL on violation (raise), never warnings, by spec.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from frasci.recovery._state import selected_hamiltonian, h0d


class _MatrixFreeDeterminantHamiltonian:
    """Exact selected-determinant Hamiltonian action without a dense matrix."""

    def __init__(self, state: dict, dets):
        from trimci import trimci_core

        normalized = [tuple(map(int, det)) for det in dets]
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("selected determinant space must be nonempty and deduplicated")
        sectors = {(a.bit_count(), b.bit_count()) for a, b in normalized}
        if len(sectors) != 1:
            raise ValueError(f"determinants span multiple electron sectors: {sectors}")
        if int(state["n_orb"]) > 64:
            raise NotImplementedError("matrix-free selected CI currently supports n_orb <= 64")
        self._alpha = np.ascontiguousarray([a for a, _ in normalized], dtype=np.uint64)
        self._beta = np.ascontiguousarray([b for _, b in normalized], dtype=np.uint64)
        self._h1 = np.ascontiguousarray(state["h1"], dtype=float)
        self._eri = np.ascontiguousarray(state["eri"], dtype=float)
        self._n_orb = int(state["n_orb"])
        self._e_const = float(state.get("e_const", 0.0))
        self._backend = trimci_core.fast_expansion
        self._index = self._backend.ABIndex()
        self._index.build(self._alpha, self._beta, self._n_orb)
        self._electronic_diagonal = np.asarray(
            self._backend.compute_diagonals(
                self._alpha, self._beta, self._h1, self._eri, self._n_orb), dtype=float)
        self.diagonal = self._electronic_diagonal + self._e_const

    @property
    def size(self) -> int:
        return len(self._alpha)

    def matvec(self, vector) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=float)
        electronic = self._backend.matvec_row_sliced(
            self._index, self._alpha, self._beta, vector,
            self._electronic_diagonal, self._h1, self._eri, self._n_orb,
            0, self.size)
        return np.asarray(electronic, dtype=float) + self._e_const * vector


def projected_coupling(h0d_value, overlap, h00):
    """Return ``<Phi|(H-H00)|D>`` for a normalized contracted reference.

    This is the coupling of the part of ``D`` orthogonal to ``Phi``.  Raw
    ``H0D`` includes the parallel ``H00*S0D`` contribution and is therefore not
    a bridge-importance measure when the determinant overlaps the LAS state.
    Scalars and NumPy arrays are both accepted.
    """
    result = np.asarray(h0d_value) - float(h00) * np.asarray(overlap)
    return float(result) if result.ndim == 0 else result


def structural_connectivity(hdd: np.ndarray, projected: np.ndarray,
                            target_indices, *, edge_threshold: float = 1e-10) -> dict:
    """Thresholded reachability from the contracted LAS node to target dets.

    ``target_indices`` indexes the determinant block (zero-based, excluding the
    LAS node).  LAS--determinant edges use projected couplings; determinant
    edges use off-diagonal Slater--Condon matrix elements.
    """
    hdd = np.asarray(hdd, dtype=float)
    projected = np.asarray(projected, dtype=float)
    n_det = len(projected)
    if hdd.shape != (n_det, n_det):
        raise ValueError("hdd/projected dimensions disagree")
    targets = tuple(sorted({int(i) for i in target_indices}))
    if any(i < 0 or i >= n_det for i in targets):
        raise IndexError("target determinant index is out of range")
    if edge_threshold < 0:
        raise ValueError("edge_threshold must be nonnegative")

    # Node 0 is Phi_LAS; determinant i is node i+1.
    adjacency = [set() for _ in range(n_det + 1)]
    for i, value in enumerate(projected):
        if abs(float(value)) > edge_threshold:
            adjacency[0].add(i + 1)
            adjacency[i + 1].add(0)
    for i in range(n_det):
        for j in range(i + 1, n_det):
            if abs(float(hdd[i, j])) > edge_threshold:
                adjacency[i + 1].add(j + 1)
                adjacency[j + 1].add(i + 1)

    reachable = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                stack.append(neighbor)
    reachable_targets = tuple(i for i in targets if i + 1 in reachable)
    return {
        "structurally_connected": bool(targets) and len(reachable_targets) == len(targets),
        "n_targets": len(targets),
        "n_targets_reachable": len(reachable_targets),
        "reachable_target_indices": reachable_targets,
        "n_determinants_reachable": len(reachable) - 1,
        "las_edge_count": len(adjacency[0]),
        "edge_threshold_ha": float(edge_threshold),
    }


def generalized_populations(s: np.ndarray, coefficients: np.ndarray,
                            target_indices=()) -> dict:
    """Projection and symmetric-Lowdin populations for a generalized root.

    The direct LAS projection is ``|<Phi|Psi>|^2 = |(S c)_0|^2``.  It is not
    ``c[0]^2`` when explicit determinants overlap the contracted reference.
    Symmetric-Lowdin populations provide a normalized partition across the
    nonorthogonal basis and are used for the target-space diagnostic.
    """
    s = np.asarray(s, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    sw, su = eigh(0.5 * (s + s.T), check_finite=False)
    if sw[0] < -1e-8:
        raise AssertionError(f"overlap has negative eigenvalue {sw[0]:.3e}")
    sqrt_s = (su * np.sqrt(np.clip(sw, 0.0, None))[None, :]) @ su.T
    amplitudes = sqrt_s @ c
    populations = np.square(np.abs(amplitudes))
    total = float(populations.sum())
    if total <= 0.0:
        raise AssertionError("generalized eigenvector has zero Lowdin population")
    populations /= total
    target_nodes = [int(i) + 1 for i in target_indices]
    return {
        "las_projection_weight": float(abs((s @ c)[0]) ** 2),
        "lowdin_las_weight": float(populations[0]),
        "lowdin_target_weight": float(populations[target_nodes].sum()) if target_nodes else 0.0,
        "lowdin_explicit_weight": float(populations[1:].sum()),
        "lowdin_populations": populations,
    }


def connectivity_status(connectivity: dict, delta_mix: float, *,
                        meaningful_delta_ha: float = 1e-6,
                        mixing_noise_ha: float = 1e-10) -> str:
    """Classify a valid generalized solve without conflating edges and energy."""
    if meaningful_delta_ha < 0.0 or mixing_noise_ha < 0.0:
        raise ValueError("mixing thresholds must be nonnegative")
    if not connectivity["structurally_connected"]:
        return "disconnected"
    if delta_mix <= max(float(mixing_noise_ha), float(meaningful_delta_ha)):
        return "connected_but_negligible"
    return "connected_and_mixed"


def canonical_orthogonalization_solve(h: np.ndarray, s: np.ndarray,
                                      *, threshold: float = 1e-10) -> dict:
    if not np.allclose(h, h.T, atol=1e-8) or not np.allclose(s, s.T, atol=1e-10):
        raise AssertionError("hybrid H/S must be symmetric")
    sw, su = eigh(s, check_finite=False)
    if sw[0] < -1e-8:
        raise AssertionError(f"overlap has negative eigenvalue {sw[0]:.3e}")
    keep = sw > threshold
    x = su[:, keep] / np.sqrt(sw[keep])[None, :]
    ho = x.T @ h @ x
    ew, ev = eigh(0.5 * (ho + ho.T), subset_by_index=[0, 0], check_finite=False)
    c = x @ ev[:, 0]
    norm = float(c @ s @ c)
    if abs(norm - 1.0) > 1e-6:
        raise AssertionError(f"generalized eigenvector norm is {norm}")
    return dict(e_hybrid=float(ew[0]), coefficients=c, s_min=float(sw[0]),
                s_max=float(sw[-1]), n_removed=int((~keep).sum()), c_dagger_s_c=norm,
                s_min_eigenvalue=float(sw[0]), s_max_eigenvalue=float(sw[-1]),
                n_overlap_directions_removed=int((~keep).sum()), c_dagger_S_c=norm)


def build_overlap_and_hamiltonian(state: dict, dets, s_vec: np.ndarray, h0d_vec: np.ndarray):
    """Build H and S in the nonorthogonal {Phi_LAS,D_1,...,D_N} basis."""
    hdd = selected_hamiltonian(state["h1"], state["eri"], dets, n_orb=state["n_orb"])
    hdd += np.eye(len(dets)) * state.get("e_const", 0.0)
    h = np.zeros((len(dets) + 1, len(dets) + 1))
    h[0, 0] = state["h00"]
    h[0, 1:] = h0d_vec; h[1:, 0] = h0d_vec; h[1:, 1:] = hdd
    s = np.eye(len(dets) + 1)
    s[0, 1:] = s_vec; s[1:, 0] = s_vec
    return h, s, hdd


def solve_contracted_hybrid(state: dict, eligible: dict, *, phase_energy: float | None = None,
                            overlap_threshold: float = 1e-10,
                            meaningful_delta_ha: float = 1e-6,
                            connectivity_threshold_ha: float = 1e-10,
                            mixing_noise_ha: float = 1e-10,
                            max_memory_gb: float = 24.0,
                            coupling_vector=None,
                            on_progress=None) -> dict:
    """Solve the exact generalized {Phi_LAS,D} variational problem and gate it.

    ``on_progress(phase, done, total, **fields)``, if given, is called through
    the h0d loop -- the one genuinely slow step here with an exactly knowable
    length -- and once before the dense eigensolve, which is a single LAPACK
    call that cannot self-report and so is announced by size instead.

    ``coupling_vector`` mirrors :func:`solve_contracted_hybrid_matrix_free`'s own
    parameter of the same name: pass the already-computed ``<Phi_LAS|H|D>`` values
    (e.g. from a vectorized ``h0d_batch`` call a caller already ran) to skip the
    per-determinant ``h0d`` loop below entirely. ``None`` (default) preserves the
    original behavior exactly.

    ``meaningful_delta_ha`` is the threshold, in Ha, above which the mixing gain
    (see ``status`` below) counts as chemically real rather than numerical noise.
    Default is 1 microHartree -- clearly above float64 noise on an O(1) Ha
    Hamiltonian, clearly below anything that would matter chemically, deliberately
    conservative so it flags weak-but-real effects rather than only large ones.
    Override per system/context if that default doesn't fit.
    """
    dets = list(eligible)
    if not dets:
        raise ValueError("contracted hybrid candidate set is empty")
    estimated_peak_gb = 4 * 8 * (len(dets) + 1) ** 2 / 1e9
    if estimated_peak_gb > max_memory_gb:
        raise MemoryError(
            f"contracted hybrid solve needs about {estimated_peak_gb:.2f} GB for H/S work arrays; "
            f"configured limit is {max_memory_gb:.2f} GB")
    s_vec = np.asarray([eligible[d]["s"] for d in dets])
    if coupling_vector is not None:
        h0d_vec = np.asarray(coupling_vector, dtype=float)
        if h0d_vec.shape != (len(dets),):
            raise ValueError("coupling_vector shape does not match eligible determinant space")
    elif on_progress is None:
        h0d_vec = np.asarray([h0d(d, state) for d in dets])
    else:
        h0d_vec = np.empty(len(dets))
        for i, d in enumerate(dets):
            h0d_vec[i] = h0d(d, state)
            on_progress("h0d", i + 1, len(dets))
        on_progress("eigh", 0, 0, dim=len(dets) + 1,
                    mem=f"{estimated_peak_gb:.2f}GB")
    h, s, hdd = build_overlap_and_hamiltonian(state, dets, s_vec, h0d_vec)
    solved = canonical_orthogonalization_solve(h, s, threshold=overlap_threshold)
    e_c = float(eigh(hdd, subset_by_index=[0, 0], check_finite=False)[0][0])
    gates = dict(
        E_hybrid_le_H00=bool(solved["e_hybrid"] <= state["h00"] + 1e-6),
        E_hybrid_le_E_C=bool(solved["e_hybrid"] <= e_c + 1e-6),
        S_positive_semidefinite=bool(solved["s_min"] >= -1e-8),
        c_dagger_S_c_eq_1=bool(abs(solved["c_dagger_s_c"] - 1.0) < 1e-6),
        no_pt2_mixed_with_variational=True,
    )
    failed = [k for k, v in gates.items() if v is not True]
    if failed:
        raise AssertionError(f"contracted-hybrid gates failed: {failed}")
    # Structural reachability and energetic mixing answer different questions.  Both
    # are diagnostics, not replacements for the hard numerical gates above.
    delta_mix = float(min(state["h00"], e_c) - solved["e_hybrid"])
    projected = projected_coupling(h0d_vec, s_vec, state["h00"])
    strict_labels = {"dispersion", "ct1", "ct2", "ct_gt2"}
    target_indices = [i for i, d in enumerate(dets)
                      if eligible[d].get("is_strict_target",
                                          eligible[d].get("label") in strict_labels)]
    # A generic caller may provide an unlabeled determinant set.  In that case the
    # explicit set itself is the requested endpoint; production recovery candidates
    # are labeled and therefore take the strict-only branch above.
    if not target_indices and not any("label" in eligible[d] for d in dets):
        target_indices = list(range(len(dets)))
    connectivity = structural_connectivity(
        hdd, projected, target_indices, edge_threshold=connectivity_threshold_ha)
    populations = generalized_populations(s, solved["coefficients"], target_indices)
    status = connectivity_status(
        connectivity, delta_mix, meaningful_delta_ha=meaningful_delta_ha,
        mixing_noise_ha=mixing_noise_ha)
    connected = status == "connected_and_mixed"
    return dict(e_tot=solved["e_hybrid"], e_c=e_c, h00=state["h00"],
                n_contracted_basis=1, n_explicit_dets=len(dets),
                recovery_mha=(state["h00"] - solved["e_hybrid"]) * 1000,
                phase_energy=phase_energy, overlap=solved, gates=gates,
                estimated_peak_memory_gb=estimated_peak_gb,
                h0d_max_abs=float(np.max(np.abs(h0d_vec))),
                projected_coupling_max_abs=float(np.max(np.abs(projected))),
                las_weight=populations["las_projection_weight"],
                lowdin_las_weight=populations["lowdin_las_weight"],
                strict_target_weight=populations["lowdin_target_weight"],
                connectivity=connectivity, delta_mix=delta_mix,
                status=status, connected=connected)


def _overlap_action(vector: np.ndarray, overlap_vector: np.ndarray) -> np.ndarray:
    out = np.asarray(vector, dtype=float).copy()
    out[0] += float(overlap_vector @ vector[1:])
    out[1:] += overlap_vector * vector[0]
    return out


def _canonical_overlap_action(vector: np.ndarray, overlap_vector: np.ndarray, *,
                              threshold: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm_s = float(np.linalg.norm(overlap_vector))
    if norm_s == 0.0:
        return vector.copy()
    unit_s = overlap_vector / norm_s
    parallel = float(unit_s @ vector[1:])
    perpendicular = vector[1:] - unit_s * parallel
    plus = (vector[0] + parallel) / np.sqrt(2.0 * (1.0 + norm_s))
    minus = ((vector[0] - parallel) / np.sqrt(2.0 * (1.0 - norm_s))
             if 1.0 - norm_s > threshold else 0.0)
    out = np.empty_like(vector)
    out[0] = (plus + minus) / np.sqrt(2.0)
    out[1:] = perpendicular + unit_s * (plus - minus) / np.sqrt(2.0)
    return out


def _smallest_eigenpair(operator: LinearOperator, *, tol: float, max_iter: int,
                        max_subspace: int, initial_guess=None):
    dimension = operator.shape[0]
    if dimension <= 3:
        basis = np.eye(dimension)
        matrix = np.column_stack([operator @ basis[:, i] for i in range(dimension)])
        values, vectors = eigh(0.5 * (matrix + matrix.T), subset_by_index=[0, 0],
                               check_finite=False)
        return float(values[0]), vectors[:, 0]
    initial_guess = np.ones(dimension) if initial_guess is None else np.asarray(
        initial_guess, dtype=float)
    initial_norm = float(np.linalg.norm(initial_guess))
    if initial_norm == 0.0:
        initial_guess = np.ones(dimension)
        initial_norm = np.sqrt(dimension)
    ncv = min(dimension, max(3, int(max_subspace)))
    values, vectors = eigsh(
        operator, k=1, which="SA", tol=tol, maxiter=max_iter, ncv=ncv,
        v0=initial_guess / initial_norm)
    return float(values[0]), vectors[:, 0]


def solve_contracted_hybrid_matrix_free(
        state: dict, eligible: dict, *, phase_energy: float | None = None,
        overlap_threshold: float = 1e-10, max_memory_gb: float = 24.0,
        tol: float = 1e-9, max_iter: int = 1000, max_subspace: int = 40,
        coupling_vector=None, initial_coefficients=None, on_progress=None) -> dict:
    """Linear-memory exact ``{Phi_LAS,D}`` variational solve.

    ``coupling_vector`` may be supplied by a sweep runner so nested caps reuse the
    already-computed ``<Phi_LAS|H|D>`` values.  Only the linear algebra differs from
    :func:`solve_contracted_hybrid`; the determinant space and variational ansatz are
    identical.

    ``initial_coefficients``, if given, is a ``[c0, d_1, ..., d_R]`` vector (shape
    ``(len(dets)+1,)``) in the ORIGINAL non-orthogonal basis -- e.g. a previous rotation
    step's converged ``coefficients`` -- used as the Davidson warm start instead of the
    default ``[1, 0, ..., 0]``. ``None`` preserves the original cold-start behavior
    exactly.
    """
    dets = list(eligible)
    if not dets:
        raise ValueError("contracted hybrid candidate set is empty")
    if max_iter < 1 or max_subspace < 3:
        raise ValueError("matrix-free max_iter must be >=1 and max_subspace >=3")

    overlap_vector = np.asarray([eligible[det]["s"] for det in dets], dtype=float)
    overlap_norm = float(np.linalg.norm(overlap_vector))
    s_min, s_max = 1.0 - overlap_norm, 1.0 + overlap_norm
    if s_min < -1e-8:
        raise AssertionError(f"overlap has negative eigenvalue {s_min:.3e}")
    n_removed = int(s_min <= overlap_threshold)
    if coupling_vector is None:
        coupling_vector = np.asarray([h0d(det, state) for det in dets])
    else:
        coupling_vector = np.asarray(coupling_vector, dtype=float)
        if coupling_vector.shape != (len(dets),):
            raise ValueError("coupling_vector shape does not match eligible determinant space")

    dimension = len(dets) + 1
    krylov_vectors = min(dimension, max_subspace) + 12
    estimated_peak_gb = (8 * krylov_vectors * dimension + 8 * 12 * len(dets)) / 1e9
    if estimated_peak_gb > max_memory_gb:
        raise MemoryError(
            f"matrix-free hybrid vector workspace needs about {estimated_peak_gb:.2f} GB; "
            f"configured limit is {max_memory_gb:.2f} GB")

    determinant_hamiltonian = _MatrixFreeDeterminantHamiltonian(state, dets)
    hybrid_calls = 0

    def hybrid_h_action(vector):
        nonlocal hybrid_calls
        hybrid_calls += 1
        vector = np.asarray(vector, dtype=float)
        out = np.empty_like(vector)
        out[0] = float(state["h00"]) * vector[0] + float(coupling_vector @ vector[1:])
        out[1:] = coupling_vector * vector[0] + determinant_hamiltonian.matvec(vector[1:])
        return out

    def orthogonalized_action(vector):
        contracted = _canonical_overlap_action(
            vector, overlap_vector, threshold=overlap_threshold)
        out = _canonical_overlap_action(
            hybrid_h_action(contracted), overlap_vector, threshold=overlap_threshold)
        if n_removed:
            unit_s = overlap_vector / overlap_norm
            null_projection = (vector[0] - float(unit_s @ vector[1:])) / np.sqrt(2.0)
            spectral_shift = (
                abs(float(state["h00"]))
                + float(np.max(np.abs(determinant_hamiltonian.diagonal)))
                + 2.0 * float(np.linalg.norm(coupling_vector)) + 1.0) * dimension
            out[0] += spectral_shift * null_projection / np.sqrt(2.0)
            out[1:] -= spectral_shift * null_projection * unit_s / np.sqrt(2.0)
        return out

    operator = LinearOperator((dimension, dimension), matvec=orthogonalized_action,
                              dtype=float)
    if initial_coefficients is None:
        initial_vector = np.r_[1.0, np.zeros(len(dets))]
    else:
        initial_vector = np.asarray(initial_coefficients, dtype=float)
        if initial_vector.shape != (dimension,):
            raise ValueError(
                f"initial_coefficients shape {initial_vector.shape} does not match "
                f"dimension ({dimension},)")
    initial = _canonical_overlap_action(
        initial_vector, overlap_vector, threshold=overlap_threshold)
    _, orthogonal_coefficients = _smallest_eigenpair(
        operator, tol=tol, max_iter=max_iter, max_subspace=max_subspace,
        initial_guess=initial)
    coefficients = _canonical_overlap_action(
        orthogonal_coefficients, overlap_vector, threshold=overlap_threshold)
    overlap_applied = _overlap_action(coefficients, overlap_vector)
    norm = float(coefficients @ overlap_applied)
    coefficients /= np.sqrt(norm)
    overlap_applied = _overlap_action(coefficients, overlap_vector)
    norm = float(coefficients @ overlap_applied)
    h_applied = hybrid_h_action(coefficients)
    hybrid_energy = float(coefficients @ h_applied / norm)
    residual_norm = float(np.linalg.norm(h_applied - hybrid_energy * overlap_applied))

    determinant_operator = LinearOperator(
        (len(dets), len(dets)), matvec=determinant_hamiltonian.matvec, dtype=float)
    determinant_initial = np.zeros(len(dets))
    determinant_initial[int(np.argmin(determinant_hamiltonian.diagonal))] = 1.0
    determinant_only_energy, _ = _smallest_eigenpair(
        determinant_operator, tol=tol, max_iter=max_iter,
        max_subspace=max_subspace, initial_guess=determinant_initial)

    overlap = dict(s_min=float(s_min), s_max=float(s_max), n_removed=n_removed,
                   c_dagger_s_c=norm, s_min_eigenvalue=float(s_min),
                   s_max_eigenvalue=float(s_max), n_overlap_directions_removed=n_removed,
                   c_dagger_S_c=norm)
    gates = dict(
        E_hybrid_le_H00=bool(hybrid_energy <= state["h00"] + 1e-6),
        E_hybrid_le_E_C=bool(hybrid_energy <= determinant_only_energy + 1e-6),
        S_positive_semidefinite=bool(s_min >= -1e-8),
        c_dagger_S_c_eq_1=bool(abs(norm - 1.0) < 1e-6),
        generalized_residual_converged=bool(residual_norm < max(1e-7, 20 * tol)),
        no_pt2_mixed_with_variational=True)
    failed = [name for name, passed in gates.items() if passed is not True]
    if failed:
        raise AssertionError(
            f"matrix-free contracted-hybrid gates failed: {failed}; residual={residual_norm:.3e}")
    return dict(
        e_tot=hybrid_energy, e_c=float(determinant_only_energy), h00=float(state["h00"]),
        n_contracted_basis=1, n_explicit_dets=len(dets), explicit_determinants=dets,
        coefficients=np.asarray(coefficients, dtype=float),
        recovery_mha=(float(state["h00"]) - hybrid_energy) * 1000,
        phase_energy=phase_energy, overlap=overlap, gates=gates,
        estimated_peak_memory_gb=estimated_peak_gb,
        h0d_max_abs=float(np.max(np.abs(coupling_vector))),
        generalized_residual_norm=residual_norm, solver_kind="matrix_free",
        hybrid_matvec_calls=hybrid_calls)


def solve_recovery_point(state: dict, dets, coeffs, *,
                         n_ct_max: int = 2, edge_kernel=None, lookup_cache=None,
                         solver_kind: str = "matrix_free",
                         dense_dimension_threshold: int = 5000,
                         max_memory_gb: float = 16.0,
                         initial_coefficients=None,
                         coupling_backend: str = "auto") -> dict:
    """Classify a fixed (dets, coeffs) support against ``state`` and solve the resulting
    ``{Phi_LAS, R}`` generalized eigenproblem.

    One-call version of the loop body ``fixed_frame_sweep.run_system`` already repeats per
    cap: ``retain_candidates`` -> ``h0d_batch`` -> a hybrid solver. If ``dets``/``coeffs``
    classify to an empty eligible set (no interfragment determinants), returns the parent
    H00 itself as the energy (same "no interfragment determinants exist" convention
    ``fixed_frame_sweep.py`` already uses), never raises.

    ``initial_coefficients``, if given, is a ``[c0, d_1, ..., d_R]`` warm-start vector
    (shape ``(len(eligible)+1,)``, in the order ``eligible`` iterates) -- e.g. a previous
    recovery rung's converged generalized coefficients, zero-padded for any newly admitted
    determinant. Forwarded to :func:`solve_contracted_hybrid_matrix_free`'s own
    ``initial_coefficients`` parameter; has no effect on the ``"dense"`` path, which always
    solves via a direct ``eigh`` call that needs no starting vector.

    ``solver_kind``:
    - ``"matrix_free"`` (default -- unchanged production behavior): the linear-memory
      iterative (Lanczos-type) solve. Robust to memory at any size, but its convergence
      rate depends on the spectral gap near the ground state -- pathologically slow on
      systems with dense, near-degenerate low-lying spectra (observed: an Fe3 cluster's
      spin ladder can leave it running far past what a dense solve needs).
    - ``"dense"``: the exact ``scipy.linalg.eigh`` solve (:func:`solve_contracted_hybrid`).
      Immune to spectral-gap slowness, but O(N^2) memory -- only safe up to a few thousand
      explicit determinants.
    - ``"auto"``: dense when ``len(eligible) <= dense_dimension_threshold``, else
      matrix_free. Both backends return the identical schema (``gates``,
      ``generalized_residual_norm``, ``solver_kind``) so callers never branch on which ran;
      dense's exact solve trivially satisfies the residual-convergence gate
      (``generalized_residual_norm=0.0``, ``generalized_residual_converged=True``), added
      to its gates dict for parity with the matrix-free one.

    ``coupling_backend`` picks how the ``<Phi_LAS|H|D>`` coupling vector is computed --
    purely an implementation/resource choice, never a change to candidate selection,
    Parent contraction, A-prime ordering, or the generalized solve itself:
    - ``"auto"`` (default): :func:`frasci.recovery._state.parent_support_preflight`
      decides, per call, whether the complete factorized Parent's own Cartesian-product
      support is small enough to use the exact row-sliced-matvec backend; falls back to
      ``h0d_batch`` otherwise.
    - ``"row_sliced_matvec"``: force :func:`frasci.recovery._state.h0d_batch_via_parent_matvec`
      regardless of what the preflight would have recommended -- explicit caller opt-in
      only (e.g. for the small-system validation tests below).
    - ``"h0d_batch"``: force the original per-candidate neighbor-enumeration path.
    """
    if solver_kind not in ("matrix_free", "dense", "auto"):
        raise ValueError(
            f"solver_kind must be 'matrix_free', 'dense', or 'auto', got {solver_kind!r}")
    if coupling_backend not in ("auto", "row_sliced_matvec", "h0d_batch"):
        raise ValueError(
            "coupling_backend must be 'auto', 'row_sliced_matvec', or 'h0d_batch', "
            f"got {coupling_backend!r}")
    from frasci.recovery._slater_condon import RealSlaterCondon
    from frasci.recovery._state import (h0d_batch, h0d_batch_via_parent_matvec,
                                        parent_support_preflight,
                                        prepare_owned_coefficient_lookup, retain_candidates)

    eligible, classification = retain_candidates(
        dets, coeffs, state, n_ct_max=n_ct_max, include_same_sector=False)
    if not eligible:
        return dict(e_tot=float(state["h00"]), n_recovery=0, eligible=eligible,
                   classification=classification, gates={"E_hybrid_le_H00": True},
                   generalized_residual_norm=0.0, solver_kind="none",
                   energy_note="no interfragment determinants exist/found; parent retained",
                   explicit_determinants=(), coefficients=np.asarray([1.0]),
                   coupling_backend="none", coupling_preflight={}, coupling_timings={})

    eligible_dets = list(eligible)
    preflight = parent_support_preflight(state, len(eligible_dets))
    use_matvec = (coupling_backend == "row_sliced_matvec"
                 or (coupling_backend == "auto" and preflight["use_row_sliced_matvec"]))
    if use_matvec:
        coupling, coupling_timings = h0d_batch_via_parent_matvec(eligible_dets, state)
        coupling_backend_used = "row_sliced_matvec"
    else:
        if edge_kernel is None:
            edge_kernel = RealSlaterCondon(state["h1"], state["eri"], e_const=state["e_const"])
        if lookup_cache is None:
            lookup_cache = prepare_owned_coefficient_lookup(state)
        coupling = h0d_batch(eligible_dets, state, edge_kernel, lookup_cache=lookup_cache)
        coupling_timings = {}
        coupling_backend_used = "h0d_batch"

    effective_kind = solver_kind
    if effective_kind == "auto":
        effective_kind = ("dense" if len(eligible_dets) <= dense_dimension_threshold
                          else "matrix_free")

    if effective_kind == "dense":
        solved = solve_contracted_hybrid(state, eligible, coupling_vector=coupling,
                                         max_memory_gb=max_memory_gb)
        gates = dict(solved["gates"])
        gates["generalized_residual_converged"] = True
        residual = 0.0
        generalized_coefficients = solved["overlap"]["coefficients"]
    else:
        solved = solve_contracted_hybrid_matrix_free(
            state, eligible, coupling_vector=coupling, max_memory_gb=max_memory_gb,
            tol=1e-11, max_iter=5000, max_subspace=200,
            initial_coefficients=initial_coefficients)
        gates = solved["gates"]
        residual = float(solved["generalized_residual_norm"])
        generalized_coefficients = solved["coefficients"]

    return dict(e_tot=float(solved["e_tot"]), n_recovery=len(eligible_dets),
               eligible=eligible, classification=classification,
               gates=gates, generalized_residual_norm=residual, solver_kind=effective_kind,
               energy_note=f"{effective_kind} contracted {{Phi_LAS,R}} variational solve",
               explicit_determinants=eligible_dets,
               coefficients=np.asarray(generalized_coefficients, dtype=float),
               coupling_backend=coupling_backend_used,
               coupling_preflight=preflight,
               coupling_timings=coupling_timings)
