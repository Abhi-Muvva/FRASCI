"""Typed configuration and result objects for fixed-frame FRASCI recovery."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryConfig:
    """Fixed-parent-frame native-filtered inter-fragment recovery.

    The validated production protocol has one determinant-growth stage: start from
    ``las_seed_dets`` protected LAS product terms, grow exactly ``recovery_target``
    dispersion/allowed-CT determinants without changing orbitals, then solve the
    variational ``{Phi_LAS, R}`` problem.  There is no doorway mode, recovery COO,
    Phase-0 restart bank, or fragment rebuild.
    """
    recovery_target: int
    threshold: float = 1e-6
    n_ct_max: int = 2
    max_memory_gb: float = 16.0
    las_seed_dets: int = 100
    growth_factor: float = 1.5
    max_rounds: int = 40
    solver_kind: str = "auto"  # "auto" | "dense" | "matrix_free"
    dense_dimension_threshold: int = 5000
    las_reconstruction_tol: float = 1e-8
    union_energy_tol: float = 1e-6
    las_top_k: int = 1000
    free_coo_orbital_opt_max_iter: int = 20
    allow_nonconverged_parent: bool = False
    """``solve()`` refuses a parent recorded as ``converged=False`` by default (before any
    growth, generalized solve, union solve, or free-COO begins) -- Fe2S2 and Fe4S4 pilot
    runs on an unconverged parent produced multi-Ha "recovery" energies that were artifacts
    of the non-convergence, not genuine physics. Set this to ``True`` only for an explicitly
    labeled diagnostic; every ``Result`` it produces carries ``diagnostic_only=True`` and
    ``parent_converged=False`` so the caller cannot mistake it for a validated result.
    Release notebooks must never enable this."""

    def __post_init__(self):
        if self.recovery_target < 1:
            raise ValueError(
                f"recovery_target must be >= 1, got {self.recovery_target}")
        if self.las_seed_dets < 1:
            raise ValueError(f"las_seed_dets must be >= 1, got {self.las_seed_dets}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1, got {self.growth_factor}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if self.solver_kind not in ("auto", "dense", "matrix_free"):
            raise ValueError(
                "solver_kind must be 'auto', 'dense', or 'matrix_free', "
                f"got {self.solver_kind!r}")
        if self.dense_dimension_threshold < 1:
            raise ValueError(
                "dense_dimension_threshold must be >= 1, got "
                f"{self.dense_dimension_threshold}")
        if self.las_top_k < 1:
            raise ValueError(
                f"las_top_k must be >= 1, got {self.las_top_k}")


@dataclass(frozen=True)
class Result:
    """What frasci.recovery.solve() returns.

    ``energy`` is the rigorous variational ``{Phi_LAS, R}`` total in the fixed
    parent frame. ``union_*`` holds the exact ``P_LAS union R`` solve and
    ``free_coo_*`` the independently rediagonalized top-K free-orbital result.
    """
    hybrid_produced: bool
    energy: float
    n_dets: int
    converged: bool
    gates: dict
    h00: float
    wall_s: float
    diagnostics: dict
    union_produced: bool = False
    union_energy: float | None = None
    union_n_dets: int | None = None
    union_dets: list | None = None
    union_coefficients: object = None
    union_residual_norm: float | None = None
    free_coo_produced: bool = False
    free_coo_energy: float | None = None
    free_coo_n_dets: int | None = None
    free_coo_gain_mha: float | None = None
    parent_converged: bool = True
    """Whether the parent this recovery solve was built on was itself converged. Always
    populated -- ``True`` for the ordinary path. ``False`` only appears when the caller
    explicitly passed ``RecoveryConfig(allow_nonconverged_parent=True)``, in which case
    ``diagnostic_only`` is also ``True``."""
    diagnostic_only: bool = False
    """``True`` only when this result was produced from a non-converged parent via the
    explicit ``allow_nonconverged_parent=True`` opt-out -- a conspicuous marker that these
    energies are a convergence diagnostic, not a validated physical result."""
