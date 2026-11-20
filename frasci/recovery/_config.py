"""Typed configuration and result objects for fixed-frame FRASCI recovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecoveryConfig:
    """Fixed-parent-frame native-filtered inter-fragment recovery.

    The validated public protocol is: complete contracted Parent -> corrected A-prime
    recovery at ONE user-selected ``recovery_target`` -> an optional post-recovery COO
    refinement. Start from ``las_seed_dets`` protected LAS product terms, grow exactly
    ``recovery_target`` dispersion/allowed-CT determinants without changing orbitals,
    then solve the variational ``{Phi_LAS, R}`` problem. ``solve()`` performs this ONE
    direct calculation and never makes an automatic stopping decision -- a sequence of
    targets, a plateau rule, and campaign checkpointing are the separate
    ``frasci.recovery.benchmark_ladder`` driver's job (used for the project's own
    all-molecule recovery-limit study), not part of this configuration or this
    function. There is no doorway mode, Phase-0 restart bank, fragment rebuild,
    multiparent/retained-guide machinery, or exposed selector/boundary-shell choice.
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
    checkpoint_directory: Path | None = None
    """When set, ``solve()`` atomically persists the corrected-A' growth reservoir for
    THIS ``recovery_target`` and reuses it only when its recorded Parent/frame/A-prime
    fingerprint matches the current call AND its capacity already covers
    ``recovery_target`` -- instead of regrowing from the protected seed. An
    incompatible or undersized checkpoint is never silently loaded: it is rebuilt (and
    overwritten), never silently reused. This is a production convenience for
    resuming/re-running one target cheaply -- it is NOT the separate benchmark
    ladder's per-rung campaign checkpointing (``frasci.recovery.benchmark_ladder``
    uses distinctly-named files and tracks a whole target sequence); the two are
    never confused with each other even when pointed at the same directory."""
    run_post_recovery_coo: bool = False
    """Run the top-K free-orbital COO refinement (``Result.free_coo_*``) exactly once,
    after the requested ``recovery_target`` recovery, on that support -- reporting the
    matched-support energy immediately before (``Result.energy``) and after
    (``Result.free_coo_energy``, with ``Result.free_coo_gain_mha`` the difference) the
    orbital rotation. Defaults to ``False``: the default public protocol is complete
    contracted Parent + corrected A-prime recovery only.

    Backward-compatibility note -- this is an intentional behavior change, not a drop-in
    migration: earlier releases of this function always ran the top-K free-orbital COO
    step (and the exact explicit union) on every call. The exact union is UNCHANGED and
    still always runs; COO specifically is now opt-in. Any caller that relied on
    ``Result.free_coo_*`` being populated unconditionally must now pass
    ``run_post_recovery_coo=True`` explicitly to keep getting it."""
    expand_parent_k: int | None = None
    """When set, run the Expanded-Parent diagnostic once: exactly diagonalize the top-``k``
    explicit LAS product terms (``frasci.recovery._expand_parent.compute_expanded_parent``)
    as independent, freely-variable determinants -- no recovery determinants, no orbital
    rotation. Reports how much energy is available just from relaxing the Parent's own
    rigid tensor-product contraction, independent of and reported alongside the recovery
    result (``Result.expand_parent_*``). Defaults to ``None`` (disabled): the default
    public protocol is complete contracted Parent + corrected A-prime recovery only, same
    as ``run_post_recovery_coo``.

    This is a Parent-only diagnostic, not a recovery step: it never feeds into growth, the
    generalized solve, or either union, and its own result is not fed back into them
    either. For 4 of the 6 campaign systems this probe plateaus near-trivially; for the
    remaining systems it can be a real, if expensive, effect -- see the released
    ``09_expanding_parent_ladder`` notebook for the per-system ladder this promotes."""
    allow_nonconverged_parent: bool = False
    """``solve()`` refuses a parent recorded as ``converged=False`` by default (before any
    growth, generalized solve, union solve, or free-COO begins) -- Fe2S2 and Fe4S4 pilot
    runs on an unconverged parent produced multi-Ha "recovery" energies that were artifacts
    of the non-convergence, not genuine physics. A parent whose ``ParentConfig.tail_convergence``
    confirmed a ``stationary_plateau`` (RDM-stable, extension-confirmed -- see
    ``frasci.parent._convergence``) is accepted automatically without this flag; anything
    else needs this set to ``True``, only for an explicitly labeled diagnostic. Every
    ``Result`` produced via this opt-out carries ``diagnostic_only=True`` and
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
        if self.expand_parent_k is not None and self.expand_parent_k < 1:
            raise ValueError(
                f"expand_parent_k must be >= 1 or None, got {self.expand_parent_k}")
        if self.checkpoint_directory is not None:
            object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))


@dataclass(frozen=True)
class Result:
    """What frasci.recovery.solve() returns.

    ``energy``/``gates``/``converged`` describe the requested ``RecoveryConfig.
    recovery_target``'s own ``{Phi_LAS, R}`` generalized solve. ``diagnostics["growth"]``
    carries the corrected-A' growth record (requested N, native actual N, termination
    reason, checkpoint status) and ``diagnostics["spin"]`` the diagnostic-only spin-orbit
    analysis (see that key's own fields -- never a claim of spin-group closure).
    ``union_*`` holds the exact ``P_LAS union R`` solve (always computed, unchanged from
    earlier releases) and ``free_coo_*`` the independently rediagonalized top-K
    free-orbital COO result, produced only when ``RecoveryConfig.run_post_recovery_coo=
    True`` (see that flag's docstring for the always-on -> default-off migration note).
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
    spin_square: float | None = None
    """``<Psi|S^2|Psi>`` for the COMPLETE hybrid wavefunction (contracted Parent block,
    Parent-determinant cross terms, AND the explicit recovery-determinant block
    together), independently validated. ``None`` ("unavailable") in this release: the
    solver does not compute that complete-wavefunction quantity, and discarding the
    recovery support's "owned"-space spin-exchange partners would NOT make the support
    spin-closed either -- a one-dimensional contracted Parent cannot span those owned
    partners, so genuine S^2 closure is architecturally blocked, not merely unverified.
    See ``diagnostics["spin"]`` for the diagnostic-only orbit analysis (orbit
    completeness, which orbits are blocked by requiring an owned partner, and a
    NON-CERTIFYING partial S^2 of the explicit recovery block alone when computable) --
    none of that is ever promoted to this field."""
    parent_converged: bool = True
    """Whether the parent this recovery solve was built on itself reached STRICT
    convergence (mrh's own flag). Always populated -- ``True`` for the ordinary path.
    ``False`` for a non-strict parent regardless of *why* solve() proceeded -- including a
    confirmed ``stationary_plateau``, where ``diagnostic_only`` is still ``False``. See
    ``parent_convergence_status`` for the finer-grained reason."""
    parent_convergence_status: str | None = None
    """The parent's ``convergence_status`` at solve time (one of the six
    ``frasci.parent._convergence.classify_trace`` labels), or ``None`` when the parent's
    ``ParentConfig.tail_convergence`` was unset/off. A confirmed ``"stationary_plateau"``
    is why ``solve()`` can proceed with ``parent_converged=False`` and
    ``diagnostic_only=False`` at the same time -- it is not strict convergence, but it is
    not an unvetted guess either."""
    diagnostic_only: bool = False
    """``True`` only when this result was produced from a parent that was neither strictly
    converged nor a confirmed ``stationary_plateau``, via the explicit
    ``allow_nonconverged_parent=True`` opt-out -- a conspicuous marker that these energies
    are a convergence diagnostic, not a validated physical result."""
    expand_parent_produced: bool = False
    expand_parent_k_requested: int | None = None
    expand_parent_n_dets: int | None = None
    """Achieved support size for the Expanded-Parent diagnostic -- may be smaller than
    ``expand_parent_k_requested`` if the fragment supports admit fewer distinct product
    terms, exactly like achieved recovery ``N`` elsewhere in this package. ``None`` when
    ``RecoveryConfig.expand_parent_k`` was not set."""
    expand_parent_energy: float | None = None
    expand_parent_gain_mha: float | None = None
    """``(H00 - expand_parent_energy) * 1000``: positive means the expanded, uncontracted
    support is lower (better) than the contracted Parent. Independent of
    ``free_coo_gain_mha`` -- the two diagnostics are not additive without rerunning COO on
    the expanded support itself, which this release does not do automatically."""
