"""Typed configuration and result objects for frasci.parent -- the phased
LASSCF-based TrimCI solver (Layer A). Only one method exists here by design
(see the plan's Global Constraints): the flat/non-phased kernel is not exposed.

Defaults below are the "frozen effective phased defaults" read directly from
engine/lasscf/runners/run_lasscf_coo.py::run()'s own body (the values it locks
in when the corresponding phased_* argument is left at its Python default of
None) -- verified 2026-08-28, not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from frasci.parent._convergence import TailConvergencePolicy


@dataclass(frozen=True)
class ParentConfig:
    """Configuration for the whole phased LASSCF-COO solve.

    ``threshold`` and ``max_cycle_macro`` have no default -- they are the
    caller's precision/budget choices, like
    ``frasci.recovery.RecoveryConfig.recovery_target``.
    """
    threshold: float
    max_cycle_macro: int
    orbopt: bool = True
    orbopt_cycles: int = 10
    phase0_dets: int = 100
    #: Default lowered 2000 -> 1000 (2026-09-22): frasci.recovery's Milestone G
    #: free-orbital-COO final step gets its own headroom from las_top_k/P_R growth,
    #: not from a larger stored fragment support -- a smaller/cheaper parent is
    #: sufficient (see frasci.recovery._free_coo).
    phase1_dets: int = 1000
    growth_factor_phase1: float = 1.1
    discovery_num_runs: int = 64
    refine_num_runs: int = 1
    rediscover_tol: float = 1e-2
    rng_seed: int = 0
    pt2: bool = False
    dm_mixing_alpha: float = 1.0
    enable_phase2: bool = False
    phase2_dets: int = 1_000_000
    growth_factor_phase2: float = 2.0
    invariance_atol: float = 1e-4
    warm_start_kappa: bool = True
    adaptive_phase_policy: bool = False
    adaptive_fixed_support_tol: float = 1e-4
    """No longer governs same-episode dispatch (frasci/parent/_kernel.py's
    ``make_phased_kernel`` under ``adaptive_phase_policy=True`` dispatches on
    cached-support presence, not on this threshold) -- retained only for
    backward-compatible construction of existing config records/callers.
    Still validated below; otherwise inert."""
    adaptive_phase0_tol: float = 1e-2
    """Same status as ``adaptive_fixed_support_tol`` above: no longer a
    same-episode rediscovery trigger, retained for backward compatibility."""
    adaptive_phase2_after_plateau: bool = False
    phase2_plateau_delta_e_tol: float = 1e-6
    phase2_plateau_delta_rdm_tol: float = 1e-5
    frame_leakage_tol: float = 5e-2
    frame_strict: bool = True
    failure_capture_dir: str | None = None
    rdm_backend: str = "auto"
    conv_tol_grad: float = 1e-4
    max_cycle_rdmjk: int | None = None
    n_cpu_workers: int | None = None
    """``None`` (default) uses one process per fragment. ``N>0`` uses
    ``min(N, n_frags)`` processes; ``0`` is an explicit serial diagnostic
    override."""
    max_memory_mb: int | None = None
    escape_max_attempts: int | None = None
    """Opt-in macro-boundary escape rule (2026-09-17 contract; see
    frasci/parent/__init__.py's ``_install_safeguarded_escape_controller``).
    ``None`` (default) installs the plain ``_install_frame_publisher`` with
    ZERO behavior change from before this field existed. An integer N>=1
    additionally installs the energy-backtracking safeguard (every orbital
    micro-step independently re-checked against the true global post-step
    energy; a step where every halving remains uphill is rejected and the
    keyframe restored exactly) plus a bounded, global escape controller: on
    a full rejection with ``|g_int|`` still above ``conv_tol_grad``, force
    exactly one fresh Phase-0+1 rediscovery -- at a distinct, reproducibly
    derived episode seed, never a repeat of the same bank -- at the next
    macro boundary, up to N total support attempts (1 = original support
    only, i.e. the safeguard with no rescue; 2 = one rescue). Requires
    ``adaptive_phase_policy=True`` and ``orbopt=True``. Suppressed once any
    fragment has started Phase-2 finalization. Exhaustion does not raise,
    but ``Parent.escape_exhausted`` and ``Parent.escape_log`` distinguish a
    genuinely stalled result from ordinary non-convergence."""
    tail_convergence: TailConvergencePolicy | None = None
    """Opt-in bounded adaptive tail-confirmation protocol (2026-09-29
    contract; see ``frasci.parent._convergence`` and
    ``frasci.parent.__init__``'s ``_run_tail_confirmation``). ``None``
    (default) is behaviorally inert: no macrocycle trace is captured, no
    extra ``las.kernel(...)`` call is ever made, and every new ``Parent``
    field this feature adds stays at its neutral default -- byte-identical
    to a parent that predates this field. A configured policy lets the
    solve inspect the trend of the final macrocycle window after the
    ordinary ``max_cycle_macro`` budget ends and, in ``mode="extend"``,
    request bounded additional macrocycles -- but it can never automatically
    loosen ``conv_tol_grad``, and a confirmed ``stationary_plateau`` still
    leaves ``Parent.converged=False``."""

    def __post_init__(self):
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.max_cycle_macro < 1:
            raise ValueError(f"max_cycle_macro must be >= 1, got {self.max_cycle_macro}")
        if self.n_cpu_workers is not None and self.n_cpu_workers < 0:
            raise ValueError(f"n_cpu_workers must be >= 0, got {self.n_cpu_workers}")
        if self.adaptive_fixed_support_tol < 0:
            raise ValueError("adaptive_fixed_support_tol must be non-negative")
        if self.adaptive_phase0_tol < self.adaptive_fixed_support_tol:
            raise ValueError(
                "adaptive_phase0_tol must be >= adaptive_fixed_support_tol")
        if self.frame_leakage_tol < 0:
            raise ValueError("frame_leakage_tol must be non-negative")
        if self.escape_max_attempts is not None and self.escape_max_attempts < 1:
            raise ValueError(
                f"escape_max_attempts must be >= 1, got {self.escape_max_attempts}")
        if self.escape_max_attempts is not None and not (
                self.adaptive_phase_policy and self.orbopt):
            raise ValueError(
                "escape_max_attempts requires adaptive_phase_policy=True and orbopt=True")
        if self.escape_max_attempts is not None and self.n_cpu_workers != 0:
            raise ValueError(
                "escape_max_attempts requires explicit n_cpu_workers=0: the "
                "Phase-2 guard reads live kernel_fn.state, which parallel dispatch's "
                "proxy kernels do not reflect (their real state lives in a worker "
                "process). Not yet supported, not silently wrong.")
        if (self.tail_convergence is not None
                and self.tail_convergence.gradient_upper <= self.conv_tol_grad):
            raise ValueError(
                "tail_convergence.gradient_upper must be > conv_tol_grad, got "
                f"gradient_upper={self.tail_convergence.gradient_upper}, "
                f"conv_tol_grad={self.conv_tol_grad}")


@dataclass(frozen=True)
class FragmentResult:
    """One fragment's converged phased-CI state -- the in-memory equivalent of
    a phase1_state/fragment_{i}.npz file (see engine/lasscf/coo_kernel.py:687-709,
    the exact npz schema this mirrors field-for-field)."""
    determinants: np.ndarray
    coefficients: np.ndarray
    orbital_rotation: np.ndarray
    n_orb: int
    n_alpha: int
    n_beta: int
    energy: float


@dataclass(frozen=True)
class Parent:
    """The converged whole-system phased LASSCF-COO state (Phi_LAS), in memory.

    ``mo_coeff`` is the full (n_orb, n_orb) molecular-orbital matrix (the
    ``checkpoint.npz`` equivalent). ``fragments`` holds one FragmentResult per
    fragment, in fragment order. ``ncas_sub``/``nelec_per_frag``/``spin_sub``
    describe the partition this parent was solved against (the
    ``checkpoint_metadata.json`` equivalent) -- frasci.recovery needs these to
    interpret ``fragments`` correctly.
    """
    mo_coeff: np.ndarray
    fragments: tuple[FragmentResult, ...]
    ncas_sub: tuple[int, ...]
    nelec_per_frag: tuple[tuple[int, int], ...]
    spin_sub: tuple[int, ...]
    e_tot: float
    converged: bool
    dets_per_frag: tuple[int, ...] = field(default_factory=tuple)
    escape_exhausted: bool = False
    """True iff the (opt-in) escape controller forced its full attempt
    budget and never found a downhill step -- ``converged=False`` alone
    does not distinguish this genuinely stalled outcome from ordinary
    non-convergence (still progressing, just out of macro budget)."""
    escape_log: tuple = field(default_factory=tuple)
    """One dict per macro-boundary escape decision (empty when
    ``ParentConfig.escape_max_attempts`` is ``None``): ``macro``,
    ``invalidated``, ``reason``, ``attempts_used``, ``master_seed``,
    ``episode_seed``, ``g_int``, and (only when ``invalidated``)
    ``next_episode_seed``."""
    convergence_status: str | None = None
    """One of the six tail-confirmation statuses (see
    ``frasci.parent._convergence.classify_trace``), or ``None`` when
    ``ParentConfig.tail_convergence`` is unset/``mode="off"``. Note
    ``convergence_status="stationary_plateau"`` is NOT strict convergence --
    ``converged`` above stays ``False`` for it."""
    needs_more_iterations: bool = False
    """``True`` when the tail policy judged the trace still worth more
    macrocycles but stopped anyway (``mode="signal"``, or ``mode="extend"``
    exhausting ``max_extra_cycles``). Always ``False`` when the tail policy
    is unset/off."""
    initial_macro_budget: int = 0
    """``ParentConfig.max_cycle_macro`` for this solve -- always populated
    (purely informational, independent of the tail policy)."""
    extra_macro_cycles: int = 0
    """Total additional macrocycles actually run by the tail policy's
    ``mode="extend"`` continuation blocks. ``0`` when the tail policy is
    unset/off/``"signal"``, or when it never found an extendable trace."""
    total_macro_cycles: int = 0
    """``initial_macro_budget + extra_macro_cycles``."""
    convergence_diagnostics: dict | None = None
    """The full ``ConvergenceDiagnostics`` from the final classification,
    as a plain (JSON-serializable) dict -- raw classification inputs and
    reasons, not just the label. ``None`` when the tail policy is unset/off."""
