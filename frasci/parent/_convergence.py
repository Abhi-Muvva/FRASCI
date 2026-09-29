"""Pure, deterministic policy for frasci.parent's bounded adaptive
tail-confirmation protocol (2026-09-29 contract).

No PySCF, MRH, or TrimCI imports here by design: every function operates on
plain floats/dataclasses so the whole convergence policy is testable without
running a single macrocycle. The orchestration that actually drives ``las``
(the continuation ``las.kernel(...)`` calls, RDM snapshotting) lives in
``frasci.parent.__init__`` and calls into this module.

CORE SCIENTIFIC RULE (do not weaken): the parent may automatically extend its
macrocycle budget, but it must never automatically loosen ``conv_tol_grad``.
A ``stationary_plateau`` is not strict convergence -- callers must keep
``Parent.converged is False`` for it.

Trace precision note: the only per-macrocycle values this module ever sees
come from parsing mrh's own ``LASSCF macro N : E = %.15g ; |g_int| = %.15g``
log line (see ``frasci.parent.__init__._GIntTap``) -- mrh keeps no
per-macrocycle attribute on the ``las`` object itself, so stdout parsing is
the only existing reliable hook for a full trace. ``%.15g`` is 15
significant digits, effectively full float64 precision for the gradient/
energy magnitudes and thresholds (1e-4 .. 1e-5 Ha) this policy classifies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

_VALID_MODES = ("off", "signal", "extend")

#: Statuses for which another extension block is ever justified.
_EXTENDABLE_STATUSES = ("near_threshold_progressing", "stationary_plateau_candidate")


@dataclass(frozen=True)
class TailConvergencePolicy:
    """Configuration for the bounded adaptive tail-confirmation protocol.

    ``mode="off"`` (default) is behaviorally inert: no trace capture, no
    classification, no extension -- byte-identical to a parent that has never
    heard of this feature. ``mode="signal"`` classifies the final trace and
    reports whether more iterations look useful, but never runs one. Only
    ``mode="extend"`` ever issues additional ``las.kernel(...)`` calls, and
    only ``max_extra_cycles`` total, ``extension_cycles`` at a time (the last
    block is shortened to whatever remains of the budget -- ``max_extra_cycles``
    need not be an exact multiple of ``extension_cycles``).
    """
    mode: str = "off"
    gradient_upper: float = 5e-4
    window: int = 10
    extension_cycles: int = 10
    max_extra_cycles: int = 50
    energy_span_tol: float = 1e-5
    log_gradient_slope_tol: float = 1e-2
    gradient_relative_range_tol: float = 0.20
    median_gradient_change_tol: float = 0.10
    rdm_change_tol: float = 1e-5
    confirmation_blocks: int = 1

    def __post_init__(self):
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        for name in ("gradient_upper", "energy_span_tol", "log_gradient_slope_tol",
                     "gradient_relative_range_tol", "median_gradient_change_tol",
                     "rdm_change_tol"):
            value = getattr(self, name)
            if not (value > 0):
                raise ValueError(f"{name} must be positive, got {value}")
        if self.window < 3:
            raise ValueError(f"window must be >= 3, got {self.window}")
        if self.extension_cycles < 1:
            raise ValueError(f"extension_cycles must be >= 1, got {self.extension_cycles}")
        if self.max_extra_cycles < self.extension_cycles:
            raise ValueError(
                "max_extra_cycles must be >= extension_cycles, got "
                f"max_extra_cycles={self.max_extra_cycles}, "
                f"extension_cycles={self.extension_cycles}")
        if self.confirmation_blocks < 1:
            raise ValueError(
                f"confirmation_blocks must be >= 1, got {self.confirmation_blocks}")


@dataclass(frozen=True)
class MacroRecord:
    """One captured LASSCF macrocycle line.

    ``local_macro`` is mrh's own counter (restarts at 0 on every fresh
    ``las.kernel(...)`` call). ``segment`` counts continuation calls (0 for
    the initial solve, 1 for the first tail-extension block, ...).
    ``global_macro`` is monotonically increasing across the whole trace
    regardless of segment/local-macro resets.
    """
    segment: int
    local_macro: int
    global_macro: int
    e_tot: float
    g_int: float


@dataclass(frozen=True)
class ConvergenceDiagnostics:
    """Structured output of :func:`classify_trace` -- the raw inputs behind
    the label, not just the label itself."""
    status: str
    needs_more_iterations: bool
    reasons: tuple[str, ...]
    g_int_final: float
    g_int_median: float
    log_gradient_slope: float
    gradient_relative_range: float
    energy_span: float
    median_gradient_change: float
    rdm_change: float | None
    rdm_status: str
    insufficient_history: bool
    extension_blocks_run: int
    consecutive_plateau_blocks: int
    las_declared_converged: bool
    window: tuple[MacroRecord, ...]


def _pair_envelope(values: Sequence[float]) -> list[float]:
    """Adjacent-pair-averaged envelope: kills a small odd/even sawtooth
    while preserving the real trend, without a distribution assumption."""
    if len(values) < 2:
        return list(values)
    return [(a + b) / 2.0 for a, b in zip(values, values[1:])]


def _log_slope(values: Sequence[float]) -> float:
    """Least-squares slope of ln(values) vs. index. ~0 for a flat series,
    materially negative for a genuinely descending one."""
    n = len(values)
    if n < 2:
        return 0.0
    ys = [math.log(v) for v in values]
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    num = sum((x - xbar) * (y - ybar) for x, y in enumerate(ys))
    den = sum((x - xbar) ** 2 for x in range(n))
    return num / den if den else 0.0


def _relative_range(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    lo, hi = min(values), max(values)
    return (hi - lo) / hi if hi else 0.0


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _median_change_frac(values: Sequence[float]) -> float:
    """Fractional drop between the window's first-half and second-half
    medians (positive = improving). Self-contained (no cross-call state):
    avoids the odd/even sawtooth the same way the envelope does, by
    comparing two multi-point blocks rather than two single points."""
    n = len(values)
    half = n // 2
    if half < 1:
        return 0.0
    first = _median(values[:half])
    second = _median(values[-half:])
    if first == 0:
        return 0.0
    return (first - second) / first


def rdm_block_change(previous, current) -> float | None:
    """Max per-fragment Frobenius-norm change between two RDM snapshots
    (each a sequence of same-shaped arrays, one per fragment -- e.g. two
    ``las.make_casdm1s_sub()`` calls). ``None`` when ``previous`` is
    ``None`` (fewer than two snapshots exist yet) -- callers must report
    this as ``not_yet_confirmed``, never silently as "stable"."""
    if previous is None:
        return None
    import numpy as np
    return max(float(np.linalg.norm(np.asarray(c) - np.asarray(p)))
                for p, c in zip(previous, current))


def classify_trace(
    records: Sequence[MacroRecord],
    *,
    conv_tol_grad: float,
    policy: TailConvergencePolicy,
    las_declared_converged: bool,
    extension_blocks_run: int = 0,
    consecutive_plateau_blocks: int = 0,
    rdm_change: float | None = None,
    external_instability: bool = False,
) -> ConvergenceDiagnostics:
    """Classify a captured macrocycle trace into exactly one of six
    statuses: ``strict_converged``, ``near_threshold_progressing``,
    ``stationary_plateau_candidate``, ``stationary_plateau``,
    ``unstable_nonconverged``, ``budget_exhausted``.

    Never returns a plateau status when ``len(records) < policy.window``
    (insufficient history) or when ``extension_blocks_run == 0`` (a
    plateau must be *confirmed* by at least one real extension block, never
    declared from the initial trace alone).
    """
    if not records:
        raise ValueError("classify_trace requires at least one macro record")

    g_final = records[-1].g_int
    insufficient_history = len(records) < policy.window
    window = tuple(records[-policy.window:]) if not insufficient_history else tuple(records)
    g_values = [r.g_int for r in window]
    e_values = [r.e_tot for r in window]

    envelope = _pair_envelope(g_values)
    log_slope = _log_slope(envelope)
    rel_range = _relative_range(envelope)
    median_change = _median_change_frac(g_values)
    e_span = max(e_values) - min(e_values)
    g_median = _median(g_values)
    rdm_status = ("not_yet_confirmed" if rdm_change is None
                  else "stable" if rdm_change <= policy.rdm_change_tol else "unstable")

    reasons: list[str] = []
    if las_declared_converged and g_final < conv_tol_grad:
        status, needs_more = "strict_converged", False
        reasons.append("las_reported_converged_and_gradient_below_conv_tol_grad")
    elif external_instability:
        status, needs_more = "unstable_nonconverged", False
        reasons.append("external_instability_reported")
    elif g_final >= policy.gradient_upper:
        status, needs_more = "budget_exhausted", True
        reasons.append("gradient_at_or_above_gradient_upper_when_budget_ended")
    elif insufficient_history:
        status, needs_more = "near_threshold_progressing", True
        reasons.append("insufficient_history_for_plateau_confirmation")
    else:
        gradient_flat = (abs(log_slope) <= policy.log_gradient_slope_tol
                         and rel_range <= policy.gradient_relative_range_tol)
        energy_flat = e_span <= policy.energy_span_tol
        progressing_ok = (log_slope <= -policy.log_gradient_slope_tol
                          or median_change > policy.median_gradient_change_tol)

        if gradient_flat and not energy_flat:
            status, needs_more = "unstable_nonconverged", False
            reasons.append("gradient_flat_but_energy_drifting")
        elif gradient_flat and energy_flat:
            effective_blocks = consecutive_plateau_blocks + 1
            if (extension_blocks_run >= 1 and effective_blocks >= policy.confirmation_blocks
                    and rdm_change is not None and rdm_change <= policy.rdm_change_tol):
                status, needs_more = "stationary_plateau", False
                reasons.append("plateau_confirmed_by_extension_block_and_rdm_stability")
            else:
                status, needs_more = "stationary_plateau_candidate", True
                reasons.append("gradient_and_energy_flat_awaiting_confirmation")
        elif progressing_ok:
            status, needs_more = "near_threshold_progressing", True
            reasons.append("gradient_envelope_still_descending")
        else:
            status, needs_more = "unstable_nonconverged", False
            reasons.append("gradient_neither_descending_nor_flat")

    return ConvergenceDiagnostics(
        status=status,
        needs_more_iterations=needs_more,
        reasons=tuple(reasons),
        g_int_final=g_final,
        g_int_median=g_median,
        log_gradient_slope=log_slope,
        gradient_relative_range=rel_range,
        energy_span=e_span,
        median_gradient_change=median_change,
        rdm_change=rdm_change,
        rdm_status=rdm_status,
        insufficient_history=insufficient_history,
        extension_blocks_run=extension_blocks_run,
        consecutive_plateau_blocks=consecutive_plateau_blocks,
        las_declared_converged=las_declared_converged,
        window=window,
    )


def decide_extension(
    diagnostics: ConvergenceDiagnostics, *, policy: TailConvergencePolicy,
    extra_cycles_used: int,
) -> int:
    """How many more macrocycles the next extension block should run, or 0
    to stop. Zero unless ``policy.mode == "extend"``, the classification is
    still extendable (progressing or an unconfirmed plateau candidate --
    strict convergence, a confirmed plateau, instability, and budget
    exhaustion are all terminal), and the budget has room left."""
    if policy.mode != "extend":
        return 0
    if diagnostics.status not in _EXTENDABLE_STATUSES:
        return 0
    remaining = policy.max_extra_cycles - extra_cycles_used
    if remaining <= 0:
        return 0
    return min(policy.extension_cycles, remaining)
