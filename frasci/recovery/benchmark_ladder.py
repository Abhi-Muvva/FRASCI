"""All-molecule recovery-limit benchmark driver -- NOT the public FRASCI protocol.

``frasci.recovery.solve()`` performs ONE direct corrected-A' recovery calculation at a
single user-selected ``RecoveryConfig.recovery_target`` and never makes an automatic
stopping decision (see ``frasci.recovery._growth``). This module is the SEPARATE
benchmarking/testing tool that grows a full recovery-cap ladder across a sequence of
targets and applies this project's own 0.01 mHa plateau criterion -- a benchmark
CAMPAIGN convergence criterion used to decide where a system's recoverable
inter-fragment correlation saturates, not a property, default, or stopping rule of the
released solver itself.

Checkpointing here is a SEPARATE, differently-named "ladder rung" checkpoint per
target, tracking a whole sequence plus the campaign's own stop-reason bookkeeping. It
reuses ``frasci.recovery._growth``'s reservoir fingerprinting (so a campaign and a
single production ``solve()`` call against the same Parent can share warm-start
reservoir growth) but never shares files with, or is confused with, that module's
single-target reservoir checkpoint.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import frasci
from frasci.recovery._growth import (
    RESERVOIR_CAPACITY_MULTIPLIER, _atomic_write_json, _atomic_write_npz,
    _support_sha256, analyze_rung_spin, build_fingerprint, fingerprint_hash,
    load_reservoir, save_reservoir,
)
from frasci.recovery._native_stream import iter_fixed_filtered_ladder
from frasci.recovery._solve import solve_recovery_point
from frasci.recovery._state import retain_candidates

Determinant = tuple[int, int]
_SCHEMA_VERSION = 1

# The shared venv's editable `frasci` install is pinned to the outer (reference-only)
# research tree, not this release package -- see frasci/recovery/benchmark_ladder.py
# module docstring context. Every benchmark entry point must verify it actually got
# this tree before running, not the outer one.
_RELEASE_FRASCI_DIR = Path(__file__).resolve().parents[1]


def _assert_release_tree() -> None:
    resolved = Path(frasci.__file__).resolve().parent
    if resolved != _RELEASE_FRASCI_DIR:
        raise RuntimeError(
            f"frasci resolved to {resolved}, expected {_RELEASE_FRASCI_DIR} "
            "(release/FRASCI/frasci). The outer frasci/ tree is reference-only and "
            "must never be imported or modified for development, tests, or "
            "benchmarks. Aborting.")


@dataclass(frozen=True)
class BenchmarkLadderConfig:
    """Settings for one all-molecule recovery-limit benchmark run. Deliberately
    separate from ``RecoveryConfig`` -- ``recovery_targets`` and
    ``plateau_threshold_mha`` are benchmark-campaign concepts, not public solver
    configuration."""
    recovery_targets: tuple[int, ...]
    threshold: float = 1e-6
    n_ct_max: int = 2
    max_memory_gb: float = 16.0
    las_seed_dets: int = 100
    growth_factor: float = 1.5
    max_rounds: int = 40
    solver_kind: str = "auto"
    dense_dimension_threshold: int = 5000
    plateau_threshold_mha: float = 0.01
    checkpoint_directory: Path | None = None

    def __post_init__(self):
        targets = tuple(int(t) for t in self.recovery_targets)
        if not targets or any(t < 1 for t in targets) or tuple(sorted(set(targets))) != targets:
            raise ValueError("recovery_targets must be unique, positive, and increasing")
        object.__setattr__(self, "recovery_targets", targets)
        if self.plateau_threshold_mha <= 0:
            raise ValueError(
                f"plateau_threshold_mha must be positive, got {self.plateau_threshold_mha}")
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
        if self.checkpoint_directory is not None:
            object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))


def _rung_paths(directory: Path, target: int) -> tuple[Path, Path]:
    stem = directory / f"benchmark_ladder_rung_{int(target):012d}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _save_rung(directory: Path, target: int, fp_hash: str, point: dict, solved: dict,
               record: dict) -> None:
    json_path, npz_path = _rung_paths(directory, target)
    _atomic_write_npz(
        npz_path, dets=np.asarray(point["dets"], dtype=object),
        coeffs=np.asarray(point["coeffs"], dtype=float),
        explicit_determinants=np.asarray(solved["explicit_determinants"], dtype=object),
        generalized_coefficients=np.asarray(solved["coefficients"], dtype=float))
    _atomic_write_json(json_path, {
        "schema_version": _SCHEMA_VERSION, "fingerprint_hash": fp_hash,
        "target": int(target), "record": record})


def _load_rung(directory: Path, target: int, fp_hash: str, state: dict, *, n_ct_max: int):
    """Returns ``{"record": ..., "point": ..., "solved": ...}`` reconstructed from disk,
    or ``None`` on any mismatch/corruption -- never a partial or stale load. ``point``'s
    ``eligible`` classification is recomputed (cheap, deterministic, no native TrimCI
    call) rather than trusted from disk."""
    json_path, npz_path = _rung_paths(directory, target)
    if not (json_path.is_file() and npz_path.is_file()):
        return None
    try:
        metadata = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (metadata.get("schema_version") != _SCHEMA_VERSION
            or metadata.get("fingerprint_hash") != fp_hash
            or int(metadata.get("target", -1)) != int(target)):
        return None
    arrays = np.load(npz_path, allow_pickle=True)
    point_dets = [tuple(map(int, d)) for d in arrays["dets"]]
    point_coeffs = np.asarray(arrays["coeffs"], dtype=float)
    record = dict(metadata["record"])
    if record.get("support_sha256") != _support_sha256(point_dets):
        return None
    eligible, classification = retain_candidates(
        point_dets, point_coeffs, state, n_ct_max=n_ct_max, include_same_sector=False)
    point = {
        "dets": point_dets, "coeffs": point_coeffs,
        "target_r": int(target), "n_recovery": int(record["native_actual_n"]),
        "target_met": bool(record["native_target_met"]),
        "termination_reason": record["termination_reason"],
        "frame_errors": record["frame_errors"],
        "eligible": eligible, "classification": classification,
    }
    solved = {
        "explicit_determinants": [
            tuple(map(int, d)) for d in arrays["explicit_determinants"]],
        "coefficients": np.asarray(arrays["generalized_coefficients"], dtype=float),
        "e_tot": float(record["energy_ha"]), "gates": dict(record["gates"]),
        "generalized_residual_norm": float(record["generalized_residual_norm"]),
        "solver_kind": record["solver_kind"],
        # classification is the freshly recomputed point["classification"] (always
        # present, cheap/deterministic); the rest mirror what a reused rung's own
        # saved record carries, honestly marked unknown if it predates these fields --
        # _build_record and the identical-support short-circuit in
        # run_recovery_ladder_benchmark both read solved[...], not just record[...],
        # so this dict must stay structurally complete on every path, not just the
        # freshly-computed one.
        "classification": dict(classification),
        "coupling_backend": record.get("coupling_backend", "unknown_legacy_checkpoint"),
        "coupling_preflight": dict(record.get("coupling_preflight", {})),
        "coupling_timings": dict(record.get("coupling_timings", {})),
    }
    return {"record": record, "point": point, "solved": solved}


def warm_start_vector(previous_solved: dict | None, eligible_order: list[Determinant]):
    """``[c0, d_1, ..., d_R]`` for the NEW rung's eligible order (the exact order
    :func:`solve_recovery_point`'s own ``retain_candidates`` call will reproduce for
    this same ``point["dets"]``), built by mapping the previous rung's converged
    generalized coefficients onto the new order BY DETERMINANT IDENTITY, zero-padded
    for every newly admitted determinant. ``None`` (cold start) if there is no previous
    solve to warm-start from."""
    if previous_solved is None:
        return None
    previous_dets = previous_solved.get("explicit_determinants")
    if not previous_dets:
        return None
    amplitudes = dict(zip(previous_dets, previous_solved["coefficients"][1:]))
    c0 = float(previous_solved["coefficients"][0])
    return np.asarray(
        [c0, *(float(amplitudes.get(det, 0.0)) for det in eligible_order)], dtype=float)


def evaluate_stop(previous: dict | None, current: dict, *,
                  plateau_threshold_mha: float) -> str | None:
    """Return why the BENCHMARK CAMPAIGN should stop after ``current``'s rung, or
    ``None`` to continue to the next scheduled rung. This is the campaign's own
    convergence criterion, not a property of ``frasci.recovery.solve()``.

    ``gain_mHa = 1000 * (E_previous - E_current)``, ALWAYS recomputed here from
    ``previous``/``current``'s own ``energy_ha`` -- never trusted from a stored
    ``incremental_gain_mha`` field, which may have been computed against a DIFFERENT
    adjacent rung under a different target schedule. The first rung (``previous is
    None``) can never plateau -- it has no preceding rung to compare against. A gain
    below zero beyond float noise is a correctness failure (nested recovery energy must
    never increase), raised rather than reported as any stop reason.
    """
    if (not current["native_target_met"]
            and current["termination_reason"] == "filtered_saturated"):
        return "candidate_pool_exhausted"
    if previous is None:
        return None
    if (not current["native_target_met"]
            and int(current["native_actual_n"]) <= int(previous["native_actual_n"])):
        return "candidate_pool_exhausted"
    gain_mha = 1000.0 * (float(previous["energy_ha"]) - float(current["energy_ha"]))
    if gain_mha < -1e-6:
        raise AssertionError(
            f"nested recovery energy increased by {-gain_mha:.6f} mHa between rungs "
            f"(previous={previous['energy_ha']!r} -> current={current['energy_ha']!r}) "
            "-- this violates the ladder's variational-nesting invariant")
    if gain_mha < float(plateau_threshold_mha):
        return "energy_plateau"
    return None


def _build_record(point: dict, solved: dict, spin_info: dict, previous: dict | None,
                  *, wall_s: float) -> dict:
    import resource
    import sys

    def _peak_rss_mib() -> float:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0

    energy_ha = float(solved["e_tot"])
    gain_mha = (None if previous is None
                else 1000.0 * (float(previous["energy_ha"]) - energy_ha))
    return {
        "requested_n": int(point["target_r"]),
        "native_actual_n": int(point["n_recovery"]),
        "native_target_met": bool(point["target_met"]),
        "energy_ha": energy_ha,
        "incremental_gain_mha": gain_mha,
        "generalized_residual_norm": float(solved["generalized_residual_norm"]),
        "gates": dict(solved["gates"]),
        "solver_kind": solved["solver_kind"],
        "termination_reason": point["termination_reason"],
        "frame_errors": point["frame_errors"],
        "support_sha256": _support_sha256(point["dets"]),
        "spin": spin_info,
        "classification": dict(solved["classification"]),
        "coupling_backend": solved["coupling_backend"],
        "coupling_preflight": dict(solved["coupling_preflight"]),
        "coupling_timings": dict(solved["coupling_timings"]),
        "wall_s": float(wall_s),
        "peak_memory_mib": _peak_rss_mib(),
        "checkpoint_status": "not_configured",
    }


def run_recovery_ladder_benchmark(state: dict, seed_dets, seed_coeffs,
                                  config: BenchmarkLadderConfig, *,
                                  parent_dir=None) -> dict:
    """Run the all-molecule recovery-limit benchmark ladder: grow (the UNCHANGED,
    validated corrected-A' determinant selection), warm-started generalized-solve,
    plateau/exhaustion-stop per THIS CAMPAIGN'S OWN criterion, diagnostic-only
    spin-orbit analysis, and -- when ``config.checkpoint_directory`` is set -- reuse of
    compatible "benchmark_ladder_rung_*" checkpoints with adjacency-dependent fields
    (the incremental gain feeding the plateau decision) always recomputed fresh against
    the ACTUAL previous rung in this run's sequence, never trusted from disk.

    Returns ``{"rungs": [...], "stop_reason": str, "final_point": dict,
    "final_solved": dict, "checkpoint_status": dict}``.
    """
    n_alpha, n_beta = int(state["n_elec"][0]), int(state["n_elec"][1])
    targets = tuple(config.recovery_targets)
    fingerprint = build_fingerprint(state, parent_dir, config)
    fp_hash = fingerprint_hash(fingerprint)
    checkpoint_dir = config.checkpoint_directory

    checkpoint_status = {"reservoir": "not_configured", "rungs_reused": 0,
                         "rungs_computed": 0, "fingerprint_hash": fp_hash}
    working_dets = working_coeffs = None
    rungs: list[dict] = []
    previous_record = None
    previous_solved = None
    previous_r_set: frozenset = frozenset()
    final_point = final_solved = None
    stop_reason = None
    remaining_targets = list(targets)

    if checkpoint_dir is not None:
        for target in list(remaining_targets):
            loaded = _load_rung(checkpoint_dir, target, fp_hash, state,
                               n_ct_max=config.n_ct_max)
            if loaded is None:
                break
            point, solved = loaded["point"], loaded["solved"]
            r_set = frozenset(point["eligible"].keys())
            if not previous_r_set <= r_set:
                raise AssertionError(
                    "checkpointed recovery space is not nested across rungs")
            # Adjacency-dependent fields (the gain feeding the plateau decision) are
            # ALWAYS recomputed against the actual previous rung in THIS run's
            # sequence -- a reused rung's stored incremental_gain_mha may have been
            # computed against a different adjacent rung under a different schedule.
            record = dict(loaded["record"])
            record["incremental_gain_mha"] = (
                None if previous_record is None
                else 1000.0 * (float(previous_record["energy_ha"]) - float(record["energy_ha"])))
            record["checkpoint_status"] = "reused"
            # Backfill bookkeeping fields a checkpoint saved before this field existed
            # won't have. classification is cheap/deterministic to recompute (see
            # _load_rung); coupling_backend/preflight/timings describe an
            # implementation choice made when the rung was originally solved and
            # cannot be reconstructed, so an old checkpoint is honestly labeled
            # unknown rather than backfilled with today's dispatch decision.
            record.setdefault("classification", dict(point["classification"]))
            record.setdefault("coupling_backend", "unknown_legacy_checkpoint")
            record.setdefault("coupling_preflight", {})
            record.setdefault("coupling_timings", {})
            previous_r_set = r_set
            rungs.append(record)
            final_point, final_solved = point, solved
            working_dets, working_coeffs = point["dets"], point["coeffs"]
            previous_solved = solved
            stop_reason = evaluate_stop(
                previous_record, record, plateau_threshold_mha=config.plateau_threshold_mha)
            previous_record = record
            checkpoint_status["rungs_reused"] += 1
            remaining_targets.pop(0)
            if stop_reason is not None:
                break

    if stop_reason is None and remaining_targets:
        reservoir_capacity = max(remaining_targets) * RESERVOIR_CAPACITY_MULTIPLIER
        if checkpoint_dir is not None:
            loaded_reservoir = load_reservoir(
                checkpoint_dir, fp_hash, min_capacity=reservoir_capacity)
            if loaded_reservoir is not None:
                working_dets, working_coeffs = loaded_reservoir
                checkpoint_status["reservoir"] = "reused"
            else:
                checkpoint_status["reservoir"] = "rebuilt"

        growth = iter_fixed_filtered_ladder(
            state, seed_dets, seed_coeffs, recovery_targets=tuple(remaining_targets),
            n_ct_max=config.n_ct_max, threshold=config.threshold,
            growth_factor=config.growth_factor, max_rounds=config.max_rounds,
            initial_working_dets=working_dets, initial_working_coeffs=working_coeffs,
            global_topk=True, candidate_reservoir_capacity=reservoir_capacity)

        for point in growth:
            if checkpoint_dir is not None:
                save_reservoir(
                    checkpoint_dir, fingerprint, fp_hash, reservoir_capacity,
                    point["dets"], point["coeffs"])
            r_set = frozenset(point["eligible"].keys())
            if not previous_r_set <= r_set:
                raise AssertionError(
                    "corrected A' recovery space is not nested across rungs")
            if previous_solved is not None and r_set == previous_r_set:
                # A reservoir that was already grown past every remaining target (e.g.
                # reused from a prior completed run) makes consecutive targets request
                # the IDENTICAL determinant support -- re-solving would reproduce the
                # same energy up to floating-point noise, whose sign is not guaranteed
                # negative, so waiting for evaluate_stop's plateau check here could
                # burn a full re-solve per remaining target for zero new information.
                # Reuse the previous solve wholesale instead of calling
                # solve_recovery_point again; spin analysis is cheap and still
                # recomputed for an honest per-rung record.
                solved = previous_solved
                wall_s = 0.0
            else:
                eligible_order = list(point["eligible"].keys())
                initial_coefficients = warm_start_vector(previous_solved, eligible_order)
                started = time.perf_counter()
                solved = solve_recovery_point(
                    state, point["dets"], point["coeffs"], n_ct_max=config.n_ct_max,
                    solver_kind=config.solver_kind,
                    dense_dimension_threshold=config.dense_dimension_threshold,
                    max_memory_gb=config.max_memory_gb,
                    initial_coefficients=initial_coefficients)
                wall_s = time.perf_counter() - started
            spin_info = analyze_rung_spin(point, state, n_alpha=n_alpha, n_beta=n_beta)
            record = _build_record(point, solved, spin_info, previous_record, wall_s=wall_s)
            if checkpoint_dir is not None:
                _save_rung(checkpoint_dir, point["target_r"], fp_hash, point, solved, record)
                record = {**record, "checkpoint_status": "computed_and_saved"}
                checkpoint_status["rungs_computed"] += 1
            rungs.append(record)
            final_point, final_solved = point, solved
            previous_r_set = r_set
            previous_solved = solved
            stop_reason = evaluate_stop(
                previous_record, record, plateau_threshold_mha=config.plateau_threshold_mha)
            previous_record = record
            if stop_reason is not None:
                break
            working_dets, working_coeffs = point["dets"], point["coeffs"]

    if stop_reason is None:
        stop_reason = "schedule_exhausted_not_plateaued"

    return {
        "rungs": rungs, "stop_reason": stop_reason, "final_point": final_point,
        "final_solved": final_solved, "checkpoint_status": checkpoint_status,
        "requested_targets": targets,
    }


def run(parent_dir, bundle, config: BenchmarkLadderConfig) -> dict:
    """Friendly entry point mirroring ``frasci.recovery.solve()``'s call shape, for the
    all-molecule recovery-limit campaign driver. Builds the contracted reference and
    protected seed itself, then delegates to :func:`run_recovery_ladder_benchmark`.

    Requires a STRICTLY converged or an accepted ``stationary_plateau`` Parent --
    unlike ``frasci.recovery.solve()``, this benchmark driver has no
    ``allow_nonconverged_parent`` opt-out: a benchmark campaign result is never
    produced from an unvetted, non-converged Parent.

    Aborts immediately if ``frasci`` did not resolve to release/FRASCI/frasci (see
    ``_assert_release_tree``) rather than silently benchmarking against the outer,
    reference-only tree.
    """
    _assert_release_tree()
    from frasci.recovery._state import build_contracted_reference, top_product_terms

    state = build_contracted_reference(parent_dir, bundle)
    convergence_status = state.get("convergence_status")
    parent_accepted = state["parent_converged"] or convergence_status == "stationary_plateau"
    if not parent_accepted:
        raise ValueError(
            "frasci.recovery.benchmark_ladder.run() refuses to benchmark on top of a "
            "Parent recorded as converged=False and not a confirmed stationary_plateau "
            f"(convergence_status={convergence_status!r}) -- a benchmark-campaign "
            "result is only meaningful against a validated Parent. There is no "
            "diagnostic opt-out here (unlike RecoveryConfig.allow_nonconverged_parent); "
            "fix or re-run the Parent instead.")
    seed_dets, seed_coeffs = top_product_terms(state, config.las_seed_dets)
    return run_recovery_ladder_benchmark(state, seed_dets, seed_coeffs, config,
                                        parent_dir=parent_dir)
