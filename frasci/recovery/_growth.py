"""Production corrected-A' recovery growth: ONE direct native-filtered growth call at
a single user-selected ``RecoveryConfig.recovery_target``, with an optional reservoir
checkpoint to resume construction of THAT target.

This is deliberately NOT a ladder. ``frasci.recovery.solve()`` performs one recovery
calculation and never makes an automatic stopping decision. A sequence of targets, the
0.01 mHa plateau rule, and per-rung campaign checkpoints belong to the separate
``frasci.recovery.benchmark_ladder`` driver (used for the all-molecule recovery-limit
study), which reuses this module's reservoir fingerprinting but keeps its own,
differently-named rung checkpoints -- the two are never confused with each other.

Spin handling, precisely: the validated corrected A' recovery support
(``point["eligible"]``) is used EXACTLY as produced -- never augmented, trimmed, or
otherwise changed on the basis of spin analysis. A one-dimensional contracted Parent
cannot be extended to arbitrarily span the "owned" (same-fragment-sector) spin partners
that full S^2 closure of a dispersion determinant's orbit would require, and this
module does not attempt to solve that by adding another contracted parent (which would
be exactly the forbidden multiparent/M-parameter machinery). ``analyze_rung_spin``
below is read-only: it reports which spin orbits are complete within the recovery
space and which are blocked by requiring an "owned" partner the single contracted
Parent cannot supply. ``Result.spin_square`` stays ``None`` in this release.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from frasci.recovery._native_stream import iter_fixed_filtered_ladder
from frasci.recovery._spin import spin_exchange_partners
from frasci.recovery._state import classify

Determinant = tuple[int, int]
_SCHEMA_VERSION = 1
RESERVOIR_CAPACITY_MULTIPLIER = 2  # execution/capacity control, not a scientific knob.


def _support_sha256(dets) -> str:
    digest = hashlib.sha256()
    for alpha, beta in sorted((int(a), int(b)) for a, b in dets):
        digest.update(alpha.to_bytes(64, "little", signed=False))
        digest.update(beta.to_bytes(64, "little", signed=False))
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def build_fingerprint(state: dict, parent_dir, config) -> dict:
    """Identity of everything a checkpoint reuse must match: integrals/frame, Parent
    fingerprint, fragment partition and electron sectors, A-prime settings, and this
    module's schema version. Deliberately excludes ``recovery_target`` itself -- a
    reservoir grown for a smaller target is a valid, reusable warm start for a larger
    one (checked separately by capacity, not by fingerprint equality)."""
    integral_hash = hashlib.sha256(
        np.ascontiguousarray(state["h1"], dtype=float).tobytes()
        + np.ascontiguousarray(state["eri"], dtype=float).tobytes()).hexdigest()
    try:
        parent_identity = str(Path(parent_dir).resolve())
    except TypeError:
        parent_identity = None  # an in-memory Parent object has no filesystem identity
    return {
        "schema_version": _SCHEMA_VERSION,
        "parent_dir": parent_identity,
        "parent_h00_ha": float(state["h00"]),
        "integral_hash": integral_hash,
        "n_orb": int(state["n_orb"]),
        "n_elec": [int(state["n_elec"][0]), int(state["n_elec"][1])],
        "blocks": [[int(o) for o in block] for block in state["blocks"]],
        "ref_sectors": [[int(x) for x in sector] for sector in state["ref_sectors"]],
        "threshold": float(config.threshold),
        "n_ct_max": int(config.n_ct_max),
        "growth_factor": float(config.growth_factor),
        "max_rounds": int(config.max_rounds),
        "las_seed_dets": int(config.las_seed_dets),
        "global_topk": True,
    }


def fingerprint_hash(fingerprint: dict) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def _reservoir_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "reservoir.json", directory / "reservoir.npz"


def save_reservoir(directory: Path, fingerprint: dict, fp_hash: str, capacity: int,
                   dets, coeffs) -> None:
    json_path, npz_path = _reservoir_paths(directory)
    _atomic_write_npz(npz_path, dets=np.asarray(dets, dtype=object),
                      coeffs=np.asarray(coeffs, dtype=float))
    _atomic_write_json(json_path, {
        "fingerprint": fingerprint, "fingerprint_hash": fp_hash,
        "capacity": int(capacity), "n_dets": len(dets),
        "support_sha256": _support_sha256(dets)})


def load_reservoir(directory: Path, fp_hash: str, *, min_capacity: int):
    """Returns ``(dets, coeffs)`` only if the stored fingerprint matches EXACTLY and the
    stored capacity already covers ``min_capacity``; otherwise ``None`` -- the caller
    rebuilds that stage from scratch. Never returns a partially-matching or corrupt
    checkpoint."""
    json_path, npz_path = _reservoir_paths(directory)
    if not (json_path.is_file() and npz_path.is_file()):
        return None
    try:
        metadata = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (metadata.get("fingerprint_hash") != fp_hash
            or int(metadata.get("capacity", 0)) < int(min_capacity)):
        return None
    arrays = np.load(npz_path, allow_pickle=True)
    dets = [tuple(map(int, d)) for d in arrays["dets"]]
    if _support_sha256(dets) != metadata.get("support_sha256"):
        return None
    return dets, np.asarray(arrays["coeffs"], dtype=float)


def analyze_rung_spin(point: dict, state: dict, *, n_alpha: int, n_beta: int,
                      max_orbit_size: int = 2000) -> dict:
    """Diagnostic-only spin-orbit analysis of the UNMODIFIED corrected-A' recovery
    support (``point["eligible"]``). Never changes the determinant selection.

    Groups the recovery (R) determinants by spatial-occupation orbit (same doubly- and
    singly-occupied orbitals) and, for each orbit actually represented in R, computes
    its full theoretical spin-exchange orbit. An orbit is ``complete`` if every member
    is already in R. If a missing member classifies as "owned" against the parent's own
    reference sectors, true completion is architecturally impossible without adding
    another contracted parent to represent that owned partner -- this is reported as
    ``spin_closure_status="blocked_by_owned_space"``, never silently worked around.

    A spin-exchange orbit's size is C(s, s//2) for s singly-occupied orbitals and can
    reach tens of thousands for large active spaces (observed: 48,620 for Fe4S4's
    18-orbital fragments at s=18), each requiring an unvectorized ``classify()`` call
    per missing partner. This size is computed from ``math.comb`` BEFORE calling
    ``spin_exchange_partners`` at all -- materializing the orbit just to measure it
    reintroduces the same blowup this is meant to avoid. An orbit whose size exceeds
    ``max_orbit_size`` skips both the materialization and the per-partner scan, counted
    honestly as ``orbits_skipped_too_large`` -- never silently folded into "complete",
    "owned_blocked", or "incomplete_other", since this function never actually checked
    those orbits' missing partners.
    """
    r_dets = tuple(point["eligible"].keys())
    r_set = frozenset(r_dets)
    blocks, ref_sectors = state["blocks"], state["ref_sectors"]
    seen_orbits: set[tuple[int, int]] = set()
    complete = 0
    owned_blocked = 0
    incomplete_other = 0
    skipped_too_large = 0
    for det in r_dets:
        alpha, beta = int(det[0]), int(det[1])
        orbit_key = (alpha & beta, alpha ^ beta)
        if orbit_key in seen_orbits:
            continue
        seen_orbits.add(orbit_key)
        n_singles = bin(alpha ^ beta).count("1")
        n_single_alpha = n_alpha - bin(alpha & beta).count("1")
        orbit_size = math.comb(n_singles, n_single_alpha)
        if orbit_size > max_orbit_size:
            skipped_too_large += 1
            continue
        full_orbit = frozenset(spin_exchange_partners(det, n_alpha=n_alpha, n_beta=n_beta))
        missing = full_orbit - r_set
        if not missing:
            complete += 1
        elif any(classify(d, ref_sectors, blocks)[0] == "owned" for d in missing):
            owned_blocked += 1
        else:
            incomplete_other += 1
    total = len(seen_orbits)
    if owned_blocked:
        status = "blocked_by_owned_space"
    elif incomplete_other:
        status = "incomplete_within_recovery_space"
    elif skipped_too_large:
        status = "partial_too_expensive_to_fully_check"
    else:
        status = "complete"
    return {
        "total_spin_orbits": total,
        "complete_spin_orbits": complete,
        "orbits_intersecting_owned_space": owned_blocked,
        "orbits_incomplete_within_recovery_space": incomplete_other,
        "orbits_skipped_too_large": skipped_too_large,
        "spin_closure_status": status,
    }


def grow_recovery_target(state: dict, seed_dets, seed_coeffs, config, *,
                         parent_dir=None) -> dict:
    """One direct corrected-A' recovery growth call at ``config.recovery_target``.

    When ``config.checkpoint_directory`` is set, resumes from (and updates) a reservoir
    checkpoint for THIS single target. This is a production-path convenience for
    resuming or re-running one requested target cheaply -- it is NOT the separate
    benchmark ladder's per-rung campaign checkpointing (see
    ``frasci.recovery.benchmark_ladder``), which uses distinctly-named files and tracks
    a whole sequence of targets plus the campaign's own stopping decisions.

    Returns ``{"point": dict, "checkpoint_status": dict}``.
    """
    checkpoint_dir = config.checkpoint_directory
    fingerprint = build_fingerprint(state, parent_dir, config)
    fp_hash = fingerprint_hash(fingerprint)
    capacity = int(config.recovery_target) * RESERVOIR_CAPACITY_MULTIPLIER
    status = {"reservoir": "not_configured", "fingerprint_hash": fp_hash}
    working_dets = working_coeffs = None
    if checkpoint_dir is not None:
        # Any fingerprint-matching checkpoint is a USEFUL resume point regardless of
        # its own stored size relative to the current target: iter_fixed_filtered_ladder
        # already handles "stored support already covers this target" (skips growth)
        # and "needs more growth" (extends it) correctly either way -- requiring the
        # stored capacity to already be >= this call's target would defeat reuse for
        # every strictly-growing sequence of single-target calls, which is exactly the
        # resume case this checkpoint exists for.
        loaded = load_reservoir(checkpoint_dir, fp_hash, min_capacity=0)
        if loaded is not None:
            working_dets, working_coeffs = loaded
            status["reservoir"] = "reused"
        else:
            status["reservoir"] = "rebuilt"

    point = next(iter(iter_fixed_filtered_ladder(
        state, seed_dets, seed_coeffs, recovery_targets=(int(config.recovery_target),),
        n_ct_max=config.n_ct_max, threshold=config.threshold,
        growth_factor=config.growth_factor, max_rounds=config.max_rounds,
        initial_working_dets=working_dets, initial_working_coeffs=working_coeffs,
        global_topk=True, candidate_reservoir_capacity=capacity)))

    if checkpoint_dir is not None:
        save_reservoir(checkpoint_dir, fingerprint, fp_hash, capacity,
                       point["dets"], point["coeffs"])
        if status["reservoir"] == "not_configured":
            status["reservoir"] = "computed_and_saved"

    return {"point": point, "checkpoint_status": status}
