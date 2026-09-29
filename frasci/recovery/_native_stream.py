"""Native fixed-frame recovery growth and fixed-support orbital optimization."""
from __future__ import annotations

import numpy as np
import trimci.trimci_core as tc

from frasci.recovery._state import _embed_bits, retain_candidates

fe = tc.fast_expansion
Determinant = tuple[int, int]

_WIDTHS = (64, 128, 192, 256, 320, 384, 448, 512)


def build_allowed_orbital_pairs(blocks: list[list[int]]) -> list[tuple[int, int]]:
    """Every (p, q) with p < q, both GLOBAL orbital indices, and both inside the SAME
    fragment block. Cross-fragment pairs are excluded by construction -- there is no
    filtering step, they simply never get generated.
    """
    pairs = []
    for block in blocks:
        sorted_block = sorted(int(o) for o in block)
        for i, p in enumerate(sorted_block):
            for q in sorted_block[i + 1:]:
                pairs.append((p, q))
    return pairs


def _storage_suffix(n_orb: int) -> str:
    for width in _WIDTHS:
        if n_orb <= width:
            return str(width)
    raise ValueError(
        f"n_orb={n_orb} exceeds the largest natively supported determinant width "
        f"({_WIDTHS[-1]})")


def _det_cls(suffix: str):
    return tc.Determinant if suffix == "64" else getattr(tc, f"Determinant{suffix}")


def _pack(value: int, n_words: int):
    """Native per-string storage: a plain int at 1 word, a little-endian tuple wider."""
    if n_words == 1:
        return int(value)
    mask = (1 << 64) - 1
    return tuple((int(value) >> (64 * i)) & mask for i in range(n_words))


def _to_native_det(det: Determinant, suffix: str):
    cls = _det_cls(suffix)
    n_words = int(suffix) // 64
    alpha, beta = det
    return cls(_pack(alpha, n_words), _pack(beta, n_words))


def _pack_array(dets: list[Determinant], n_words: int) -> tuple[np.ndarray, np.ndarray]:
    """``(N,)`` uint64 arrays at 1 word, ``(N, n_words)`` little-endian-word uint64
    arrays wider -- the exact input shape ``run_expansion_phased``/``_128`` expect
    (confirmed empirically: ``get_alpha_beta()`` returns the same shape back)."""
    if n_words == 1:
        alpha = np.array([int(d[0]) for d in dets], dtype=np.uint64)
        beta = np.array([int(d[1]) for d in dets], dtype=np.uint64)
        return alpha, beta
    mask = (1 << 64) - 1
    alpha = np.array([[(int(d[0]) >> (64 * w)) & mask for w in range(n_words)]
                      for d in dets], dtype=np.uint64)
    beta = np.array([[(int(d[1]) >> (64 * w)) & mask for w in range(n_words)]
                     for d in dets], dtype=np.uint64)
    return alpha, beta


def _unpack_array(alpha, beta) -> list[Determinant]:
    """Inverse of ``_pack_array``, from ``get_alpha_beta()``'s returned arrays."""
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    if alpha.ndim == 1:
        return [(int(a), int(b)) for a, b in zip(alpha, beta)]

    def _combine(row) -> int:
        out = 0
        for i, word in enumerate(row):
            out |= int(word) << (64 * i)
        return out

    return [(_combine(a), _combine(b)) for a, b in zip(alpha, beta)]


# Only 64-bit and 128-bit run_expansion_phased bindings exist in this TrimCI build
# (confirmed by introspection: fe.run_expansion_phased / _128, no _192/_256/... variants),
# even though SectorConstraint/iterative_workflow_filtered support up to 512-bit for
# Phase 0. Phase 1/2 cannot silently fall back to the wrong representation.
_EXPANSION_BINDINGS = {"64": ("ExpansionConfig", "run_expansion_phased"),
                       "128": ("ExpansionConfig128", "run_expansion_phased_128")}


def _expansion_bindings(suffix: str):
    """(ExpansionConfig class, run_expansion_phased function) for this storage width."""
    if suffix not in _EXPANSION_BINDINGS:
        raise NotImplementedError(
            f"n_orb requiring {suffix}-bit determinants has no matching "
            f"run_expansion_phased_{suffix} binding in this TrimCI build -- only 64-bit "
            "and 128-bit Phase 1/2 expansion exist (Phase 0 alone supports up to 512-bit "
            f"via iterative_workflow_filtered_{suffix}, but Phase 1/2 cannot continue). "
            "Add the matching expansion bindings before running this system's recovery.")
    cfg_name, fn_name = _EXPANSION_BINDINGS[suffix]
    return getattr(fe, cfg_name), getattr(fe, fn_name)


def build_sector_constraint(state: dict, protected_dets: list[Determinant], *,
                            n_ct_max: int):
    """Native ``SectorConstraint`` mirroring ``state``'s fragment blocks/reference
    sectors/per-fragment local support, translated into native determinant objects.
    """
    n_orb = int(state["n_orb"])
    suffix = _storage_suffix(n_orb)
    factory = getattr(tc, f"make_sector_constraint_{suffix}")
    # frag["dets"] are fragment-LOCAL bitstrings (0-indexed within that fragment's own
    # block, the same convention _state.py's top_product_terms/owned_coefficient use) --
    # the native fragment_support parameter expects them embedded at their actual GLOBAL
    # orbital positions, matching how a full-system Determinant is built everywhere else.
    fragment_support = [
        [_to_native_det(
            (_embed_bits(int(a), block), _embed_bits(int(b), block)), suffix)
         for a, b in frag["dets"]]
        for block, frag in zip(state["blocks"], state["frags"])
    ]
    protected = [_to_native_det(d, suffix) for d in protected_dets]
    constraint = factory(
        n_orb=n_orb,
        fragment_blocks=[list(map(int, block)) for block in state["blocks"]],
        reference_sectors=[tuple(map(int, s)) for s in state["ref_sectors"]],
        fragment_support=fragment_support,
        protected_determinants=protected,
        n_ct_max=int(n_ct_max))
    return constraint, suffix


def _recovery_policy(recovery_target: int, *, max_work_dets: int | None):
    policy = tc.RecoveryPolicy()
    policy.recovery_target = int(recovery_target)
    if max_work_dets is not None:
        policy.max_work_dets = int(max_work_dets)
    return policy


def run_fixed_filtered_ladder(
        state: dict, protected_dets: list[Determinant], protected_coeffs, *,
        recovery_targets, n_ct_max: int = 2, threshold: float = 1e-6,
        growth_factor: float = 1.5, max_rounds: int = 40) -> list[dict]:
    """Grow nested native-filtered recovery spaces in the parent's fixed frame.

    There is no Phase 0 discovery, orbital optimization, or fragment rebuild.  At
    each rung the complete preceding working set is made immutable in the native
    constraint and only the missing number of strict interfragment determinants is
    requested.  This makes the scientific ``R`` spaces rigorously nested even if
    TrimCI would otherwise trim an older recovery determinant.
    """
    targets = tuple(map(int, recovery_targets))
    if not targets or any(t < 1 for t in targets) or tuple(sorted(set(targets))) != targets:
        raise ValueError("recovery_targets must be unique, positive, and increasing")
    n_orb = int(state["n_orb"])
    current_dets = [tuple(map(int, d)) for d in protected_dets]
    current_coeffs = np.asarray(protected_coeffs, dtype=float)
    scientific_seed = frozenset(current_dets)
    snapshots = []
    previous_r = frozenset()

    for target in targets:
        if len(previous_r) < target:
            constraint, suffix = build_sector_constraint(
                state, list(scientific_seed), n_ct_max=n_ct_max)
            cfg_cls, run_fn = _expansion_bindings(suffix)
            cfg = cfg_cls()
            cfg.verbose = 0
            cfg.growth_factor = float(growth_factor)
            cfg.max_expansion_rounds = int(max_rounds)
            cfg.threshold = float(threshold)
            cfg.dets_conv_ratio = -1.0
            cfg.expansion_energy_tol = 0.0
            cfg.orbital_optimization = False
            cfg.sector_constraint = constraint
            cfg.recovery = _recovery_policy(
                target, max_work_dets=len(scientific_seed) + target)
            cfg.strict_target_size = True
            cfg.phase_index = 2
            n_words = int(suffix) // 64
            alpha, beta = _pack_array(current_dets, n_words)
            res = run_fn(
                alpha, beta, current_coeffs,
                np.ascontiguousarray(state["h1"]),
                np.ascontiguousarray(state["eri"]).ravel(), n_orb, [cfg])
            fa, fb = res.get_alpha_beta()
            new_dets = _unpack_array(fa, fb)
            new_coeffs = np.asarray(res.get_coefficients(), dtype=float)
            rotation = np.asarray(res.get_U_total(), dtype=float)
            h1_final = np.asarray(res.get_h1_optimized(), dtype=float)
            eri_final = np.asarray(res.get_eri_optimized(), dtype=float).reshape((n_orb,) * 4)
            frame_errors = {
                "rotation": float(np.max(np.abs(rotation - np.eye(n_orb)))),
                "h1": float(np.max(np.abs(h1_final - state["h1"]))),
                "eri": float(np.max(np.abs(eri_final - state["eri"]))),
            }
            if any(error >= 1e-10 for error in frame_errors.values()):
                raise AssertionError(f"fixed-frame native expansion changed the frame: {frame_errors}")

            # TrimCI protects P_K, not older R.  Make the SCIENTIFIC ladder nested
            # explicitly: keep every prior R and fill the remaining slots from this
            # rung's coefficient-ranked native candidates, then rediagonalize that
            # exact fixed support before using it as the next warm start.
            proposed, _ = retain_candidates(
                new_dets, new_coeffs, state, n_ct_max=n_ct_max,
                include_same_sector=False)
            ranked_new = sorted(
                (det for det in proposed if det not in previous_r),
                key=lambda det: abs(float(proposed[det]["coeff"])), reverse=True)
            chosen_r = list(previous_r)
            chosen_r.extend(ranked_new[:max(0, target - len(chosen_r))])
            coefficient_guess = {
                det: float(coeff) for det, coeff in zip(current_dets, current_coeffs)}
            coefficient_guess.update(
                {det: float(coeff) for det, coeff in zip(new_dets, new_coeffs)})
            current_dets = list(scientific_seed) + chosen_r
            guess = np.asarray([coefficient_guess.get(det, 0.0) for det in current_dets])
            if not np.any(guess):
                guess[0] = 1.0
            from frasci.parent._kernel import _fixed_support_ground_state
            rediagonalized = _fixed_support_ground_state(
                state["h1"], state["eri"], current_dets, guess, n_orb)
            current_dets = [tuple(map(int, d)) for d in rediagonalized.determinants]
            current_coeffs = np.asarray(rediagonalized.coefficients, dtype=float)
            native_composition = res.filtered_composition
            native_telemetry = res.filtered_telemetry
            termination_reason = res.termination_reason
        else:
            frame_errors = {"rotation": 0.0, "h1": 0.0, "eri": 0.0}
            native_composition, native_telemetry = {}, []
            termination_reason = "target_already_present"

        eligible, classification = retain_candidates(
            current_dets, current_coeffs, state, n_ct_max=n_ct_max,
            include_same_sector=False)
        r_set = frozenset(eligible)
        if not previous_r <= r_set:
            raise AssertionError("scientific recovery space is not nested")
        if scientific_seed & r_set:
            raise AssertionError("original protected LAS seed leaked into scientific R")
        previous_r = r_set
        snapshots.append({
            "target_r": target, "dets": list(current_dets),
            "coeffs": np.asarray(current_coeffs, dtype=float),
            "eligible": eligible, "classification": classification,
            "n_protected_original": len(scientific_seed), "n_recovery": len(r_set),
            "n_work": len(current_dets), "target_met": len(r_set) >= target,
            "frame_errors": frame_errors, "native_composition": native_composition,
            "native_telemetry": native_telemetry,
            "termination_reason": termination_reason,
        })
    return snapshots


def run_fixed_support_constrained_coo(
        state: dict, dets: list[Determinant], coeffs, *,
        allowed_orbital_pairs: list[tuple[int, int]],
        orbital_opt_max_iter: int = 20, orbital_opt_grad_tol: float = 1e-6,
        orbital_opt_energy_tol: float = 1e-10) -> dict:
    """Rotate orbitals ONLY within fragment blocks, on a FIXED determinant support.

    No Phase 0, no expansion, no sector constraint: ``evaluate_only=True`` runs exactly
    one Davidson diagonalization on ``dets`` unchanged, followed (since
    ``orbital_optimization=True``) by a full BFGS orbital optimization restricted to
    ``allowed_orbital_pairs`` -- the native mechanism Part 1 added. The determinant
    support is asserted unchanged (requirement: "must not add or remove determinants").
    """
    n_orb = int(state["n_orb"])
    suffix = _storage_suffix(n_orb)
    cfg_cls, run_fn = _expansion_bindings(suffix)
    n_words = int(suffix) // 64

    cfg = cfg_cls()
    cfg.verbose = 0
    cfg.evaluate_only = True
    cfg.orbital_optimization = True
    cfg.orbital_opt_max_iter = int(orbital_opt_max_iter)
    cfg.orbital_opt_grad_tol = float(orbital_opt_grad_tol)
    cfg.orbital_opt_energy_tol = float(orbital_opt_energy_tol)
    cfg.allowed_orbital_pairs = list(allowed_orbital_pairs)
    cfg.phase_index = 1

    input_dets = [tuple(map(int, d)) for d in dets]
    alpha, beta = _pack_array(input_dets, n_words)
    res = run_fn(alpha, beta, np.asarray(coeffs, dtype=float),
                np.ascontiguousarray(state["h1"]), np.ascontiguousarray(state["eri"]).ravel(),
                n_orb, [cfg])

    fa, fb = res.get_alpha_beta()
    out_dets = _unpack_array(fa, fb)
    # evaluate_only never adds/removes determinants, but the native call is free to
    # return them in a different order than the input (observed: internally re-sorted)
    # -- compare as sets, not ordered lists. out_dets/out_coeffs stay mutually
    # consistent (both come from the same res call), just not in the caller's order.
    if set(out_dets) != set(input_dets) or len(out_dets) != len(input_dets):
        raise AssertionError(
            "fixed-support constrained COO changed the determinant support: expected "
            f"the same {len(input_dets)} dets, native call returned a different set of "
            f"{len(out_dets)} -- evaluate_only must not add or remove determinants")
    out_coeffs = np.asarray(res.get_coefficients(), dtype=float)

    rotation = np.asarray(res.get_U_total(), dtype=float)
    h1_final = np.asarray(res.get_h1_optimized(), dtype=float)
    eri_final = np.asarray(res.get_eri_optimized(), dtype=float).reshape((n_orb,) * 4)

    h1_check = rotation.T @ state["h1"] @ rotation
    eri_check = np.einsum(
        "ip,jq,kr,ls,ijkl->pqrs", rotation, rotation, rotation, rotation, state["eri"],
        optimize=True)
    orth_err = float(np.max(np.abs(rotation.T @ rotation - np.eye(n_orb))))
    h1_err = float(np.max(np.abs(h1_final - h1_check)))
    eri_err = float(np.max(np.abs(eri_final - eri_check)))

    cross_norm_sq = 0.0
    for i, left in enumerate(state["blocks"]):
        for j, right in enumerate(state["blocks"]):
            if i != j:
                cross_norm_sq += float(np.linalg.norm(rotation[np.ix_(left, right)])) ** 2
    cross_fragment_rotation_norm = float(np.sqrt(cross_norm_sq))

    return {
        "dets": out_dets, "coeffs": out_coeffs, "rotation": rotation,
        "h1_final": h1_final, "eri_final": eri_final,
        "cross_fragment_rotation_norm": cross_fragment_rotation_norm,
        "rotation_errors": {"orthogonality": orth_err, "h1": h1_err, "eri": eri_err},
        "termination_reason": res.termination_reason,
    }
