"""
Experimental gamma generation with checkpointed, damped MFA updates.

This is intentionally separate from ``gamma_bootstrap.py``.  It is for trying
Phase-B-style density iterations without overwriting the current production
``gamma_mixed_final.npy`` artifact.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

_DEFAULT_CONFIG = {
    "threshold": 0.06,
    "max_final_dets": "auto",
    "max_rounds": 2,
    "num_runs": 1,
    "pool_build_strategy": "heat_bath",
    "verbose": False,
}


def _reference_gamma(ref_alpha_bits: int, ref_beta_bits: int, n_orb: int) -> np.ndarray:
    return np.array(
        [((ref_alpha_bits >> p) & 1) + ((ref_beta_bits >> p) & 1) for p in range(n_orb)],
        dtype=np.float64,
    )


def _normalize_gamma(gamma: np.ndarray, n_elec: int) -> np.ndarray:
    gamma = np.clip(np.asarray(gamma, dtype=np.float64), 0.0, 2.0)
    gamma_sum = float(gamma.sum())
    if gamma_sum > 0.0:
        gamma = gamma * (float(n_elec) / gamma_sum)
    return np.clip(gamma, 0.0, 2.0)


def _write_history_csv(path: Path, history: Iterable[dict]) -> None:
    rows = list(history)
    if not rows:
        return
    fieldnames = [
        "iteration",
        "mixing",
        "max_delta_gamma",
        "raw_max_delta_gamma",
        "gamma_sum",
        "gamma_min",
        "gamma_max",
        "total_dets",
        "checkpoint_path",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_gamma_experiment(
    fcidump_path: str,
    dets_npz_path: str,
    output_dir: str,
    config: dict | None = None,
    max_iter: int = 30,
    tol: float = 1e-4,
    mixing: float = 0.10,
    min_mixing: float = 0.02,
    adaptive: bool = True,
    start_gamma_path: str | None = None,
    save_every: int = 1,
) -> dict:
    """
    Run a safer experimental gamma loop and save every useful candidate.

    Compared with ``bootstrap_gamma``, this keeps all checkpoints, can start
    from a previous gamma, and reduces the mixing when an update starts moving
    farther away.  The final candidate is saved under ``output_dir`` only.
    """
    import trimci
    from FRASCI.core.fragment import (
        extract_fragment_integrals,
        fragment_by_sliding_window,
        fragment_electron_count,
    )
    from FRASCI.core.trimci_adapter import solve_fragment_trimci
    from FRASCI.mfa.helpers import (
        assemble_global_rdm1_diag,
        compute_fragment_rdm1,
        dress_integrals_meanfield,
    )
    from FRASCI.mfa.solver import load_ref_det

    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")
    if not (0.0 < mixing <= 1.0):
        raise ValueError(f"mixing must be in (0, 1], got {mixing}")
    if not (0.0 < min_mixing <= mixing):
        raise ValueError(f"min_mixing must be in (0, mixing], got {min_mixing}")

    out_dir = Path(output_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    h1, eri, n_elec, n_orb, _e_nuc, _na, _nb, _psym = trimci.read_fcidump(fcidump_path)
    h1 = np.asarray(h1, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_npz_path, row=0)

    if start_gamma_path:
        gamma_cur = np.load(start_gamma_path).astype(np.float64)
        if gamma_cur.shape != (n_orb,):
            raise ValueError(
                f"start gamma shape must be {(n_orb,)}, got {gamma_cur.shape}"
            )
        gamma_cur = _normalize_gamma(gamma_cur, n_elec)
        start_source = str(start_gamma_path)
    else:
        gamma_cur = _reference_gamma(ref_alpha_bits, ref_beta_bits, n_orb)
        start_source = "reference_determinant_row_0"

    order = np.argsort(np.diag(h1))
    fragments = [
        fo for fo in fragment_by_sliding_window(n_orb, order, 15, 10)
        if (lambda na, nb: na > 0 and nb > 0 and na <= len(fo) and nb <= len(fo))(
            *fragment_electron_count(ref_alpha_bits, ref_beta_bits, fo)
        )
    ]

    print(
        "[gamma-experiment] "
        f"start={start_source} fragments={len(fragments)} "
        f"max_iter={max_iter} mixing={mixing} tol={tol}"
    )

    history: list[dict] = []
    previous_delta: float | None = None
    current_mixing = float(mixing)
    converged = False

    for it in range(1, max_iter + 1):
        rdm1_list = []
        frag_orbs_list = []
        dets_per_frag = []

        for frag_orbs in fragments:
            na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
            h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
            frag_set = set(frag_orbs)
            ext_orbs = [r for r in range(n_orb) if r not in frag_set]
            ext_gamma = gamma_cur[np.asarray(ext_orbs, dtype=np.intp)]
            h1_use = dress_integrals_meanfield(h1_f, eri, frag_orbs, ext_gamma, ext_orbs)
            res = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), cfg)
            rdm1_list.append(compute_fragment_rdm1(res.dets, res.coeffs, res.n_orb_frag))
            frag_orbs_list.append(frag_orbs)
            dets_per_frag.append(int(res.n_dets))

        gamma_raw = assemble_global_rdm1_diag(
            rdm1_list,
            frag_orbs_list,
            n_orb,
            n_elec,
            ref_alpha_bits,
            ref_beta_bits,
        )
        raw_delta = float(np.max(np.abs(gamma_raw - gamma_cur)))

        trial_mixing = current_mixing
        while True:
            gamma_next = _normalize_gamma(
                (1.0 - trial_mixing) * gamma_cur + trial_mixing * gamma_raw,
                n_elec,
            )
            delta = float(np.max(np.abs(gamma_next - gamma_cur)))
            if (
                not adaptive
                or previous_delta is None
                or delta <= previous_delta * 1.05
                or trial_mixing <= min_mixing
            ):
                break
            trial_mixing = max(min_mixing, trial_mixing * 0.5)

        current_mixing = trial_mixing
        gamma_cur = gamma_next
        previous_delta = delta

        checkpoint_path: str | None = None
        if it % save_every == 0 or delta <= tol or it == max_iter:
            checkpoint = checkpoint_dir / f"gamma_iter_{it:03d}.npy"
            np.save(checkpoint, gamma_cur)
            checkpoint_path = str(checkpoint)

        row = {
            "iteration": it,
            "mixing": float(current_mixing),
            "fragment_n_dets": dets_per_frag,
            "total_dets": int(sum(dets_per_frag)),
            "max_delta_gamma": delta,
            "raw_max_delta_gamma": raw_delta,
            "gamma_sum": float(gamma_cur.sum()),
            "gamma_min": float(gamma_cur.min()),
            "gamma_max": float(gamma_cur.max()),
            "checkpoint_path": checkpoint_path,
        }
        history.append(row)
        print(
            "[gamma-experiment] "
            f"iter {it}/{max_iter} mix={current_mixing:.4f} "
            f"max|dγ|={delta:.6f} raw={raw_delta:.6f} "
            f"dets={dets_per_frag} sum={gamma_cur.sum():.4f}"
        )

        if delta <= tol:
            converged = True
            break

    final_path = out_dir / "gamma_mixed_final_candidate.npy"
    np.save(final_path, gamma_cur)

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fcidump_path": str(fcidump_path),
        "dets_npz_path": str(dets_npz_path),
        "output_dir": str(out_dir),
        "final_gamma_path": str(final_path),
        "start_gamma_path": str(start_gamma_path) if start_gamma_path else None,
        "start_source": start_source,
        "config": cfg,
        "max_iterations": int(max_iter),
        "tol": float(tol),
        "initial_mixing": float(mixing),
        "final_mixing": float(current_mixing),
        "min_mixing": float(min_mixing),
        "adaptive": bool(adaptive),
        "converged": bool(converged),
        "iterations_completed": len(history),
        "n_orb": int(n_orb),
        "n_elec": int(n_elec),
        "final_gamma_sum": float(gamma_cur.sum()),
        "history": history,
    }
    metadata_path = out_dir / "gamma_experiment_metadata.json"
    history_path = out_dir / "gamma_experiment_history.csv"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    _write_history_csv(history_path, history)
    print(f"[gamma-experiment] saved final candidate to {final_path}")
    print(f"[gamma-experiment] saved metadata to {metadata_path}")
    return metadata


def evaluate_gamma_candidate_d2(
    fcidump_path: str,
    gamma_path: str,
    ref_dets_path: str,
    output_dir: str,
    trimci_config: dict | None = None,
    partition: str = "h1diag",
) -> dict:
    """Run one D2 calculation for a saved gamma candidate."""
    from FRASCI.mfa.solver import run_mfa_d2

    return run_mfa_d2(
        fcidump_path=fcidump_path,
        gamma_path=gamma_path,
        ref_dets_path=ref_dets_path,
        trimci_config=trimci_config or _DEFAULT_CONFIG,
        output_dir=output_dir,
        partition=partition,
    )
