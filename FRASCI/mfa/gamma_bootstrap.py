"""
mfa/gamma_bootstrap.py
======================
Bootstrap a fractional diagonal 1-RDM from the reference determinant.

No pre-computed files required — only fcidump and dets.npz as inputs.

Pipeline (one iteration)
------------------------
1. Build integer occupation vector from the correlated reference determinant
   (row 0 of dets.npz).
2. Partition into overlapping sliding-window fragments (W=15, S=10,
   h1-diagonal ordering) — same geometry as Phase B.
3. Dress each fragment's h1 with the mean-field contribution from all other
   orbitals using the current occupation vector.
4. Run TrimCI on each fragment; extract the diagonal of its 1-RDM.
5. Average overlapping contributions; normalize to n_elec.

Multiple iterations (n_iter > 1) repeat steps 3-5 with the updated
occupation vector, converging toward the Phase B self-consistent solution.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

_DEFAULT_CONFIG = {
    "threshold":           0.06,
    "max_final_dets":      "auto",
    "max_rounds":          2,
    "num_runs":            1,
    "pool_build_strategy": "heat_bath",
    "verbose":             False,
}


def bootstrap_gamma(
    fcidump_path: str,
    dets_npz_path: str,
    config: dict | None = None,
    n_iter: int = 1,
    tol: float | None = None,
    mixing: float = 1.0,
    output_path: str | None = None,
    metadata_path: str | None = None,
) -> np.ndarray:
    """
    Compute a fractional diagonal 1-RDM for use in Phase D2 MFA embedding.

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file
    dets_npz_path : path to dets.npz (reference determinants)
    config        : optional TrimCI config overrides (default: threshold=0.06)
    n_iter        : maximum number of extraction iterations
    tol           : optional convergence tolerance on max |Δγ|
    mixing        : linear mixing fraction for each update, in (0, 1]
    output_path   : optional .npy path to save the final gamma
    metadata_path : optional .json path to save run metadata

    Returns
    -------
    gamma_diag : (n_orb,) float64 — fractional occupation vector, sum == n_elec
    """
    import trimci
    from FRASCI.core.fragment import (
        fragment_by_sliding_window,
        extract_fragment_integrals,
        fragment_electron_count,
    )
    from FRASCI.core.trimci_adapter import solve_fragment_trimci
    from FRASCI.mfa.helpers import (
        compute_fragment_rdm1,
        dress_integrals_meanfield,
        assemble_global_rdm1_diag,
    )
    from FRASCI.mfa.solver import load_ref_det

    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    if not (0.0 < mixing <= 1.0):
        raise ValueError(f"mixing must be in (0, 1], got {mixing}")

    h1, eri, n_elec, n_orb, _e_nuc, _na, _nb, _psym = trimci.read_fcidump(fcidump_path)
    h1  = np.asarray(h1,  dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)

    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_npz_path, row=0)

    # Integer occupation vector from reference determinant
    gamma_cur = np.array(
        [((ref_alpha_bits >> p) & 1) + ((ref_beta_bits >> p) & 1) for p in range(n_orb)],
        dtype=np.float64,
    )
    print(f"[bootstrap] γ_init  sum={gamma_cur.sum():.1f}  "
          f"D={int((gamma_cur == 2).sum())}  "
          f"S={int((gamma_cur == 1).sum())}  "
          f"V={int((gamma_cur == 0).sum())}")

    # Overlapping partition: W=15, S=10, h1-diagonal ordering
    order     = np.argsort(np.diag(h1))
    fragments = [
        fo for fo in fragment_by_sliding_window(n_orb, order, 15, 10)
        if (lambda na, nb: na > 0 and nb > 0 and na <= len(fo) and nb <= len(fo))(
            *fragment_electron_count(ref_alpha_bits, ref_beta_bits, fo)
        )
    ]
    print(f"[bootstrap] {len(fragments)} overlapping fragments (W=15 S=10)")

    history = []
    converged = False

    for it in range(n_iter):
        rdm1_list      = []
        frag_orbs_list = []
        dets_per_frag  = []

        for idx, frag_orbs in enumerate(fragments):
            na, nb    = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
            h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
            ext_orbs  = [r for r in range(n_orb) if r not in set(frag_orbs)]
            ext_gamma = gamma_cur[np.asarray(ext_orbs, dtype=np.intp)]
            h1_use    = dress_integrals_meanfield(h1_f, eri, frag_orbs, ext_gamma, ext_orbs)
            res       = solve_fragment_trimci(h1_use, eri_f, na, nb, len(frag_orbs), cfg)
            rdm1      = compute_fragment_rdm1(res.dets, res.coeffs, res.n_orb_frag)
            rdm1_list.append(rdm1)
            frag_orbs_list.append(frag_orbs)
            dets_per_frag.append(res.n_dets)

        gamma_new = assemble_global_rdm1_diag(
            rdm1_list, frag_orbs_list, n_orb, n_elec,
            ref_alpha_bits, ref_beta_bits,
        )

        gamma_next = (1.0 - mixing) * gamma_cur + mixing * gamma_new
        gamma_sum = gamma_next.sum()
        if gamma_sum > 0:
            gamma_next *= n_elec / gamma_sum
        gamma_next = np.clip(gamma_next, 0.0, 2.0)

        delta = float(np.max(np.abs(gamma_next - gamma_cur)))
        raw_delta = float(np.max(np.abs(gamma_new - gamma_cur)))
        gamma_cur = gamma_next
        history.append({
            "iteration": it + 1,
            "fragment_n_dets": [int(x) for x in dets_per_frag],
            "total_dets": int(sum(dets_per_frag)),
            "max_delta_gamma": delta,
            "raw_max_delta_gamma": raw_delta,
            "gamma_sum": float(gamma_cur.sum()),
        })
        print(f"[bootstrap] iter {it + 1}/{n_iter}  "
              f"dets={dets_per_frag}  total={sum(dets_per_frag)}  "
              f"max|Δγ|={delta:.4f}  raw={raw_delta:.4f}  "
              f"sum={gamma_cur.sum():.4f}")

        if tol is not None and delta <= tol:
            converged = True
            print(f"[bootstrap] converged: max|Δγ|={delta:.6g} <= tol={tol:.6g}")
            break

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, gamma_cur)
        print(f"[bootstrap] saved gamma to {out}")

    if metadata_path:
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fcidump_path": str(fcidump_path),
            "dets_npz_path": str(dets_npz_path),
            "output_path": str(output_path) if output_path else None,
            "config": cfg,
            "max_iterations": int(n_iter),
            "tol": tol,
            "mixing": mixing,
            "converged": converged,
            "iterations_completed": len(history),
            "n_orb": int(n_orb),
            "n_elec": int(n_elec),
            "final_gamma_sum": float(gamma_cur.sum()),
            "history": history,
        }
        meta_out = Path(metadata_path)
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        meta_out.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[bootstrap] saved metadata to {meta_out}")

    return gamma_cur
