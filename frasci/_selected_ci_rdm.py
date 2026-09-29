"""
_selected_ci_rdm.py
====================
Sparse selected-CI 1/2-RDM construction for the LASSCF/TrimCI fragment kernels.

Port of engine/lasscf/selected_ci_rdm.py -- new module boundary, same logic.
The two engine/lasscf/trimci_to_civec.py functions the `dense` backend needs
(`_extract_alpha_beta_bits`, `trimci_result_to_pyscf_civec`) are inlined below
as private helpers rather than porting a separate module: nothing else in
frasci needs trimci_to_civec's other functions.

`make_selected_rdm12s` is a drop-in for `pyscf.fci.direct_spin1.make_rdm12s`
that consumes a sparse selected-CI wavefunction ``(dets, coeffs)`` instead of a
dense FCI vector. On the ``sparse`` backend it never allocates the full FCI
vector ``C(norb,nalpha) x C(norb,nbeta)``: it wraps the selected determinants
into a PySCF ``SCIvector`` over the *unique* alpha/beta strings and calls the
public ``SelectedCI().make_rdm12s`` (link-table, connection-based).

Return convention is that of ``direct_spin1.make_rdm12s`` verbatim:
    ((dm1a, dm1b), (dm2aa, dm2ab, dm2bb)),   2pdm[p,q,r,s] = <p+ r+ s q>.
The caller keeps ownership of back-rotation and the spin-sum
``dm2aa + dm2ab + dm2ab.transpose(2,3,0,1) + dm2bb``.

Verified (engine original, 2026-08-03): `SelectedCI().make_rdm12s` reproduces
`direct_spin1.make_rdm12s` to <=3.3e-16 on non-Cartesian selected sets and is
*truly sparse* -- a norb=24 (18a,18b) 300-det RDM (full FCI vector = 135 GiB)
peaks at 96 MiB. Both memory and runtime track the unique-string rectangle
(u_alpha . u_beta), not the full sector.
"""
from __future__ import annotations

import time
from collections.abc import MutableMapping
from math import comb

import numpy as np
from pyscf.fci import cistring, direct_spin1, selected_ci

_DEFAULT_MAX_DENSE_BYTES = 128 * 1024 ** 2      # 128 MiB
_FALLBACK_RAM_CAP_BYTES = 2 * 1024 ** 3         # used when psutil is unavailable
_NATIVE_BUFFER_SAFETY = 1.5                     # slack for selected_ci C buffers

# One stateless solver reused across calls (make_rdm12s only reads its args).
_SCI = selected_ci.SelectedCI()


# ---------------------------------------------------------------------------
# inlined from engine/lasscf/trimci_to_civec.py (only the 2 functions the
# dense backend below needs)
# ---------------------------------------------------------------------------
def _extract_alpha_beta_bits(det, norb: int):
    """Extract (alpha_bits, beta_bits) as plain Python ints from a TrimCI
    determinant. Handles both the scalar (norb<=64) and list-of-uint64
    (norb>64) cases."""
    alpha_raw = det.alpha
    beta_raw = det.beta

    if isinstance(alpha_raw, (list, tuple)):
        alpha_bits = 0
        beta_bits = 0
        for chunk_idx, (a_chunk, b_chunk) in enumerate(zip(alpha_raw, beta_raw)):
            shift = chunk_idx * 64
            alpha_bits |= int(a_chunk) << shift
            beta_bits |= int(b_chunk) << shift
    else:
        alpha_bits = int(alpha_raw)
        beta_bits = int(beta_raw)

    return alpha_bits, beta_bits


def _trimci_result_to_pyscf_civec(dets, coeffs, norb: int, nelec: tuple) -> np.ndarray:
    """Pack a 3-phase ``trimci.Result`` wavefunction (``Result.determinants``,
    an ``(n_dets,2)`` int array of ``[alpha_bits, beta_bits]`` rows) into a
    dense PySCF FCI vector. Only ``norb <= 64`` is supported (scalar bitmask
    rows)."""
    n_alpha, n_beta = int(nelec[0]), int(nelec[1])
    dets = np.asarray(dets)
    if dets.ndim != 2 or dets.shape[1] != 2:
        raise ValueError(
            f"expected Result.determinants shape (n_dets, 2), got {dets.shape} "
            f"(norb > 64 packed bitmasks are not supported here)"
        )
    na = cistring.num_strings(norb, n_alpha)
    nb = cistring.num_strings(norb, n_beta)
    civec = np.zeros((na, nb), dtype=np.float64)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    for (alpha_bits, beta_bits), c in zip(dets, coeffs_arr):
        addr_a = cistring.str2addr(norb, n_alpha, int(alpha_bits))
        addr_b = cistring.str2addr(norb, n_beta, int(beta_bits))
        civec[addr_a, addr_b] = float(c)
    return civec


# ---------------------------------------------------------------------------
# size estimators
# ---------------------------------------------------------------------------
def estimate_dense_civec_bytes(norb: int, nelec) -> int:
    na, nb = int(nelec[0]), int(nelec[1])
    return comb(norb, na) * comb(norb, nb) * 8


def estimate_sparse_working_bytes(n_ua: int, n_ub: int, norb: int, nelec) -> int:
    """Working-set model: SCIvector block + one transpose copy + link tables +
    RDM outputs, times a native-buffer safety factor."""
    na, nb = int(nelec[0]), int(nelec[1])
    block = 2 * n_ua * n_ub * 8
    links = (n_ua * na * (norb - na) + n_ub * nb * (norb - nb)) * 8 * 2
    outputs = (2 * norb ** 2 + 3 * norb ** 4) * 8
    return int((block + links + outputs) * _NATIVE_BUFFER_SAFETY)


def _available_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return _FALLBACK_RAM_CAP_BYTES


# ---------------------------------------------------------------------------
# input normalization + validation
# ---------------------------------------------------------------------------
def _to_bitmask_arrays(dets, norb: int):
    """Return (alpha, beta) int64 arrays from either an (n_dets,2) integer array
    or a sequence of objects exposing .alpha/.beta (flat-kernel Determinants)."""
    arr = np.asarray(dets)
    if arr.ndim == 2 and arr.shape[1] == 2 and np.issubdtype(arr.dtype, np.integer):
        return arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64)
    ab = np.asarray([_extract_alpha_beta_bits(d, norb) for d in dets], dtype=np.int64)
    if ab.ndim != 2 or ab.shape[1] != 2:
        raise ValueError(f"could not normalize dets to (n_dets,2); got shape {ab.shape}")
    return ab[:, 0], ab[:, 1]


def _validate(alpha, beta, coeffs, norb, na, nb):
    if norb > 63:
        raise ValueError(f"norb={norb} > 63 unsupported (scalar-int string addressing)")
    if not (0 <= na <= norb and 0 <= nb <= norb):
        raise ValueError(f"invalid electron sector ({na},{nb}) for norb={norb}")
    n = alpha.shape[0]
    if beta.shape[0] != n or coeffs.shape[0] != n:
        raise ValueError(f"length mismatch: dets={n} coeffs={coeffs.shape[0]}")
    if n == 0:
        raise ValueError("empty determinant set")
    limit = 1 << norb
    seen = set()
    for a, b in zip(alpha.tolist(), beta.tolist()):
        if not (0 <= a < limit and 0 <= b < limit):
            raise ValueError(f"bitmask out of range for norb={norb}: ({a},{b})")
        if int(a).bit_count() != na or int(b).bit_count() != nb:
            raise ValueError(
                f"electron-sector mismatch: det ({a},{b}) is "
                f"({int(a).bit_count()},{int(b).bit_count()})e, expected ({na},{nb})e")
        if (a, b) in seen:
            raise ValueError(f"duplicate determinant ({a},{b})")
        seen.add((a, b))


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------
def choose_rdm_backend(alpha, beta, norb, nelec, *, backend,
                       max_dense_bytes, mem_safety_fraction, logger=None,
                       diagnostics: MutableMapping[str, object] | None = None) -> str:
    if backend not in ("auto", "dense", "sparse"):
        raise ValueError(f"backend must be 'auto'|'dense'|'sparse', got {backend!r}")
    dense_bytes = estimate_dense_civec_bytes(norb, nelec)
    if backend == "auto":
        chosen = "dense" if dense_bytes <= max_dense_bytes else "sparse"
    else:
        chosen = backend
    if diagnostics is not None:
        diagnostics["chosen_backend"] = chosen
    if chosen == "sparse":
        n_ua = int(np.unique(alpha).size)
        n_ub = int(np.unique(beta).size)
        need = estimate_sparse_working_bytes(n_ua, n_ub, norb, nelec)
        available = _available_ram_bytes()
        cap = int(mem_safety_fraction * available)
        if diagnostics is not None:
            diagnostics.update({
                "available_ram_bytes": int(available),
                "memory_cap_bytes": int(cap),
            })
        if need > cap:
            raise MemoryError(
                f"sparse selected-CI RDM working set ~{need/1024**3:.2f} GiB exceeds cap "
                f"~{cap/1024**3:.2f} GiB (rectangle {n_ua}x{n_ub}, {alpha.shape[0]} dets, "
                f"norb={norb}, nelec={tuple(int(x) for x in nelec)}). Reduce the determinant "
                f"budget or add the n_dets-scaling fallback.")
        if logger:
            logger(f"[selected_ci_rdm] backend=sparse rect={n_ua}x{n_ub} "
                   f"dense_would_be={dense_bytes/1024**3:.3g}GiB work~{need/1024**2:.0f}MiB")
    elif logger:
        logger(f"[selected_ci_rdm] backend=dense civec={dense_bytes/1024**2:.1f}MiB")
    return chosen


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def _dense_rdm12s(alpha, beta, coeffs, norb, nelec):
    dets2 = np.stack([alpha, beta], axis=1)
    civec = _trimci_result_to_pyscf_civec(dets2, coeffs, norb, nelec)
    return direct_spin1.make_rdm12s(civec, norb, tuple(int(x) for x in nelec))


def _sparse_rdm12s(alpha, beta, coeffs, norb, nelec):
    ua = np.unique(alpha)          # ascending -> cistring address order for the subset
    ub = np.unique(beta)
    ia = {int(s): i for i, s in enumerate(ua)}
    ib = {int(s): j for j, s in enumerate(ub)}
    block = np.zeros((ua.size, ub.size), dtype=np.float64)
    for a, b, c in zip(alpha.tolist(), beta.tolist(), coeffs.tolist()):
        block[ia[a], ib[b]] = c          # absent pairs stay 0 (non-Cartesian closure)
    sciv = selected_ci._as_SCIvector(block, (ua, ub))
    return _SCI.make_rdm12s(sciv, norb, tuple(int(x) for x in nelec))


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def make_selected_rdm12s(dets, coeffs, norb, nelec, *, backend="auto",
                         max_dense_bytes=_DEFAULT_MAX_DENSE_BYTES,
                         mem_safety_fraction=0.5, normalize=True, logger=None,
                         diagnostics: MutableMapping[str, object] | None = None):
    started = time.perf_counter()
    na, nb = int(nelec[0]), int(nelec[1])
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "status": "started",
            "requested_backend": backend,
            "chosen_backend": None,
            "n_orb": int(norb),
            "n_alpha": na,
            "n_beta": nb,
            "n_dets": None,
            "n_unique_alpha": None,
            "n_unique_beta": None,
            "rectangle_entries": None,
            "rectangle_per_det": None,
            "dense_civec_bytes": None,
            "sparse_working_bytes": None,
            "available_ram_bytes": None,
            "memory_cap_bytes": None,
            "mem_safety_fraction": float(mem_safety_fraction),
            "coefficient_norm_before_normalization": None,
            "normalized": False,
            "wall_s": None,
            "error_type": None,
            "error_message": None,
        })

    phase = "validation"
    try:
        alpha, beta = _to_bitmask_arrays(dets, norb)
        coeffs = np.asarray(coeffs, dtype=np.float64).ravel()
        if diagnostics is not None:
            n_dets = int(alpha.size)
            n_ua = int(np.unique(alpha).size)
            n_ub = int(np.unique(beta).size)
            rectangle = n_ua * n_ub
            diagnostics.update({
                "n_dets": n_dets,
                "n_unique_alpha": n_ua,
                "n_unique_beta": n_ub,
                "rectangle_entries": int(rectangle),
                "rectangle_per_det": (float(rectangle / n_dets)
                                      if n_dets else None),
                "dense_civec_bytes": int(estimate_dense_civec_bytes(
                    norb, (na, nb))),
                "sparse_working_bytes": int(estimate_sparse_working_bytes(
                    n_ua, n_ub, norb, (na, nb))),
            })
        _validate(alpha, beta, coeffs, norb, na, nb)

        nrm = float(np.linalg.norm(coeffs))
        if diagnostics is not None:
            diagnostics["coefficient_norm_before_normalization"] = nrm
        if nrm < 1e-12:
            raise ValueError(f"near-zero coefficient norm ({nrm:.3e})")
        if normalize:
            coeffs = coeffs / nrm
            if diagnostics is not None:
                diagnostics["normalized"] = True

        phase = "preflight"
        chosen = choose_rdm_backend(
            alpha, beta, norb, (na, nb), backend=backend,
            max_dense_bytes=max_dense_bytes,
            mem_safety_fraction=mem_safety_fraction, logger=logger,
            diagnostics=diagnostics)
        if diagnostics is not None:
            diagnostics["chosen_backend"] = chosen
        phase = "backend"
        if chosen == "dense":
            result = _dense_rdm12s(alpha, beta, coeffs, norb, (na, nb))
        else:
            result = _sparse_rdm12s(alpha, beta, coeffs, norb, (na, nb))
        if diagnostics is not None:
            diagnostics["status"] = "ok"
        return result
    except Exception as exc:
        if diagnostics is not None:
            diagnostics["status"] = (
                "preflight_failed"
                if phase == "preflight" and isinstance(exc, MemoryError)
                else "error"
            )
            diagnostics["error_type"] = type(exc).__name__
            diagnostics["error_message"] = str(exc)
        raise
    finally:
        if diagnostics is not None:
            diagnostics["wall_s"] = float(time.perf_counter() - started)
