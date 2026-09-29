"""Contracted-reference state construction and candidate classification.

Fresh-authored rewrite of the non-solve, non-growth primitives from
``engine/interfrag_hybrid.py``: ``build_contracted_reference``, ``top_product_terms``,
``classify``, ``determinant_diagonal``, ``owned_coefficient``, ``h0d``,
``retain_candidates``, ``selected_hamiltonian``. Same math and semantics as the
original -- new module boundary only.

Out of scope, not ported here (see the plan's Global Constraints):
``grow_fixed_orbital_candidates`` (fixed-orbital-only path, not this method), and
``canonical_orthogonalization_solve`` / ``build_overlap_and_hamiltonian`` /
``solve_contracted_hybrid`` / ``rebuild_exact_contracted_reference`` (H/S assembly and
the generalized eigensolve live in ``frasci.recovery._solve``, Task 3).
"""
from __future__ import annotations

import ctypes
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pyscf import ao2mo, lib
from pyscf.fci import cistring, direct_spin1

from frasci._mock_scf import build_mock_scf_from_fcidump
from frasci._selected_ci_rdm import make_selected_rdm12s
from frasci.parent._config import Parent

Determinant = tuple[int, int]
Sector = tuple[int, int]


def _bit_chunks_to_int(value) -> int:
    """Return a Python integer from a scalar or little-endian uint64 chunks."""
    if np.isscalar(value):
        return int(value)
    out = 0
    for word_index, word in enumerate(np.asarray(value).reshape(-1)):
        out |= int(word) << (64 * word_index)
    return out


def _determinant_to_ints(det) -> tuple[int, int]:
    """Normalize a TrimCI determinant object or an ``(alpha, beta)`` pair."""
    alpha = det.alpha if hasattr(det, "alpha") else det[0]
    beta = det.beta if hasattr(det, "beta") else det[1]
    return _bit_chunks_to_int(alpha), _bit_chunks_to_int(beta)


def _embed_bits(local_bits: int, orbitals: Sequence[int]) -> int:
    """Scatter a fragment-local occupation bitstring into the global orbital indices."""
    if int(local_bits) >> len(orbitals):
        raise ValueError("local bitstring occupies an orbital outside its block")
    out = 0
    for local, global_orb in enumerate(orbitals):
        if (int(local_bits) >> local) & 1:
            out |= 1 << int(global_orb)
    return out


def _block_sectors(det: Determinant, blocks: Sequence[Sequence[int]]) -> list[Sector]:
    """Per-fragment (n_alpha, n_beta) occupation of a global determinant."""
    alpha, beta = map(int, det)
    sectors = []
    for block in blocks:
        mask = sum(1 << int(o) for o in block)
        sectors.append(((alpha & mask).bit_count(), (beta & mask).bit_count()))
    return sectors


def classify(det: Determinant, ref_sectors: Sequence[Sector],
             blocks: Sequence[Sequence[int]]) -> tuple[str, int]:
    """Classify a determinant against the LAS reference's per-fragment sectors.

    Returns ``("owned", 0)`` if every fragment's (n_alpha, n_beta) exactly matches
    ``ref_sectors``; otherwise ``n_ct`` counts the half-sum of absolute per-fragment
    electron-count deviations, and the label is ``("dispersion", 0)`` when ``n_ct == 0``
    (same total per fragment, different spin split) or ``("ct", n_ct)`` for a genuine
    charge-transfer excitation of rank ``n_ct``.
    """
    sectors = _block_sectors(det, blocks)
    if sectors == list(ref_sectors):
        return "owned", 0
    n_ct = sum(abs((na + nb) - (ra + rb))
               for (na, nb), (ra, rb) in zip(sectors, ref_sectors)) // 2
    return ("dispersion", 0) if n_ct == 0 else ("ct", n_ct)


def determinant_diagonal(h1: np.ndarray, eri: np.ndarray,
                         determinants: Iterable[Determinant]) -> np.ndarray:
    """Slater-Condon diagonal <D|H_1e+H_2e|D> for a batch of determinants."""
    h1_diag = np.diagonal(h1)
    eri_j = np.einsum("ppqq->pq", eri)
    eri_k = np.einsum("pqqp->pq", eri)
    values = []
    for alpha, beta in map(_determinant_to_ints, determinants):
        occ_a = [p for p in range(h1.shape[0]) if (alpha >> p) & 1]
        occ_b = [p for p in range(h1.shape[0]) if (beta >> p) & 1]
        e = float(h1_diag[occ_a].sum() + h1_diag[occ_b].sum())
        e += 0.5 * float((eri_j[np.ix_(occ_a, occ_a)] -
                          eri_k[np.ix_(occ_a, occ_a)]).sum())
        e += 0.5 * float((eri_j[np.ix_(occ_b, occ_b)] -
                          eri_k[np.ix_(occ_b, occ_b)]).sum())
        e += float(eri_j[np.ix_(occ_a, occ_b)].sum())
        values.append(e)
    return np.asarray(values)


def selected_hamiltonian(h1: np.ndarray, eri: np.ndarray,
                         determinants: Iterable[Determinant], *, n_orb: int) -> np.ndarray:
    """Exact selected-space Slater-Condon Hamiltonian (dense, n_orb < 64)."""
    dets = [_determinant_to_ints(d) for d in determinants]
    if not dets or len(dets) != len(set(dets)):
        raise ValueError("selected determinant space must be nonempty and deduplicated")
    if n_orb >= 64:
        raise NotImplementedError("PySCF selected p-space requires n_orb < 64")
    sectors = {(a.bit_count(), b.bit_count()) for a, b in dets}
    if len(sectors) != 1:
        raise ValueError(f"determinants span multiple electron sectors: {sectors}")
    h1c = np.ascontiguousarray(h1, dtype=float)
    eric = np.ascontiguousarray(ao2mo.restore(1, eri, n_orb), dtype=float)
    stra = np.ascontiguousarray([a for a, _ in dets], dtype=np.uint64)
    strb = np.ascontiguousarray([b for _, b in dets], dtype=np.uint64)
    h0 = np.zeros((len(dets), len(dets)))
    direct_spin1.libfci.FCIpspace_h0tril(
        h0.ctypes.data_as(ctypes.c_void_p), h1c.ctypes.data_as(ctypes.c_void_p),
        eric.ctypes.data_as(ctypes.c_void_p), stra.ctypes.data_as(ctypes.c_void_p),
        strb.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_orb), ctypes.c_int(len(dets)))
    h0 = lib.hermi_triu(h0)
    idx = np.arange(len(dets))
    h0[idx, idx] = determinant_diagonal(h1c, eric, dets)
    return h0


def _prepare_selected_integrals(h1: np.ndarray, eri: np.ndarray, n_orb: int):
    """Contiguous arrays accepted by PySCF's exact selected-space C kernel."""
    if n_orb >= 64:
        raise NotImplementedError("PySCF selected p-space requires n_orb < 64")
    return (np.ascontiguousarray(h1, dtype=float),
            np.ascontiguousarray(ao2mo.restore(1, eri, n_orb), dtype=float))


def selected_hamiltonian_element(det_left: Determinant, det_right: Determinant, *,
                                 n_orb: int, prepared_integrals) -> float:
    """Exact off-diagonal Slater--Condon element using constant-size work arrays.

    ``prepared_integrals`` comes from ``_prepare_selected_integrals``.  This path is
    intentionally rectangular-in-spirit: it evaluates one requested edge without
    constructing the dense Hamiltonian over every candidate pair.
    """
    left = _determinant_to_ints(det_left)
    right = _determinant_to_ints(det_right)
    if left == right:
        h1c, eric = prepared_integrals
        return float(determinant_diagonal(h1c, eric, [left])[0])
    if (left[0].bit_count(), left[1].bit_count()) != (
            right[0].bit_count(), right[1].bit_count()):
        raise ValueError("determinants span different electron sectors")
    h1c, eric = prepared_integrals
    stra = np.ascontiguousarray([left[0], right[0]], dtype=np.uint64)
    strb = np.ascontiguousarray([left[1], right[1]], dtype=np.uint64)
    h0 = np.zeros((2, 2))
    direct_spin1.libfci.FCIpspace_h0tril(
        h0.ctypes.data_as(ctypes.c_void_p), h1c.ctypes.data_as(ctypes.c_void_p),
        eric.ctypes.data_as(ctypes.c_void_p), stra.ctypes.data_as(ctypes.c_void_p),
        strb.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_orb), ctypes.c_int(2))
    return float(h0[1, 0])


def top_product_terms(state: dict, k: int) -> tuple[list[Determinant], np.ndarray]:
    """Heap-select the top-k LAS product terms by |product of local coefficients|.

    Never materializes the full Cartesian product of fragment determinants.
    """
    factors = []
    for frag in state["frags"]:
        coeffs = np.asarray(frag["coeffs"], dtype=float)
        order = np.argsort(-np.abs(coeffs), kind="stable")
        factors.append([(tuple(map(int, frag["dets"][i])), float(coeffs[i]))
                        for i in order if coeffs[i] != 0.0])
    origin = (0,) * len(factors)

    def score(indices):
        return float(np.prod([abs(factors[j][i][1]) for j, i in enumerate(indices)]))

    heap = [(-score(origin), origin)]
    visited = {origin}
    dets, coeffs = [], []
    while heap and len(dets) < k:
        _, indices = heapq.heappop(heap)
        alpha = beta = 0
        coeff = 1.0
        for j, (idx, block) in enumerate(zip(indices, state["orbital_lists"])):
            (la, lb), lc = factors[j][idx]
            alpha |= _embed_bits(la, block)
            beta |= _embed_bits(lb, block)
            coeff *= lc
        dets.append((alpha, beta))
        coeffs.append(coeff)
        for j in range(len(factors)):
            neighbor = list(indices)
            neighbor[j] += 1
            neighbor = tuple(neighbor)
            if neighbor[j] < len(factors[j]) and neighbor not in visited:
                visited.add(neighbor)
                heapq.heappush(heap, (-score(neighbor), neighbor))
    coeffs = np.asarray(coeffs)
    coeffs /= np.linalg.norm(coeffs)
    return dets, coeffs


def _extract_exact_fragments(parent_dir: Path, bundle, meta: dict, mo_coeff: np.ndarray):
    """Safe fallback for old parents that predate phase1_state persistence."""
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

    ncas_sub = tuple(map(int, meta["ncas_sub"]))
    nelec = [tuple(map(int, x)) for x in meta["nelec_per_frag"]]
    spin_sub = list(map(int, meta["spin_sub"]))
    dims = [math.comb(n, na) * math.comb(n, nb)
            for n, (na, nb) in zip(ncas_sub, nelec)]
    if any(dim > 1_000_000 for dim in dims):
        raise RuntimeError(
            f"parent has no phase1_state and exact local fallback dimensions are {dims}; "
            "rerun the phased parent with sparse-state persistence enabled")
    mf = build_mock_scf_from_fcidump(str(bundle.fcidump_path))
    las = LASSCF(mf, ncas_sub, nelec, spin_sub=spin_sub)
    las.verbose = 0
    conv, e_lasci, _es, _ecas, _lexc, ci = las.lasci(mo_coeff)
    frags = []
    for i, (norb, (na, nb)) in enumerate(zip(ncas_sub, nelec)):
        ca = cistring.make_strings(range(norb), na)
        cb = cistring.make_strings(range(norb), nb)
        vec = np.asarray(ci[i][0])
        terms = [(int(a), int(b), float(vec[ia, ib]))
                 for ia, a in enumerate(ca) for ib, b in enumerate(cb)
                 if abs(vec[ia, ib]) > 1e-14]
        dets = np.asarray([(a, b) for a, b, _ in terms], dtype=np.int64)
        coeffs = np.asarray([c for _, _, c in terms])
        frags.append(dict(dets=dets, coeffs=coeffs, R=np.eye(norb), na=na, nb=nb,
                          lookup=dict(zip(map(tuple, dets.tolist()), coeffs.tolist()))))
    return frags, float(e_lasci), bool(conv), dims


def h00_from_frags(h1: np.ndarray, eri: np.ndarray, blocks: Sequence[Sequence[int]],
                   frags: list[dict], *, e_const: float) -> float:
    """<Phi_LAS|H|Phi_LAS> = e_const + sum of intra-fragment RDM energies + the
    inter-fragment Coulomb/exchange mean-field terms between fragment 1-RDMs.

    Exactly the formula ``_assemble_contracted_reference`` always used -- factored out
    so a rotated-frame rebuild (``rebuild_state_after_block_diagonal_rotation``) can
    recompute H00 from RDMs in the NEW frame, not propagate a cached scalar.
    """
    gammas_a, gammas_b, gammas = [], [], []
    e_intra = 0.0
    for block, frag in zip(blocks, frags):
        (d1a, d1b), (d2aa, d2ab, d2bb) = make_selected_rdm12s(
            frag["dets"], frag["coeffs"], len(block), (frag["na"], frag["nb"]))
        d2 = d2aa + d2ab + d2ab.transpose(2, 3, 0, 1) + d2bb
        e_intra += float(np.einsum("pq,pq->", h1[np.ix_(block, block)], d1a + d1b)
                         + 0.5 * np.einsum("pqrs,pqrs->",
                                            eri[np.ix_(block, block, block, block)], d2))
        ga = np.zeros_like(h1); gb = np.zeros_like(h1)
        ga[np.ix_(block, block)] = d1a; gb[np.ix_(block, block)] = d1b
        gammas_a.append(ga); gammas_b.append(gb); gammas.append(ga + gb)
    e_inter = 0.0
    for i in range(len(frags)):
        for j in range(i + 1, len(frags)):
            j_term = np.einsum("pq,pq->", gammas[i],
                               np.einsum("rs,pqrs->pq", gammas[j], eri, optimize=True))
            k_term = (np.einsum("ps,ps->", gammas_a[i],
                                np.einsum("qr,pqrs->ps", gammas_a[j], eri, optimize=True))
                      + np.einsum("ps,ps->", gammas_b[i],
                                  np.einsum("qr,pqrs->ps", gammas_b[j], eri, optimize=True)))
            e_inter += float(j_term - k_term)
    return float(e_const + e_intra + e_inter)


def densify_fragment_ci(dets: np.ndarray, coeffs: np.ndarray, *, norb: int,
                        na: int, nb: int) -> np.ndarray:
    """Sparse fragment-local (dets, coeffs) -> dense (n_strings_a, n_strings_b) FCI array,
    in PySCF's own string-address ordering (index i in ``cistring.make_strings`` IS
    address i -- the same convention ``_extract_exact_fragments`` already relies on when
    it goes the other direction).
    """
    ca = cistring.make_strings(range(norb), na)
    cb = cistring.make_strings(range(norb), nb)
    addr_a = {int(s): i for i, s in enumerate(ca)}
    addr_b = {int(s): i for i, s in enumerate(cb)}
    dense = np.zeros((len(ca), len(cb)), dtype=float)
    for (la, lb), c in zip(dets, coeffs):
        dense[addr_a[int(la)], addr_b[int(lb)]] = float(c)
    return dense


def sparsify_fragment_ci(dense: np.ndarray, *, norb: int, na: int,
                         nb: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``densify_fragment_ci``: the COMPLETE dense support as sparse
    (dets, coeffs) arrays -- every (alpha_string, beta_string) pair, no thresholding.
    Per the plan's Global Constraints, transformed fragment CI is never truncated.
    """
    ca = cistring.make_strings(range(norb), na)
    cb = cistring.make_strings(range(norb), nb)
    dets = np.array([(int(a), int(b)) for a in ca for b in cb], dtype=np.int64)
    coeffs = np.ascontiguousarray(dense, dtype=float).reshape(-1)
    return dets, coeffs


def transform_fragment_ci_block(frag: dict, u_block: np.ndarray) -> dict:
    """Covariantly rotate one fragment's CI into a new LOCAL orbital basis.

    ``u_block`` is that fragment's own (len(block), len(block)) orbital-rotation
    submatrix (``U[np.ix_(block, block)]`` for a block-diagonal global ``U``).  Returns a
    NEW fragment dict (dets/coeffs/lookup/na/nb; ``R`` composed as ``frag["R"] @ u_block``)
    -- never mutates ``frag`` in place, matching every other builder in this module.
    """
    from pyscf.fci import addons

    norb = u_block.shape[0]
    na, nb = int(frag["na"]), int(frag["nb"])
    dense_old = densify_fragment_ci(frag["dets"], frag["coeffs"], norb=norb, na=na, nb=nb)
    dense_new = addons.transform_ci(dense_old, (na, nb), np.ascontiguousarray(u_block))
    dets, coeffs = sparsify_fragment_ci(dense_new, norb=norb, na=na, nb=nb)
    norm = float(np.linalg.norm(coeffs))
    if abs(norm - 1.0) > 1e-8:
        raise AssertionError(
            f"transformed fragment CI norm is {norm}, expected 1.0 (orbital rotation "
            "must be unitary)")
    return dict(dets=dets, coeffs=coeffs, R=frag["R"] @ u_block, na=na, nb=nb,
               lookup=dict(zip(map(tuple, dets.tolist()), coeffs.tolist())))


def _assemble_contracted_reference(bundle, meta: dict, mo0: np.ndarray, frags: list[dict],
                                   *, parent_dir: Path, parent_e_tot: float, source: str,
                                   parent_converged: bool,
                                   fallback_lasci_energy=None, fallback_converged=None,
                                   fallback_dims=None) -> dict:
    """Assemble H00 and the sparse LAS reference once fragment wavefunctions are known."""
    ncas_sub = list(map(int, meta["ncas_sub"]))
    starts = np.cumsum([0] + ncas_sub)
    blocks = [list(range(int(starts[i]), int(starts[i + 1])))
              for i in range(len(ncas_sub))]
    ref_sectors = [tuple(map(int, x)) for x in meta["nelec_per_frag"]]

    mo_final = mo0.copy()
    for block, frag in zip(blocks, frags):
        mo_final[:, block] = mo0[:, block] @ frag["R"]
    orth_err = float(np.max(np.abs(mo_final.T @ mo_final - np.eye(mo_final.shape[1]))))
    if orth_err > 1e-8:
        raise AssertionError(f"contracted-reference orbitals are not orthogonal: {orth_err:.3e}")

    mf = build_mock_scf_from_fcidump(str(bundle.fcidump_path))
    # e_const is the FCIDUMP core constant (already carries any frozen-core contribution),
    # not a freshly recomputed nuclear repulsion energy -- recomputing would double-count
    # the core on systems with a frozen core (e.g. Fe4S4).
    e_const = float(mf.mol.energy_nuc())
    h1 = mo_final.T @ mf.get_hcore() @ mo_final
    eri = ao2mo.incore.full(mf._eri, mo_final, compact=False).reshape((bundle.n_orb,) * 4)

    h00 = h00_from_frags(h1, eri, blocks, frags, e_const=e_const)
    return dict(
        parent_dir=str(parent_dir), n_orb=bundle.n_orb, n_frag=len(frags),
        n_elec=(int(bundle.n_alpha), int(bundle.n_beta)), blocks=blocks, orbital_lists=blocks,
        ref_sectors=ref_sectors, frags=frags, mo_final=mo_final, h1=h1, eri=eri,
        e_const=e_const, h00=h00, checkpoint_e_tot=parent_e_tot,
        source=source, parent_converged=bool(parent_converged),
        fallback_lasci_energy=fallback_lasci_energy,
        fallback_converged=fallback_converged, fallback_dimensions=fallback_dims,
        gates=dict(orbitals_orthonormal=True,
                   fragment_coefficients_normalized=all(
                       abs(float(np.linalg.norm(f["coeffs"])) - 1.0) < 1e-6 for f in frags),
                   h00_finite=bool(np.isfinite(h00))),
    )


def build_contracted_reference(parent: Parent | Path, bundle) -> dict:
    """Build the sparse contracted LAS reference in one common orbital frame.

    Accepts either an in-memory ``frasci.parent.Parent`` (default handoff from
    ``frasci.parent.solve()``) or a ``Path`` to a persisted parent directory
    (``checkpoint.npz``/``checkpoint_metadata.json``/``phase1_state/`` --
    backward-compatible / for resuming from disk). Both branches produce the
    identical ``state`` dict shape.
    """
    if isinstance(parent, Parent):
        return _build_contracted_reference_from_parent_object(parent, bundle)
    return _build_contracted_reference_from_path(parent, bundle)


def rebuild_state_after_block_diagonal_rotation(state: dict, rotation: np.ndarray, *,
                                                 cross_fragment_tol: float = 1e-12) -> dict:
    """Covariantly rebuild a contracted-reference ``state`` after a fragment-block-
    constrained orbital rotation ``rotation`` (an (n_orb, n_orb) matrix returned by a
    fixed-support/evaluate-only native COO call restricted to intra-fragment pairs).

    This is a pure basis-change of an EXISTING converged wavefunction -- not a new
    LASSCF/fragment solve. Every fragment's CI
    is covariantly transformed (never reused unchanged), h1/eri are rotated by the SAME
    global ``rotation``, and H00 is recomputed from RDMs in the new frame via
    ``h00_from_frags`` -- never propagated as a cached scalar.

    Raises ``AssertionError`` if ``rotation`` is not (to ``cross_fragment_tol``) exactly
    block-diagonal on ``state["blocks"]`` -- the "no-rebuild invariance gate" a
    deliberately cross-fragment rotation must fail.
    """
    n_orb = int(state["n_orb"])
    rotation = np.ascontiguousarray(rotation, dtype=float)
    if rotation.shape != (n_orb, n_orb):
        raise ValueError(f"rotation must be ({n_orb},{n_orb}), got {rotation.shape}")

    blocks = state["blocks"]
    cross_norm_sq = 0.0
    for i, left in enumerate(blocks):
        for j, right in enumerate(blocks):
            if i != j:
                cross_norm_sq += float(np.linalg.norm(rotation[np.ix_(left, right)])) ** 2
    cross_norm = float(np.sqrt(cross_norm_sq))
    if cross_norm > cross_fragment_tol:
        raise AssertionError(
            f"rotation is not block-diagonal on the fragment partition: cross-fragment "
            f"norm {cross_norm:.3e} exceeds tol {cross_fragment_tol:.0e} -- this is not a "
            "fragment-gauge-preserving rotation; refusing to rebuild the contracted state")

    h1_new = rotation.T @ state["h1"] @ rotation
    eri_new = np.einsum("ip,jq,kr,ls,ijkl->pqrs", rotation, rotation, rotation, rotation,
                        state["eri"], optimize=True)
    mo_new = state["mo_final"] @ rotation

    new_frags = []
    for block, frag in zip(blocks, state["frags"]):
        u_block = rotation[np.ix_(block, block)]
        new_frags.append(transform_fragment_ci_block(frag, u_block))

    h00_new = h00_from_frags(h1_new, eri_new, blocks, new_frags, e_const=state["e_const"])

    new_state = dict(state)
    new_state.update(mo_final=mo_new, h1=h1_new, eri=eri_new, h00=h00_new, frags=new_frags,
                     source="constrained_coo_block_diagonal_rebuild",
                     cross_fragment_rotation_norm=cross_norm)
    return new_state


def _build_contracted_reference_from_parent_object(parent: Parent, bundle) -> dict:
    """In-memory equivalent of ``_build_contracted_reference_from_path``'s
    non-fallback branch. A ``Parent`` always carries every fragment's complete
    state, so there is no fallback case here.
    """
    meta = dict(ncas_sub=list(parent.ncas_sub),
               nelec_per_frag=[list(x) for x in parent.nelec_per_frag])
    mo0 = np.asarray(parent.mo_coeff, dtype=float)
    ref_sectors = [tuple(map(int, x)) for x in meta["nelec_per_frag"]]

    if len(parent.fragments) != len(ref_sectors):
        raise ValueError(
            f"Parent has {len(parent.fragments)} fragments but the checkpoint "
            f"metadata expects {len(ref_sectors)}")

    frags = []
    for i, (sector, frag) in enumerate(zip(ref_sectors, parent.fragments)):
        dets = np.asarray(frag.determinants, dtype=np.int64)
        coeffs = np.asarray(frag.coefficients, dtype=float)
        rot = np.asarray(frag.orbital_rotation, dtype=float)
        if (int(frag.n_alpha), int(frag.n_beta)) != sector:
            raise AssertionError(f"fragment {i} sector disagrees with parent metadata")
        frags.append(dict(dets=dets, coeffs=coeffs, R=rot,
                          na=sector[0], nb=sector[1],
                          lookup=dict(zip(map(tuple, dets.tolist()), coeffs.tolist()))))

    return _assemble_contracted_reference(
        bundle, meta, mo0, frags, parent_dir="<in-memory Parent object>",
        parent_e_tot=float(parent.e_tot), source="in_memory_parent_object",
        parent_converged=bool(parent.converged),
        fallback_lasci_energy=None, fallback_converged=None, fallback_dims=None)


def _build_contracted_reference_from_path(parent_dir: Path, bundle) -> dict:
    """Build the sparse contracted LAS reference in one common orbital frame.

    Loads each fragment's CI vector and orbital rotation from
    ``parent_dir/phase1_state/fragment_{i}.npz``, rotates the parent MOs block-by-block
    (``mo_final[:, block_i] = mo0[:, block_i] @ R_i``), asserts the result is orthonormal,
    and assembles h1/eri/H00 in that frame.
    """
    parent_dir = Path(parent_dir)
    meta = json.loads((parent_dir / "checkpoint_metadata.json").read_text())
    checkpoint = np.load(parent_dir / "checkpoint.npz", allow_pickle=False)
    mo0 = np.asarray(checkpoint["mo_coeff"], dtype=float)
    ncas_sub = list(map(int, meta["ncas_sub"]))
    ref_sectors = [tuple(map(int, x)) for x in meta["nelec_per_frag"]]
    state_dir = parent_dir / "phase1_state"
    fallback = not all((state_dir / f"fragment_{i}.npz").exists()
                       for i in range(len(ncas_sub)))
    parent_result = json.loads((parent_dir / "result.json").read_text())
    parent_e_tot = float(parent_result["result"]["e_tot"])

    if fallback:
        frags, fallback_lasci_energy, fallback_converged, fallback_dims = \
            _extract_exact_fragments(parent_dir, bundle, meta, mo0)
        parent_counts = parent_result.get("result", {}).get("dets_per_frag_final")
        if not fallback_converged:
            raise RuntimeError("exact LASCI fallback did not converge")
        if parent_counts is None or list(map(int, parent_counts)) != list(map(int, fallback_dims)):
            raise RuntimeError(
                "exact LASCI fallback would change the parent fragment spaces: "
                f"parent counts={parent_counts}, exact dimensions={fallback_dims}; rerun the "
                "parent with persisted phase1_state")
        source = "exact_lasci_fallback"
    else:
        frags = []
        for i, sector in enumerate(ref_sectors):
            raw = np.load(state_dir / f"fragment_{i}.npz", allow_pickle=False)
            dets = np.asarray(raw["determinants"], dtype=np.int64)
            coeffs = np.asarray(raw["coefficients"], dtype=float)
            rot = np.asarray(raw["orbital_rotation"], dtype=float)
            if tuple(map(int, (raw["n_alpha"], raw["n_beta"]))) != sector:
                raise AssertionError(f"fragment {i} saved sector disagrees with parent metadata")
            frags.append(dict(dets=dets, coeffs=coeffs, R=rot,
                              na=sector[0], nb=sector[1],
                              lookup=dict(zip(map(tuple, dets.tolist()), coeffs.tolist()))))
        fallback_lasci_energy = None
        fallback_converged, fallback_dims = None, None
        source = "persisted_sparse_phase1_state"

    return _assemble_contracted_reference(
        bundle, meta, mo0, frags, parent_dir=parent_dir, parent_e_tot=parent_e_tot,
        source=source, parent_converged=bool(meta["converged"]),
        fallback_lasci_energy=fallback_lasci_energy,
        fallback_converged=fallback_converged, fallback_dims=fallback_dims)


def owned_coefficient(det: Determinant, state: dict) -> float | None:
    """S_0D = product of per-fragment local CI coefficients, or None if any fragment's
    sector doesn't match the reference or the fragment-local determinant is absent from
    that fragment's sparse CI lookup.
    """
    alpha, beta = map(int, det)
    coeff = 1.0
    for block, sector, frag in zip(state["blocks"], state["ref_sectors"], state["frags"]):
        mask = sum(1 << o for o in block)
        if ((alpha & mask).bit_count(), (beta & mask).bit_count()) != sector:
            return None
        la = lb = 0
        for local, orb in enumerate(block):
            la |= ((alpha >> orb) & 1) << local
            lb |= ((beta >> orb) & 1) << local
        local_coeff = frag["lookup"].get((la, lb))
        if local_coeff is None:
            return None
        coeff *= local_coeff
    return float(coeff)


def _iter_rank_le2_excitations(det: Determinant, n_orb: int):
    """Yield singles and doubles without materializing the full neighbor set."""
    alpha, beta = det
    occ_a = [p for p in range(n_orb) if (alpha >> p) & 1]
    occ_b = [p for p in range(n_orb) if (beta >> p) & 1]
    vir_a = [p for p in range(n_orb) if not (alpha >> p) & 1]
    vir_b = [p for p in range(n_orb) if not (beta >> p) & 1]
    for occ, vir, spin in ((occ_a, vir_a, "a"), (occ_b, vir_b, "b")):
        for i in occ:
            for a in vir:
                yield ((alpha ^ (1 << i) ^ (1 << a), beta) if spin == "a"
                       else (alpha, beta ^ (1 << i) ^ (1 << a)))
        for ii in range(len(occ)):
            for jj in range(ii + 1, len(occ)):
                for aa in range(len(vir)):
                    for bb in range(aa + 1, len(vir)):
                        bits = (1 << occ[ii]) ^ (1 << occ[jj]) ^ (1 << vir[aa]) ^ (1 << vir[bb])
                        yield ((alpha ^ bits, beta) if spin == "a"
                               else (alpha, beta ^ bits))
    for i in occ_a:
        for a in vir_a:
            for j in occ_b:
                for b in vir_b:
                    yield (alpha ^ (1 << i) ^ (1 << a), beta ^ (1 << j) ^ (1 << b))


def _rank_le2_excitations(det: Determinant, n_orb: int) -> set[Determinant]:
    """All rank-<=2 excitations; retained for existing small-space callers."""
    return set(_iter_rank_le2_excitations(det, n_orb))


def h0d(det: Determinant, state: dict) -> float:
    """Exact <Phi_LAS|H|D>, including the same-sector self term, via rank <= 2 excitations."""
    total = 0.0
    self_coeff = owned_coefficient(det, state)
    if self_coeff:
        total += self_coeff * float(determinant_diagonal(state["h1"], state["eri"], [det])[0])
    matched, coeffs = [], []
    for phi in _iter_rank_le2_excitations(det, state["n_orb"]):
        c = owned_coefficient(phi, state)
        if c:
            matched.append(phi); coeffs.append(c)
    if matched:
        h = selected_hamiltonian(state["h1"], state["eri"], [det] + matched,
                                 n_orb=state["n_orb"])
        total += float(np.dot(coeffs, h[0, 1:]))
    return float(total + (state["e_const"] * self_coeff if self_coeff else 0.0))


def _fragment_lookup_table(frag: dict, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted ``(local_key, coeff)`` arrays for vectorized batch membership lookup.

    ``local_key = la * 2**block_size + lb`` packs a fragment-local (alpha, beta) pair
    into one sortable integer (block_size <= ~30 in practice, well under int64 range).
    """
    dets = np.asarray(frag["dets"], dtype=np.int64)
    keys = dets[:, 0] * (1 << block_size) + dets[:, 1]
    coeffs = np.asarray(frag["coeffs"], dtype=float)
    order = np.argsort(keys)
    return keys[order], coeffs[order]


def prepare_owned_coefficient_lookup(state: dict) -> tuple[dict, ...]:
    """Prepare immutable sorted fragment lookup arrays once for repeated batches."""
    prepared = []
    for block, sector, frag in zip(state["blocks"], state["ref_sectors"], state["frags"]):
        block = tuple(map(int, block))
        table_keys, table_coeffs = _fragment_lookup_table(frag, len(block))
        table_keys = np.array(table_keys, copy=True)
        table_coeffs = np.array(table_coeffs, copy=True)
        table_keys.flags.writeable = False
        table_coeffs.flags.writeable = False
        prepared.append({
            "block": block,
            "sector": tuple(map(int, sector)),
            "mask": np.uint64(sum(1 << orbital for orbital in block)),
            "table_keys": table_keys,
            "table_coeffs": table_coeffs,
        })
    return tuple(prepared)


def owned_coefficient_batch(alpha: np.ndarray, beta: np.ndarray, state: dict, *,
                            lookup_cache=None) -> np.ndarray:
    """Vectorized ``owned_coefficient`` for a batch of determinants.

    Returns an array of S_0D values, 0.0 wherever any fragment's sector or local
    lookup misses (``owned_coefficient`` would have returned ``None``). Reused by
    ``h0d_batch`` -- see there for why this is the actual bottleneck this replaces:
    a Python dict lookup + per-orbital bit loop per candidate, repeated per h0d call.
    """
    alpha = np.asarray(alpha, dtype=np.uint64)
    beta = np.asarray(beta, dtype=np.uint64)
    n = len(alpha)
    coeff = np.ones(n, dtype=float)
    owned = np.ones(n, dtype=bool)
    prepared = (prepare_owned_coefficient_lookup(state)
                if lookup_cache is None else tuple(lookup_cache))
    if len(prepared) != len(state["frags"]):
        raise ValueError("owned-coefficient lookup cache does not match fragment count")
    for entry in prepared:
        block = entry["block"]
        sector = entry["sector"]
        block_size = len(block)
        mask = entry["mask"]
        na = np.bitwise_count(alpha & mask)
        nb = np.bitwise_count(beta & mask)
        owned &= (na == sector[0]) & (nb == sector[1])
        la = np.zeros(n, dtype=np.int64)
        lb = np.zeros(n, dtype=np.int64)
        for local, orb in enumerate(block):
            la |= (((alpha >> np.uint64(orb)) & np.uint64(1)).astype(np.int64)) << local
            lb |= (((beta >> np.uint64(orb)) & np.uint64(1)).astype(np.int64)) << local
        keys = la * (1 << block_size) + lb
        table_keys, table_coeffs = entry["table_keys"], entry["table_coeffs"]
        idx = np.searchsorted(table_keys, keys)
        idx_clipped = np.clip(idx, 0, len(table_keys) - 1)
        found = (idx < len(table_keys)) & (table_keys[idx_clipped] == keys)
        owned &= found
        coeff = coeff * np.where(found, table_coeffs[idx_clipped], 1.0)
    return np.where(owned, coeff, 0.0)


def h0d_batch(candidates: Sequence[Determinant], state: dict, kernel, *,
              lookup_cache=None, telemetry: dict | None = None) -> np.ndarray:
    """Vectorized ``h0d`` for many determinants at once.

    ``h0d`` is exact but was the dominant runtime cost in bridge-search candidate
    ranking: it re-enumerates a candidate's full rank<=2 neighbor set (up to ~85k at
    Fe4S4 scale) and calls ``owned_coefficient`` -- a per-orbital Python bit loop plus
    a dict lookup -- once per neighbor, per candidate. With thousands of candidates
    ranked per round this was hundreds of millions of Python-level operations.

    This keeps the same per-candidate neighbor enumeration (already fast via
    ``iter_excitation_batches``, the same generator the vectorized edge kernel uses)
    but replaces the per-neighbor Python dict lookup with one vectorized
    ``owned_coefficient_batch`` call per candidate's full neighbor batch, and evaluates
    the exact Hamiltonian coupling only for the (typically few) nonzero hits via the
    vectorized ``kernel`` (a ``frasci.recovery._slater_condon.RealSlaterCondon``)
    instead of a dense ``selected_hamiltonian`` call. Mathematically identical to
    calling ``h0d`` per candidate -- validated against it directly in tests.
    """
    from frasci.recovery._slater_condon import iter_excitation_batches

    n_orb = int(state["n_orb"])
    candidates = list(candidates)
    self_coeffs = owned_coefficient_batch(
        np.asarray([int(d[0]) for d in candidates], dtype=np.uint64),
        np.asarray([int(d[1]) for d in candidates], dtype=np.uint64), state,
        lookup_cache=lookup_cache)
    results = np.zeros(len(candidates), dtype=float)
    neighbor_checks = batch_calls = owned_hits = 0
    for i, det in enumerate(candidates):
        total = 0.0
        if self_coeffs[i]:
            total += self_coeffs[i] * float(
                determinant_diagonal(state["h1"], state["eri"], [det])[0])
        for _cls, batch in iter_excitation_batches(det, n_orb):
            batch_calls += 1
            neighbor_checks += len(batch)
            neigh_a = np.fromiter((p[0] for p in batch), dtype=np.uint64)
            neigh_b = np.fromiter((p[1] for p in batch), dtype=np.uint64)
            neighbor_coeffs = owned_coefficient_batch(
                neigh_a, neigh_b, state, lookup_cache=lookup_cache)
            hit_idx = np.flatnonzero(neighbor_coeffs)
            owned_hits += len(hit_idx)
            if len(hit_idx):
                hits = [batch[j] for j in hit_idx]
                edge_values = kernel.elements(det, hits)
                total += float(np.dot(neighbor_coeffs[hit_idx], edge_values))
        results[i] = total + (state["e_const"] * self_coeffs[i] if self_coeffs[i] else 0.0)
    if telemetry is not None:
        telemetry.update(candidates=len(candidates), neighbor_checks=neighbor_checks,
                         batch_calls=batch_calls, owned_hits=owned_hits)
    return results


def retain_candidates(dets: Sequence[Determinant], coeffs: Sequence[float], state: dict,
                      *, n_ct_max: int = 2, include_same_sector: bool = False) -> tuple[dict, dict]:
    """Classify every candidate determinant and drop those with n_ct > n_ct_max.

    By default (``include_same_sector=False``), same-sector-present/-absent ("owned")
    determinants are excluded from the returned ``eligible`` set entirely -- they are still
    tallied in ``counts`` so the classification breakdown stays complete; they simply never
    enter the Hamiltonian/overlap block built from ``eligible``. Pass ``include_same_sector=True``
    to restore old behavior and include owned determinants in the eligible set. Dispersion/ct1/ct2
    are unaffected either way.

    Returns ``(eligible, counts)``: ``eligible`` maps each retained determinant to its
    proposal coefficient, S_0D (owned dets only), label, and n_ct; ``counts`` tallies how
    many candidates fell into each classification bucket (including dropped/duplicate ones).
    """
    eligible = {}
    counts = dict(same_sector_present=0, same_sector_absent=0,
                  dispersion=0, ct1=0, ct2=0, ct_gt2=0, duplicates=0)
    for det, c in zip(dets, coeffs):
        det = tuple(map(int, det))
        if det in eligible:
            counts["duplicates"] += 1
            continue
        label, n_ct = classify(det, state["ref_sectors"], state["blocks"])
        s = owned_coefficient(det, state)
        if label == "owned":
            key = "same_sector_present" if s is not None else "same_sector_absent"
            counts[key] += 1
            if include_same_sector:
                eligible[det] = dict(coeff=float(c), s=float(s or 0.0), label=key, n_ct=0)
        elif label == "dispersion":
            counts["dispersion"] += 1
            eligible[det] = dict(coeff=float(c), s=0.0, label=label, n_ct=0)
        elif n_ct <= 2:
            counts[f"ct{n_ct}"] += 1
            eligible[det] = dict(coeff=float(c), s=0.0, label=f"ct{n_ct}", n_ct=n_ct)
        elif n_ct <= n_ct_max:
            # Rank > 2 charge transfer, admitted because n_ct_max was raised above 2. Always
            # labeled "ct_gt2" regardless of the exact rank (3, 4, ...) -- matches the fixed
            # 4-label taxonomy {dispersion, ct1, ct2, ct_gt2} that _solve.py's strict_labels
            # already assumes; a dynamic f"ct{n_ct}" key here caused a KeyError against the
            # counts dict for any n_ct > 2 (counts only ever pre-declares ct1/ct2/ct_gt2).
            counts["ct_gt2"] += 1
            eligible[det] = dict(coeff=float(c), s=0.0, label="ct_gt2", n_ct=n_ct)
        else:
            counts["ct_gt2"] += 1
    return eligible, counts
