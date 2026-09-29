"""Bridge a converged ``frasci.parent.Parent`` into an mrh LASSIS-ready LAS object.

``frasci.parent.solve()`` drives ``mrh.my_pyscf.mcscf.lasscf_rdm.LASSCF`` internally, whose
RDM-based fragment solvers never populate ``.ci`` (by design, for memory efficiency -- they
convert each fragment's wavefunction straight to 1-/2-RDMs and discard the CI vector).
``mrh.my_pyscf.lassi.lassi.LASSI``/``LASSIS`` require an object with real ``.ci`` vectors
(``LASSI.__init__`` reads ``las.mo_coeff``, ``las.ci``, ``las.ncore``, ``las.ncas``,
``las.nfrags``, ``las.nroots``, ``las.ncas_sub``, ``las.nelecas_sub``, ``las.fciboxes``,
``las.weights``, ``las.e_states``, ``las.e_lexc``, ``las.converged``). This module
reconstructs exactly that, from ``Parent``'s own persisted sparse per-fragment
``(determinants, coefficients)``, WITHOUT re-running any TrimCI kernel: it builds a fresh
``mrh.my_pyscf.mcscf.lasscf_o0.LASSCF`` shell (the CI-vector-preserving mrh implementation --
the same one ``frasci.recovery._state._extract_exact_fragments`` already uses), densifies each
fragment's sparse selected-CI vector via ``frasci.recovery._state.densify_fragment_ci``, and
assigns it directly.

Scope: this bridges the parent's OWN reference state (``|Phi_LAS>``) into LASSIS's required
input shape. It does not make LASSIS itself tractable on a large system -- LASSIS still needs
one dense CI array per fragment, so ``max_dense_elements`` exists specifically to fail loudly,
before allocating anything, on a fragment whose dense dimension would be unreasonable. Fe4S4's
own ``h1diag2`` partition is two 18-orbital fragments (not one 24-orbital fragment), with dense
dimensions 1,914,762,564 and 324 respectively -- the larger of the two alone is already far past
``max_dense_elements``'s default. This adapter does not imply Fe4S4/h1diag2 LASSIS is feasible;
it only makes the bridge itself correct for whichever systems are small enough to attempt it on.
"""
from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np

from frasci._mock_scf import build_mock_scf_from_fcidump
from frasci.parent._config import Parent
from frasci.recovery._state import densify_fragment_ci

__all__ = ["build_lassis_ready_las"]


def _dense_dimension(n_orb: int, na: int, nb: int) -> int:
    return comb(n_orb, na) * comb(n_orb, nb)


def build_lassis_ready_las(parent: Parent, fcidump_path: str | Path, *,
                           max_dense_elements: int = 10_000_000,
                           allow_nonconverged: bool = False):
    """Reconstruct a ``.ci``-populated, LASSIS-compatible LAS object from ``parent``.

    Every partition field (``ncas_sub``, ``nelec_per_frag``, ``spin_sub``) is read directly
    from ``parent`` -- the caller supplies only the FCIDUMP path, since ``Parent`` already
    records the orbital frame (``mo_coeff``) and the partition it was solved against.

    Raises ``ValueError`` if ``parent`` is not converged (unless ``allow_nonconverged=True``),
    if any fragment's shape/electron count is internally inconsistent with the recorded
    partition, or if any fragment's dense CI dimension would exceed ``max_dense_elements``
    (checked for every fragment before allocating any of them -- a preflight, not a
    first-fragment-then-crash failure).
    """
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF as LASSCF_o0

    if not allow_nonconverged and not parent.converged:
        raise ValueError(
            "build_lassis_ready_las: parent.converged is False -- LASSIS built on an "
            "unconverged reference state is not a meaningful comparison. Pass "
            "allow_nonconverged=True to build one anyway (e.g. for an explicitly-labeled "
            "diagnostic/stress-test use, never for a production result).")

    n_frags = len(parent.fragments)
    if not (len(parent.ncas_sub) == len(parent.nelec_per_frag) == len(parent.spin_sub) == n_frags):
        raise ValueError(
            f"build_lassis_ready_las: partition metadata length mismatch -- "
            f"ncas_sub has {len(parent.ncas_sub)}, nelec_per_frag has {len(parent.nelec_per_frag)}, "
            f"spin_sub has {len(parent.spin_sub)}, fragments has {n_frags} entries; "
            "all four must agree.")

    for i, (frag, norb, (na, nb), spin) in enumerate(
            zip(parent.fragments, parent.ncas_sub, parent.nelec_per_frag, parent.spin_sub)):
        if frag.n_orb != norb:
            raise ValueError(
                f"build_lassis_ready_las: fragment {i} has n_orb={frag.n_orb}, but "
                f"parent.ncas_sub[{i}]={norb} -- the recorded partition disagrees with the "
                "fragment's own stored shape.")
        if (frag.n_alpha, frag.n_beta) != (na, nb):
            raise ValueError(
                f"build_lassis_ready_las: fragment {i} has (n_alpha, n_beta)="
                f"({frag.n_alpha}, {frag.n_beta}), but parent.nelec_per_frag[{i}]=({na}, {nb}) "
                "-- the recorded partition disagrees with the fragment's own stored electron count.")
        expected_spin = abs(na - nb) + 1
        if spin != expected_spin:
            raise ValueError(
                f"build_lassis_ready_las: fragment {i} has nelec=({na},{nb}) implying spin "
                f"multiplicity {expected_spin}, but parent.spin_sub[{i}]={spin} disagrees.")

    # Preflight EVERY fragment's dense CI dimension before densifying any of them -- some
    # Fe4S4-scale fragments have dense dimensions no array can hold; fail before allocating,
    # not after the first few fragments have already consumed memory.
    dense_dims = [_dense_dimension(norb, na, nb)
                 for norb, (na, nb) in zip(parent.ncas_sub, parent.nelec_per_frag)]
    oversized = [(i, dim) for i, dim in enumerate(dense_dims) if dim > max_dense_elements]
    if oversized:
        detail = ", ".join(f"fragment {i}: {dim:,} elements" for i, dim in oversized)
        raise MemoryError(
            f"build_lassis_ready_las: {len(oversized)} fragment(s) exceed max_dense_elements="
            f"{max_dense_elements:,} -- LASSIS requires a dense CI array per fragment, and "
            f"these dimensions are not attemptable: {detail}. This is not a bug to work around; "
            "LASSIS is not feasible on this partition at this active-space size.")

    mf = build_mock_scf_from_fcidump(str(fcidump_path))
    las = LASSCF_o0(mf, tuple(parent.ncas_sub), list(parent.nelec_per_frag),
                    spin_sub=list(parent.spin_sub))
    las.mo_coeff = np.asarray(parent.mo_coeff, dtype=float)
    las.ci = [
        [densify_fragment_ci(frag.determinants, frag.coefficients,
                             norb=frag.n_orb, na=frag.n_alpha, nb=frag.n_beta)]
        for frag in parent.fragments
    ]
    las.converged = bool(parent.converged)
    las.e_states = np.array([float(parent.e_tot)])
    las.e_lexc = [[0.0] for _ in range(n_frags)]
    las.weights = [1.0]
    return las
