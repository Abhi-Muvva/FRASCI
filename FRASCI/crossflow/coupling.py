from __future__ import annotations

from typing import Optional

import numpy as np

from FRASCI.crossflow.types import (
    AllPairCouplingResult,
    FragmentSpinOrbitalClasses,
    PairCouplingResult,
)


def classify_fragment_spin_orbitals(
    ref_alpha_bits: int,
    ref_beta_bits: int,
    frag_orbs: list[int],
) -> FragmentSpinOrbitalClasses:
    occ_alpha: list[int] = []
    virt_alpha: list[int] = []
    occ_beta: list[int] = []
    virt_beta: list[int] = []

    for orb in frag_orbs:
        if (ref_alpha_bits >> orb) & 1:
            occ_alpha.append(orb)
        else:
            virt_alpha.append(orb)

        if (ref_beta_bits >> orb) & 1:
            occ_beta.append(orb)
        else:
            virt_beta.append(orb)

    return FragmentSpinOrbitalClasses(
        occ_alpha=occ_alpha,
        virt_alpha=virt_alpha,
        occ_beta=occ_beta,
        virt_beta=virt_beta,
    )


def compute_pt2_pair_coupling(
    eri_full: np.ndarray,
    h1_solver_I: np.ndarray,
    h1_solver_J: np.ndarray,
    frag_orbs_I: list[int],
    frag_orbs_J: list[int],
    spin_classes_I: FragmentSpinOrbitalClasses,
    spin_classes_J: FragmentSpinOrbitalClasses,
    frag_idx_I: int,
    frag_idx_J: int,
) -> PairCouplingResult:
    local_I = {orb: idx for idx, orb in enumerate(frag_orbs_I)}
    local_J = {orb: idx for idx, orb in enumerate(frag_orbs_J)}
    eps_I = np.diag(h1_solver_I)
    eps_J = np.diag(h1_solver_J)

    E_pt2 = 0.0
    n_terms = 0
    max_abs_M = 0.0
    min_gap: Optional[float] = None
    delta_h_I = np.zeros(len(frag_orbs_I))
    delta_h_J = np.zeros(len(frag_orbs_J))

    channels = [
        (
            spin_classes_I.occ_alpha,
            spin_classes_I.virt_alpha,
            spin_classes_J.occ_alpha,
            spin_classes_J.virt_alpha,
            True,
        ),
        (
            spin_classes_I.occ_beta,
            spin_classes_I.virt_beta,
            spin_classes_J.occ_beta,
            spin_classes_J.virt_beta,
            True,
        ),
        (
            spin_classes_I.occ_alpha,
            spin_classes_I.virt_alpha,
            spin_classes_J.occ_beta,
            spin_classes_J.virt_beta,
            False,
        ),
        (
            spin_classes_I.occ_beta,
            spin_classes_I.virt_beta,
            spin_classes_J.occ_alpha,
            spin_classes_J.virt_alpha,
            False,
        ),
    ]

    for occ_I, virt_I, occ_J, virt_J, same_spin in channels:
        if not (occ_I and virt_I and occ_J and virt_J):
            continue

        for i in occ_I:
            li = local_I[i]
            for a in virt_I:
                la = local_I[a]
                for j in occ_J:
                    lj = local_J[j]
                    for b in virt_J:
                        lb = local_J[b]
                        M = float(eri_full[a, i, b, j])
                        if same_spin:
                            M -= float(eri_full[a, j, b, i])

                        gap = (float(eps_I[la]) + float(eps_J[lb])) - (
                            float(eps_I[li]) + float(eps_J[lj])
                        )
                        if gap <= 0.0:
                            raise ValueError(
                                f"Non-positive gap in PT2 denominator: gap={gap:.6e} "
                                f"(i={i}, a={a}, j={j}, b={b}). "
                                "Check that virtual orbital energies exceed occupied."
                            )

                        contribution = M * M / gap
                        E_pt2 -= contribution
                        n_terms += 1
                        max_abs_M = max(max_abs_M, abs(M))
                        if min_gap is None or gap < min_gap:
                            min_gap = gap

                        delta_h_I[la] -= contribution
                        delta_h_J[lb] -= contribution
                        delta_h_I[li] += contribution
                        delta_h_J[lj] += contribution

    return PairCouplingResult(
        frag_I=frag_idx_I,
        frag_J=frag_idx_J,
        E_pt2=E_pt2,
        n_terms=n_terms,
        max_abs_M=max_abs_M,
        min_gap=min_gap,
        delta_h_I=delta_h_I,
        delta_h_J=delta_h_J,
    )


def compute_all_pair_coupling(
    eri_full: np.ndarray,
    h1_solver_by_frag: dict[int, np.ndarray],
    frag_orbs_list: list[list[int]],
    spin_classes_list: list[FragmentSpinOrbitalClasses],
    coupled_pairs: list[tuple[int, int]],
) -> AllPairCouplingResult:
    delta_h_by_frag = {
        frag_idx: np.zeros(len(frag_orbs))
        for frag_idx, frag_orbs in enumerate(frag_orbs_list)
    }
    pair_results: list[PairCouplingResult] = []
    E_pt2_total = 0.0
    min_gap_global: Optional[float] = None
    max_abs_M_global = 0.0
    n_terms_total = 0

    for I, J in coupled_pairs:
        pair_result = compute_pt2_pair_coupling(
            eri_full,
            h1_solver_by_frag[I],
            h1_solver_by_frag[J],
            frag_orbs_list[I],
            frag_orbs_list[J],
            spin_classes_list[I],
            spin_classes_list[J],
            I,
            J,
        )
        pair_results.append(pair_result)
        delta_h_by_frag[I] += pair_result.delta_h_I
        delta_h_by_frag[J] += pair_result.delta_h_J
        E_pt2_total += pair_result.E_pt2
        n_terms_total += pair_result.n_terms
        max_abs_M_global = max(max_abs_M_global, pair_result.max_abs_M)
        if pair_result.min_gap is not None:
            if min_gap_global is None or pair_result.min_gap < min_gap_global:
                min_gap_global = pair_result.min_gap

    max_abs_delta_h = max(
        (float(np.max(np.abs(delta_h))) for delta_h in delta_h_by_frag.values()),
        default=0.0,
    )

    return AllPairCouplingResult(
        E_pt2_cross_total=E_pt2_total,
        pair_results=pair_results,
        delta_h_by_frag=delta_h_by_frag,
        min_gap_global=min_gap_global,
        max_abs_M_global=max_abs_M_global,
        max_abs_delta_h=max_abs_delta_h,
        n_terms_total=n_terms_total,
    )


def apply_damped_delta_h(
    delta_h_new: np.ndarray,
    delta_h_prev: Optional[np.ndarray],
    damping: float,
) -> np.ndarray:
    if delta_h_prev is None:
        return damping * delta_h_new
    return damping * delta_h_new + (1.0 - damping) * delta_h_prev
