"""
fragments.py
============
Fragment orbital lists and electron counts for LASSCF on the Fe4S4 FCIDUMP
(36 orbitals, 27α + 27β electrons, E_nuc=0).

h1diag partition (12/12/12)
----------------------------
Orbitals are sorted by the diagonal of the one-electron Hamiltonian h1 from
the FCIDUMP, then divided into three equal contiguous groups of 12.  This is
the same partition used in the best SC MFA result (+0.281 Ha above full TrimCI
reference).

Source: FRASCI/mfa/solver.py::make_nonoverlapping_partition
        FRASCI/crossflow/partition.py::make_equal_nonoverlapping_partition

Per-fragment electron counts are taken from the correlated reference
determinant in data/dets.npz (row 0), exactly as done in
the SC MFA runner (mfa/solver.py::run_mfa_d2 calls load_ref_det then
fragment_electron_count for each fragment).

For the Fe4S4 FCIDUMP at cycle 6 these yield:
    F0: orbs=[2,3,6,7,8,11,24,25,26,27,29,32], na=6,  nb=6
    F1: orbs=[4,5,9,10,12,13,20,21,23,28,31,33], na=9,  nb=9
    F2: orbs=[0,1,14,15,16,17,18,19,22,30,34,35], na=12, nb=12

spin_sub default (1,1,1): singlet per fragment.  This is a working default for
Phase 2 (csf_solver control).  Phase 5 will sweep over multiplicities.

Stub functions strong_pair_fragments() and mi_min_cut_fragments() are placeholders
for Phases 5+ and raise NotImplementedError when called.
"""

from __future__ import annotations

import numpy as np

# Read-only imports — these modules must not be modified.
from FRASCI.lasscf.support import (
    fragment_electron_count,
    load_ref_det,
    make_nonoverlapping_partition,
)


# ---------------------------------------------------------------------------
# Default paths — can be overridden by callers
# ---------------------------------------------------------------------------
_DEFAULT_FCIDUMP = "data/fcidump_cycle_6"
_DEFAULT_DETS    = "data/dets.npz"


def h1diag_fragments(
    fcidump_path: str = _DEFAULT_FCIDUMP,
    dets_path: str = _DEFAULT_DETS,
) -> tuple[list[list[int]], list[tuple[int, int]], list[int]]:
    """
    Return the h1diag 12/12/12 fragment specification for LASSCF.

    Reads the FCIDUMP to get h1 (for orbital ordering) and the dets.npz file
    to get per-fragment electron counts from the reference determinant.  This
    exactly replicates the partition used by the SC MFA runner.

    Parameters
    ----------
    fcidump_path : str
        Path to the FCIDUMP file.
    dets_path : str
        Path to the dets.npz file containing reference determinants.

    Returns
    -------
    orbital_lists : list of 3 lists of int
        Orbital indices for each fragment (0-indexed, sorted ascending).
        Outer list length = 3; each inner list has 12 entries.
    nelec_per_frag : list of 3 tuples (n_alpha, n_beta)
        Electron counts per fragment from the reference determinant (row 0).
    spin_sub : list of 3 ints
        Spin multiplicity (2S+1) per fragment.  Default: [1, 1, 1] (singlet).
    """
    import trimci  # read-only use of TrimCI's FCIDUMP reader

    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(
        fcidump_path
    )

    # h1diag partition: sort orbitals by h1 diagonal, split into 3 equal blocks
    orbital_lists = make_nonoverlapping_partition(h1, n_orb)

    # Per-fragment electron counts from reference determinant row 0
    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_path, row=0)
    nelec_per_frag = []
    for frag_orbs in orbital_lists:
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        nelec_per_frag.append((int(na), int(nb)))

    # Validate total electron count matches FCIDUMP
    total_na = sum(na for na, nb in nelec_per_frag)
    total_nb = sum(nb for na, nb in nelec_per_frag)
    if total_na != n_alpha or total_nb != n_beta:
        raise RuntimeError(
            f"Electron count mismatch after partitioning: "
            f"sum(na)={total_na} vs FCIDUMP n_alpha={n_alpha}, "
            f"sum(nb)={total_nb} vs FCIDUMP n_beta={n_beta}"
        )

    # Default: singlet per fragment (Phase 5 will sweep multiplicities)
    spin_sub = [1, 1, 1]

    return orbital_lists, nelec_per_frag, spin_sub


def strong_pair_fragments(
    fcidump_path: str = _DEFAULT_FCIDUMP,
    dets_path: str = _DEFAULT_DETS,
) -> tuple[list[list[int]], list[tuple[int, int]], list[int]]:
    """
    Return the strong_pair 12/12/12 fragment specification for LASSCF.

    Partition origin
    ----------------
    The orbital lists were produced by the strong_pair clustering scheme and
    are identical across all six SC MFA Gauss-Seidel runs stored under:

        Outputs/mfa/outs_gs_strongpair_*/gs_metadata.json

    Canonical source (most recent run):

        Outputs/mfa/
            outs_gs_strongpair_20260609_152510/gs_metadata.json
            (keys: "fragment_orbs", "partition": "strong_pair")

    All earlier runs (20260531_100318, 20260531_101138, 20260531_102033,
    20260605_105313, 20260605_222218) carry identical orbital lists,
    confirming the partition is deterministic.

    Per-fragment electron counts
    ----------------------------
    Taken from the correlated reference determinant in dets.npz (row 0),
    exactly as done in the SC MFA runner.  Values recorded from
    fragment_electron_count() applied to the reference determinant:

        F0: orbs=[1,4,5,6,9,10,11,13,17,25,32,34],   na=10, nb=8
        F1: orbs=[2,3,7,12,15,16,18,21,22,27,30,35],  na=10, nb=9
        F2: orbs=[0,8,14,19,20,23,24,26,28,29,31,33], na=7,  nb=10

    spin_sub
    --------
    Derived from the minimum spin allowed by |na - nb| per fragment:
        F0: |10-8| = 2  ->  triplet (smult = 3)
        F1: |10-9| = 1  ->  doublet (smult = 2)
        F2: |7-10| = 3  ->  quartet (smult = 4)

    This partition is more spin-polarised than h1diag (all-singlet) or
    mi_min_cut.  Phase 5 will sweep multiplicities; this function returns
    the minimum-spin values as the working default.

    SC MFA baseline energies (for LASSCF comparison)
    -------------------------------------------------
    GS optimiser converged at E_total_diag = -326.7114 Ha (iter 42,
    outs_gs_strongpair_20260609_152510/gs_metadata.json).
    D2 run with SC gamma (strong_pair gamma, h1diag D2 partition):
        E_total = -326.7053 Ha
        (outs_d2_gs_strongpair_h1part_20260609_152510/results.json)
    V2 pre-SC-gamma fixed-gamma result: E = -326.687 Ha.
    Under SC gamma (strong_pair GS): E = -326.7119 Ha.
    Compare with h1diag SC MFA: E = -326.9107 Ha (best SC MFA to date).

    Parameters
    ----------
    fcidump_path : str
        Path to the FCIDUMP file (used only for the total-electron sanity
        check; orbital lists are hardcoded from the SC MFA canonical run).
    dets_path : str
        Path to the dets.npz file (used to compute per-fragment electron
        counts from the reference determinant row 0).

    Returns
    -------
    orbital_lists : list of 3 lists of int
        Orbital indices for each fragment (0-indexed, sorted ascending).
        Sizes: [12, 12, 12].
    nelec_per_frag : list of 3 tuples (n_alpha, n_beta)
        Electron counts per fragment from the reference determinant (row 0).
    spin_sub : list of 3 ints
        Spin multiplicity (2S+1) per fragment: [3, 2, 4].
    """
    # Canonical orbital lists from SC MFA canonical run.
    # Source: Outputs/mfa/
    #   outs_gs_strongpair_20260609_152510/gs_metadata.json
    # Verified identical across all 6 runs (20260531 through 20260609).
    orbital_lists = [
        [1, 4, 5, 6, 9, 10, 11, 13, 17, 25, 32, 34],
        [2, 3, 7, 12, 15, 16, 18, 21, 22, 27, 30, 35],
        [0, 8, 14, 19, 20, 23, 24, 26, 28, 29, 31, 33],
    ]

    # Sanity: orbital lists must partition range(36) exactly.
    flat = sorted(o for frag in orbital_lists for o in frag)
    assert flat == list(range(36)), (
        "strong_pair orbital lists do not cover range(36): "
        f"got {flat!r}"
    )
    # Sanity: fragment sizes must be 12/12/12.
    assert [len(f) for f in orbital_lists] == [12, 12, 12], (
        "strong_pair fragment sizes mismatch: "
        f"got {[len(f) for f in orbital_lists]}"
    )

    # Per-fragment electron counts from reference determinant row 0.
    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_path, row=0)
    nelec_per_frag = []
    for frag_orbs in orbital_lists:
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        nelec_per_frag.append((int(na), int(nb)))

    # Sanity: total electrons must equal (27, 27).
    total_na = sum(na for na, nb in nelec_per_frag)
    total_nb = sum(nb for na, nb in nelec_per_frag)
    assert (total_na, total_nb) == (27, 27), (
        f"strong_pair electron count mismatch: got ({total_na}, {total_nb}), "
        "expected (27, 27)"
    )

    # Spin sub: minimum multiplicity from |na - nb| per fragment.
    # F0: |10-8|=2 -> smult=3; F1: |10-9|=1 -> smult=2; F2: |7-10|=3 -> smult=4
    spin_sub = [
        int(abs(na - nb)) + 1
        for na, nb in nelec_per_frag
    ]

    return orbital_lists, nelec_per_frag, spin_sub


def mi_min_cut_fragments(
    fcidump_path: str = _DEFAULT_FCIDUMP,
    dets_path: str = _DEFAULT_DETS,
) -> tuple[list[list[int]], list[tuple[int, int]], list[int]]:
    """
    Return the MI-minimum-cut 6/12/18 fragment specification for LASSCF.

    Partition origin
    ----------------
    The orbital lists were identified by a min-cut analysis on the mutual-
    information matrix in MI_Partition_Size_Sweep.ipynb (sweep run
    20260607_102501) and are the canonical 3-fragment partition with the
    lowest inter-fragment MI (3.75 nat vs 7.39 nat for h1diag).

    The partition was consumed verbatim by GS_D2_MI_MinCut.ipynb (runs
    20260607_110509 and 20260607_110727) and is stored canonically in:

        Outputs/mfa/
            outs_gs_d2_mi_min_cut_6_12_18_20260607_110509/
                fragments_mi_min_cut_6_12_18.json

    with source tracing back to:

        Outputs/mi_analysis/
            sweep_20260607_102501/sweep_results.json
            (results_3frag entry with sizes=[6,12,18], strategy 'mi_min_cut')

    Per-fragment electron counts
    ----------------------------
    Taken from the correlated reference determinant in dets.npz (row 0),
    exactly as done in the SC MFA runner.  Values recorded from the
    GS_D2_MI_MinCut.ipynb cell 5 output:

        F0: orbs=[0,1,17,31,34,35],                    na=5,  nb=6
        F1: orbs=[2,5,7,11,12,14,15,16,18,20,24,25],  na=10, nb=9
        F2: orbs=[3,4,6,8,9,10,13,19,21,22,23,26,     na=12, nb=12
                  27,28,29,30,32,33]

    spin_sub
    --------
    Derived from the minimum spin allowed by |na - nb| per fragment:
        F0: |5-6| = 1  →  doublet  (smult = 2)
        F1: |10-9| = 1 →  doublet  (smult = 2)
        F2: |12-12| = 0 → singlet  (smult = 1)

    This differs from h1diag_fragments, whose balanced electron counts
    allow all-singlet [1,1,1].  Phase 5 will sweep multiplicities; this
    function returns the minimum-spin values as the working default.

    Parameters
    ----------
    fcidump_path : str
        Path to the FCIDUMP file (used only for the total-electron sanity
        check; orbital lists are hardcoded from the SC MFA canonical run).
    dets_path : str
        Path to the dets.npz file (used to compute per-fragment electron
        counts from the reference determinant row 0).

    Returns
    -------
    orbital_lists : list of 3 lists of int
        Orbital indices for each fragment (0-indexed, sorted ascending).
        Sizes: [6, 12, 18].
    nelec_per_frag : list of 3 tuples (n_alpha, n_beta)
        Electron counts per fragment from the reference determinant (row 0).
    spin_sub : list of 3 ints
        Spin multiplicity (2S+1) per fragment: [2, 2, 1].
    """
    # Canonical orbital lists from SC MFA canonical run.
    # Source: Outputs/mfa/
    #   outs_gs_d2_mi_min_cut_6_12_18_20260607_110509/
    #   fragments_mi_min_cut_6_12_18.json
    orbital_lists = [
        [0, 1, 17, 31, 34, 35],
        [2, 5, 7, 11, 12, 14, 15, 16, 18, 20, 24, 25],
        [3, 4, 6, 8, 9, 10, 13, 19, 21, 22, 23, 26, 27, 28, 29, 30, 32, 33],
    ]

    # Sanity: orbital lists must partition range(36) exactly.
    flat = sorted(o for frag in orbital_lists for o in frag)
    assert flat == list(range(36)), (
        "mi_min_cut orbital lists do not cover range(36): "
        f"got {flat!r}"
    )
    # Sanity: fragment sizes must be 6/12/18.
    assert [len(f) for f in orbital_lists] == [6, 12, 18], (
        "mi_min_cut fragment sizes mismatch: "
        f"got {[len(f) for f in orbital_lists]}"
    )

    # Per-fragment electron counts from reference determinant row 0.
    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_path, row=0)
    nelec_per_frag = []
    for frag_orbs in orbital_lists:
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        nelec_per_frag.append((int(na), int(nb)))

    # Sanity: total electrons must equal (27, 27).
    total_na = sum(na for na, nb in nelec_per_frag)
    total_nb = sum(nb for na, nb in nelec_per_frag)
    assert (total_na, total_nb) == (27, 27), (
        f"mi_min_cut electron count mismatch: got ({total_na}, {total_nb}), "
        "expected (27, 27)"
    )

    # Spin sub: minimum multiplicity from |na - nb| per fragment.
    # F0: |5-6|=1 -> smult=2; F1: |10-9|=1 -> smult=2; F2: |12-12|=0 -> smult=1
    spin_sub = [
        int(abs(na - nb)) + 1
        for na, nb in nelec_per_frag
    ]

    return orbital_lists, nelec_per_frag, spin_sub
