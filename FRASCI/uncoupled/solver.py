"""
uncoupled/solver.py
===================
Fragmented TrimCI solve with no inter-fragment coupling.

Each fragment sees only its own bare integrals — no knowledge of the electron
density in neighbouring fragments. This is the simplest fragmentation scheme
and establishes the determinant-count baseline.

Fe4S4 result: 3 fragments, total 118 determinants vs 10,095 brute-force (1.2%).
Regression-locked: must always return total_dets=118, fragment_n_dets=[51,51,16].
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from FRASCI.core.results import FragmentedRunResult
from FRASCI.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from FRASCI.core.trimci_adapter import solve_fragment_trimci


def run_fragmented_trimci(
    fcidump_path: str,
    window_size: int = 15,
    stride: int = 10,
    trimci_config: Optional[dict] = None,
) -> FragmentedRunResult:
    """
    Fragment-and-solve without inter-fragment coupling.

    Steps
    -----
    1. Read FCIDUMP (h1, eri in chemist notation)
    2. Sort orbitals by h1 diagonal (orbital energy)
    3. Create overlapping fragments via sliding window
    4. For each fragment: extract bare integrals, count electrons, run TrimCI
    5. Return aggregated result with per-fragment energies and det counts

    Parameters
    ----------
    fcidump_path  : path to FCIDUMP file (e.g. fcidump_cycle_6)
    window_size   : orbitals per fragment (default 15)
    stride        : sliding window stride (default 10)
    trimci_config : optional TrimCI config overrides

    Notes
    -----
    - Do NOT sum fragment energies — double-counting makes it meaningless.
      Compare total_dets vs brute_force_dets (10,095) instead.
    - Fe4S4: E_nuc = 0.0 (already absorbed in FCIDUMP). Do not add it.
    - dets.npz row 0 is used as the reference determinant for electron counting.
      HF fallback gives 1-det fragments; always prefer dets.npz.
    """
    import trimci

    h1, eri, n_elec, n_orb, E_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(fcidump_path)
    order = np.argsort(np.diag(h1))

    _dets_path = os.path.join(os.path.dirname(os.path.abspath(fcidump_path)), "dets.npz")
    if os.path.exists(_dets_path):
        _ref_data = np.load(_dets_path)
        ref_alpha_bits = int(_ref_data["dets"][0, 0])
        ref_beta_bits  = int(_ref_data["dets"][0, 1])
    else:
        ref_alpha_bits = int(sum(1 << int(order[i]) for i in range(n_alpha)))
        ref_beta_bits  = int(sum(1 << int(order[i]) for i in range(n_beta)))

    fragments = fragment_by_sliding_window(n_orb, order, window_size, stride)

    fragment_energies = []
    fragment_n_dets   = []
    fragment_orbs_out = []

    for frag_orbs in fragments:
        na, nb = fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag_orbs)
        n_frag = len(frag_orbs)
        if na == 0 or nb == 0 or na > n_frag or nb > n_frag:
            print(f"  Skipping fragment {frag_orbs[:3]}... "
                  f"(n_alpha={na}, n_beta={nb}, n_orb={n_frag})")
            continue

        h1_f, eri_f = extract_fragment_integrals(h1, eri, frag_orbs)
        result = solve_fragment_trimci(h1_f, eri_f, na, nb, n_frag, trimci_config)

        fragment_energies.append(result.energy)
        fragment_n_dets.append(result.n_dets)
        fragment_orbs_out.append(frag_orbs)
        print(f"  Fragment orbs {frag_orbs[0]}..{frag_orbs[-1]}: "
              f"n_dets={result.n_dets}, energy={result.energy:.4f} Ha")

    return FragmentedRunResult(
        fragment_energies=fragment_energies,
        fragment_n_dets=fragment_n_dets,
        fragment_orbs=fragment_orbs_out,
        total_dets=sum(fragment_n_dets),
    )
