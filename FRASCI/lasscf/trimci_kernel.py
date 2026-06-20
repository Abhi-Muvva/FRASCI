"""
trimci_kernel.py
================
TrimCI -> mrh LASSCF fragment-kernel closure.

mrh contract (lasscf_rdm.make_fcibox kernel): kernel(norb, nelec, h0, h1s, h2)
must return (etot, dm1s, dm2_spinsum) where
  - etot   : float, total fragment energy (h0 + electronic)
  - dm1s   : ndarray (2, norb, norb), [alpha, beta] 1-RDMs
  - dm2    : ndarray (norb,)*4, spin-summed 2-RDM in Mulliken order

Important mrh note (from lasscf_rdm.py:RDMSolver.kernel):
  mrh calls ``h2 = ao2mo.restore(1, h2, norb)`` on h2 before dispatching to the
  user kernel, so h2 is always fully-restored 4D shape (norb,)*4 by the time our
  closure receives it.

h1 averaging (from lasscf_rdm.py §6.2, design doc):
  TrimCI takes a single restricted h1[norb, norb].  mrh gives h1s[2, norb, norb].
  We use h1 = 0.5 * (h1s[0] + h1s[1]); this is exact for closed-shell (h1s[0] ==
  h1s[1]) and a minimal-spin-contamination approximation for open-shell.

dm2 spin-sum (confirmed vs lasscf_rdm.py:_ci2rdm, line 400):
  dm2 = dm2[0] + dm2[1] + dm2[1].transpose(2,3,0,1) + dm2[2]
  where indices 0,1,2 correspond to aa, ab, bb from direct_spin1.make_rdm12s.
  Our formula: dm2 = dm2aa + dm2ab + dm2ab.transpose(2,3,0,1) + dm2bb  (identical).

Strategy: average h1s -> restricted h1, hand h1 and h2 to TrimCI via
solve_fragment_trimci, decode the resulting sparse (dets, coeffs) into a PySCF FCI
vector via trimci_to_pyscf_civec, then call pyscf.fci.direct_spin1.make_rdm12s
to get (dm1a, dm1b), (dm2aa, dm2ab, dm2bb).
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from pyscf.fci import direct_spin1

from FRASCI.lasscf.trimci_adapter import solve_fragment_trimci
from FRASCI.lasscf.trimci_to_civec import trimci_to_pyscf_civec


def make_trimci_kernel_for_fragment(
    *,
    threshold: float = 0.01,
    log_callback=None,
    **trimci_kwargs,
) -> Callable:
    """Return a closure with signature kernel(norb, nelec, h0, h1s, h2) -> (etot, dm1s, dm2).

    The closure is stateless and can be reused across LASSCF outer iterations.

    Parameters
    ----------
    threshold : float
        TrimCI selection threshold (passed as config['threshold']).
        Use 1e-6 or smaller for near-exact results; 0.01 for production Fe4S4.
    log_callback : callable or None
        If provided, called once per kernel invocation with
        (norb, nelec, n_dets, energy_electronic, wall_time_sec).
        No-op by default.
    **trimci_kwargs
        Additional TrimCI config overrides forwarded to solve_fragment_trimci as
        the 'config' dict (e.g. max_rounds=3, max_final_dets=2000).

    Returns
    -------
    kernel : callable
        Closure satisfying the mrh lasscf_rdm.make_fcibox kernel contract:
        kernel(norb, nelec, h0, h1s, h2) -> (etot, dm1s, dm2).
    """
    # Build TrimCI config dict from threshold + any forwarded kwargs
    trimci_config = {"threshold": threshold, **trimci_kwargs}

    def kernel(norb, nelec, h0, h1s, h2):
        """
        mrh LASSCF fragment kernel backed by TrimCI.

        Parameters
        ----------
        norb : int
        nelec : tuple(int, int) or int
        h0 : float
            Inactive-core energy + nuclear repulsion.  Added to returned etot.
        h1s : ndarray, shape (2, norb, norb)
            Spin-resolved 1e integrals.  h1s[0]=alpha, h1s[1]=beta.
        h2 : ndarray, shape (norb, norb, norb, norb)
            2e integrals in Mulliken order, fully restored (ao2mo.restore(1,...)).
            Note: mrh's RDMSolver.kernel already calls ao2mo.restore(1,...) before
            dispatching here, so h2 is always 4D on entry.

        Returns
        -------
        etot : float
        dm1s : ndarray, shape (2, norb, norb)
        dm2  : ndarray, shape (norb, norb, norb, norb)
            Spin-summed in Mulliken order: aaaa + aabb + bbaa + bbbb.
        """
        t_start = time.perf_counter()

        # --- 1. Unpack nelec ---
        if isinstance(nelec, (int, np.integer)):
            n = int(nelec)
            na = (n + 1) // 2
            nb = n // 2
        else:
            na, nb = int(nelec[0]), int(nelec[1])

        # --- 2. Average spin-resolved h1s to restricted h1 ---
        # Exact for closed-shell (h1s[0] == h1s[1]).
        # Safe minimal-spin-contamination approximation for open-shell.
        h1 = 0.5 * (h1s[0] + h1s[1])

        # --- 3. Call TrimCI ---
        result = solve_fragment_trimci(
            h1_frag=h1,
            eri_frag=h2,
            n_alpha_frag=na,
            n_beta_frag=nb,
            n_orb_frag=norb,
            config=trimci_config,
        )

        # --- 4. Sanity guard ---
        if result.n_dets == 0 or not np.isfinite(result.energy):
            raise RuntimeError(
                f"TrimCI returned invalid result for fragment "
                f"(norb={norb}, nelec=({na},{nb})): "
                f"n_dets={result.n_dets}, energy={result.energy}. "
                f"Check TrimCI config and input integrals."
            )

        # --- 5. Decode sparse (dets, coeffs) into dense PySCF FCI vector ---
        civec = trimci_to_pyscf_civec(result.dets, result.coeffs, norb, (na, nb))

        # Normalize civec (TrimCI should return near-unit-norm, but guard against
        # truncation effects that shift the norm away from 1.0)
        norm = np.sqrt(np.sum(civec ** 2))
        if norm < 1e-12:
            raise RuntimeError(
                f"TrimCI civec has near-zero norm ({norm:.3e}) for "
                f"(norb={norb}, nelec=({na},{nb})). "
                f"n_dets={result.n_dets}."
            )
        civec_norm = civec / norm

        # --- 6. Compute RDMs via PySCF ---
        # make_rdm12s returns ((dm1a, dm1b), (dm2aa, dm2ab, dm2bb))
        # All in Mulliken (chemist) order: dm1[p,q] = <p†q>, dm2[p,q,r,s] = <p†q r†s> ...
        # See pyscf.fci.direct_spin1.make_rdm12s for exact convention.
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = direct_spin1.make_rdm12s(
            civec_norm, norb, (na, nb)
        )
        dm1s = np.stack([dm1a, dm1b], axis=0)  # shape (2, norb, norb)

        # Spin-summed 2-RDM per lasscf_rdm.py:_ci2rdm line 400:
        #   dm2 = dm2[0] + dm2[1] + dm2[1].transpose(2,3,0,1) + dm2[2]
        # where dm2[0]=aaaa, dm2[1]=aabb, dm2[2]=bbbb.
        dm2 = dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb  # shape (norb,)*4

        # --- 7. Total energy: h0 (core/nuclear) + TrimCI electronic energy ---
        etot = float(h0) + result.energy

        # --- 8. Optional logging side-channel ---
        if callable(log_callback):
            wall_time = time.perf_counter() - t_start
            log_callback(norb, (na, nb), result.n_dets, result.energy, wall_time)

        return etot, dm1s, dm2

    return kernel
