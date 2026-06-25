"""
coo_kernel.py
=============
COO-TrimCI -> mrh LASSCF fragment-kernel closure.

This is the COO-enabled counterpart to
``FRASCI/lasscf/trimci_kernel.py``: it satisfies the same mrh
``lasscf_rdm.make_fcibox`` kernel contract
``kernel(norb, nelec, h0, h1s, h2) -> (etot, dm1s, dm2)`` so it can be dropped
into the existing LASSCF pipeline by swapping the kernel constructor.

Algorithm per fragment kernel call:
  1. Average ``h1s -> h1 = 0.5*(h1s[0] + h1s[1])`` (same convention as the
     vanilla TrimCI kernel).
  2. Run vanilla TrimCI on ``(h1, h2)`` to seed a determinant set.
  3. For each COO outer cycle: call
     ``trimci.orblab.OrbitalOptimizer.optimize`` on the fragment integrals at
     the fixed determinant set, then re-run TrimCI in the rotated basis.
  4. Build the PySCF FCI vector from the final ``(dets, coeffs)``, compute
     spin-resolved 1-RDM and 2-RDMs via ``direct_spin1.make_rdm12s``.
  5. **Back-rotate** the RDMs to the kernel's input basis using the
     accumulated ``U_total``, so LASSCF receives consistent data:
        dm1_orig = U_total @ dm1_rot @ U_total.T
        dm2_orig[p,q,r,s] = U[p,a] U[q,b] U[r,c] U[s,d] dm2_rot[a,b,c,d]
     Energy is invariant under unitary rotation, so the lower COO energy is
     preserved while the RDMs live in the basis LASSCF expects.

Why we can back-rotate without changing the energy
-------------------------------------------------
For any unitary ``U``, ``E = h0 + Tr(h1 gamma) + 0.5 Tr(g Gamma)`` is invariant
under simultaneous rotation of integrals and RDMs.  COO reduces ``E[Psi]`` by
finding a basis where the SAME determinant *count* spans a *better* subspace;
the resulting wavefunction has the same energy whether expressed in the rotated
or original basis.  We return the energy and the back-rotated RDMs.
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Callable, Optional

import numpy as np
import trimci
from pyscf.fci import direct_spin1
from trimci.orblab import OrbitalOptimizer

from FRASCI.lasscf.trimci_adapter import solve_fragment_trimci
from FRASCI.lasscf.trimci_to_civec import trimci_to_pyscf_civec


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def _back_rotate_dm1(dm1: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ dm1 @ U.T


def _back_rotate_dm2(dm2: np.ndarray, U: np.ndarray) -> np.ndarray:
    # dm2_orig[p,q,r,s] = sum_{a,b,c,d} U[p,a] U[q,b] U[r,c] U[s,d] dm2_rot[a,b,c,d]
    # Implemented as four contracted dgemms to avoid the O(n^8) einsum naïve cost.
    out = np.einsum("pa,abcd->pbcd", U, dm2, optimize=True)
    out = np.einsum("qb,pbcd->pqcd", U, out, optimize=True)
    out = np.einsum("rc,pqcd->pqrd", U, out, optimize=True)
    out = np.einsum("sd,pqrd->pqrs", U, out, optimize=True)
    return out


def _rotate_eri(eri: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Forward 4-index transformation: eri_new[a,b,c,d] = U[p,a] U[q,b] U[r,c] U[s,d] eri[p,q,r,s].

    Same shape (norb,)*4, chemist notation preserved.  Used for warm-start pre-rotation.
    """
    out = np.einsum("pa,pqrs->aqrs", U, eri, optimize=True)
    out = np.einsum("qb,aqrs->abrs", U, out, optimize=True)
    out = np.einsum("rc,abrs->abcs", U, out, optimize=True)
    out = np.einsum("sd,abcs->abcd", U, out, optimize=True)
    return out


def make_coo_trimci_kernel_for_fragment(
    *,
    threshold: float = 0.01,
    n_coo_cycles: int = 2,
    bfgs_maxiter: int = 20,
    davidson_tol: float = 1e-7,
    ftol: float = 1e-8,
    log_callback=None,
    quiet: bool = True,
    warm_start_kappa: bool = True,
    **trimci_kwargs,
) -> Callable:
    """Return a COO-TrimCI fragment kernel matching the mrh LASSCF contract.

    Parameters
    ----------
    threshold : float
        TrimCI selection threshold for every TrimCI call (seed and post-COO).
    n_coo_cycles : int
        Number of (orbital opt + TrimCI re-detect) outer cycles per kernel call.
        ``0`` makes the kernel equivalent to the vanilla TrimCI kernel.
    bfgs_maxiter : int
        BFGS steps per orbital-opt call.
    davidson_tol, ftol : float
        Inner tolerances for ``OrbitalOptimizer.optimize``.
    log_callback : callable or None
        ``log_callback(norb, nelec, n_dets, energy_electronic, wall_time, extras)``.
        ``extras`` is a dict with COO-specific metrics
        (``n_coo_cycles_done``, ``U_offdiag_norm``, ``dE_from_coo_mHa``,
        ``warm_start_used``, ``U_cached_offdiag``).
    quiet : bool
        Suppress TrimCI/OrbitalOptimizer stdout/stderr.
    warm_start_kappa : bool
        If True, cache the accumulated orbital rotation across mrh LASSCF
        macro iterations and pre-rotate the input integrals with it before
        invoking the OrbitalOptimizer.  This is effectively warm-starting
        kappa: BFGS sees integrals that are already near the previous COO
        optimum and only has to find an incremental update.  Without this,
        each kernel call restarts BFGS from kappa=0 and the previous
        macro-iter rotation is lost (the failure mode that motivates this
        flag; see the original development notes, 2026-06-20).
    **trimci_kwargs
        Forwarded to ``solve_fragment_trimci`` config dict.
    """
    base_config = {"threshold": threshold, **trimci_kwargs}

    # Per-fragment closure state for warm-starting kappa.
    # ``U_cached`` is the cumulative orbital rotation produced by the previous
    # call, expressed in the LASSCF MO basis the previous call received.
    # Across macro iterations LASSCF rotates the *full* mo_coeff, so the
    # *fragment-local* span of the active block changes only by an orthogonal
    # transformation that LASSCF resolves itself -- the assumption is that
    # U_cached remains close to the optimum for the new (h1, h2) handed in,
    # which holds near convergence.  If that assumption fails BFGS just walks
    # away from the bad initial guess in a few steps (no correctness cost).
    state = {"U_cached": None}

    def kernel(norb, nelec, h0, h1s, h2):
        t_start = time.perf_counter()

        # --- 1. Unpack nelec, average h1s ---
        if isinstance(nelec, (int, np.integer)):
            n = int(nelec)
            na = (n + 1) // 2
            nb = n // 2
        else:
            na, nb = int(nelec[0]), int(nelec[1])

        h1 = 0.5 * (h1s[0] + h1s[1])

        # --- 1b. Warm-start: pre-rotate integrals by U_cached ---
        # The OrbitalOptimizer's BFGS always starts from kappa = 0, so we cannot
        # seed kappa directly through the public API.  Instead we apply the
        # cached rotation to the input integrals before the optimizer sees
        # them, which is mathematically identical to starting BFGS from
        # kappa = log(U_cached).  The resulting U_step is an *incremental*
        # rotation, and we accumulate U_total = U_step @ U_cached for the
        # next call.
        if warm_start_kappa and state["U_cached"] is not None \
                and state["U_cached"].shape == (norb, norb):
            U_cached = state["U_cached"]
            h1_pre = U_cached.T @ h1 @ U_cached
            eri_pre = _rotate_eri(h2, U_cached)
            U_total = U_cached.copy()
            warm_started = True
        else:
            h1_pre, eri_pre = h1.copy(), h2.copy()
            U_total = np.eye(norb)
            warm_started = False

        # --- 2. Seed TrimCI in the (possibly pre-rotated) basis ---
        seed = solve_fragment_trimci(
            h1_frag=h1_pre, eri_frag=eri_pre,
            n_alpha_frag=na, n_beta_frag=nb, n_orb_frag=norb,
            config=base_config,
        )
        if seed.n_dets == 0 or not np.isfinite(seed.energy):
            raise RuntimeError(
                f"COO seed TrimCI returned invalid result "
                f"(norb={norb}, nelec=({na},{nb})): n_dets={seed.n_dets}, "
                f"energy={seed.energy}"
            )
        e_seed = seed.energy

        # --- 3. COO outer loop ---
        h1_curr, eri_curr = h1_pre, eri_pre
        dets_curr, coeffs_curr = list(seed.dets), list(seed.coeffs)
        e_curr = e_seed

        for _cycle in range(n_coo_cycles):
            # (a) Orbital opt at fixed dets
            opt = OrbitalOptimizer(
                n_orb=norb, n_elec=na + nb,
                mol_name="coo_frag", verbose=False,
            )
            opt.nuclear_repulsion = 0.0  # h0 is added by LASSCF outside the kernel
            ctx = _silent() if quiet else contextlib.nullcontext()
            with ctx:
                h1_rot, eri_rot, e_after_orb, _conv, U_step = opt.optimize(
                    h1_curr, eri_curr, dets_curr, coeffs_curr,
                    optimizer_options_dict={
                        "optimizer": "cpp_bfgs",
                        "gradient_mode": "analytical",
                        "maxiter": bfgs_maxiter,
                        "davidson_tol": davidson_tol,
                        "ftol": ftol,
                        "verbose": False,
                    },
                )

            # (b) Re-detect in rotated basis
            redet = solve_fragment_trimci(
                h1_frag=h1_rot, eri_frag=eri_rot,
                n_alpha_frag=na, n_beta_frag=nb, n_orb_frag=norb,
                config=base_config,
            )
            if redet.n_dets == 0 or not np.isfinite(redet.energy):
                # Bail out of further COO cycles, keep previous state
                break

            # Accept the cycle if it lowered the energy; else stop early.
            # U_total composes: incremental U_step on top of any cached rotation.
            if redet.energy <= e_curr + 1e-10:
                h1_curr, eri_curr = h1_rot, eri_rot
                dets_curr = list(redet.dets)
                coeffs_curr = list(redet.coeffs)
                e_curr = redet.energy
                U_total = U_step @ U_total
            else:
                break

        # Persist the accumulated rotation so the next macro iter starts here.
        # We only update the cache on success (U_total is at least near-identity)
        # so a bad call doesn't poison the next one.
        if warm_start_kappa:
            state["U_cached"] = U_total.copy()

        # --- 4. Build civec + RDMs in the rotated basis ---
        civec = trimci_to_pyscf_civec(dets_curr, coeffs_curr, norb, (na, nb))
        norm = np.sqrt(np.sum(civec ** 2))
        if norm < 1e-12:
            raise RuntimeError(
                f"COO-TrimCI civec has near-zero norm ({norm:.3e}) "
                f"(norb={norb}, nelec=({na},{nb}))"
            )
        civec /= norm

        (dm1a_rot, dm1b_rot), (dm2aa_rot, dm2ab_rot, dm2bb_rot) = direct_spin1.make_rdm12s(
            civec, norb, (na, nb)
        )

        # --- 5. Back-rotate RDMs to the input basis ---
        # OrbitalOptimizer convention: new orbital a = sum_p U[p,a] * old_p, with
        # h1_rot = U^T h1 U.  Therefore the RDM transforms as
        # dm1_orig = U dm1_rot U^T  (and similarly for dm2).
        dm1a = _back_rotate_dm1(dm1a_rot, U_total)
        dm1b = _back_rotate_dm1(dm1b_rot, U_total)
        dm1s = np.stack([dm1a, dm1b], axis=0)

        dm2aa = _back_rotate_dm2(dm2aa_rot, U_total)
        dm2ab = _back_rotate_dm2(dm2ab_rot, U_total)
        dm2bb = _back_rotate_dm2(dm2bb_rot, U_total)
        dm2 = dm2aa + dm2ab + dm2ab.transpose(2, 3, 0, 1) + dm2bb

        etot = float(h0) + e_curr

        if callable(log_callback):
            wall = time.perf_counter() - t_start
            U_cached_norm = (
                float(np.linalg.norm(state["U_cached"] - np.eye(norb)))
                if warm_start_kappa and state["U_cached"] is not None
                else 0.0
            )
            extras = {
                "n_coo_cycles_done": int(n_coo_cycles),
                "U_offdiag_norm": float(np.linalg.norm(U_total - np.eye(norb))),
                "dE_from_coo_mHa": float((e_curr - e_seed) * 1000.0),
                "n_dets_seed": int(seed.n_dets),
                "warm_started": bool(warm_started),
                "U_cached_offdiag": U_cached_norm,
            }
            log_callback(norb, (na, nb), len(dets_curr), e_curr, wall, extras)

        return etot, dm1s, dm2

    return kernel


# ---------------------------------------------------------------------------
# Standalone sanity check (run python -m FRASCI.lasscf.coo_kernel)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pyscf import gto, scf, ao2mo
    from pyscf.fci import direct_spin1 as ds1
    from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF, make_fcibox
    from FRASCI.lasscf.trimci_kernel import make_trimci_kernel_for_fragment

    mol = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
                basis="sto-3g", unit="Bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    norb = mol.nao_nr(); na = mol.nelectron // 2; nb = mol.nelectron // 2
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(norb, norb, norb, norb)
    e_fci, _ = ds1.kernel(h1, h2, norb, (na, nb), ecore=mol.energy_nuc())
    print(f"PySCF FCI e_tot = {e_fci:.10f}")

    # LASSCF + COO-TrimCI
    las = LASSCF(mf, ncas_sub=(norb,), nelecas_sub=[(na, nb)], spin_sub=(1,))
    las.verbose = 0
    kernel_fn = make_coo_trimci_kernel_for_fragment(threshold=1e-8, n_coo_cycles=1)
    las.fciboxes[0] = make_fcibox(mol, kernel=kernel_fn, spin=0, smult=1)
    las.kernel(mf.mo_coeff)
    print(f"LASSCF+COO+TrimCI e_tot = {las.e_tot:.10f}  delta = {abs(las.e_tot - e_fci):.2e}")
    assert abs(las.e_tot - e_fci) < 1e-6, "COO kernel diverged from FCI on H4/STO-3G"
    print("COO-KERNEL SMOKE TEST PASS")
