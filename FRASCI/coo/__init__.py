"""
FRASCI.coo
=====================

Wrappers around TrimCI's Core-Optimized Orbitals (COO) machinery.

Paper: Zhang & Otten, "Absorbing Many-Body Correlations into Core-Optimized
Orbitals", arXiv:2605.22977 (2026).

Algorithm in one paragraph
--------------------------
COO co-optimizes a sparse selected-CI wavefunction with the underlying
orbital basis.  At each outer iteration:

  1.  Solve TrimCI in the current orbital basis  ->  ``(dets, coeffs)``.
  2.  Hold the determinant set fixed and minimize  ``E[kappa, c] = <Psi|H[U]|Psi>``
      jointly over the CI coefficients ``c`` and the antisymmetric matrix
      ``kappa`` that parameterises ``U = exp(kappa)``.  BFGS over kappa; at
      every line-search point the CI block is re-diagonalised so ``c`` is
      always variationally optimal for the trial orbitals.
  3.  Apply ``U`` to the integrals:  ``h_new = U^T h U``,
      ``V_new = U^T (4-index transform) V``.
  4.  Re-run TrimCI in the new basis (the rotation typically opens up
      better, more compact determinant sets) and loop.

For Fe4S4 the published result is ``~1e9`` determinants reaching FCI quality
that LMOs need ``~3e14`` for -- the basis absorbs most of the dynamical
correlation that LMOs leak into long Slater-determinant tails.

What lives here
---------------
* ``coo_adapter`` -- thin Python wrappers (no new physics) around
  ``trimci.run_full_calculation`` and ``trimci.orblab.OrbitalOptimizer``.
  Exposes one manual outer loop (``run_outer_loop``) for clear,
  step-by-step inspection inside notebooks, and one passthrough to the
  built-in driver (``run_end_to_end``) for production runs.
* ``runners/run_coo.py`` -- CLI runner mirroring the LASSCF runners style.

The original ``core/trimci_adapter.py`` is read-only and untouched.
"""

from FRASCI.coo.coo_adapter import (
    COOCycle,
    COOOuterLoopResult,
    BaselineResult,
    run_baseline,
    run_outer_loop,
    run_end_to_end,
    silent,
)

__all__ = [
    "COOCycle",
    "COOOuterLoopResult",
    "BaselineResult",
    "run_baseline",
    "run_outer_loop",
    "run_end_to_end",
    "silent",
]
