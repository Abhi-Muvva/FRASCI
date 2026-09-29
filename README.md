# FRASCI

FRASCI (**FRA**gment-based **S**elected **CI**) combines a phased TrimCI–LASSCF parent
with fixed-parent-frame interfragment selected-CI recovery. This document describes the
framework as shipped in this checkout.

## 1. Status and dependency notice

This is a **framework release**, not a comprehensive results release. It ships the
current parent and production recovery code, a focused test suite, and four example
notebooks (see §9) with real, gate-verified runs on several systems — it does not claim
a comprehensive scientific benchmark suite across the full molecule registry (see §8).

FRASCI's recovery layer (`frasci.recovery`) requires native TrimCI capabilities —
`RecoveryPolicy`, sector constraints (`make_sector_constraint_64`),
`ExpansionConfig.allowed_orbital_pairs`, and reproducible-CPU-RNG restart seeding — that
do not exist in a stock TrimCI release. They exist only in the FRASCI-specific TrimCI
fork, published at <https://github.com/Abhi-Muvva/TrimCI-FRASCI>. `pyproject.toml` and
`requirements.txt` pin `trimci` to that fork's exact published commit (not a branch, and
not the stock `trimci` package) — that repository is private, so installing FRASCI
requires GitHub credentials with access to that repository.

The newer fixed-top-K **contracted-LAS COO** experiment (a tied-amplitude K-term ansatz
with an analytic orbital gradient) is **not included** in this release. It remains under
independent scientific validation in the development tree and is intentionally excluded
here. Nothing in the shipped `frasci.recovery` package imports, exposes a flag for, or
returns a result field belonging to that experiment.

## 2. Inputs

FRASCI takes three things:

- **An FCIDUMP file** — the one- and two-electron integrals, electron count, and scalar
  core energy of a physical active space, in an already-fixed molecular-orbital basis.
  `FCIDUMPInput.from_file(path)` reads its orbital/electron-count metadata.
- **Orbital blocks** (`orbital_lists`) — a partition of the FCIDUMP's orbital indices
  into disjoint fragments, e.g. `[[0, 1], [2, 3]]` for a 4-orbital system cut into two
  2-orbital fragments. Every orbital must belong to exactly one fragment.
- **Electrons and spin per fragment** (`nelec_per_frag`, `spin_sub`) — how many alpha and
  beta electrons live in each fragment's orbitals, and each fragment's spin multiplicity
  (`spin_sub[i] = |n_alpha_i - n_beta_i| + 1`). These must sum to the FCIDUMP's own total
  electron count.

The partition is a physical/chemical choice the caller makes (e.g. one fragment per
transition-metal site); FRASCI does not choose it automatically.

## 3. The parent

A large active space is expensive to solve as one block. The parent instead splits it
into the orbital fragments above, solves each fragment's local determinant space with
selected CI (TrimCI), and uses LASSCF to make the fragments' orbitals and embedding
mutually self-consistent — each fragment sees a mean-field-like potential from the
others, iterated to convergence. Optionally, orbital rotations (COO) inside each
fragment's own block compress that fragment's selected-CI description further, without
mixing orbitals across fragments during this step.

The converged result is a single antisymmetrized product of fragment wavefunctions, the
**LAS reference**:

```
|Phi_LAS> = A * prod_I |Phi_I>
```

where `A` antisymmetrizes the product of the per-fragment states `|Phi_I>` so the whole
thing is a valid many-electron wavefunction. `frasci.parent.solve(...)` returns this as
an in-memory `Parent` (one `FragmentResult` per fragment: its determinants, coefficients,
and orbital rotation) and can additionally persist it to disk (`save_dets_to=...`) for
`frasci.recovery.solve` to read back later.

The parent by itself has no representation of correlation *between* fragments — that is
what recovery adds.

**Optional: comparing against mrh LASSIS.** `frasci.parent.build_lassis_ready_las(parent,
fcidump_path, *, max_dense_elements=10_000_000)` reconstructs a `.ci`-populated LAS object
from a converged `Parent`, suitable for `mrh.my_pyscf.lassi.lassis.LASSIS(...)`, without
re-running any TrimCI kernel. This exists because `frasci.parent.solve()`'s own LASSCF
backend keeps only fragment RDMs (never CI vectors, for memory efficiency), while LASSIS
needs explicit dense CI arrays; the adapter derives every partition field directly from
`Parent` and preflights each fragment's dense CI dimension so an infeasibly large fragment
(most transition-metal-cluster partitions, at scale) fails loudly before allocating
anything, rather than silently exhausting memory. It rejects a nonconverged parent by
default (`allow_nonconverged=True` opts out, e.g. for an explicitly-labeled diagnostic).
It does not make LASSIS itself tractable on a large system — that is bounded by each
fragment's own dense CI size, independent of this adapter.

## 4. Recovery

Recovery starts from the converged parent and holds its orbital frame **fixed** — no
further orbital rotation, and no re-running of Phase-0 discovery — while it protects the
most important LAS product determinants and grows a bounded set of *interfragment*
determinants: dispersion terms (same total charge per fragment, different spin
arrangement) and charge-transfer terms up to a configurable excitation rank
(`RecoveryConfig.n_ct_max`). Same-fragment-sector determinants that duplicate what
`|Phi_LAS>` already represents are excluded from this growth by construction.

`|Phi_LAS>` stays factorized (never expanded into an explicit list of determinants) and
is combined with the grown determinant set `R` into one nonorthogonal generalized
eigenproblem:

```
|Psi_rec> = c_0 |Phi_LAS> + sum_{D in R} c_D |D>,        H c = E S c
```

`S` is not the identity here because `|Phi_LAS>` and the explicit determinants in `R` are
not mutually orthogonal — some determinants in `R` can have nonzero overlap with the
factorized parent. `frasci.recovery.solve(...)` builds and solves this generalized
problem (dense `scipy.linalg.eigh` or a matrix-free iterative solve, selectable via
`RecoveryConfig.solver_kind`) and returns the result as `Result.energy`.

## 5. Two independently gated refinements

Beyond the generalized `{Phi_LAS, R}` solve, two further, separately-reported
refinements are always attempted:

- **Exact explicit union.** `|Phi_LAS>` is expanded into its full Cartesian product of
  fragment determinants (certified against the parent's own energy first) and combined
  with `R` into one ordinary (not generalized) Hermitian eigenproblem over the union
  `P_LAS ∪ R`, where every coefficient — including the ones that came from `|Phi_LAS>` —
  is independently re-optimized. This is exact but scales with the full fragment
  Cartesian product size, so it is only tractable up to a few hundred thousand
  determinants.
- **Top-K free-orbital COO.** The same union, but with `|Phi_LAS>` truncated to its
  top-K product terms by coefficient magnitude (`RecoveryConfig.las_top_k`), on which
  orbitals are then rotated **freely across fragment boundaries** (not restricted to
  intra-fragment pairs) to minimize the Rayleigh quotient:

  ```
  E_COO = min_{c, kappa}  <Psi(c, kappa)| H |Psi(c, kappa)> / <Psi(c, kappa)|Psi(c, kappa)>
  ```

  where `kappa` parameterizes the orbital rotation and the allowed orbital pairs for that
  rotation are supplied through the FRASCI-specific TrimCI extension
  (`ExpansionConfig.allowed_orbital_pairs`). The energy and coefficients this returns are
  never taken from TrimCI's own paired output — FRASCI always independently
  rediagonalizes at the rotated frame before reporting `free_coo_energy`.

## 6. Public API example

```python
from pathlib import Path

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery import RecoveryConfig, solve as solve_recovery

fcidump = Path("/path/to/FCIDUMP")
system = FCIDUMPInput.from_file(fcidump)
parent_dir = Path("parent_state")

parent = solve_parent(
    fcidump_path=str(fcidump),
    orbital_lists=[[0, 1], [2, 3]],
    nelec_per_frag=[(1, 1), (1, 1)],
    spin_sub=[1, 1],
    config=ParentConfig(threshold=1e-4, max_cycle_macro=50),
    save_dets_to=parent_dir,
)

result = solve_recovery(
    parent_dir,
    system,
    RecoveryConfig(recovery_target=5000),
)
print(result.energy, result.union_energy, result.free_coo_energy)
assert result.hybrid_produced, result.diagnostics
```

The partition and electron counts above are illustrative; they must match the supplied
physical FCIDUMP in real use. There is no `RebuildConfig`, no `phase2_dets` field on
`RecoveryConfig`, and no `discover_phase0`/`solve_from_phase0` functions in this API —
those belonged to an earlier, now-removed recovery implementation.

## 7. Output interpretation

`frasci.recovery.solve` returns one `Result` with three distinct energies, each
belonging to a different variational space — never interchange them:

- **`Result.energy`** — the generalized `{Phi_LAS, R}` solve (§4). `Phi_LAS` stays
  factorized here.
- **`Result.union_energy`** — the exact explicit `P_LAS ∪ R` solve (§5), where every
  coefficient is independently variable. Also exposes `union_dets`,
  `union_coefficients`, `union_n_dets`, `union_residual_norm`.
- **`Result.free_coo_energy`** — the top-K explicit-union, freely orbital-optimized,
  independently rediagonalized result (§5). Also exposes `free_coo_n_dets`,
  `free_coo_gain_mha` (the energy lowering from the orbital rotation, in mHa).

`Result.converged` and `Result.gates` report whether the generalized `{Phi_LAS, R}`
solve's own variational/residual gates passed — this is a correctness signal, not a
growth-completeness signal. Whether the requested `RecoveryConfig.recovery_target`
determinant count was actually reached is a *separate* fact, in
`Result.diagnostics["target_met"]`/`Result.diagnostics["n_recovery"]`; a solve can
converge cleanly while still reporting `target_met=False` if the classified candidate
pool ran out before reaching the target. `Result.h00` is the parent's own `|Phi_LAS>`
energy, useful as the baseline every one of the above energies is measured against
(all three are guaranteed `<= h00` up to solver tolerance).

## 8. Current validation status

This checkout proves the framework — the parent, the fixed-frame recovery solve, the
exact union, and the free-orbital COO refinement — is present, importable in isolation
from the development tree, and covered by a focused unit/component test suite (see
`tests/`). Beyond the unit tests, the four example notebooks (§9) run the real framework
end to end on seven physical systems, with variational/residual gates checked at every
reported point and machine-precision H00 certification against `parent.e_tot` where
applicable. It does **not** claim that every physical system in the wider molecule
registry converges under this release's default settings, that every `recovery_target` is
reachable in general, or that a comprehensive scientific benchmark suite has been
reproduced by this checkout — see each notebook's own scope notes (in particular,
notebook 01's AlFe2(11e,20o) LASSIS-adapter panel is explicitly marked blocked validation,
and notebook 03's Fe2S2 panel is explicitly a non-convergent stress case with no recovery
energies).

## 9. Example notebooks

`notebooks/` contains four self-contained example notebooks. Each starts with the
release-first import bootstrap (§ below) and reads its own static FCIDUMP data from
`notebooks/data/<system>/` (provenance and SHA-256 hashes alongside each FCIDUMP). Raw,
per-point result JSON lives under `notebooks/data/results/<notebook>/`; frozen initial
records are named `*_pilot.json` and kept read-only.

- **`01_parent_recovery_lassis.ipynb`** — three workflows (LASSCF-CAS+LASSIS, phased
  parent+LASSIS via `build_lassis_ready_las`, phased parent+FRASCI recovery) compared
  across AlFe2 (11e,10o and 11e,20o), bisdiazene, and tetrazene.
- **`02_size_extensivity.ipynb`** — parent-energy size extensivity on synthetic (H2)_N
  chains, N=1 through 32, plus a labeled ablation showing why top-K truncation is
  unsafe for systems built from repeated, physically-equivalent fragments.
- **`03_free_coo_bfgs_sweeps.ipynb`** — a full one-factor + combined-best free-COO/
  recovery-target/BFGS-budget sweep on AlFe2 (11e,20o) and the FeS-dimer-3- system
  (published DMRG reference), plus a Fe2S2 non-convergence stress case.
- **`04_fe4s4_stress_test.ipynb`** — the largest system in this release (54e, 36o):
  a certified converged h1diag2 parent (H00 == `parent.e_tot` to machine precision) and
  a bounded three-point recovery ladder against it, with an h1diag3 partition kept as a
  matched-cost comparison point only.



No license file is included because no owner-approved license choice was present in the
source project.
