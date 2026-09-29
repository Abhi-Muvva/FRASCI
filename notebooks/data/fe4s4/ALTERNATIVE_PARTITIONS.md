# Fe4S4 alternative three-fragment partitions (2026-09-28)

Two new, deterministic 3x12-orbital partitions of the Fe4S4 (54e,36o) active space,
independent of the historical `h1diag3` partition (which is excluded entirely from
this construction -- confirmed the resulting orbital lists below are distinct from it).
Full data: `alternative_three_fragment_partitions.json`.

## Reference occupation (never an HF assumption)

Orbitals are split into 27 "reference-occupied" and 9 "reference-virtual" using the
total (alpha+beta) natural-occupation-number diagonal of the certified `h1diag2` parent's
own correlated 1-RDM, per fragment (`make_selected_rdm12s` on each fragment's converged
`(dets, coeffs)`), transformed back into the FCIDUMP's original (unrotated) orbital
indexing via that fragment's own rotation `R` (`dm1_orig_block = R @ dm1_local @ R.T`).
"Occupied" = the 27 orbitals with the highest resulting occupation; "virtual" = the
remaining 9.

**Caveat, stated plainly:** this cluster's correlated reference does *not* show a clean
doubly-occupied/empty (occ~2 / occ~0) split. 19 orbitals sit near occ 1.84-2.0, but the
rest -- including all 9 orbitals labeled "virtual" here -- cluster tightly near occ~1.0
(open-shell/radical character, consistent with a high-spin Fe-S cluster). Only one
orbital (index 13, occ=0.446) is clearly low. The rank-27/28 boundary sits in a
congested, nearly-flat region of the occupation spectrum (occ=1.0036 at rank 27 vs
1.0030 at rank 28, a gap of only 0.0006) -- deterministic and reproducible for this
exact parent instance, but not a robust physical gap. "Occupied"/"virtual" here means
top-27/bottom-9 by this ranking, not a physically crisp doubly-occ/empty split.

**Also note:** the parent checkpoint this is derived from is a *fresh* reproduction of
`h1diag2` run this session (the archived certified checkpoint no longer exists on disk).
It converged with all gates passing and H00 self-consistent, but to
`e_tot=-327.05282363243236 Ha`, 0.34 mHa below the archived `-327.05248402672197 Ha` --
see the phase final report for the reproducibility discussion. This is a real, if small,
downstream dependency of both candidates below.

## h1-diagonal ordering

Within each occupied/virtual group, orbitals are sorted ascending by `h1[i,i]`, the
diagonal one-electron integral (the diagonal-Fock-approximation orbital-energy proxy
already used elsewhere in this FCIDUMP-only pipeline -- see
`docs/superpowers/plans/2026-08-29-frasci-stage4-zero-engine-dependency.md`).

## Candidate 1: `balanced_interleaved3`

Round-robin distribution of the h1-diagonal-ordered occupied group (9 per fragment)
and virtual group (3 per fragment) across the 3 fragments.

- `orbital_lists`: `[[1,7,8,9,11,14,16,20,22,24,29,33], [0,4,6,10,12,15,19,25,27,30,31,32], [2,3,5,13,17,18,21,23,26,28,34,35]]`
- `nelec_per_frag`: `[(9,9), (9,9), (9,9)]`, `spin_sub`: `[1,1,1]`
- Local CI dimension per fragment: `C(12,9)^2 = 48,400` (independently verified via
  `math.comb`, not hardcoded)
- Total intrafragment coupling objective: **33.04**

## Candidate 2: `strong_pair3`

Starts from `balanced_interleaved3`. At each step, considers every deterministic
same-occupancy-class swap (an occupied orbital in fragment A for an occupied orbital
in fragment B, or a virtual-for-virtual swap; never occ-for-virtual, which would break
the 9+3 sector) and applies the single swap that increases total intrafragment coupling
the most (ties broken by lexical `(i,j)` order). Stops when no swap improves the
objective. No random optimizer.

**Coupling metric.** For orbitals `i != j`:

```
coupling[i,j] = |h1[i,j]| + ||eri[i,j,:,:]||_F
```

using chemist-notation two-electron integrals `(ij|kl)`. The two-electron term is the
Frobenius norm of the `eri[i,j,:,:]` slice, `sqrt(sum_kl (ij|kl)^2)` -- symmetric in
`(i,j)` because `(ij|kl) = (ji|kl)` for real orbitals (the bra product `phi_i*phi_j` is
commutative), so no extra symmetrization is needed. The maximized objective is the sum
of `coupling[i,j]` over every same-fragment orbital pair.

- `orbital_lists`: `[[2,3,4,6,7,9,11,12,14,15,16,18], [8,17,19,20,24,25,27,29,31,32,34,35], [0,1,5,10,13,21,22,23,26,28,30,33]]`
- `nelec_per_frag`: `[(9,9), (9,9), (9,9)]`, `spin_sub`: `[1,1,1]`
- Local CI dimension per fragment: `C(12,9)^2 = 48,400`
- 11 deterministic swaps applied (full log in the JSON)
- Total intrafragment coupling objective: **53.68** (+62.5% over `balanced_interleaved3`)

## Coverage / nonoverlap

Both candidates independently verified: every orbital 0-35 assigned to exactly one
fragment, each fragment exactly 12 orbitals (9 occupied-class + 3 virtual-class),
totals 27 alpha + 27 beta electrons.

## Not used

The historical `h1diag3` partition (`docs/reports/large_system_scaling_phase_b_parent_sweep_2026-09-20.py`,
also present in `fe4s4_lowdet_combo_sweep.ipynb`) plays no role in this construction and
was not read for orbital-list values; both candidates above are confirmed distinct from it.
