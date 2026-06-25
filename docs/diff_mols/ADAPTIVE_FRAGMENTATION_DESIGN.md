# Adaptive Fragmentation for LASSCF from FCIDUMP and Pilot Correlation Data

## Implementation design, equations, diagnostics, validation plan, and literature basis

**Status:** proposed design  
**Scope:** `FRASCI/diff_mols` with reuse of existing FRASCI graph and fragment-sweep code  
**Primary objective:** generate molecule-dependent LASSCF fragments that retain strong correlation inside fragments while limiting fragment solver cost  
**Near-term deliverable:** an FCIDUMP-only integral-graph candidate generator followed by optional low-cost LASSCF/LASSIS refinement  

---

## 1. Motivation

The `diff_mols` results show that fragmentation is not a minor technical choice. For
tetrazene, the literature-inspired `chem_bond` partition is close to the best observed
energy, whereas atom-wise and energy-diagonal partitions can be tens to hundreds of
milli-Hartree higher. The LASSCF/TrimCI/COO machinery is therefore not the only important
approximation. The partition determines which correlations the product-state LAS ansatz
can represent internally and which correlations must be recovered later through LASSI,
LASSIS, orbital optimization, or another interfragment correction.

The central design principle is

\[
\boxed{
\text{retain strong correlation inside fragments and minimize important correlation
crossing fragment boundaries}
}
\]

subject to solver-cost, electron-count, spin, locality, and reproducibility constraints.

No established paper currently supplies a universally reliable, fully automatic
LASSCF fragment selector. The LAS literature states the physical requirement clearly,
but fragments are still usually chosen chemically. This leaves room for a systematic
candidate-generation and adaptive-refinement procedure.

This document separates the problem into three information levels:

1. **FCIDUMP-only:** integral-based orbital coupling graph.
2. **FCIDUMP + orbital metadata:** chemically/spatially constrained graph using
   `mo_coeff`, geometry, IAO/IBO populations, and orbital centroids.
3. **Pilot correlated calculation:** mutual information, cumulant, LASSIS correction,
   charge-transfer weight, and RDM mismatch.

The first level can be implemented immediately and cheaply. The later levels refine,
rather than replace, the FCIDUMP candidates.

---

## 2. What FCIDUMP contains—and what it does not

A standard active-space FCIDUMP represents

\[
\hat H =
E_\mathrm{const}
+ \sum_{pq,\sigma} h_{pq}\,
  a^\dagger_{p\sigma}a_{q\sigma}
+ \frac{1}{2}
  \sum_{pqrs,\sigma\tau}
  (pq|rs)\,
  a^\dagger_{p\sigma}a^\dagger_{r\tau}
  a_{s\tau}a_{q\sigma},
\]

where this repository stores the two-electron integrals in chemist notation,

\[
(pq|rs)
=
\int\!\!\int
\phi_p(\mathbf r_1)\phi_q(\mathbf r_1)
\frac{1}{r_{12}}
\phi_r(\mathbf r_2)\phi_s(\mathbf r_2)
d\mathbf r_1d\mathbf r_2.
\]

The FCIDUMP header normally provides:

- number of spatial orbitals \(K\);
- total active electrons \(N\);
- \(M_S\), usually through `MS2 = N_\alpha-N_\beta`;
- optional orbital symmetry labels;
- one- and two-electron integrals;
- the constant/core/nuclear contribution.

Therefore,

\[
N_\alpha = \frac{N+\mathrm{MS2}}{2},
\qquad
N_\beta = \frac{N-\mathrm{MS2}}{2}.
\]

### 2.1 FCIDUMP can provide

- orbital-orbital Hamiltonian coupling proxies;
- integral graph strength and cut scores;
- energy-order and spectral-order candidates;
- balanced or cost-constrained graph partitions;
- a deterministic Aufbau-like reference determinant obtained by occupying the
  lowest eigenvectors or lowest diagonal entries of \(h\);
- fragment electron counts derived from that approximate determinant;
- fragment Hilbert-space cost estimates.

### 2.2 FCIDUMP cannot provide by itself

- atom labels or molecular connectivity;
- orbital centroids or spatial locality;
- IAO, IBO, Pipek-Mezey, or Boys populations;
- bonding/antibonding identities;
- correlated one- or two-particle density matrices;
- single-orbital entropy or mutual information;
- LASSI/LASSIS charge-transfer or spin-recoupling weights;
- state-averaged behavior across geometries or electronic states.

Consequently, an FCIDUMP-only partition is an **integral partition**, not automatically a
chemical partition. This distinction must remain explicit in names, metadata, and plots.

---

## 3. Existing implementation that should be reused

The repository already contains most of the low-level machinery:

### 3.1 Integral coupling graph

File:

```text
FRASCI/lasscf/fragment_sweep.py
```

Current implementation:

\[
W^{\mathrm{current}}_{pq}
=
|h_{pq}| + |(pp|qq)| + |(pq|qp)|.
\]

The code is:

```python
weights = np.abs(h1)
coulomb = np.abs(np.einsum("ppqq->pq", eri))
exchange = np.abs(np.einsum("pqqp->pq", eri))
weights = weights + coulomb + exchange
```

It also provides:

- `make_integral_graph_partition`;
- `make_strong_pair_partition`;
- occupancy-aware post-balancing;
- partition validation.

### 3.2 Candidate sweeps and spectral order

File:

```text
FRASCI/lasscf/fragment_sweep.py
```

It already provides:

- graph Laplacian/Fiedler ordering;
- multiple fragment-size patterns;
- block and round-robin candidates;
- cut-strength calculation;
- deterministic stochastic controls;
- candidate metadata and catalogs.

### 3.3 Result analysis

File:

```text
FRASCI/lasscf/analyze_fragment_sweep.py
```

It already includes plots of integral cut strength against energy error.

### 3.4 Mutual-information work

Existing Fe4S4 work includes:

- `mi_min_cut` partitions;
- MI size sweeps;
- interfragment/intrafragment MI analysis;
- evidence that minimizing MI alone does not ensure self-consistent convergence;
- evidence that unequal fragment sizes can lower interfragment MI substantially.

This history is important. The adaptive method must jointly consider:

- separability;
- fragment solver difficulty;
- self-consistent stability;
- local electron/spin structure.

It must not optimize only a graph cut.

---

## 4. Definition of the orbital graph

Let each active spatial orbital be a node:

\[
V=\{0,\ldots,K-1\}.
\]

Let \(W\in\mathbb R^{K\times K}\) be a symmetric, nonnegative matrix with

\[
W_{pq}=W_{qp},\qquad W_{pp}=0.
\]

A large \(W_{pq}\) means that orbitals \(p\) and \(q\) should preferably remain in the
same LASSCF fragment.

The graph degree is

\[
d_p = \sum_{q\ne p}W_{pq},
\]

and the graph Laplacian is

\[
L=D-W,\qquad D=\operatorname{diag}(d_0,\ldots,d_{K-1}).
\]

For scale-independent spectral clustering, use the symmetric normalized Laplacian

\[
L_{\mathrm{sym}}
=
I-D^{-1/2}WD^{-1/2}.
\]

---

## 5. FCIDUMP-only edge weights

No single integral expression is guaranteed to represent correlation. The implementation
should calculate separate channels, normalize them, and preserve each channel in output
metadata. Combining raw terms without normalization can allow one channel—especially
Coulomb terms—to dominate only because of scale.

### 5.1 One-electron mixing

\[
H_{pq}=|h_{pq}|,\qquad p\ne q.
\]

This detects direct one-electron coupling in the current orbital basis.

Limitations:

- it is strongly basis dependent;
- it can vanish for canonical orbitals;
- it does not measure two-electron correlation;
- after localization, it can become more chemically meaningful.

### 5.2 Coulomb coupling

\[
J_{pq}=|(pp|qq)|.
\]

This measures density-density electrostatic interaction.

Important caution: \(J_{pq}\) is not itself a correlation measure. It may be large for
spatially separated occupied orbitals and may produce a dense graph. It should not be
used alone.

### 5.3 Exchange coupling

\[
K_{pq}=|(pq|qp)|.
\]

Exchange requires spatial overlap and is usually more local than \(J_{pq}\). It is often
a more useful indicator of shared spatial/bond character.

### 5.4 Pair-transfer/pair-hopping coupling

\[
P_{pq}=|(pp|qq)|,
\]

when interpreted as pair transfer in a second-quantized pairing representation.
Numerically this equals the Coulomb-index expression above, so it must not be counted
twice in a combined weight. Its physical interpretation depends on the operator
decomposition.

### 5.5 General two-electron coupling norm

To capture interactions that are not visible in \(J_{pq}\) and \(K_{pq}\), define

\[
T_{pq}^{2}
=
\sum_{rs}
\left[
|(pr|qs)|^2
+ |(ps|qr)|^2
\right].
\]

Then

\[
T_{pq}=\sqrt{T_{pq}^{2}}.
\]

This quantity measures how strongly orbitals \(p\) and \(q\) participate together in
the full two-electron tensor. It is more expensive than extracting \(J\) and \(K\), but
for the small `diff_mols` active spaces its \(O(K^4)\) contraction is negligible
relative to electronic-structure runs.

Alternative lower-cost contractions include

\[
T^{(1)}_{pq}
=
\sqrt{\sum_r |(pr|qr)|^2},
\]

and

\[
T^{(\mathrm{as})}_{pq}
=
\sqrt{\sum_{rs}
\left|
(pr|qs)-(ps|qr)
\right|^2}.
\]

The antisymmetrized version is physically appealing, but the implementation must verify
the integral index convention carefully before use.

### 5.6 Energy-denominator-weighted proxy

If orbital-energy estimates are available from \(h\), define

\[
\varepsilon_p = h_{pp}
\]

or, more robustly, diagonalize \(h\):

\[
hU=U\varepsilon.
\]

A perturbative-style coupling can be formed as

\[
Q_{pq}
=
\frac{K_{pq}+T_{pq}}
{\Delta_{pq}+\epsilon},
\qquad
\Delta_{pq}=|\varepsilon_p-\varepsilon_q|,
\]

where \(\epsilon>0\) prevents division by zero.

This favors strongly coupled near-degenerate orbitals. It is only a heuristic because
correlation denominators normally involve occupied/virtual many-orbital energy
differences rather than a single \(\Delta_{pq}\).

### 5.7 Robust normalization

For each nonnegative channel \(X\), use one of:

#### Frobenius normalization

\[
\widetilde X
=
\frac{X}{\|X\|_F+\epsilon}.
\]

#### Maximum normalization

\[
\widetilde X
=
\frac{X}{\max_{p<q}X_{pq}+\epsilon}.
\]

#### Robust quantile normalization

\[
\widetilde X_{pq}
=
\min\left(
\frac{X_{pq}}{Q_{0.95}(X_{ij}:i<j)+\epsilon},
1
\right).
\]

Quantile normalization is recommended for graphs with a few extreme integral values.

### 5.8 Recommended first composite weight

The first implementation should remain interpretable:

\[
\boxed{
W^{\mathrm{int}}_{pq}
=
\alpha_H\widetilde H_{pq}
+\alpha_K\widetilde K_{pq}
+\alpha_T\widetilde T_{pq}
}
\]

with default

\[
(\alpha_H,\alpha_K,\alpha_T)=(0.25,0.35,0.40).
\]

These are initial hyperparameters, not established physical constants. They must be
validated against benchmark outcomes.

A compatibility mode should reproduce the existing graph:

\[
W^{\mathrm{legacy}}_{pq}
=
\widetilde{|h_{pq}|}
+\widetilde{|(pp|qq)|}
+\widetilde{|(pq|qp)|}.
\]

The normalized form is preferable to the current raw sum.

---

## 6. Optional chemical/spatial edge channels

These require geometry and `mo_coeff.npz`; they cannot be reconstructed from FCIDUMP
alone.

### 6.1 IAO population vector

Let

\[
P_{Ap}
\]

be the population of active orbital \(p\) on atom \(A\), with

\[
P_{Ap}\ge 0,
\qquad
\sum_A P_{Ap}\approx 1.
\]

The current implementation assigns an orbital to

\[
A^\star(p)=\arg\max_A P_{Ap}.
\]

That hard assignment discards uncertainty.

### 6.2 Shared atomic-support similarity

Define

\[
S^{\mathrm{IAO}}_{pq}
=
\sum_A \sqrt{P_{Ap}P_{Aq}},
\]

the Bhattacharyya overlap between atomic population vectors.

Alternatively,

\[
S^{\mathrm{dot}}_{pq}
=
\frac{\sum_A P_{Ap}P_{Aq}}
{\sqrt{\sum_A P_{Ap}^2}\sqrt{\sum_A P_{Aq}^2}+\epsilon}.
\]

Large values indicate that two orbitals occupy similar atomic regions.

### 6.3 Orbital assignment ambiguity

For orbital \(p\), sort group populations:

\[
g_{(1)p}\ge g_{(2)p}\ge\cdots.
\]

Define margin

\[
m_p=g_{(1)p}-g_{(2)p}
\]

and ambiguity

\[
a_p = 1-m_p.
\]

An orbital should be flagged when, for example,

\[
g_{(2)p}>0.15
\]

or

\[
m_p<0.20.
\]

These values are starting thresholds and require calibration.

### 6.4 Orbital centroids

If real-space moments are available,

\[
\mathbf R_p
=
\langle\phi_p|\mathbf r|\phi_p\rangle.
\]

Define spatial affinity

\[
S^{\mathrm{centroid}}_{pq}
=
\exp\left[
-\frac{\|\mathbf R_p-\mathbf R_q\|^2}{2\sigma_R^2}
\right].
\]

This prevents a graph algorithm from grouping strong but spatially remote Coulomb
interactions unless another channel strongly supports that grouping.

### 6.5 Chemical composite graph

\[
W^{\mathrm{chem}}_{pq}
=
\beta_I\widetilde S^{\mathrm{IAO}}_{pq}
+\beta_R\widetilde S^{\mathrm{centroid}}_{pq}
+\beta_B B_{pq},
\]

where \(B_{pq}\) is a must-link or bond-manifold affinity derived from IBO/vvIBO
analysis or atom-group rules.

---

## 7. Correlation-derived edge channels

These require a pilot wavefunction and are not FCIDUMP-only.

### 7.1 Single-orbital entropy

For the one-orbital reduced density matrix \(\rho_p\),

\[
s_p
=
-\operatorname{Tr}\left(\rho_p\ln\rho_p\right)
=
-\sum_\alpha \omega_{\alpha,p}\ln\omega_{\alpha,p}.
\]

For a spatial orbital, \(\rho_p\) has the local states

\[
|0\rangle,\quad |\alpha\rangle,\quad|\beta\rangle,\quad|\alpha\beta\rangle.
\]

Large \(s_p\) means orbital \(p\) is strongly entangled with the remainder of the
active space.

### 7.2 Two-orbital entropy

For the two-orbital RDM \(\rho_{pq}\),

\[
s_{pq}
=
-\operatorname{Tr}\left(\rho_{pq}\ln\rho_{pq}\right).
\]

### 7.3 Orbital mutual information

Conventions differ by a factor of \(1/2\). Store the convention explicitly.

One common convention is

\[
\boxed{
I_{pq}
=
\frac{1}{2}\left(s_p+s_q-s_{pq}\right)
}
\qquad (p\ne q).
\]

Another uses the same expression without \(1/2\). Relative graph rankings are
unchanged, but reported numerical values differ.

For fragmentation:

\[
W^{\mathrm{MI}}_{pq}=I_{pq}.
\]

Strong MI edges should not be cut unless fragment-cost constraints make it unavoidable.

### 7.4 Two-body cumulant correlation

Given the spin-orbital or spatial-orbital 1-RDM \(\gamma\) and 2-RDM \(\Gamma\), define
the connected two-body cumulant schematically as

\[
\lambda^{(2)}
=
\Gamma-\mathcal A(\gamma\gamma),
\]

where \(\mathcal A\) antisymmetrizes the product.

An orbital-pair cumulant score can be defined as

\[
C_{pq}
=
\left[
\sum_{\substack{\mu\in p\\\nu\in q\\\kappa,\lambda}}
\left|\lambda^{(2)}_{\mu\nu,\kappa\lambda}\right|^2
\right]^{1/2}.
\]

The exact spin summation and orbital block definition must be fixed in code and tested
against simple determinants, for which the connected cumulant should vanish.

### 7.5 Multi-state/geometry robust correlation graph

For a partition intended to remain valid over states \(s\) and geometries \(g\), use

\[
W^{\mathrm{robust}}_{pq}
=
(1-\eta)
\sum_{g,s}\omega_{gs}I^{(g,s)}_{pq}
+\eta\max_{g,s}I^{(g,s)}_{pq}.
\]

The maximum term prevents a boundary from cutting a bond that becomes strongly
correlated only at one stretched geometry or spin state.

---

## 8. Partition objectives

Let

\[
\mathcal P=\{F_1,\ldots,F_M\}
\]

be a nonoverlapping partition of all active orbitals.

### 8.1 Raw cut

\[
\operatorname{Cut}(\mathcal P;W)
=
\sum_{a<b}
\sum_{p\in F_a}
\sum_{q\in F_b}
W_{pq}.
\]

This is already implemented as `cut_strength`.

Problem: unconstrained raw min-cut favors isolating weakly connected single orbitals.

### 8.2 Fractional cut

\[
R_{\mathrm{cut}}
=
\frac{\operatorname{Cut}(\mathcal P;W)}
{\sum_{p<q}W_{pq}+\epsilon}.
\]

This is dimensionless and comparable across molecules if the graph definition is fixed.

Define retained affinity:

\[
R_{\mathrm{intra}}=1-R_{\mathrm{cut}}.
\]

### 8.3 Normalized cut

For fragment \(F\), define volume

\[
\operatorname{vol}(F)
=
\sum_{p\in F}d_p.
\]

Then

\[
\operatorname{NCut}(\mathcal P)
=
\sum_{a=1}^{M}
\frac{
\sum_{p\in F_a,q\notin F_a}W_{pq}
}{
\operatorname{vol}(F_a)+\epsilon
}.
\]

This discourages trivial singleton cuts.

### 8.4 Ratio cut

\[
\operatorname{RatioCut}(\mathcal P)
=
\sum_{a=1}^{M}
\frac{
\sum_{p\in F_a,q\notin F_a}W_{pq}
}{
|F_a|
}.
\]

### 8.5 Combined optimization objective

The implementation-ready objective is

\[
\boxed{
\mathcal J(\mathcal P)
=
\lambda_{\mathrm{cut}}R_{\mathrm{cut}}
+\lambda_{\mathrm{cost}}C_{\mathrm{cost}}
+\lambda_{\mathrm{electron}}C_{\mathrm{electron}}
+\lambda_{\mathrm{spin}}C_{\mathrm{spin}}
+\lambda_{\mathrm{local}}C_{\mathrm{local}}
+\lambda_{\mathrm{ambiguity}}C_{\mathrm{ambiguity}}
+\lambda_{\mathrm{singleton}}C_{\mathrm{singleton}}
}
\]

with hard rejection for invalid partitions.

Each component must be saved separately. A single opaque score is insufficient for
scientific analysis.

---

## 9. Solver-cost constraints

Equal orbital counts are not equal computational cost.

For fragment \(F\) with \(K_F\) spatial orbitals, \(N_{\alpha,F}\) alpha electrons, and
\(N_{\beta,F}\) beta electrons, the numbers of alpha and beta occupation strings are

\[
N_F^\alpha
=
\binom{K_F}{N_{\alpha,F}},
\qquad
N_F^\beta
=
\binom{K_F}{N_{\beta,F}}.
\]

If a solver stores alpha and beta strings separately, a simple storage proxy is

\[
D_F^{\mathrm{strings}}
=
\binom{K_F}{N_{\alpha,F}}
+
\binom{K_F}{N_{\beta,F}}.
\]

For a conventional determinant product basis, the actual number of Slater determinants is

\[
\boxed{
D_F^{\mathrm{product}}
=
\binom{K_F}{N_{\alpha,F}}
\times
\binom{K_F}{N_{\beta,F}}.
}
\]

The implementation must use the cost model matching the fragment solver. For TrimCI,
the full product dimension is a useful worst-case measure even though only selected
determinants are retained.

Use a log-cost:

\[
c_F = \log\left(D_F^{\mathrm{product}}+1\right).
\]

Possible aggregate penalties:

\[
C_{\max}=\max_F c_F,
\]

\[
C_{\mathrm{sum}}=\sum_F c_F,
\]

\[
C_{\mathrm{imbalance}}
=
\operatorname{Var}(c_{F_1},\ldots,c_{F_M}).
\]

Recommended:

\[
C_{\mathrm{cost}}
=
C_{\max}
+\xi C_{\mathrm{imbalance}}.
\]

This allows unequal orbital counts when the actual electronic dimensions remain
manageable.

---

## 10. Electron and spin feasibility

### 10.1 FCIDUMP-only approximate determinant

If `dets.npz` is unavailable, derive an Aufbau seed.

Option A: occupy lowest diagonal elements:

\[
\mathcal O_\alpha
=
\operatorname{arg\,sort}_{N_\alpha}(h_{pp}),
\qquad
\mathcal O_\beta
=
\operatorname{arg\,sort}_{N_\beta}(h_{pp}).
\]

Option B: diagonalize \(h\), occupy the lowest eigenvectors, and rotate all integrals
into that basis. This changes the orbital basis and therefore changes the partitioning
problem. It should be a separate explicit mode.

For the existing localized FCIDUMPs, Option A preserves the current basis and is the
appropriate compatibility choice.

### 10.2 Fragment electron counts

\[
N_{\alpha,F}
=
\sum_{p\in F}n^\alpha_p,
\qquad
N_{\beta,F}
=
\sum_{p\in F}n^\beta_p.
\]

### 10.3 Multiplicity seed

The current code uses

\[
2S_F+1
=
|N_{\alpha,F}-N_{\beta,F}|+1.
\]

This is a reasonable initial multiplicity, not necessarily the physically optimal
fragment spin. LASSI/LASSIS or a local spin-state sweep may need alternatives.

### 10.4 Electron feasibility penalty

With fractional occupations \(n_p\) from a pilot 1-RDM:

\[
C_{\mathrm{electron}}
=
\sum_F
\left|
\sum_{p\in F}n_p
-
\operatorname{round}\left(\sum_{p\in F}n_p\right)
\right|.
\]

With only a determinant seed, counts are integers by construction. Instead penalize
pathological cases:

- empty-electron fragments;
- fully occupied fragments with no particle space;
- empty fragments with no hole space;
- one-orbital fragments unless explicitly allowed;
- fragments whose selected determinant dimension is trivial.

---

## 11. Hard constraints and must-link groups

The partition generator should support union-find must-link groups before graph
optimization.

Potential must-link rules:

1. bonding/antibonding pairs identified from IBO/vvIBO or pilot MI;
2. \(\sigma/\sigma^\ast\) and \(\pi/\pi^\ast\) manifolds of the same multiple bond;
3. metal \(d\) and correlating \(d'\) partners;
4. orbitals belonging to a multicenter IBO;
5. symmetry-equivalent local orbital sets;
6. orbital pairs above a correlation threshold:

   \[
   I_{pq}>\tau_{\mathrm{MI}};
   \]

7. ambiguous orbitals with significant support on two chemical groups.

Must-not-link constraints may also be useful:

- distant orbital centroids with negligible exchange;
- orbitals assigned to disconnected molecular components;
- fragment-size/cost violations.

Hard constraints should be used sparingly. Incorrect must-link groups can force one
giant fragment.

---

## 12. Candidate-generation algorithms

No single graph algorithm should be trusted initially. Generate a diverse candidate set
and rank it.

### 12.1 Strong-pair greedy

Already implemented.

1. Sort all edges by decreasing \(W_{pq}\).
2. Place unassigned strongly coupled pairs together.
3. Attach remaining nodes by accumulated fragment affinity.
4. Enforce size/cost constraints.

Useful as a baseline, but early greedy decisions can be irreversible and suboptimal.

### 12.2 Agglomerative maximum-affinity merging

Recommended first adaptive implementation.

Initialize one cluster per orbital or per must-link group.

For clusters \(A\) and \(B\), define

\[
\operatorname{Affinity}(A,B)
=
\frac{
\sum_{p\in A,q\in B}W_{pq}
}{
(|A||B|)^\zeta
},
\]

with \(0\le\zeta\le1\).

At each step:

1. select the valid pair with maximum affinity;
2. merge if cost/electron/locality constraints remain valid;
3. record each hierarchy level as a candidate;
4. stop when no valid merge remains.

Advantages:

- deterministic;
- naturally produces different fragment counts;
- easy to add constraints;
- hierarchy can be benchmarked without choosing \(M\) in advance.

### 12.3 Recursive spectral partitioning

Solve the Fiedler problem for a cluster:

\[
L_{\mathrm{sym}}\mathbf v_2
=
\lambda_2\mathbf v_2.
\]

Split by the sign or a threshold of \(\mathbf v_2\), choosing the threshold that
minimizes normalized cut under constraints.

Recursively split the cluster with the weakest internal bottleneck until:

- target number of fragments is reached;
- maximum fragment cost is satisfied;
- further splitting exceeds a cut threshold.

### 12.4 Constrained \(k\)-way search

For `diff_mols` active spaces of roughly 8–12 orbitals, exhaustive enumeration with
canonical labeling and pruning may be practical.

Prune when:

- a partial cluster exceeds maximum cost;
- must-link constraints are violated;
- the lower bound on cut score exceeds the best current score;
- electron/spin feasibility becomes impossible.

This provides a valuable exact reference for evaluating heuristic graph methods on
small molecules.

### 12.5 Local search

Given partition \(\mathcal P\), evaluate:

- single-orbital moves;
- pair swaps;
- fragment merges;
- fragment splits;
- boundary-group moves.

Accept a move when

\[
\Delta\mathcal J<0.
\]

Use deterministic best improvement first. Random restarts can be added later.

---

## 13. FCIDUMP-only scoring outputs

For each candidate, write:

```json
{
  "name": "integral_agglomerative_k3_rank02",
  "source": "fcidump_only",
  "graph_version": "integral_graph_v1",
  "weight_formula": {
    "h1": 0.25,
    "exchange": 0.35,
    "tensor_norm": 0.40,
    "normalization": "q95"
  },
  "fragments": [[0, 1, 4, 5], [2, 3], [6, 7]],
  "nelec_per_frag": [[2, 2], [1, 1], [1, 1]],
  "spin_sub": [1, 1, 1],
  "scores": {
    "cut": 0.123,
    "cut_fraction": 0.081,
    "normalized_cut": 0.214,
    "max_log_dimension": 5.31,
    "cost_imbalance": 0.42,
    "singleton_count": 0
  },
  "warnings": []
}
```

Also save:

- `weight_h1.npy`;
- `weight_exchange.npy`;
- `weight_tensor_norm.npy`;
- `weight_combined.npy`;
- `candidate_catalog.json`;
- graph heatmap;
- graph/network visualization;
- candidate score table.

---

## 14. Pilot refinement using existing LASSCF/LASSIS machinery

The FCIDUMP graph generates hypotheses. A low-cost pilot decides whether the graph cut is
compatible with the LAS ansatz and solver.

### 14.1 Pilot protocol

Suggested default:

```text
method: lasscf_trimci + lassis
trimci_max_dets: 50 or 100 per fragment
trimci_max_rounds: 1 or 2
coo_cycles: 0 initially
max_cycle_macro: 20
one LASSIS nspin value for closed-shell organics
```

COO can be added only for the top candidates.

### 14.2 LASSIS lowering

\[
\Delta E_{\mathrm{SI}}
=
E_{\mathrm{LASSCF}}-E_{\mathrm{LASSIS}}.
\]

Interpretation:

- small \(\Delta E_{\mathrm{SI}}\): the LAS product is relatively separable;
- large \(\Delta E_{\mathrm{SI}}\): substantial physics is crossing boundaries.

Caution: for magnetic systems, large spin-recoupling lowering may be expected and does
not automatically mean two metal fragments should be merged.

### 14.3 Convergence penalty

Define:

\[
C_{\mathrm{conv}}
=
\begin{cases}
0,&\text{converged},\\
c_0+c_1N_{\mathrm{macro}}/N_{\max},&\text{not converged}.
\end{cases}
\]

Also record:

- maximum orbital gradient;
- energy oscillation amplitude;
- repeated fragment solver failures;
- determinant-count saturation;
- wall time.

### 14.4 Pilot ranking score

If an external/FCI reference exists:

\[
\mathcal J_{\mathrm{pilot}}
=
w_E|E-E_{\mathrm{ref}}|
+w_{\mathrm{SI}}\Delta E_{\mathrm{SI}}
+w_{\mathrm{conv}}C_{\mathrm{conv}}
+w_{\mathrm{cost}}C_{\mathrm{runtime}}.
\]

Without a reference:

\[
\mathcal J_{\mathrm{pilot}}
=
w_E(E-E_{\mathrm{best}})
+w_{\mathrm{SI}}\Delta E_{\mathrm{SI}}
+w_{\mathrm{conv}}C_{\mathrm{conv}}
+w_{\mathrm{cost}}C_{\mathrm{runtime}}.
\]

The plot label must say **best observed**, not reference, when no independent reference
exists.

---

## 15. Adaptive merge/split refinement

### 15.1 Pairwise boundary score

For fragment pair \(A,B\):

\[
B_{AB}^{\mathrm{int}}
=
\sum_{p\in A,q\in B}W^{\mathrm{int}}_{pq}.
\]

With MI:

\[
B_{AB}^{\mathrm{MI}}
=
\sum_{p\in A,q\in B}I_{pq}.
\]

With full diagnostics:

\[
B_{AB}
=
\eta_I\widetilde B_{AB}^{\mathrm{int}}
+\eta_M\widetilde B_{AB}^{\mathrm{MI}}
+\eta_C\widetilde B_{AB}^{\mathrm{CT}}
+\eta_R\widetilde B_{AB}^{\mathrm{RDM}}.
\]

### 15.2 Merge rule

Merge \(A,B\) if:

\[
B_{AB}>\tau_B,
\]

the merged fragment remains below the cost limit, and the pilot objective improves.

### 15.3 Split rule

Split fragment \(F\) when:

- its determinant dimension or selected determinant count dominates runtime;
- it has an internal weak graph cut;
- its internal correlation graph has separable communities;
- the split does not produce pathological local electron counts.

Choose split:

\[
(F_1,F_2)
=
\arg\min
\operatorname{NCut}(F_1,F_2)
\]

subject to cost and electron constraints.

### 15.4 Termination

Stop when:

1. no valid merge/split improves the pilot objective;
2. \(\Delta E_{\mathrm{SI}}<\tau_{\mathrm{SI}}\);
3. cross-fragment MI fraction is below threshold;
4. all candidates exceed the maximum allowed cost;
5. one fragment remains.

A one-fragment result is valid. It means the selected active space does not admit a
useful weakly coupled LAS factorization under the imposed accuracy threshold.

---

## 16. Recommended implementation architecture

### 16.1 New module

```text
FRASCI/diff_mols/adaptive_fragmentation.py
```

Proposed interfaces:

```python
@dataclass
class OrbitalGraph:
    weights: np.ndarray
    channels: dict[str, np.ndarray]
    metadata: dict


@dataclass
class AdaptivePartitionCandidate:
    name: str
    orbital_lists: list[list[int]]
    nelec_per_frag: list[tuple[int, int]]
    spin_sub: list[int]
    scores: dict[str, float]
    warnings: list[str]
    provenance: dict


def build_integral_graph(
    fcidump_path: Path,
    *,
    formula: str = "h1_exchange_tensor",
    normalization: str = "q95",
    coefficients: dict[str, float] | None = None,
) -> OrbitalGraph:
    ...


def generate_integral_candidates(
    graph: OrbitalGraph,
    *,
    alpha_bits: int,
    beta_bits: int,
    max_fragments: int,
    max_orbitals_per_fragment: int | None,
    max_log_dimension: float | None,
) -> list[AdaptivePartitionCandidate]:
    ...


def rank_pilot_results(
    candidates: list[AdaptivePartitionCandidate],
    run_rows: pd.DataFrame,
) -> pd.DataFrame:
    ...
```

### 16.2 Config schema

Extend `FragmentationSpec` with optional fields:

```yaml
- name: adaptive_integral
  orbital_lists: adaptive_integral
  graph_formula: h1_exchange_tensor
  graph_normalization: q95
  graph_coefficients: {h1: 0.25, exchange: 0.35, tensor: 0.40}
  min_fragments: 1
  max_fragments: 4
  max_orbitals_per_fragment: 8
  max_log_dimension: 10.0
  candidate_count: 8
  candidate_selector: agglomerative
```

An adaptive entry generates several concrete partition names:

```text
adaptive_integral_k2_r01
adaptive_integral_k3_r01
adaptive_integral_k3_r02
...
```

The resolved concrete partitions should be persisted so reruns are reproducible.

### 16.3 CLI phases

Proposed commands:

```bash
# Generate candidates only
./FRASCIenv/bin/python -m FRASCI.diff_mols.fragment_search \
  --mol me2n2 \
  --source fcidump \
  --candidate-family integral \
  --max-fragments 4

# Run cheap pilots
./FRASCIenv/bin/python -m FRASCI.diff_mols.fragment_search \
  --mol me2n2 \
  --run-pilots \
  --pilot-methods lasscf_trimci,lassis \
  --pilot-max-dets 100

# Promote top partitions to production protocols
./FRASCIenv/bin/python -m FRASCI.diff_mols.fragment_search \
  --mol me2n2 \
  --promote-top 3
```

### 16.4 Results layout

```text
Outputs/diff_mols/<mol>/fragment_search/<search_id>/
├── search_config.yaml
├── integral_graph/
│   ├── weight_h1.npy
│   ├── weight_exchange.npy
│   ├── weight_tensor.npy
│   ├── weight_combined.npy
│   ├── heatmap.png
│   └── graph.png
├── candidates/
│   ├── candidate_catalog.json
│   ├── adaptive_integral_k2_r01.json
│   └── ...
├── pilot_matrix.csv
├── pilot_ranking.csv
├── score_vs_energy.png
├── score_vs_lassis_gain.png
├── score_vs_convergence.png
└── README.md
```

Production method outputs should continue using:

```text
Outputs/diff_mols/<mol>/<method>/<partition>/<protocol>/run_<timestamp>/
```

---

## 17. Validation strategy

### 17.1 Scientific questions

1. Does integral cut fraction predict final energy error?
2. Does it predict convergence?
3. Does it predict LASSIS lowering?
4. Does adding chemical/IAO channels improve those correlations?
5. Does pilot MI improve candidate ranking enough to justify its cost?
6. Does the optimal number of fragments vary by molecule?

### 17.2 Required baselines

For every molecule where applicable:

- `full` selected-CI controls;
- `chem`;
- `chem_bond`;
- `h1diag_2`;
- `h1diag_4`;
- legacy `integral_graph`;
- `strong_pair`;
- new normalized integral candidates;
- later MI-refined candidates.

### 17.3 Metrics

Energy:

\[
\epsilon_E
=
1000|E-E_{\mathrm{ref}}|\quad\mathrm{mHa}.
\]

If no independent reference exists:

\[
\epsilon_E^{\mathrm{obs}}
=
1000(E-E_{\mathrm{best\ observed}}).
\]

Convergence:

- binary converged fraction;
- macroiteration count;
- energy/gradient oscillation;
- runtime.

Fragment separability:

\[
R_{\mathrm{cut}}^{\mathrm{int}},
\qquad
R_{\mathrm{cut}}^{\mathrm{MI}},
\qquad
\Delta E_{\mathrm{SI}}.
\]

Cost:

- maximum determinant-space dimension;
- actual selected determinants per fragment;
- total wall time;
- imbalance across fragments.

### 17.4 Statistical analysis

Across candidates and molecules compute:

- Spearman rank correlation;
- Kendall \(\tau\);
- Pearson correlation where linearity is plausible;
- top-\(k\) recall: does the graph score retain the best-energy partition?
- Pareto fronts of energy error versus runtime;
- calibration of cut thresholds against chemical accuracy.

The key test is ranking, not merely linear correlation.

### 17.5 Ablation study

Compare:

\[
\widetilde H,
\quad
\widetilde K,
\quad
\widetilde T,
\quad
\widetilde H+\widetilde K,
\quad
\widetilde K+\widetilde T,
\quad
\widetilde H+\widetilde K+\widetilde T.
\]

Then add:

\[
S^{\mathrm{IAO}},
\qquad
I_{pq}.
\]

This reveals which information actually predicts good LASSCF fragments.

---

## 18. Unit and integration tests

### 18.1 Graph tests

- symmetry:

  \[
  W=W^\mathrm T;
  \]

- nonnegativity;
- zero diagonal;
- invariance to FCIDUMP integral permutation symmetries;
- no NaN/Inf for zero matrices;
- normalization behavior;
- expected high edge for a synthetic strongly coupled orbital pair.

### 18.2 Partition tests

- every orbital appears exactly once;
- no empty fragments;
- must-link constraints are satisfied;
- electron counts sum to global \(N_\alpha,N_\beta\);
- spin multiplicities are positive integers;
- cost limits hold;
- deterministic output under fixed inputs;
- canonical candidate names independent of fragment ordering.

### 18.3 Objective tests

- one fragment has zero cut;
- singleton isolation is penalized by NCut/constraints;
- merging two fragments cannot increase raw cut;
- scaling all graph weights leaves normalized cut unchanged;
- equivalent fragment relabeling leaves all scores unchanged.

### 18.4 End-to-end tests

Use small H4/H6 fixtures:

- disconnected strong pairs should be grouped;
- a dominant synthetic \(W_{01}\) must keep orbitals 0 and 1 together;
- candidate JSON must load through `resolve_fragmentation`;
- dry-run matrix must place results under the generated partition name;
- generated partitions must appear in diagnostic plots.

---

## 19. Failure modes and safeguards

### 19.1 Basis dependence

Integral and MI graphs depend on the orbital basis. Canonical orbitals may produce
nonlocal partitions; aggressively localized orbitals may distort near-degenerate
manifolds.

Safeguard:

- record orbital-localization method and hash;
- compare Boys and IAO-Pipek-Mezey for selected benchmarks;
- never compare graph scores generated in different orbital bases as if identical.

### 19.2 Coulomb domination

Raw \(J_{pq}\) can make the graph dense and nonlocal.

Safeguard:

- normalize channels independently;
- report ablations;
- default to exchange and tensor norms rather than Coulomb-heavy weights.

### 19.3 Trivial min-cut fragments

Raw min-cut isolates weak nodes.

Safeguard:

- use NCut or cost-constrained agglomeration;
- minimum fragment size;
- singleton penalty.

### 19.4 Unequal fragment pathology

MI minimization may produce one large correlated fragment and several nearly trivial
fragments. Existing Fe4S4 results show that this can reduce interfragment MI but worsen
self-consistent stability.

Safeguard:

- determinant-space cost penalty;
- minimum internal correlation or minimum selected-determinant count;
- pilot convergence screening;
- reject fragments lacking meaningful hole/particle spaces where required.

### 19.5 Incorrect local spin assignment

Aufbau-based fragment counts may not reflect the correct local spin coupling.

Safeguard:

- retain alternative local multiplicity candidates;
- use LASSIS spin sectors;
- for metals, distinguish charge-transfer/covalent coupling from spin recoupling.

### 19.6 Geometry transfer

A partition selected at equilibrium may cut a bond that becomes strongly correlated
during dissociation.

Safeguard:

- aggregate graph data across geometries;
- use maximum MI/coupling across the path;
- track orbitals consistently;
- require a single robust partition or explicitly permit geometry-dependent partitions.

### 19.7 “Best observed” mistaken for reference

Safeguard:

- use `reference` only for independent FCI/CASSCF/literature values;
- label zero baselines as `best observed`;
- store reference provenance.

---

## 20. Staged implementation roadmap

### Phase A — FCIDUMP graph diagnostics

1. Extract current graph construction into a reusable API.
2. Add separately normalized \(H,K,T\) channels.
3. Add raw cut, fractional cut, NCut, and solver-cost metrics.
4. Score all existing `diff_mols` partitions without running new chemistry.
5. Plot graph score against existing energy and convergence data.

**Decision gate:** determine whether any FCIDUMP score ranks `chem_bond` above bad
partitions for me2n2 and tetrazene.

### Phase B — Candidate generation

1. Implement constrained agglomerative hierarchy.
2. Implement recursive spectral candidates.
3. Add exact/pruned enumeration for \(K\le 12\) as a validation oracle.
4. Persist candidate catalogs.
5. Add dry-run matrix support.

### Phase C — Cheap pilot screen

1. Run `lasscf_trimci` with 50–100 determinants.
2. Run matching LASSIS.
3. Rank by energy, convergence, LASSIS lowering, and runtime.
4. Promote top candidates to production grids.

### Phase D — Chemical graph

1. Add IAO population-vector similarities.
2. Add assignment ambiguity.
3. Add bond/motif must-links.
4. Compare Boys against IAO-Pipek-Mezey localization.

### Phase E — Correlation refinement

1. Obtain pilot 1-/2-RDM or low-\(M\) DMRG MI.
2. Add MI/cumulant graph channels.
3. Implement adaptive merge/split.
4. Test state/geometry robust graphs.

---

## 21. Concrete first experiment

Before implementing an adaptive runner, perform a retrospective analysis:

1. Load each molecule's FCIDUMP.
2. Compute \(H,K,T\) channels.
3. Score every existing partition:
   - `chem`;
   - `chem_bond`;
   - `h1diag_2`;
   - `h1diag_4`;
   - metal `h1diag`;
   - full excluded from fragmented ranking.
4. Join scores with `runs_index.csv`.
5. For each LASSCF workflow, calculate:
   - best converged error;
   - convergence rate;
   - median wall time;
   - best/median LASSIS lowering.
6. Test whether lower graph cut predicts better outcomes.

This experiment requires no new expensive electronic-structure runs and will determine
whether the FCIDUMP-only graph is promising enough to justify candidate generation.

---

## 22. Primary literature

### Localized active spaces

1. M. R. Hermes and L. Gagliardi, “The Localized Active Space Self-Consistent
   Field Method,” *J. Chem. Theory Comput.* **15**, 972–986 (2019).
   [DOI: 10.1021/acs.jctc.8b01009](https://doi.org/10.1021/acs.jctc.8b01009)

2. M. R. Hermes, R. Pandharkar, and L. Gagliardi, “Variational Localized Active
   Space Self-Consistent Field Method,” *J. Chem. Theory Comput.* **16**, 4923–4937
   (2020).
   [DOI: 10.1021/acs.jctc.0c00222](https://doi.org/10.1021/acs.jctc.0c00222)

3. R. Pandharkar et al., “Localized Active Space-State Interaction: A
   Multireference Method for Chemical Insight,” *J. Chem. Theory Comput.* (2022).
   [DOI: 10.1021/acs.jctc.2c00536](https://doi.org/10.1021/acs.jctc.2c00536)

4. A. Agarawal et al., “Automatic State Interaction with Large Localized Active
   Spaces for Multimetallic Systems,” *J. Chem. Theory Comput.* (2024).
   [DOI: 10.1021/acs.jctc.4c00376](https://doi.org/10.1021/acs.jctc.4c00376)

5. M. R. Hermes et al., “Localized Active Space State Interaction Singles,”
   *J. Chem. Theory Comput.* (2025).
   [DOI: 10.1021/acs.jctc.5c00387](https://doi.org/10.1021/acs.jctc.5c00387)

### Active-space and orbital construction

6. E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan, and G. Knizia, “Automated
   Construction of Molecular Active Spaces from Atomic Valence Orbitals,”
   *J. Chem. Theory Comput.* **13**, 4063–4078 (2017).
   [DOI: 10.1021/acs.jctc.7b00128](https://doi.org/10.1021/acs.jctc.7b00128)

7. G. Knizia, “Intrinsic Atomic Orbitals: An Unbiased Bridge between Quantum
   Theory and Chemical Concepts,” *J. Chem. Theory Comput.* **9**, 4834–4843
   (2013).
   [DOI: 10.1021/ct400687b](https://doi.org/10.1021/ct400687b)

8. S. Lehtola and H. Jónsson, “Unitary Optimization of Localized Molecular
   Orbitals,” *J. Chem. Theory Comput.* (2014).
   [DOI: 10.1021/ct401016x](https://doi.org/10.1021/ct401016x)

### Orbital entanglement and automated correlation analysis

9. J. Rissler, R. M. Noack, and S. R. White, “Measuring Orbital Interaction
   Using Quantum Information Theory,” *Chem. Phys.* **323**, 519–531 (2006).
   [DOI: 10.1016/j.chemphys.2005.10.018](https://doi.org/10.1016/j.chemphys.2005.10.018)

10. K. Boguslawski et al., “Entanglement Measures for Single- and
    Multireference Correlation Effects,” *J. Phys. Chem. Lett.* (2012).
    [DOI: 10.1021/jz301319v](https://doi.org/10.1021/jz301319v)

11. C. J. Stein and M. Reiher, “Automated Selection of Active Orbital Spaces,”
    *J. Chem. Theory Comput.* **12**, 1760–1771 (2016).
    [DOI: 10.1021/acs.jctc.6b00156](https://doi.org/10.1021/acs.jctc.6b00156)

12. Ö. Legeza and J. Sólyom, “Optimizing the Density-Matrix Renormalization
    Group Method Using Quantum Information Entropy,” *Phys. Rev. B* **68**,
    195116 (2003).
    [DOI: 10.1103/PhysRevB.68.195116](https://doi.org/10.1103/PhysRevB.68.195116)

13. M. Ali, “On the Ordering of Sites in the Density Matrix Renormalization
    Group Using Quantum Mutual Information” (2021).
    [arXiv:2103.01111](https://arxiv.org/abs/2103.01111)

14. L. Ding, I. K. M. Knecht, and C. Schilling, “Quantum
    Information-Assisted Complete Active Space Optimization” (2023).
    [arXiv:2309.01676](https://arxiv.org/abs/2309.01676)

### Embedding and adaptive boundary diagnostics

15. G. Knizia and G. K.-L. Chan, “Density Matrix Embedding: A Simple
    Alternative to Dynamical Mean-Field Theory,” *Phys. Rev. Lett.* **109**,
    186404 (2012).
    [DOI: 10.1103/PhysRevLett.109.186404](https://doi.org/10.1103/PhysRevLett.109.186404)

16. G. Knizia and G. K.-L. Chan, “Density Matrix Embedding: A Strong-Coupling
    Quantum Embedding Theory,” *J. Chem. Theory Comput.* **9**, 1428–1432
    (2013).
    [DOI: 10.1021/ct301044e](https://doi.org/10.1021/ct301044e)

17. M. Welborn, T. Tsuchimochi, and T. Van Voorhis, “Bootstrap Embedding: An
    Internally Consistent Fragment-Based Method,” *J. Chem. Phys.* **145**,
    074102 (2016).
    [DOI: 10.1063/1.4960986](https://doi.org/10.1063/1.4960986)

These embedding methods do not directly select LASSCF fragments. Their transferable
idea is to diagnose and reduce boundary error using Schmidt coupling, overlapping
fragments, or RDM consistency.

---

## 23. Final recommendation

Implement the project in this order:

1. **Retrospective FCIDUMP scoring** of existing partitions.
2. **Normalized integral graph** with \(H,K,T\) channels.
3. **Constrained agglomerative candidate hierarchy.**
4. **Cheap LASSCF/LASSIS pilot ranking.**
5. **IAO/chemical must-link refinement.**
6. **MI/cumulant adaptive merging only if it materially improves ranking.**

The first research hypothesis should be:

\[
\boxed{
\text{lower normalized interfragment integral cut predicts lower LASSCF error,
better convergence, or smaller LASSIS correction}
}
\]

If this fails, the failure itself is informative: FCIDUMP integrals alone are not enough,
and the next model must include chemical locality or correlated RDM information.
