# FRASCI

Fragmented Selected Configuration Interaction for Fe₄S₄ using a self-consistent mean-field embedding.

**Target:** match the brute-force TrimCI energy (−327.1920 Ha) with far fewer determinants.  
**System:** Fe₄S₄, 36 spatial orbitals, 27α + 27β electrons.  
**Best result:** −326.911 Ha, error +0.281 Ha, 4001 total determinants.

---

## Background

### QFlow

QFlow is a fragmentation framework for large active-space problems. The core idea comes from quantum embedding methods like DMET: split a large active space into smaller fragments, solve each independently with a sub-solver, and couple them through a shared mean-field density matrix until self-consistent. The original QFlow papers use VQE on quantum hardware; this project replaces VQE with TrimCI to stress-test the fragmentation on a classically hard system.

### TrimCI

TrimCI is a selected configuration interaction solver. It builds the wavefunction iteratively from a reference determinant using a heat-bath selection scheme, converging to a compact representation with far fewer determinants than full CI. For Fe₄S₄ with a 36-orbital active space, TrimCI converges around 10,000 determinants where full CI would require billions.

### Why Combine Them?

TrimCI still scales steeply with system size. The group's target systems have 50–80 orbital active spaces where brute-force TrimCI is infeasible. Fragmenting into larger ~12-orbital windows keeps each sub-problem tractable regardless of total system size. This project validates the approach and measures the cost reduction on a system where the exact answer is known.

---

## What Is Being Compared?

Three related but distinct approaches:

1. **Full TrimCI** — one global selected-CI solve
2. **Original QFlow** — many small SES fragments + amplitude iteration
3. **FRASCI** — three larger fragments with self-consistent TrimCI embedding

### Shared Input: FCIDUMP

`data/fcidump_cycle_6` contains the one- and two-electron integrals for the 36-orbital Fe₄S₄ active space:

```
h1[p,q], eri[p,q,r,s] = (pq|rs)
n_orb = 36, n_elec = 54, E_nuc = 0.0
```

Both full TrimCI and FRASCI read this same FCIDUMP. `data/dets.npz` is the output from the full TrimCI solve — used for reference electron counting and benchmarking, not as input to the embedding.

### Full TrimCI

```
FCIDUMP → one 36-orbital selected-CI solve → E = −327.1920 Ha, ~10,095 dets
```

This is the reference. It is full-active-space selected CI, not full FCI; TrimCI selects the important determinants adaptively.

### Original QFlow

Original QFlow builds effective Hamiltonians via a similarity transform in the full determinant space (`H_eff = exp(−σ) H exp(+σ)`), which is not tractable for Fe₄S₄ at this scale. FRASCI keeps the fragment idea but replaces the full-space transform with a mean-field 1-RDM dressing.

### FRASCI

```
FCIDUMP → three 12-orbital non-overlapping fragments
→ dress each fragment's h1 with mean-field environment from gamma (self-consistent)
→ solve each fragment with TrimCI
→ assemble total energy
→ compare against full TrimCI
```

Energy formula (the only valid one):
```
E_total = E_mf_global + Σ_I (E_TrimCI_I − E_mf_emb_I)
```

The gamma (spin-summed diagonal 1-RDM) is computed self-consistently via a Gauss-Seidel outer loop with Anderson/DIIS acceleration, starting from zero. Each sweep updates fragment gamma blocks sequentially and repeats until `max|Δγ| < 1e-4`. This self-consistent embedding brought the fragmentation error from +0.624 Ha (fixed gamma) down to +0.281 Ha.

### Cost-Accuracy Tradeoff

| Method | What is solved | Determinants | Energy (Ha) | Error vs −327.1920 |
|--------|----------------|-------------:|------------:|-------------------:|
| Full TrimCI reference | One full 36-orbital selected-CI problem | 10,095 | −327.1920 | 0.000 |
| SC MFA h1diag, loose threshold | Three 12-orbital SC-embedded fragments | 147 | −326.511 | +0.681 |
| SC MFA h1diag, 1000 dets/frag | Three 12-orbital SC-embedded fragments | 2,017 | -326.908656 | +0.283 |
| SC MFA h1diag, 2000 dets/frag | Three 12-orbital SC-embedded fragments | 4,001 | -326.910993 | +0.281 |

The best result uses 4001 determinants, with an energy error of +0.281 Ha.

---

## Results


### Self-Consistent MFA

Three non-overlapping 12-orbital fragments. Gamma is converged via the GS self-consistent loop before the final D2 energy evaluation.

| Gamma | Partition | Max dets/frag | Dets | E_total (Ha) | Error |
|-------|-----------|:---:|------|--------------|-------|
| SC h1diag (20 iters) | h1diag 12/12/12 | 1000 | 2,017 | −326.9073 | +0.285 Ha |
| SC h1diag (20 iters) | h1diag 12/12/12 | 2000 | **4,001** | **−326.9107** | **+0.281 Ha** |
| SC strong_pair (42 iters) | strong_pair 12/12/12 | 2000 | 6,000 | −326.7119 | +0.480 Ha |
| SC strong_pair | h1diag 12/12/12 | 2000 | 4,001 | −326.7053 | +0.487 Ha |
| SC h1diag | strong_pair 12/12/12 | 2000 | 6,000 | −326.5854 | +0.607 Ha |

The h1diag partition with self-consistent h1diag gamma is the best combination. Fragment 2 is closed-shell in this partition (1 determinant always), so the per-fragment det budget concentrates on the two correlated fragments. The energy converges quickly with budget — going from 1000 to 2000 dets/frag gains only 0.004 Ha.

### Cross-Fragment PT2 Diagnostic

Epstein-Nesbet PT2 using determinant-energy denominators — no TrimCI calls.

| Partition | n_terms | neg_gap_frac | E_PT2_cross |
|-----------|---------|--------------|-------------|
| h1diag | 3888 | 0.2492 | −0.000422 Ha |
| strong_pair | 7889 | 0.2354 | −0.013873 Ha |

Second-order cross-fragment coupling is negligible relative to the MFA gap. Closing the remaining +0.281 Ha requires higher-order inter-fragment correlation.

---

## Key Numbers

| Quantity | Value |
|----------|-------|
| Brute-force energy | −327.1920 Ha |
| Brute-force dets | 10,095 |
| Uncoupled baseline dets | 118 (1.2%) |
| SC MFA h1diag (1000 dets/frag) | −326.9073 Ha, +0.285 Ha, 2,017 dets |
| **SC MFA h1diag (2000 dets/frag)** | **−326.9107 Ha, +0.281 Ha, 4,001 dets** |
| GS convergence, h1diag | 20 outer iterations |
| GS convergence, strong_pair | 42 outer iterations |

---

## File Structure

```
FRASCI/
├── README.md
├── FRASCI_Main.ipynb          end-to-end: uncoupled baseline → SC gamma → SC MFA → PT2 → summary
├── FRASCI_Results.ipynb       executed results reference
├── generate_thesis_plots.py
├── requirements.txt
│
├── data/
│   ├── fcidump_cycle_6        FCIDUMP integrals, 36 orbs, 54 electrons (9.2 MB)
│   └── dets.npz               10,095 reference dets, row 0 = correlated ref det
│
├── FRASCI/                    Python package
│   ├── core/
│   │   ├── fragment.py        fragment_by_sliding_window, extract_fragment_integrals
│   │   ├── trimci_adapter.py  solve_fragment_trimci, FragmentResult
│   │   ├── results.py         FragmentedRunResult dataclass
│   │   └── analysis.py        determinant_summary, convergence_summary
│   ├── uncoupled/             regression-locked → 118 dets
│   │   └── solver.py          run_fragmented_trimci
│   ├── mfa/
│   │   ├── helpers.py         compute_fragment_rdm1, dress_integrals_meanfield
│   │   ├── solver.py          make_nonoverlapping_partition, run_mfa_d2, load_ref_det
│   │   ├── energy.py          mf_global_energy, mf_embedded_energy, correlation_total_energy
│   │   ├── extract_full_gamma.py
│   │   └── runners/
│   └── crossflow/
│       ├── determinant_pt2.py
│       ├── partition_candidates.py    make_strong_pair_partition, write_partition_json
│       └── runners/
│
├── tests/
└── Outputs/
    └── mfa/                   GS gamma runs and D2 results per timestamp
```

---

## How to Run

```bash
source /Users/abhimuvva/Documents/Masters_Projs/Proj_Flow/FRASCIenv/bin/activate
cd /Users/abhimuvva/Documents/Masters_Projs/Proj_Flow/FRASCI

# Regression check:
python -c "
from FRASCI.uncoupled import run_fragmented_trimci
r = run_fragmented_trimci('data/fcidump_cycle_6')
assert r.total_dets == 118
print('Uncoupled baseline regression: OK')
"

# Run full pipeline (notebook):
jupyter notebook FRASCI_Main.ipynb

# Unit tests:
python -m pytest tests/ -q
```

---

## Critical Rules

- **Energy formula:** `E_total = E_mf_global + Σ_I (E_TrimCI_I − E_mf_emb_I)` — never sum fragment energies directly.
- **E_nuc = 0.0** — already absorbed into FCIDUMP. Do not add it again.
- **Reference det:** always use `dets.npz` row 0 for fragment electron counting, not the HF reference.
- **Dressing formula:** `J − ½K` with spin-summed γ ∈ [0, 2]. Not `2J − K`.
- **Uncoupled baseline regression lock:** `run_fragmented_trimci()` must always return `total_dets=118, fragment_n_dets=[51, 51, 16]`.
