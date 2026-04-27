# FRASCI

FRASCI is a Fragmented Selected Configuration Interaction workflow for the
Fe4S4 active space. The goal is to approximate a full selected-CI calculation
by solving smaller orbital fragments and combining their energy corrections.

**System:** Fe4S4 cluster, 36 spatial orbitals, 27 alpha + 27 beta electrons  
**Reference:** brute-force TrimCI, E = -327.1920 Ha, 10,095 determinants

## How FRASCI Works

FRASCI starts from the full active-space Hamiltonian and a reference determinant
set, then replaces one large selected-CI problem with several smaller fragment
problems.

The current non-overlapping workflow uses three 12-orbital fragments:

```text
36 active orbitals -> 12 + 12 + 12 orbital fragments
```

Each fragment still needs to feel the rest of the molecule. To do that, FRASCI
first builds a mean-field density vector:

```text
Outputs/meanfield_active/outs_extraction_autodets/gamma_mixed_final.npy
```

This `gamma_mixed_final.npy` file is not a hand-copied artifact. It is rebuilt
from `data/fcidump_cycle_6` and `data/dets.npz`. In plain terms, it tells each
fragment what the other fragments look like on average.

After gamma is available, FRASCI runs selected CI on each fragment with this
mean-field environment included. The fragment results are then assembled into
one total energy:

```text
global mean-field energy + fragment selected-CI corrections
```

So the repository has two main jobs:

1. Build the mean-field density used by the fragments.
2. Run and compare overlapping and non-overlapping fragment calculations.

## Overlapping vs Non-Overlapping

FRASCI currently has two fragment styles.

The non-overlapping workflow is the completed energy workflow in this
repository. Every active orbital belongs to exactly one of the three
12-orbital fragments. Because the fragments do not overlap, the fragment
corrections can be added back to the global mean-field energy in a cleaner way.

The reported non-overlapping rows use the `h1diag` partition: orbitals are
sorted by the one-electron Hamiltonian diagonal and split into three groups. A
balanced occupation-based partition was also tested, but it did not improve the
energy for this regenerated gamma setup.

The overlapping workflow has also been tested as a diagnostic. It uses fragment
windows that share orbitals with neighboring windows, which can expose boundary
correlations more clearly. The next continuation of this project is to make the
overlapping workflow quantitative by adding a mechanism that avoids
double-counting the shared overlap regions.

## Running the Workflow

Install the Python dependencies into your environment first. The external
TrimCI package/extension must also be installed separately, because FRASCI uses
it as the selected-CI engine inside each fragment.

```bash
pip install -r requirements.txt
```

Run the notebooks from this folder in this order:

```bash
jupyter notebook Generate_Gamma_Mixed_Final.ipynb
jupyter notebook FRASCI_Results.ipynb
```

`Generate_Gamma_Mixed_Final.ipynb` rebuilds the gamma density file from scratch.
Run this when the gamma file is missing, stale, or you want the run to be fully
reproducible from the raw input files.

`FRASCI_Results.ipynb` uses that gamma file to run the main comparisons:

- overlapping diagnostic run
- non-overlapping h1diag baseline
- non-overlapping h1diag best run

## Optional Gamma Experiment

`Generate_Gamma_Mixed_Final_Experiment.ipynb` is a safer notebook for testing
improved gamma updates. It does not replace the main `gamma_mixed_final.npy`.
Instead, it writes candidates under:

```text
Outputs/meanfield_active/gamma_experiments/
```

This notebook uses smaller density mixing, adaptive damping, and saved
checkpoints. The idea is to test candidate gamma files against the
non-overlapping workflow before promoting any of them into the main output path.

## Current Results

Here `t` is the selected-CI screening threshold used inside each fragment. A
larger value keeps only the strongest determinant candidates, so the run is
cheaper but less accurate. A smaller value lets weaker candidates enter the
fragment calculation, so the determinant count and cost go up, but the energy
usually moves closer to the brute-force reference.

In that sense, `t = 0.06` is the faster/coarser setting, while `t = 0.01` is a
more expensive refinement. The values are small because they are thresholds on
Hamiltonian coupling/selection strength in Hartree-scale calculations; they are
not percentages.

| Method | Determinants | Cost | E_total (Ha) | Error |
|---|---:|---:|---:|---:|
| Brute-force TrimCI | 10,095 | 100% | -327.1920 | -- |
| Non-overlapping h1diag, t=0.06 | 147 | 1.5% | -326.3226 | +0.8694 Ha |
| Non-overlapping h1diag, t=0.01 | 2,017 | 20.0% | -326.4263 | +0.7657 Ha |

These values come from the regenerated gamma workflow. Small numerical changes
between reruns are expected because selected-CI determinant selection can vary
slightly. The table keeps the main non-overlapping h1diag path because it gives
the best regenerated result in this repository.

## Folder Structure

```text
FRASCI/
├── README.md
├── requirements.txt
├── Generate_Gamma_Mixed_Final.ipynb
├── Generate_Gamma_Mixed_Final_Experiment.ipynb
├── FRASCI_Results.ipynb
├── data/
│   ├── fcidump_cycle_6
│   └── dets.npz
├── Outputs/
│   ├── meanfield_active/
│   │   └── outs_extraction_autodets/
│   │       └── gamma_mixed_final.npy
│   └── mfa/
└── FRASCI/
    ├── core/
    │   ├── fragment.py
    │   ├── trimci_adapter.py
    │   ├── results.py
    │   └── analysis.py
    └── mfa/
        ├── solver.py
        ├── energy.py
        ├── gamma_bootstrap.py
        ├── gamma_experiment.py
        └── helpers.py
```

The main pieces are:

- `data/`: raw Hamiltonian and reference determinants.
- `Generate_Gamma_Mixed_Final.ipynb`: rebuilds the official gamma density file.
- `FRASCI_Results.ipynb`: runs the overlapping diagnostic and main
  non-overlapping FRASCI results.
- `FRASCI/core/`: shared fragment construction and selected-CI adapter code.
- `FRASCI/mfa/`: mean-field dressing, gamma generation, and fragment solvers.
- `Outputs/`: generated local results.

The repository keeps the `Outputs/` folder structure with small `.gitkeep`
placeholder files. Generated payloads inside `Outputs/`, including
`gamma_mixed_final.npy`, are intentionally ignored by git because they can be
recreated from the notebooks.
