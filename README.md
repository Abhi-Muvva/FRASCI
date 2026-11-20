# FRASCI

FRASCI (**FRA**gment-based **S**elected **CI**) is a variational method for
large active spaces. It builds a self-consistent localized-active-space Parent
from phased TrimCI fragment solvers, keeps the complete fragment product as one
contracted reference, and selects explicit determinants to recover
interfragment correlation in the Parent's fixed orbital frame.

This repository is the code-and-data release for *Fragment-based Selected
Configuration Interaction*. It contains the released implementation, focused
tests, molecular FCIDUMPs with provenance, frozen paper result records, and two
notebooks that contain both the calculation code and the plots used to inspect
the results.

## Method in one minute

Given an FCIDUMP and a user-supplied fragment partition, FRASCI performs:

1. **Parent construction.** Phased TrimCI solves the embedded fragment
   problems inside a LASSCF orbital-optimization cycle.
2. **Fixed-frame Recovery.** The converged Parent is retained as one contracted
   vector. TrimCI proposes dispersion and charge-transfer determinants, which
   are ranked and retained up to a requested determinant budget.
3. **Variational solve.** FRASCI solves the generalized eigenproblem in the
   nonorthogonal basis containing the contracted Parent and the explicit
   Recovery determinants.
4. **Optional refinements.** Expanded-Parent and post-Recovery COO calculations
   are separate, opt-in diagnostics. They are not part of the default Recovery
   result.

`frasci.recovery.solve()` evaluates one requested Recovery target. The separate
`frasci.recovery.benchmark_ladder` module runs a sequence of targets for a
convergence study. A completed schedule is not automatically a converged
scientific result; inspect the reported stop reason, achieved determinant count,
residual, and gates.

## Installation

FRASCI requires Python 3.12 and research builds of TrimCI and `mrh`.

```bash
git clone https://github.com/Abhi-Muvva/FRASCI.git
cd FRASCI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
```

The dependency pins in `pyproject.toml` are part of the numerical provenance:

- `TrimCI-FRASCI` commit `027cb57ba2cbd811bea1230ad51af76259528785`
- `mrh` commit `a65600830b8ef7be963dcc203057ebf6baf7dbc4`

The FRASCI-specific TrimCI fork provides Recovery policies and deterministic
restart seeding that are absent from stock TrimCI. A fresh installation will
therefore fail until that exact fork is publicly readable or the installer has
GitHub credentials for it. This dependency must be made accessible before the
paper's code-availability link is considered independently reproducible.

`mrh` and TrimCI compile native extensions. On macOS, their own build
instructions may require OpenMP and a non-Apple compiler.

## Minimal calculation

The example below uses the released AlFe2 `(11e,10o)` FCIDUMP. The fragment
partition, electron counts, and spin multiplicities are part of the physical
input and must be changed together for another system.

```python
from pathlib import Path

from frasci import FCIDUMPInput
from frasci.parent import ParentConfig, solve as solve_parent
from frasci.recovery import RecoveryConfig, solve as solve_recovery

fcidump = Path("notebooks/data/alfe2_11o10e/FCIDUMP")
parent_dir = Path("notebooks/runs/alfe2_11o10e/parent")

parent = solve_parent(
    fcidump_path=str(fcidump),
    orbital_lists=[list(range(5)), list(range(5, 10))],
    nelec_per_frag=[(1, 5), (5, 0)],
    spin_sub=[5, 6],
    config=ParentConfig(
        threshold=1e-5,
        max_cycle_macro=20,
        orbopt=True,
        orbopt_cycles=5,
        phase0_dets=100,
        phase1_dets=1000,
        discovery_num_runs=16,
        refine_num_runs=1,
        rng_seed=0,
    ),
    save_dets_to=parent_dir,
)
if not parent.converged:
    raise RuntimeError("Parent did not converge")

result = solve_recovery(
    parent_dir,
    FCIDUMPInput.from_file(fcidump),
    RecoveryConfig(recovery_target=1000),
)

print(f"Parent energy:   {result.h00:.12f} Ha")
print(f"Recovery energy: {result.energy:.12f} Ha")
print(f"Achieved N:      {result.diagnostics['growth']['native_actual_n']}")
print(result.gates)
```

Expect the Parent calculation to print LASSCF macroiterations and the Recovery
calculation to report the requested and achieved determinant counts. Production
systems can require hours and substantial memory; Fe4S4 is not a smoke test.

## Paper notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

- `notebooks/01_paper_benchmarks.ipynb` contains the six molecular system
  definitions, from-scratch Parent/Recovery campaign code, accepted endpoint
  tables, Recovery ladders, Expanded-Parent diagnostics, and the Fe4S4 COO
  waterfall. It loads the small frozen result manifest by default. Set
  `RECOMPUTE = True` to execute electronic-structure calculations into the
  ignored `notebooks/runs/` directory.
- `notebooks/02_size_extensivity.ipynb` builds the H6 unit Hamiltonian, shows how
  the noninteracting block-diagonal replicas and whole-system TrimCI runs are
  generated, and reproduces the paper's size-extensivity graph. Frozen values
  load by default because the largest TrimCI point uses 2.56 million
  determinants.

The notebooks display plots inline and never save figure files. Cached
calculation state is optional: deleting `notebooks/runs/` and enabling
recomputation starts from the distributed FCIDUMPs.

## Results and validation

`notebooks/data/paper_results.json` is the compact, machine-readable record used
by the paper notebooks. It stores accepted energies, determinant counts,
compatible Recovery ladders, and the H6 control values. The larger `.npz`
reservoirs, Parent checkpoints, generated HTML, and TrimCI scratch directories
are intentionally excluded from Git because they are recomputable runtime
state, not release inputs.

The release tests cover input validation, Parent-frame handling, Recovery
growth, generalized and matrix-free solves, exact-union reconstruction,
checkpoint compatibility, and the benchmark-ladder stopping rules.

```bash
python -m pytest -q
```

For a quicker check that avoids long end-to-end cases:

```bash
python -m pytest -q -m "not slow"
```

Interpret the output quantities separately:

- `Result.h00` is the complete contracted Parent expectation value.
- `Result.energy` is the fixed-frame contracted-plus-explicit Recovery energy.
- `Result.union_energy` is the independently optimized explicit determinant
  union when that calculation is feasible.
- `Result.free_coo_energy` exists only when post-Recovery COO was explicitly
  requested.
- `Result.converged` and `Result.gates` certify the stated variational solve;
  they do not prove that the selected determinant space is complete.

## Repository layout

```text
frasci/       released Python implementation
tests/        focused unit and integration tests
notebooks/    two paper-facing, runnable notebooks
notebooks/data/
              FCIDUMPs, provenance, and compact frozen result records
```

## Scope and limitations

FRASCI does not choose the fragment partition. Recovery is limited by the
admitted charge-transfer ranks, the eligible candidate reservoir, and the
requested determinant budget. The current Fe4S4 post-Recovery COO point reached
its orbital-optimization iteration limit and is reported as a variational
candidate rather than a converged orbital optimum. External DMRG values and the
same-FCIDUMP whole-system TrimCI+COO value are comparators, not exact energies.

## License

FRASCI's original code is released under the [MIT License](LICENSE). Third-party
dependencies and molecular input data remain subject to their own licenses and
provenance terms. In particular, `mrh` is GPL-licensed; installing and using the
combined software requires compliance with that license as well.

## Citation

The manuscript citation will be added when a public preprint or DOI is
available. Until then, cite this repository and the exact commit used for the
calculation.
