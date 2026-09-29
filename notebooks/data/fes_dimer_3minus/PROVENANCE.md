# fes_dimer_3minus FCIDUMP provenance

`FCIDUMP` in this directory is the (11e, 20o) active-space integral dump for the
[2Fe-2S(SCH3)4]3- mixed-valence dimer, Section 3.3.3 of the LASSIS JCTC 2025 paper
(doublet, S=1/2, ground state).

- **SHA-256 (FCIDUMP):**
  `728ead3a1b49143ad12e3c1be749afbb9131047923b34a1a8531e3dc420f62a5`
- **Verified header:** `NORB=20, NELEC=11, MS2=-1` (n_alpha=5, n_beta=6), matching the
  source config's `active_space.nelec: [5, 6]` -- checked directly against the raw file
  text, not only through the Python bundle object.

## Source

- **Config:** `mols/LASSIS/fes_dimer_3minus/config.yaml`
  (SHA-256 `d1ac45c20b14bddaca5126da7680d0fbe894d2f5a6cb23ef562a5703d2be6951`)
- **Orbital checkpoint:** `mols/LASSIS/fes_dimer_3minus/scf_orbitals.chk`,
  dataset `las/mo_coeff` (SHA-256
  `c909c9da2ef30f3bb8b8b5ff69a4cc906be7b2a9d700519abf9a98681054b535`) -- the paper
  authors' own Zenodo checkpoint (`si5_lassis_FeSDimers/2Fe2S_3/FeSdimer.py`), active
  columns 88..107 (0-indexed) of the full PySCF MO matrix.
- **Charge/spin:** charge -3, ROHF reference `mol.spin=9` (S=9/2, the paper's own SCF
  guess spin); active-space `NELEC=11, MS2=-1` is the doublet (S=1/2) the paper reports
  as the ground state -- these are two different spins at two different levels
  (SCF-reference vs. active-space) and are not in conflict.
- **Fragmentation** (`paper_fe_sites`, for context -- not baked into the FCIDUMP itself):
  2 fragments, one per Fe site: orbitals `[0..9]`/`[10..19]`,
  `nelec_per_frag=[[5,1],[0,5]]`, `spin_sub=[5,6]`.
- **Basis:** C/H cc-pvdz, S/Fe cc-pvtz; density-fit (aug_etb).

## Generation

Built once, offline, via the development-tree `engine.benchmark.MoleculeBenchmark`
(never imported at notebook runtime -- see release README section on import discipline):

```python
from engine.benchmark import MoleculeBenchmark
bench = MoleculeBenchmark(
    "mols/LASSIS/fes_dimer_3minus/config.yaml",
    results_root=<scratch dir>,
)
bundle = bench.build_integrals("eq")
```

Independently verified after generation: `bundle.n_orb == 20`, `bundle.n_alpha == 5`,
`bundle.n_beta == 6`; the raw FCIDUMP header (`NORB=`, `NELEC=`, `MS2=` line) was also
read directly as text and cross-checked against these values (not solely through the
Python bundle object).

## Software versions used to generate this file

- Python 3.12.14, Darwin 27.0.0 arm64 (macOS)
- pyscf 2.13.0
- mrh commit `a65600830b8ef7be963dcc203057ebf6baf7dbc4` (2026-06-01)
- TrimCI-FRASCI commit `027cb57ba2cbd811bea1230ad51af76259528785`
  (`github.com/Abhi-Muvva/TrimCI-FRASCI`) -- the same commit this release is pinned to;
  TrimCI itself is not actually used by `MoleculeBenchmark.build_integrals`, recorded
  here only as "installed alongside" for completeness.
- Date generated: 2026-09-25

## Scope note

Only this static FCIDUMP and this provenance record are placed here. No recovery smoke
tests or production sweeps have been run against this system in this release checkout;
that is deferred to a later phase (see the release README, section 8, and
`docs/superpowers/plans/2026-09-25-frasci-notebooks-revised-matrix.md`).
