# Molecule configuration layout

This integration includes the five complete configurations:

```text
configs/
└── diff_mols/
    ├── me2n2.yaml
    ├── c2h6n4_tetrazene.yaml
    ├── diazene_cis.yaml
    ├── diazene_trans.yaml
    ├── cr2_oh3_nh3_6.yaml
    └── geometries/
        └── cr2_oh3_nh3_6.xyz
```

`fe2_dimer.yaml` and `fe2s2.yaml` are intentionally deferred until their exact
geometries, charge/spin states, active spaces, and references are selected.

## Required schema

Each YAML must contain:

- `name`, `slug`, `description`
- `geometry`
- `electronic_structure`
- `active_space`
- `fragmentation`
- all five entries under `methods`
- `reference`

Optional molecule-specific blocks are `j_coupling` and `spin_gap`.

## Values you still need to verify

- Metal active spaces: run integrals-only first and inspect AVAS selections
  before launching methods.
- References: add a numerical `e_ref` only when you have a compatible
  geometry/basis/active-space reference. Otherwise keep `computed_inline: false`.

Organic and Cr2 geometries here are sourced from the local `mrh/examples`
material or are explicitly marked idealized scan templates.
