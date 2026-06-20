# FRASCI LASSCF

Focused four-fragment LASSCF → TrimCI → LASSI/LASSIS implementation for the
36-orbital Fe₄S₄ FCIDUMP.

## Workflow

The fixed partition is `h1diag_rev_block_6x8x10x12`:

1. CAS-LASSCF/CSF control
2. TrimCI-LASSCF cold start
3. TrimCI-LASSCF warm start from CAS orbitals
4. LASSI/LASSIS from CAS, cold, and warm checkpoints
5. LASSIS expansions with `nspin = 0, 1, 2`

[FRASCIMain.ipynb](FRASCIMain.ipynb) is the runnable pipeline. It defaults to
reuse mode and reads the saved June 16, 2026 calculation, avoiding multi-hour
solver calls. Set `RUN_EXPENSIVE = True` only when a fresh calculation is
intended.

[FRASCI_Results.ipynb](FRASCI_Results.ipynb) is the executed analysis notebook
with compact tables and millihartree-scale energy comparisons.

## Saved results: `nspin=1` and `nspin=2`

| LASSCF checkpoint | LASSIS `nspin=1` (Ha) | Lowering from LASCI (mHa) | LASSIS `nspin=2` (Ha) | Lowering from LASCI (mHa) |
|---|---:|---:|---:|---:|
| CAS | −326.961894349 | 32.486 | **−326.993920794** | 64.513 |
| TrimCI cold | −326.913929569 | 32.426 | −326.918928192 | 37.424 |
| TrimCI warm from CAS | **−326.976577039** | 50.543 | −326.991386656 | **65.353** |

For `nspin=1`, the warm checkpoint gives the lowest energy, sitting
14.683 mHa below the CAS result; the cold result is 47.965 mHa above CAS.
For `nspin=2`, CAS gives the lowest energy, while warm is nearly degenerate at
only 2.534 mHa above CAS. Cold remains 74.993 mHa above CAS.

Increasing the LASSIS expansion from `nspin=1` to `nspin=2` lowers the energy
by 32.026 mHa for CAS, 4.999 mHa for cold, and 14.810 mHa for warm.

All three LASSCF orbital optimizations reached the configured 100-cycle limit
and are recorded as `NOT_CONVERGED`. The attempted LASSI state-average
calculations also preserve their error records; missing energies are not
substituted with zeros.

## Repository layout

```text
FRASCI/
├── FRASCIMain.ipynb
├── FRASCI_Results.ipynb
├── FRASCI/
│   └── lasscf/
│       ├── runners/
│       ├── fragments.py
│       ├── fragment_sweep.py
│       ├── mock_scf.py
│       ├── support.py
│       ├── trimci_adapter.py
│       ├── trimci_kernel.py
│       └── trimci_to_civec.py
├── data/
│   ├── fcidump_cycle_6
│   ├── dets.npz
│   └── partitions/h1diag_rev_block_6x8x10x12.json
├── Outputs/lasscf/h1_4frag_pipeline_20260616_122809/
├── tests/lasscf/
├── scripts/build_lasscf_notebooks.py
└── requirements.txt
```

## Run

```bash
source ../FRASCIenv/bin/activate
jupyter notebook FRASCIMain.ipynb

# LASSCF implementation tests
python -m pytest tests/lasscf -q
```
