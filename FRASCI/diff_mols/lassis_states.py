"""State preparation and LASSIS solvers for benchmarks."""
from __future__ import annotations
from pathlib import Path
from FRASCI.diff_mols.config import MoleculeConfig


def build_lassis_kwargs(config: MoleculeConfig, lasscf_checkpoint_dir: Path) -> dict:
    """Pick lassis_ncharge / lassis_nspin / opt from config.methods.lassis."""
    spec = config.methods.lassis
    return {
        "lassis_ncharge": spec.n_charge or "s",
        "lassis_nspin": int(spec.n_spin or 0),
        "opt": int(spec.opt or 1),
    }
