"""diff_mols: multi-molecule warm-up benchmark.

See docs/superpowers/specs/2026-06-22-multi-molecule-benchmark-design.md.
"""
from FRASCI.diff_mols.benchmark import MoleculeBenchmark
from FRASCI.diff_mols.config import MoleculeConfig, load_molecule_config
from FRASCI.diff_mols.tuning import (
    LASSCF_TRIMCI_COO_PROTOCOLS, LASSCF_TRIMCI_PROTOCOLS, LASSIS_PROTOCOLS, TRIMCI_COO_PROTOCOLS,
    TRIMCI_PROTOCOLS, TUNING_PRESETS, expand_protocol_names,
    resolve_lasscf_trimci_coo_protocol, resolve_lasscf_trimci_protocol, resolve_lassis_protocol,
    resolve_tuning_overrides, resolve_trimci_coo_protocol, resolve_trimci_protocol,
)

__all__ = [
    "MoleculeBenchmark", "MoleculeConfig", "load_molecule_config",
    "LASSCF_TRIMCI_COO_PROTOCOLS", "LASSCF_TRIMCI_PROTOCOLS", "LASSIS_PROTOCOLS", "TRIMCI_COO_PROTOCOLS",
    "TRIMCI_PROTOCOLS", "TUNING_PRESETS", "expand_protocol_names",
    "resolve_lasscf_trimci_coo_protocol", "resolve_lasscf_trimci_protocol", "resolve_lassis_protocol",
    "resolve_tuning_overrides", "resolve_trimci_coo_protocol", "resolve_trimci_protocol",
]
