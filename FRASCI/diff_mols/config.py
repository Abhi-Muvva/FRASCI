"""MoleculeConfig dataclasses + YAML loader for diff_mols.

Mirrors the schema in spec §3.6 and Appendix B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GeometrySpec:
    kind: str                          # "single" or "scan"
    xyz_inline: str | None = None      # for kind=single (inline) — alternative to xyz_file
    xyz_file: str | None = None        # for kind=single (file ref) — relative to config file
    template_engine: str = "jinja2"    # for kind=scan
    template: str | None = None        # for kind=scan
    scan_param: str | None = None      # for kind=scan
    scan_points: list[dict] = field(default_factory=list)  # for kind=scan


@dataclass
class ElectronicStructureSpec:
    charge: int
    spin: int                          # 2S, NOT multiplicity
    basis: str
    scf: str                           # "RHF" | "ROHF" | "UHF→ROHF"


@dataclass
class ActiveSpaceSpec:
    kind: str                          # "explicit" | "avas" | "window"
    orb_indices: list[int] | None = None        # explicit
    nelec: list[int] | None = None              # explicit / window — [n_alpha, n_beta]
    avas_patterns: list[str] | None = None      # avas
    norb: int | None = None                     # window
    n_active_elec: int | None = None            # window
    casscf: bool = False
    boys: bool = False                   # apply Boys localization to active MOs


@dataclass
class FragmentationSpec:
    name: str                          # label used by methods.*.partitions
    orbital_lists: Any                 # "auto_per_atom" | "auto_per_metal" | "h1diag" | "chem_bond" | list[list[int]]
    nelec_per_frag: Any = "auto"
    spin_sub: Any = "auto"
    n_frags: int | None = None         # only meaningful for "h1diag": split active space into N equal-size fragments (default 2)
    atom_groups: list[list[int]] | None = None   # required for "chem_bond": list of 0-indexed atom groups; each group becomes one fragment


@dataclass
class MethodSpec:
    enabled: bool = False
    threshold: float | None = None
    max_dets: int | None = None
    num_runs: int | None = None
    max_rounds: int | None = None
    coo_cycles: int | None = None
    bfgs_maxiter: int | None = None
    bfgs_ftol: float | None = None
    davidson_tol: float | None = None
    max_cycle_macro: int | None = None
    trimci_threshold: float | None = None
    trimci_max_dets: int | None = None
    warm_start_kappa: bool | None = None
    parallel_workers: int | None = None
    process_workers: int | None = None
    omp_threads_per_frag: int | None = None
    protocols: list[str] = field(default_factory=list)
    partitions: list[str] = field(default_factory=list)
    on_top_of: list[str] = field(default_factory=list)   # lassis: list of LASSCF method names
    n_charge: str | None = None
    n_spin: int | None = None
    n_spin_values: list[int] = field(default_factory=list)
    opt: int | None = None


@dataclass
class MethodSetSpec:
    plain_trimci: MethodSpec
    trimci_coo: MethodSpec
    lasscf_cas: MethodSpec
    lasscf_trimci: MethodSpec
    lasscf_trimci_coo: MethodSpec
    lassis: MethodSpec


@dataclass
class ReferenceSpec:
    source: str = ""
    computed_inline: bool = False
    e_ref: float | None = None         # absolute reference if user supplies one


@dataclass
class JCouplingSpec:
    enabled: bool = False
    hs_state: int = 0
    bs_state: int = 1
    formula: str = "yamaguchi"


@dataclass
class SpinGapSpec:
    enabled: bool = False
    low_spin_state: int = 0
    high_spin_state: int = 1


@dataclass
class MoleculeConfig:
    name: str
    slug: str
    description: str
    geometry: GeometrySpec
    electronic_structure: ElectronicStructureSpec
    active_space: ActiveSpaceSpec
    fragmentation: list[FragmentationSpec]
    methods: MethodSetSpec
    reference: ReferenceSpec
    j_coupling: JCouplingSpec | None = None
    spin_gap: SpinGapSpec | None = None
    raw: dict = field(default_factory=dict)
    source_path: Path | None = None


_REQUIRED_TOP = ("name", "slug", "description", "geometry", "electronic_structure",
                 "active_space", "fragmentation", "methods")


def _require(d: dict, keys: tuple[str, ...], context: str):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"{context}: missing required fields: {missing}")


def load_molecule_config(path: str | Path) -> MoleculeConfig:
    """Load a molecule YAML and validate the top-level structure."""
    path = Path(path)
    with path.open() as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    _require(raw, _REQUIRED_TOP, str(path))

    geometry = GeometrySpec(**raw["geometry"])
    es = ElectronicStructureSpec(**raw["electronic_structure"])
    aspace = ActiveSpaceSpec(**raw["active_space"])
    frags = [FragmentationSpec(**f) for f in raw["fragmentation"]]

    m = raw["methods"]
    _require(m, ("plain_trimci", "trimci_coo", "lasscf_cas",
                 "lasscf_trimci_coo", "lassis"), f"{path} methods")
    methods = MethodSetSpec(
        plain_trimci=MethodSpec(**m["plain_trimci"]),
        trimci_coo=MethodSpec(**m["trimci_coo"]),
        lasscf_cas=MethodSpec(**m["lasscf_cas"]),
        lasscf_trimci=MethodSpec(**m.get("lasscf_trimci", {"enabled": False})),
        lasscf_trimci_coo=MethodSpec(**m["lasscf_trimci_coo"]),
        lassis=MethodSpec(**m["lassis"]),
    )

    ref = ReferenceSpec(**raw.get("reference", {})) if "reference" in raw else ReferenceSpec()
    j = JCouplingSpec(**raw["j_coupling"]) if "j_coupling" in raw else None
    sg = SpinGapSpec(**raw["spin_gap"]) if "spin_gap" in raw else None

    return MoleculeConfig(
        name=raw["name"], slug=raw["slug"], description=raw["description"],
        geometry=geometry, electronic_structure=es, active_space=aspace,
        fragmentation=frags, methods=methods, reference=ref,
        j_coupling=j, spin_gap=sg, raw=raw, source_path=path,
    )


# -- Geometry tag helpers -----------------------------------------------------

def geom_tag_for_single() -> str:
    return "eq"


def geom_tag_for_scan_point(point: dict, scan_param: str) -> str:
    val = point[scan_param]
    return f"{scan_param}{val:.2f}"
