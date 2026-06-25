"""Test MoleculeConfig loading + validation."""
import textwrap
from pathlib import Path

import pytest

from FRASCI.diff_mols.config import load_molecule_config, MoleculeConfig


CONFIGS_DIR = Path(__file__).parents[2] / "configs" / "diff_mols"
EXPECTED_SLUGS = {
    "me2n2", "c2h6n4_tetrazene", "diazene_cis", "diazene_trans",
    "cr2_oh3_nh3_6",
}


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_shipped_config_parses(slug: str):
    """Every YAML in configs/ must parse and expose all benchmark methods."""
    cfg = load_molecule_config(CONFIGS_DIR / f"{slug}.yaml")
    assert cfg.slug == slug
    for m in ("plain_trimci", "trimci_coo", "lasscf_cas",
              "lasscf_trimci", "lasscf_trimci_coo", "lassis"):
        assert hasattr(cfg.methods, m), f"{slug}: missing method {m!r}"


def test_lassis_nspin_sweep_is_scoped_to_metal_configs():
    """Only the open-shell metal benchmarks opt into the expensive LASSIS spin sweep."""
    metal_slugs = {"cr2_oh3_nh3_6"}
    for slug in sorted(EXPECTED_SLUGS):
        cfg = load_molecule_config(CONFIGS_DIR / f"{slug}.yaml")
        expected = [0, 1, 2, 3] if slug in metal_slugs else []
        assert cfg.methods.lassis.n_spin_values == expected


def test_load_minimal_single_point(tmp_path: Path):
    yaml_text = textwrap.dedent("""
        name: H2
        slug: h2_test
        description: "H2 single-point smoke molecule"
        geometry:
          kind: single
          xyz_inline: |
            2
            H2
            H 0.0 0.0 0.0
            H 0.0 0.0 0.74
        electronic_structure:
          charge: 0
          spin: 0
          basis: sto-3g
          scf: RHF
        active_space:
          kind: explicit
          orb_indices: [0, 1]
          nelec: [1, 1]
          casscf: false
        fragmentation:
          - name: chem
            orbital_lists: auto_per_atom
        methods:
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 100}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: true, max_cycle_macro: 10, partitions: [chem]}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference:
          source: "FCI/sto-3g inline"
          computed_inline: true
    """)
    cfg_path = tmp_path / "h2.yaml"
    cfg_path.write_text(yaml_text)

    cfg = load_molecule_config(cfg_path)

    assert isinstance(cfg, MoleculeConfig)
    assert cfg.slug == "h2_test"
    assert cfg.geometry.kind == "single"
    assert "0.74" in cfg.geometry.xyz_inline
    assert cfg.electronic_structure.basis == "sto-3g"
    assert cfg.active_space.kind == "explicit"
    assert cfg.active_space.orb_indices == [0, 1]
    assert cfg.active_space.nelec == [1, 1]
    assert cfg.fragmentation[0].name == "chem"
    assert cfg.fragmentation[0].orbital_lists == "auto_per_atom"
    assert cfg.methods.plain_trimci.enabled is True
    assert cfg.methods.lasscf_cas.partitions == ["chem"]
    assert cfg.methods.lassis.enabled is False
    assert cfg.reference.computed_inline is True


def test_load_scan_geometry(tmp_path: Path):
    yaml_text = textwrap.dedent("""
        name: H2 scan
        slug: h2_scan
        description: "H2 PES"
        geometry:
          kind: scan
          template_engine: jinja2
          template: |
            2
            H2 r={{ "%.4f"|format(r) }}
            H 0.0 0.0 0.0
            H 0.0 0.0 {{ "%.4f"|format(r) }}
          scan_param: r
          scan_points:
            - {r: 0.7}
            - {r: 1.4}
        electronic_structure: {charge: 0, spin: 0, basis: sto-3g, scf: RHF}
        active_space: {kind: explicit, orb_indices: [0,1], nelec: [1,1], casscf: false}
        fragmentation:
          - {name: chem, orbital_lists: auto_per_atom}
        methods:
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 100}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: false}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference: {source: "FCI", computed_inline: true}
    """)
    cfg_path = tmp_path / "h2_scan.yaml"
    cfg_path.write_text(yaml_text)

    cfg = load_molecule_config(cfg_path)
    assert cfg.geometry.kind == "scan"
    assert cfg.geometry.scan_param == "r"
    assert len(cfg.geometry.scan_points) == 2
    assert cfg.geometry.scan_points[0] == {"r": 0.7}
    assert cfg.geometry.scan_points[1] == {"r": 1.4}


def test_missing_required_field_raises(tmp_path: Path):
    yaml_text = "name: bad\nslug: bad\n"
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml_text)
    with pytest.raises(ValueError, match="missing required"):
        load_molecule_config(cfg_path)


def test_geom_tag_helpers(tmp_path: Path):
    """Single-point → 'eq'; scan → 'r1.40' style tags."""
    from FRASCI.diff_mols.config import geom_tag_for_single, geom_tag_for_scan_point
    assert geom_tag_for_single() == "eq"
    assert geom_tag_for_scan_point({"r": 1.4}, scan_param="r") == "r1.40"
    assert geom_tag_for_scan_point({"r": 1.0}, scan_param="r") == "r1.00"
