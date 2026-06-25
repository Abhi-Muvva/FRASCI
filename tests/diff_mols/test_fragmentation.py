"""Fragmentation resolution: h1diag + auto_per_atom + auto_per_metal."""
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from FRASCI.diff_mols.config import load_molecule_config
from FRASCI.diff_mols.integrals_builder import build_integrals
from FRASCI.diff_mols.fragmentation import (
    resolve_fragmentation, FragmentPartition,
)


def _h4_yaml() -> str:
    return textwrap.dedent("""
        name: H4 frag
        slug: h4_frag
        description: H4 chain for fragmentation
        geometry:
          kind: single
          xyz_inline: |
            4

            H 0 0 0
            H 0 0 1.4
            H 0 0 2.8
            H 0 0 4.2
        electronic_structure: {charge: 0, spin: 0, basis: sto-3g, scf: RHF}
        active_space:
          kind: avas
          avas_patterns: ["H 1s"]
          casscf: false
        fragmentation:
          - {name: per_atom, orbital_lists: auto_per_atom}
          - {name: pair, orbital_lists: [[0,1],[2,3]]}
        methods:
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 100}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: false}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference: {source: "FCI", computed_inline: true}
    """)


def test_explicit_orbital_lists_passthrough(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"
    cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")

    parts = resolve_fragmentation(cfg, bundle)
    assert "pair" in parts
    pair = parts["pair"]
    assert isinstance(pair, FragmentPartition)
    assert pair.orbital_lists == [[0, 1], [2, 3]]
    # Aufbau (2,2) split → fragments should hold 2 electrons each
    assert sum(sum(ne) for ne in pair.nelec_per_frag) == 4


def test_auto_per_atom_h4_returns_per_h(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"
    cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    parts = resolve_fragmentation(cfg, bundle)
    pa = parts["per_atom"]
    # Expect 4 fragments (one per H atom), each holding one active orbital
    assert len(pa.orbital_lists) == 4
    assert all(len(f) == 1 for f in pa.orbital_lists)
    assert sorted([orb for frag in pa.orbital_lists for orb in frag]) == [0, 1, 2, 3]


def test_auto_per_metal_raises_when_no_metal(tmp_path: Path):
    yaml_text = _h4_yaml().replace(
        "- {name: per_atom, orbital_lists: auto_per_atom}",
        "- {name: per_metal, orbital_lists: auto_per_metal}",
    )
    cfg_path = tmp_path / "h4_metal.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    with pytest.raises(ValueError, match="no metal atoms"):
        resolve_fragmentation(cfg, bundle)


# ---------------------------------------------------------------------------
# Boys-frame consistency tests (fix verification)
# ---------------------------------------------------------------------------

def _h4_explicit_only_yaml() -> str:
    """H4 config with ONLY explicit fragmentation — no auto_per_atom/metal."""
    return textwrap.dedent("""
        name: H4 explicit only
        slug: h4_explicit_only
        description: H4 chain, explicit fragmentation only (no Boys expected)
        geometry:
          kind: single
          xyz_inline: |
            4

            H 0 0 0
            H 0 0 1.4
            H 0 0 2.8
            H 0 0 4.2
        electronic_structure: {charge: 0, spin: 0, basis: sto-3g, scf: RHF}
        active_space:
          kind: avas
          avas_patterns: ["H 1s"]
          casscf: false
        fragmentation:
          - {name: pair, orbital_lists: [[0,1],[2,3]]}
        methods:
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 200}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: false}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference: {source: "FCI", computed_inline: true}
    """)


def test_explicit_fragmentation_does_not_trigger_boys(tmp_path: Path):
    """Explicit-only fragmentation must NOT apply Boys; scf_summary.json reflects this."""
    cfg_path = tmp_path / "h4_explicit_only.yaml"
    cfg_path.write_text(_h4_explicit_only_yaml())
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")

    # Check that scf_summary.json records boys_localized: false
    summary = json.loads(bundle.scf_summary_path.read_text())
    assert "boys_localized" in summary, "scf_summary.json must contain boys_localized key"
    assert summary["boys_localized"] is False, (
        f"Expected boys_localized=false for explicit-only config, got {summary['boys_localized']}"
    )

    # The MOs stored must be raw AVAS MOs (unrotated), verifiable by computing a
    # second AVAS run and confirming column space matches (up to sign/order).
    from pyscf import gto, scf as pyscf_scf
    from pyscf.mcscf import avas
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        basis="sto-3g", charge=0, spin=0, unit="Angstrom", verbose=0,
    )
    mf = pyscf_scf.RHF(mol); mf.verbose = 0; mf.kernel()
    _, _, mo_ref = avas.kernel(mf, ["H 1s"], verbose=0)
    mo_active_ref = mo_ref[:, :4]   # 4 active orbitals for H4/sto-3g

    mo_stored = np.load(bundle.mo_coeff_path)["mo_coeff"][:, :bundle.n_orb]

    # Column spaces must match: project each stored MO onto ref space — all singular
    # values should be ≈1 (each stored column lives in the AVAS span and no Boys
    # rotation has taken it elsewhere).
    overlap = mo_active_ref.T @ mol.intor_symmetric("int1e_ovlp") @ mo_stored
    sv = np.linalg.svd(overlap, compute_uv=False)
    assert np.allclose(sv, 1.0, atol=1e-6), (
        f"Stored MOs do not span the AVAS space (Boys was unexpectedly applied); singular values: {sv}"
    )


def _h4_chem_bond_yaml(atom_groups_yaml: str) -> str:
    """H4 config using chem_bond fragmentation with a parametrized atom_groups string."""
    return textwrap.dedent(f"""
        name: H4 chem_bond
        slug: h4_chem_bond
        description: H4 chain for chem_bond fragmentation test
        geometry:
          kind: single
          xyz_inline: |
            4

            H 0 0 0
            H 0 0 1.4
            H 0 0 2.8
            H 0 0 4.2
        electronic_structure: {{charge: 0, spin: 0, basis: sto-3g, scf: RHF}}
        active_space:
          kind: avas
          avas_patterns: ["H 1s"]
          casscf: false
        fragmentation:
          - name: cb
            orbital_lists: chem_bond
            atom_groups: {atom_groups_yaml}
        methods:
          plain_trimci: {{enabled: true, threshold: 0.001, max_dets: 100}}
          trimci_coo: {{enabled: false}}
          lasscf_cas: {{enabled: false}}
          lasscf_trimci_coo: {{enabled: false}}
          lassis: {{enabled: false}}
        reference: {{source: "FCI", computed_inline: true}}
    """)


def test_chem_bond_pairs_h4_two_fragments(tmp_path: Path):
    """H4 with atom_groups=[[0,1],[2,3]] → 2 fragments of 2 active MOs each."""
    cfg_path = tmp_path / "h4_cb.yaml"
    cfg_path.write_text(_h4_chem_bond_yaml("[[0,1],[2,3]]"))
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    parts = resolve_fragmentation(cfg, bundle)
    cb = parts["cb"]
    assert isinstance(cb, FragmentPartition)
    assert len(cb.orbital_lists) == 2, f"expected 2 fragments, got {len(cb.orbital_lists)}"
    assert sorted(orb for f in cb.orbital_lists for orb in f) == [0, 1, 2, 3]
    # Aufbau (2,2) split: 4 H atoms × 1 e each → fragments should hold 2 electrons each
    assert sum(sum(ne) for ne in cb.nelec_per_frag) == 4


def test_chem_bond_one_fragment_is_whole_molecule(tmp_path: Path):
    """atom_groups=[[0,1,2,3]] collapses LASSCF→CASSCF (1 fragment holds every MO)."""
    cfg_path = tmp_path / "h4_cb_one.yaml"
    cfg_path.write_text(_h4_chem_bond_yaml("[[0,1,2,3]]"))
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    parts = resolve_fragmentation(cfg, bundle)
    cb = parts["cb"]
    assert len(cb.orbital_lists) == 1
    assert sorted(cb.orbital_lists[0]) == [0, 1, 2, 3]


def test_chem_bond_validates_atom_groups(tmp_path: Path):
    """Missing/overlapping/out-of-range atom indices must raise ValueError."""
    # Missing atom 3
    cfg_path = tmp_path / "h4_cb_bad1.yaml"
    cfg_path.write_text(_h4_chem_bond_yaml("[[0,1,2]]"))
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    with pytest.raises(ValueError, match="not covered"):
        resolve_fragmentation(cfg, bundle)

    # Overlapping groups: atom 1 in both
    cfg_path = tmp_path / "h4_cb_bad2.yaml"
    cfg_path.write_text(_h4_chem_bond_yaml("[[0,1],[1,2,3]]"))
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    with pytest.raises(ValueError, match="multiple groups"):
        resolve_fragmentation(cfg, bundle)

    # Out-of-range atom 4
    cfg_path = tmp_path / "h4_cb_bad3.yaml"
    cfg_path.write_text(_h4_chem_bond_yaml("[[0,1],[2,3,4]]"))
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    with pytest.raises(ValueError, match="out-of-range"):
        resolve_fragmentation(cfg, bundle)


def test_chem_bond_requires_atom_groups(tmp_path: Path):
    """orbital_lists='chem_bond' without atom_groups must raise."""
    yaml_text = textwrap.dedent("""
        name: H4 missing groups
        slug: h4_missing
        description: chem_bond without atom_groups
        geometry:
          kind: single
          xyz_inline: |
            4

            H 0 0 0
            H 0 0 1.4
            H 0 0 2.8
            H 0 0 4.2
        electronic_structure: {charge: 0, spin: 0, basis: sto-3g, scf: RHF}
        active_space:
          kind: avas
          avas_patterns: ["H 1s"]
          casscf: false
        fragmentation:
          - {name: cb, orbital_lists: chem_bond}
        methods:
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 100}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: false}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference: {source: "FCI", computed_inline: true}
    """)
    cfg_path = tmp_path / "h4_missing.yaml"
    cfg_path.write_text(yaml_text)
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")
    with pytest.raises(ValueError, match="atom_groups"):
        resolve_fragmentation(cfg, bundle)


def test_auto_per_atom_triggers_boys(tmp_path: Path):
    """auto_per_atom fragmentation must set boys_localized=true in scf_summary.json;
    H4 should still give 4 fragments of 1 orbital each."""
    cfg_path = tmp_path / "h4.yaml"
    cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, "eq", tmp_path / "results")

    # Verify Boys was applied
    summary = json.loads(bundle.scf_summary_path.read_text())
    assert "boys_localized" in summary, "scf_summary.json must contain boys_localized key"
    assert summary["boys_localized"] is True, (
        f"Expected boys_localized=true for auto_per_atom config, got {summary['boys_localized']}"
    )

    # Fragmentation must still correctly yield 4 fragments (1 MO per H atom)
    parts = resolve_fragmentation(cfg, bundle)
    pa = parts["per_atom"]
    assert len(pa.orbital_lists) == 4, f"Expected 4 fragments, got {len(pa.orbital_lists)}"
    assert all(len(f) == 1 for f in pa.orbital_lists), (
        f"Each H atom should have exactly 1 MO; got sizes {[len(f) for f in pa.orbital_lists]}"
    )
    assert sorted([orb for frag in pa.orbital_lists for orb in frag]) == [0, 1, 2, 3]
