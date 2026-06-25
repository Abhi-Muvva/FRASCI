"""Jinja2 scan geometry rendering + build_all_geometries."""
import textwrap
from pathlib import Path

from FRASCI.diff_mols.config import load_molecule_config
from FRASCI.diff_mols.integrals_builder import build_all_geometries


def _h2_scan_yaml() -> str:
    return textwrap.dedent("""
        name: H2 scan
        slug: h2_scan_test
        description: H2 PES
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
            - {r: 2.1}
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


def test_build_all_geometries_h2_scan(tmp_path: Path):
    cfg_path = tmp_path / "h2_scan.yaml"
    cfg_path.write_text(_h2_scan_yaml())
    cfg = load_molecule_config(cfg_path)

    bundles = build_all_geometries(cfg, tmp_path / "results")
    assert set(bundles.keys()) == {"r0.70", "r1.40", "r2.10"}
    for tag, b in bundles.items():
        assert b.fcidump_path.exists()
        assert b.xyz_path.read_text().count("H 0.0 0.0") == 2

    # XYZ values differ across tags — per-geom layout: <mol>/<geom>/geometry.xyz
    r07 = (tmp_path / "results" / "h2_scan_test" / "r0.70" / "geometry.xyz").read_text()
    r21 = (tmp_path / "results" / "h2_scan_test" / "r2.10" / "geometry.xyz").read_text()
    assert "0.7000" in r07
    assert "2.1000" in r21


def test_scan_energies_increase_then_decrease_or_stay(tmp_path: Path):
    """Smoke: H2 FCI energy in active space at r=0.7 is below r=2.1 (stretched is higher)."""
    from pyscf import fci
    import trimci

    cfg_path = tmp_path / "h2_scan.yaml"
    cfg_path.write_text(_h2_scan_yaml())
    cfg = load_molecule_config(cfg_path)
    bundles = build_all_geometries(cfg, tmp_path / "results")

    def _fci(b):
        h1, eri, n_elec, n_orb, e_nuc, n_a, n_b, _ = trimci.read_fcidump(str(b.fcidump_path))
        e, _ = fci.direct_spin1.kernel(h1, eri.reshape(n_orb,n_orb,n_orb,n_orb), n_orb, (n_a, n_b), ecore=e_nuc)
        return float(e)

    e07 = _fci(bundles["r0.70"])
    e21 = _fci(bundles["r2.10"])
    assert e07 < e21    # equilibrium-ish is lower than stretched
