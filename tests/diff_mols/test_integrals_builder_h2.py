"""H2/STO-3G integrals builder: FCIDUMP roundtrip matches PySCF FCI."""
import textwrap
from pathlib import Path

from pyscf import fci, gto, scf, ao2mo

from FRASCI.diff_mols.config import load_molecule_config
from FRASCI.diff_mols.integrals_builder import build_integrals, IntegralBundle


def _h2_config_yaml() -> str:
    return textwrap.dedent("""
        name: H2 test
        slug: h2_test
        description: smoke
        geometry:
          kind: single
          xyz_inline: |
            2

            H 0.0 0.0 0.0
            H 0.0 0.0 0.74
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


def test_build_integrals_h2_roundtrip(tmp_path: Path):
    cfg_path = tmp_path / "h2.yaml"
    cfg_path.write_text(_h2_config_yaml())
    cfg = load_molecule_config(cfg_path)

    bundle = build_integrals(cfg, geom_tag="eq", output_root=tmp_path / "results", force=False)

    assert isinstance(bundle, IntegralBundle)
    assert bundle.fcidump_path.exists()
    assert bundle.mo_coeff_path.exists()
    assert bundle.scf_summary_path.exists()
    assert bundle.xyz_path.exists()
    assert bundle.n_orb == 2
    assert bundle.n_elec == 2
    assert bundle.n_alpha == 1 and bundle.n_beta == 1

    # Ground truth: PySCF FCI in the AO/MO basis of the same RHF
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(2, 2, 2, 2)
    e_fci_ref, _ = fci.direct_spin1.kernel(h1, eri, 2, (1, 1), ecore=mol.energy_nuc())

    # FCIDUMP roundtrip: read with trimci and run FCI-equivalent (full TrimCI = no truncation)
    import trimci
    h1d, erid, n_elec, n_orb, e_nuc_dump, n_a, n_b, _ = trimci.read_fcidump(str(bundle.fcidump_path))
    assert n_orb == 2 and n_elec == 2 and n_a == 1 and n_b == 1
    assert abs(e_nuc_dump - mol.energy_nuc()) < 1e-12

    e_fci_dump, _ = fci.direct_spin1.kernel(h1d, erid.reshape(2,2,2,2), n_orb, (n_a, n_b), ecore=e_nuc_dump)
    assert abs(e_fci_dump - e_fci_ref) < 1e-10


def test_build_integrals_writes_dets_npz_for_closed_shell(tmp_path: Path):
    """plain_trimci will overwrite later, but builder seeds a default Aufbau dets.npz."""
    import numpy as np

    cfg_path = tmp_path / "h2.yaml"
    cfg_path.write_text(_h2_config_yaml())
    cfg = load_molecule_config(cfg_path)
    bundle = build_integrals(cfg, geom_tag="eq", output_root=tmp_path / "results")

    dets_npz = bundle.fcidump_path.parent / "dets.npz"
    assert dets_npz.exists()
    data = np.load(dets_npz)
    assert "dets" in data
    assert data["dets"].shape == (1, 2)
    # Aufbau: alpha=0b01 (orb 0 occ), beta=0b01 (orb 0 occ)
    assert int(data["dets"][0, 0]) == 0b01
    assert int(data["dets"][0, 1]) == 0b01


def test_build_integrals_skips_when_hash_matches(tmp_path: Path):
    cfg_path = tmp_path / "h2.yaml"
    cfg_path.write_text(_h2_config_yaml())
    cfg = load_molecule_config(cfg_path)

    b1 = build_integrals(cfg, "eq", tmp_path / "results")
    mtime1 = b1.fcidump_path.stat().st_mtime

    b2 = build_integrals(cfg, "eq", tmp_path / "results", force=False)
    assert b2.fcidump_path.stat().st_mtime == mtime1   # not rebuilt

    b3 = build_integrals(cfg, "eq", tmp_path / "results", force=True)
    assert b3.fcidump_path.stat().st_mtime >= mtime1   # rebuilt
