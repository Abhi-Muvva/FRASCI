"""H4 chain integrals builder: AVAS + CASSCF paths."""
import textwrap
from pathlib import Path

import numpy as np
from pyscf import fci, gto, scf, ao2mo, mcscf

from FRASCI.diff_mols.config import load_molecule_config
from FRASCI.diff_mols.integrals_builder import build_integrals


H4_GEOM = "H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2"


def _h4_avas_yaml() -> str:
    return textwrap.dedent("""
        name: H4 AVAS
        slug: h4_avas
        description: H4 chain, AVAS (4,4)
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
          - {name: chem, orbital_lists: auto_per_atom}
        methods:
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 200}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: false}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference: {source: "FCI", computed_inline: true}
    """)


def _h4_window_yaml(casscf: bool) -> str:
    return textwrap.dedent(f"""
        name: H4 window
        slug: h4_window
        description: H4 chain, window (4,4)
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
          kind: window
          norb: 4
          n_active_elec: 4
          casscf: {str(casscf).lower()}
        fragmentation:
          - {{name: chem, orbital_lists: auto_per_atom}}
        methods:
          plain_trimci: {{enabled: true, threshold: 0.001, max_dets: 200}}
          trimci_coo: {{enabled: false}}
          lasscf_cas: {{enabled: false}}
          lasscf_trimci_coo: {{enabled: false}}
          lassis: {{enabled: false}}
        reference: {{source: "FCI", computed_inline: true}}
    """)


def _pyscf_fci_h4() -> float:
    mol = gto.M(atom=H4_GEOM, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(4, 4, 4, 4)
    e_fci, _ = fci.direct_spin1.kernel(h1, eri, 4, (2, 2), ecore=mol.energy_nuc())
    return float(e_fci)


def _fci_from_fcidump(fcidump_path: Path) -> float:
    import trimci
    h1, eri, n_elec, n_orb, e_nuc, n_a, n_b, _ = trimci.read_fcidump(str(fcidump_path))
    eri4 = eri.reshape(n_orb, n_orb, n_orb, n_orb)
    e, _ = fci.direct_spin1.kernel(h1, eri4, n_orb, (n_a, n_b), ecore=e_nuc)
    return float(e)


def test_avas_h4_full_window_matches_fci(tmp_path: Path):
    cfg_path = tmp_path / "h4_avas.yaml"
    cfg_path.write_text(_h4_avas_yaml())
    cfg = load_molecule_config(cfg_path)
    b = build_integrals(cfg, "eq", tmp_path / "results")
    assert b.n_orb == 4 and b.n_alpha == 2 and b.n_beta == 2
    e_fci_ref = _pyscf_fci_h4()
    e_fci_dump = _fci_from_fcidump(b.fcidump_path)
    assert abs(e_fci_dump - e_fci_ref) < 1e-10


def test_window_h4_no_casscf_matches_fci(tmp_path: Path):
    cfg_path = tmp_path / "h4_win.yaml"
    cfg_path.write_text(_h4_window_yaml(casscf=False))
    cfg = load_molecule_config(cfg_path)
    b = build_integrals(cfg, "eq", tmp_path / "results")
    assert b.n_orb == 4 and b.n_alpha == 2 and b.n_beta == 2
    assert abs(_fci_from_fcidump(b.fcidump_path) - _pyscf_fci_h4()) < 1e-10


def test_window_h4_with_casscf_runs(tmp_path: Path):
    """CASSCF on full window should be invariant (window == full space)."""
    cfg_path = tmp_path / "h4_win_cas.yaml"
    cfg_path.write_text(_h4_window_yaml(casscf=True))
    cfg = load_molecule_config(cfg_path)
    b = build_integrals(cfg, "eq", tmp_path / "results")
    import json
    summary = json.loads(b.scf_summary_path.read_text())
    assert summary["casscf_converged"] is True
    # CASSCF on (4,4)/sto-3g H4 is exact → energy should match FCI
    assert abs(_fci_from_fcidump(b.fcidump_path) - _pyscf_fci_h4()) < 1e-8
