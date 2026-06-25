"""End-to-end smoke tests: each method on H4/STO-3G recovers FCI."""
import textwrap
from pathlib import Path

import numpy as np
from pyscf import fci, gto, scf, ao2mo

from FRASCI.diff_mols.config import load_molecule_config
from FRASCI.diff_mols.integrals_builder import build_integrals
from FRASCI.diff_mols.fragmentation import resolve_fragmentation
from FRASCI.diff_mols.methods import (
    run_plain_trimci, run_trimci_coo, run_lasscf_cas,
    run_lasscf_trimci, run_lasscf_trimci_coo,
)


H4_GEOM = "H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2"


def _h4_yaml() -> str:
    return textwrap.dedent("""
        name: H4
        slug: h4
        description: H4 smoke
        geometry:
          kind: single
          xyz_inline: |
            4

            H 0 0 0
            H 0 0 1.4
            H 0 0 2.8
            H 0 0 4.2
        electronic_structure: {charge: 0, spin: 0, basis: sto-3g, scf: RHF}
        active_space: {kind: avas, avas_patterns: ["H 1s"], casscf: false}
        fragmentation:
          - {name: pair, orbital_lists: [[0,1],[2,3]]}
        methods:
          plain_trimci: {enabled: true, threshold: 0.0001, max_dets: 100, num_runs: 1, max_rounds: 2}
          trimci_coo:   {enabled: true, threshold: 0.0001, max_dets: 100, coo_cycles: 2, bfgs_maxiter: 20, bfgs_ftol: 1.0e-8, davidson_tol: 1.0e-7}
          lasscf_cas:   {enabled: true, max_cycle_macro: 30, partitions: [pair]}
          lasscf_trimci: {enabled: true, max_cycle_macro: 30, trimci_threshold: 0.0001, trimci_max_dets: 100, max_rounds: 2, partitions: [pair]}
          lasscf_trimci_coo: {enabled: true, max_cycle_macro: 30, trimci_threshold: 0.0001, trimci_max_dets: 100, coo_cycles: 1, bfgs_maxiter: 10, partitions: [pair]}
          lassis:       {enabled: true, on_top_of: [lasscf_cas, lasscf_trimci, lasscf_trimci_coo], n_charge: s, n_spin: 0, opt: 1}
        reference: {source: "FCI", computed_inline: true}
    """)


def _pyscf_fci_h4() -> float:
    mol = gto.M(atom=H4_GEOM, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol); mf.verbose = 0; mf.kernel()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape(4, 4, 4, 4)
    e, _ = fci.direct_spin1.kernel(h1, eri, 4, (2, 2), ecore=mol.energy_nuc())
    return float(e)


def test_plain_trimci_h4_recovers_fci(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"
    cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    results_root = tmp_path / "results"
    bundle = build_integrals(cfg, "eq", results_root)

    res = run_plain_trimci(cfg, bundle, results_root=results_root,
                           tag="default", timestamp="20260622_140000")

    e_ref = _pyscf_fci_h4()
    assert abs(res.e_tot - e_ref) < 1e-4
    assert res.run_dir.exists()
    assert (res.run_dir / "result.json").exists()

    # dets.npz now updated with TrimCI's expansion
    dets_path = bundle.fcidump_path.parent / "dets.npz"
    arr = np.load(dets_path)["dets"]
    assert arr.shape[0] >= 1
    assert arr.shape[1] == 2

    # runs_index.csv has a row
    csv_path = results_root / "h4" / "runs_index.csv"
    assert csv_path.exists()
    assert "plain_trimci" in csv_path.read_text()


def test_trimci_coo_h4_recovers_fci(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"
    cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    results_root = tmp_path / "results"
    bundle = build_integrals(cfg, "eq", results_root)

    res = run_trimci_coo(cfg, bundle, results_root=results_root,
                         tag="default", timestamp="20260622_140100")

    e_ref = _pyscf_fci_h4()
    assert abs(res.e_tot - e_ref) < 1e-3  # COO at H4(4,4) should be exact or near-exact
    assert res.run_dir.exists()
    assert (res.run_dir / "result.json").exists()

    # result has cycle_trace
    assert "cycle_trace" in res.result_json["result"]
    cycles = res.result_json["result"]["cycle_trace"]
    assert len(cycles) >= 1
    for c in cycles:
        assert "e_in" in c and "e_after_orb" in c and "e_out" in c
        assert "n_dets_in" in c and "n_dets_out" in c


def _h4_yaml_boys() -> str:
    """H4 YAML with Boys-localized active space for physically meaningful pair fragmentation."""
    return textwrap.dedent("""
        name: H4
        slug: h4
        description: H4 smoke boys
        geometry:
          kind: single
          xyz_inline: |
            4

            H 0 0 0
            H 0 0 1.4
            H 0 0 2.8
            H 0 0 4.2
        electronic_structure: {charge: 0, spin: 0, basis: sto-3g, scf: RHF}
        active_space: {kind: avas, avas_patterns: ["H 1s"], casscf: false, boys: true}
        fragmentation:
          - {name: pair, orbital_lists: [[0,2],[1,3]]}
        methods:
          plain_trimci: {enabled: true, threshold: 0.0001, max_dets: 100, num_runs: 1, max_rounds: 2}
          trimci_coo:   {enabled: true, threshold: 0.0001, max_dets: 100, coo_cycles: 2, bfgs_maxiter: 20, bfgs_ftol: 1.0e-8, davidson_tol: 1.0e-7}
          lasscf_cas:   {enabled: true, max_cycle_macro: 30, partitions: [pair]}
          lasscf_trimci: {enabled: true, max_cycle_macro: 30, trimci_threshold: 0.0001, trimci_max_dets: 100, max_rounds: 2, partitions: [pair]}
          lasscf_trimci_coo: {enabled: true, max_cycle_macro: 30, trimci_threshold: 0.0001, trimci_max_dets: 100, coo_cycles: 1, bfgs_maxiter: 10, partitions: [pair]}
          lassis:       {enabled: true, on_top_of: [lasscf_cas, lasscf_trimci, lasscf_trimci_coo], n_charge: s, n_spin: 0, opt: 1}
        reference: {source: "FCI", computed_inline: true}
    """)


def test_lasscf_cas_h4_2frag_runs(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"; cfg_path.write_text(_h4_yaml_boys())
    cfg = load_molecule_config(cfg_path)
    results_root = tmp_path / "results"
    bundle = build_integrals(cfg, "eq", results_root)
    # plain_trimci first so dets.npz exists for h1diag-style partition consumers
    from FRASCI.diff_mols.methods import run_plain_trimci
    run_plain_trimci(cfg, bundle, results_root=results_root, tag="seed", timestamp="20260622_140050")

    res = run_lasscf_cas(cfg, bundle, partition_name="pair",
                        results_root=results_root, tag="default",
                        timestamp="20260622_140200")
    e_ref = _pyscf_fci_h4()
    # LASSCF on a 2x2 product basis of H2-like fragments at 1.4 Å is exact for the dissociation curve;
    # may differ from FCI by the cross-fragment correlation energy. Sanity floor: < 0.05 Ha above FCI.
    assert res.e_tot - e_ref < 0.05
    assert res.run_dir.exists()
    assert (res.run_dir / "checkpoint.npz").exists()


from FRASCI.diff_mols.methods import run_lassis
from FRASCI.diff_mols.lassis_states import build_lassis_kwargs


def test_lasscf_trimci_h4_2frag_runs(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"; cfg_path.write_text(_h4_yaml_boys())
    cfg = load_molecule_config(cfg_path)
    results_root = tmp_path / "results"
    bundle = build_integrals(cfg, "eq", results_root)
    run_plain_trimci(cfg, bundle, results_root=results_root, tag="seed", timestamp="20260622_140050")

    res = run_lasscf_trimci(cfg, bundle, partition_name="pair",
                            results_root=results_root, tag="dets100",
                            timestamp="20260622_140250")
    e_ref = _pyscf_fci_h4()
    assert res.e_tot - e_ref < 0.05
    assert res.run_dir.exists()
    assert res.result_json["method"] == "lasscf_trimci"
    assert res.run_dir == (
        results_root / "h4" / "lasscf_trimci" / "pair"
        / "dets100" / "run_20260622_140250"
    )
    assert (res.run_dir / "checkpoint.npz").exists()
    assert (res.run_dir / "kernel_calls.json").exists()


def test_lasscf_trimci_coo_h4_2frag_runs(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"; cfg_path.write_text(_h4_yaml_boys())
    cfg = load_molecule_config(cfg_path)
    results_root = tmp_path / "results"
    bundle = build_integrals(cfg, "eq", results_root)
    from FRASCI.diff_mols.methods import run_plain_trimci
    run_plain_trimci(cfg, bundle, results_root=results_root, tag="seed", timestamp="20260622_140050")

    res = run_lasscf_trimci_coo(cfg, bundle, partition_name="pair",
                                results_root=results_root, tag="default",
                                timestamp="20260622_140300")
    e_ref = _pyscf_fci_h4()
    # COO-enabled LASSCF should match or beat plain LASSCF
    assert res.e_tot - e_ref < 0.05
    assert res.run_dir.exists()
    assert (res.run_dir / "checkpoint.npz").exists()
    assert (res.run_dir / "kernel_calls.json").exists()


def test_build_lassis_kwargs_n_spin_default(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"; cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    kwargs = build_lassis_kwargs(cfg, tmp_path)
    assert kwargs["lassis_nspin"] == 0
    assert kwargs["lassis_ncharge"] == "s"
    assert kwargs["opt"] == 1


def test_lassis_h4_runs_on_lasscf_cas_checkpoint(tmp_path: Path):
    cfg_path = tmp_path / "h4.yaml"; cfg_path.write_text(_h4_yaml())
    cfg = load_molecule_config(cfg_path)
    results_root = tmp_path / "results"
    bundle = build_integrals(cfg, "eq", results_root)
    from FRASCI.diff_mols.methods import run_plain_trimci, run_lasscf_cas
    run_plain_trimci(cfg, bundle, results_root=results_root, tag="seed", timestamp="20260622_140050")
    cas = run_lasscf_cas(cfg, bundle, partition_name="pair",
                        results_root=results_root, tag="default", timestamp="20260622_140200")

    res = run_lassis(cfg, bundle, lasscf_run_dir=cas.run_dir,
                     results_root=results_root, tag="default",
                     timestamp="20260622_140400")
    assert res.run_dir.exists()
    assert (res.run_dir / "result.json").exists()
    assert res.run_dir == (
        results_root / "h4" / "lassis_on_lasscf_cas" / "pair"
        / "default" / "run_20260622_140400"
    )
    # LASSIS energy should be at-most-equal to LASSCF (variational lower bound or equal)
    assert res.e_tot <= cas.e_tot + 1e-6
