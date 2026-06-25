"""End-to-end smoke: MoleculeBenchmark runs every enabled method on H4/STO-3G."""
import textwrap
import json
from pathlib import Path

from FRASCI.diff_mols import MoleculeBenchmark
from FRASCI.diff_mols.benchmark import _parent_run_identity
from FRASCI.diff_mols.methods import RunResult


def _write_h4_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "h4.yaml"
    # Use boys: true + [[0,2],[1,3]] (Task 10/11 pattern) to avoid degenerate fragments
    # with the [[0,1],[2,3]] partition on linear H4.
    p.write_text(textwrap.dedent("""
        name: H4 e2e
        slug: h4_e2e
        description: end-to-end
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
          trimci_coo: {enabled: true, threshold: 0.0001, max_dets: 100, coo_cycles: 2, bfgs_maxiter: 10, bfgs_ftol: 1.0e-8, davidson_tol: 1.0e-7}
          lasscf_cas: {enabled: true, max_cycle_macro: 20, partitions: [pair]}
          lasscf_trimci: {enabled: true, max_cycle_macro: 20, trimci_threshold: 0.0001, trimci_max_dets: 100, partitions: [pair]}
          lasscf_trimci_coo: {enabled: true, max_cycle_macro: 20, trimci_threshold: 0.0001, trimci_max_dets: 100, coo_cycles: 1, bfgs_maxiter: 10, partitions: [pair]}
          lassis: {enabled: true, on_top_of: [lasscf_cas, lasscf_trimci, lasscf_trimci_coo], n_charge: s, n_spin: 0, opt: 1}
        reference: {source: "FCI", computed_inline: true}
    """))
    return p


def test_benchmark_run_all_h4(tmp_path: Path):
    cfg_path = _write_h4_yaml(tmp_path)
    bm = MoleculeBenchmark(cfg_path, results_root=tmp_path / "results")
    results = bm.run_all(tag="smoke", skip_existing=False)
    methods = sorted(set(r.result_json["method"] for r in results))
    assert "plain_trimci" in methods
    assert "trimci_coo" in methods
    assert "lasscf_cas" in methods
    assert "lasscf_trimci" in methods
    assert "lasscf_trimci_coo" in methods
    assert "lassis" in methods

    # runs_index.csv has at least one row per method (lassis has one per parent LASSCF)
    csv_path = tmp_path / "results" / "h4_e2e" / "runs_index.csv"
    csv_text = csv_path.read_text()
    for m in ("plain_trimci", "trimci_coo", "lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo", "lassis"):
        assert m in csv_text


def test_benchmark_skip_existing(tmp_path: Path):
    cfg_path = _write_h4_yaml(tmp_path)
    bm = MoleculeBenchmark(cfg_path, results_root=tmp_path / "results")
    bm.run_all(tag="skip_test", skip_existing=False)
    second_pass = bm.run_all(tag="skip_test", skip_existing=True)
    # All re-runs should report skip
    assert all(r.result_json.get("skipped") for r in second_pass)


def test_benchmark_lassis_nspin_sweep_from_yaml(tmp_path: Path):
    cfg_path = _write_h4_yaml(tmp_path)
    text = cfg_path.read_text()
    text = text.replace(
        "lassis: {enabled: true, on_top_of: [lasscf_cas, lasscf_trimci, lasscf_trimci_coo], n_charge: s, n_spin: 0, opt: 1}",
        "lassis: {enabled: true, on_top_of: [lasscf_cas], n_charge: s, n_spin: 0, n_spin_values: [0, 1, 2, 3], opt: 1}",
    )
    cfg_path.write_text(text)
    bm = MoleculeBenchmark(cfg_path, results_root=tmp_path / "results")

    seen_nspin: list[int] = []
    parent_dir = (
        tmp_path / "results" / "h4_e2e" / "lasscf_cas" / "pair"
        / "spin_sweep" / "run_00000000_000000"
    )

    def fake_result(method: str, run_dir: Path) -> RunResult:
        return RunResult(
            e_tot=-1.0, n_dets=1, wall_s=0.0, run_dir=run_dir,
            result_json={"method": method}, converged=True,
        )

    bm.build_integrals = lambda geom_tag: None  # type: ignore[method-assign]
    bm.run_lasscf_cas = lambda geom_tag, partition, **kw: fake_result("lasscf_cas", parent_dir)  # type: ignore[method-assign]

    def fake_lassis(lasscf_run_dir: Path, **kw) -> RunResult:
        seen_nspin.append(kw["lassis_nspin"])
        run_dir = tmp_path / f"lassis_nspin{kw['lassis_nspin']}"
        return fake_result("lassis", run_dir)

    bm.run_lassis = fake_lassis  # type: ignore[method-assign]

    bm.run_all(
        methods=["lasscf_cas", "lassis"],
        partitions=["pair"],
        tag="spin_sweep",
        skip_existing=False,
        verbose=False,
    )

    assert seen_nspin == [0, 1, 2, 3]


def test_parent_run_identity_uses_full_nested_protocol_metadata(tmp_path: Path):
    parent_dir = (
        tmp_path / "results" / "mol" / "r1.00" / "lasscf_trimci_coo"
        / "chem" / "dets500" / "cyc2" / "bfgs40" / "run_20260623_000000"
    )
    parent_dir.mkdir(parents=True)
    (parent_dir / "result.json").write_text(json.dumps({
        "method": "lasscf_trimci_coo",
        "partition": "chem",
        "protocol": "cyc2_dets500_bfgs40",
        "tag": "cyc2_dets500_bfgs40",
    }))

    assert _parent_run_identity(parent_dir) == ("cyc2_dets500_bfgs40", "chem")
