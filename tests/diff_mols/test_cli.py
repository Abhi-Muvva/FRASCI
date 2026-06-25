"""CLI smoke test using H4 config."""
import textwrap
from pathlib import Path

from FRASCI.diff_mols.run import main as cli_main


def _write_h4(tmp_configs: Path) -> None:
    (tmp_configs / "h4_cli.yaml").write_text(textwrap.dedent("""
        name: H4 CLI
        slug: h4_cli
        description: cli
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
          plain_trimci: {enabled: true, threshold: 0.001, max_dets: 50, num_runs: 1, max_rounds: 2}
          trimci_coo: {enabled: false}
          lasscf_cas: {enabled: false}
          lasscf_trimci_coo: {enabled: false}
          lassis: {enabled: false}
        reference: {source: "FCI", computed_inline: true}
    """))


def test_cli_runs_plain_trimci_only(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    _write_h4(tmp_path / "configs")

    rc = cli_main([
        "--mol", "h4_cli",
        "--methods", "plain_trimci",
        "--tag", "cli_test",
        "--results-root", str(tmp_path / "results"),
        "--configs-dir", str(tmp_path / "configs"),
        "--no-skip-existing",
    ])
    assert rc == 0
    csv_path = tmp_path / "results" / "h4_cli" / "runs_index.csv"
    assert csv_path.exists()
    assert "plain_trimci" in csv_path.read_text()


def test_cli_build_integrals_only(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    _write_h4(tmp_path / "configs")

    rc = cli_main([
        "--mol", "h4_cli", "--build-integrals-only",
        "--results-root", str(tmp_path / "results"),
        "--configs-dir", str(tmp_path / "configs"),
    ])
    assert rc == 0
    # Single-point layout: <mol>/integrals/fcidump
    assert (tmp_path / "results" / "h4_cli" / "integrals" / "fcidump").exists()
    # Only the integrals subfolder under <mol>; no method run dirs at this point
    method_dirs = [p for p in (tmp_path / "results" / "h4_cli").iterdir()
                   if p.is_dir() and p.name not in {"integrals", "plots"}]
    assert method_dirs == []


def test_cli_override_parsed_as_numbers(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    _write_h4(tmp_path / "configs")
    rc = cli_main([
        "--mol", "h4_cli", "--methods", "plain_trimci",
        "--override", "threshold=0.002", "max_dets=42",
        "--tag", "override_test",
        "--results-root", str(tmp_path / "results"),
        "--configs-dir", str(tmp_path / "configs"),
        "--no-skip-existing",
    ])
    assert rc == 0
    # Single-point layout: <mol>/<method>/<partition>/<protocol>/run_<ts>/
    protocol_dir = tmp_path / "results" / "h4_cli" / "plain_trimci" / "full" / "override_test"
    plain_runs = list(protocol_dir.glob("run_*"))
    assert plain_runs, f"got no runs under {protocol_dir}"
    snap_path = plain_runs[0] / "config_snapshot.yaml"
    snap_text = snap_path.read_text()
    assert "0.002" in snap_text
    assert "42" in snap_text


def test_cli_dry_run_matrix_writes_csv(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    _write_h4(tmp_path / "configs")
    matrix_path = tmp_path / "matrix.csv"
    rc = cli_main([
        "--mol", "h4_cli", "--methods", "plain_trimci",
        "--plain-trimci-protocols", "dets50,dets100",
        "--dry-run-matrix", "--matrix-out", str(matrix_path),
        "--results-root", str(tmp_path / "results"),
        "--configs-dir", str(tmp_path / "configs"),
    ])
    assert rc == 0
    text = matrix_path.read_text()
    assert "dets50" in text
    assert "dets100" in text
