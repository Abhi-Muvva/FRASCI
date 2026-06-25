"""Run-folder writer + runs_index.csv atomic append."""
import csv
import json
import threading
from pathlib import Path

from FRASCI.diff_mols.run_writer import (
    make_run_dir, write_run_outputs, append_runs_index_row, RUNS_INDEX_COLUMNS,
    protocol_path_parts,
)


def test_protocol_path_parts_splits_grid_axes():
    assert protocol_path_parts("cyc1_dets50_bfgs20") == ["dets50", "cyc1", "bfgs20"]
    assert protocol_path_parts("thr005_cyc2_dets2000_bfgs40") == [
        "dets2000", "cyc2", "bfgs40", "thr005",
    ]
    assert protocol_path_parts("cyc2_dets500_bfgs40__nspin2_opt1") == [
        "dets500", "cyc2", "bfgs40", "nspin2", "opt1",
    ]


def test_make_run_dir_creates_named_folder(tmp_results_dir: Path):
    """Single-point hierarchy omits the redundant eq directory."""
    p = make_run_dir(tmp_results_dir, "h2_test", "plain_trimci",
                     "full", "eq", "cyc1_dets50_bfgs20", "20260622_140000")
    expected = (tmp_results_dir / "h2_test" / "plain_trimci" / "full"
                / "dets50" / "cyc1" / "bfgs20" / "run_20260622_140000")
    assert p == expected
    assert p.is_dir()


def test_make_run_dir_keeps_geometry_for_scan(tmp_results_dir: Path):
    p = make_run_dir(tmp_results_dir, "h2_test", "plain_trimci",
                     "full", "r1.50", "dets500", "20260622_140000")
    assert p == (
        tmp_results_dir / "h2_test" / "r1.50" / "plain_trimci" / "full"
        / "dets500" / "run_20260622_140000"
    )


def test_write_run_outputs_writes_all_files(tmp_results_dir: Path):
    p = make_run_dir(tmp_results_dir, "h2_test", "plain_trimci",
                     "full", "eq", "default", "20260622_140000")
    write_run_outputs(
        p,
        result_json={"e_tot": -1.123, "method": "plain_trimci"},
        config_snapshot={"threshold": 0.001},
        log_text="stdout line 1\nstdout line 2\n",
        readme_summary="6-line summary here",
    )
    assert json.loads((p / "result.json").read_text())["e_tot"] == -1.123
    assert (p / "config_snapshot.yaml").read_text().strip().startswith("threshold:")
    assert "stdout line 1" in (p / "log.txt").read_text()
    assert "6-line summary" in (p / "README.md").read_text()


def test_append_runs_index_atomic(tmp_results_dir: Path):
    """Concurrent writes from 2 threads → 2 rows, no corruption."""
    barrier = threading.Barrier(2)
    rows_seen = []

    def _writer(method: str, e: float):
        row = {col: "" for col in RUNS_INDEX_COLUMNS}
        row.update({
            "timestamp": "20260622_140000", "molecule": "h2_test",
            "method": method, "partition": "full", "geom_tag": "eq",
            "tag": "default", "e_tot": e, "converged": True,
            "n_dets_total": 4, "wall_s": 0.01,
            "run_dir": f"runs/{method}/full__eq__default__t",
        })
        barrier.wait()
        append_runs_index_row(tmp_results_dir, "h2_test", row)

    t1 = threading.Thread(target=_writer, args=("plain_trimci", -1.10))
    t2 = threading.Thread(target=_writer, args=("trimci_coo", -1.15))
    t1.start(); t2.start()
    t1.join(); t2.join()

    csv_path = tmp_results_dir / "h2_test" / "runs_index.csv"
    with csv_path.open() as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    assert len(rows) == 2
    methods = sorted(r["method"] for r in rows)
    assert methods == ["plain_trimci", "trimci_coo"]
