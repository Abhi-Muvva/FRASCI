"""Report helpers — load, best_per_group, plotting + J yamaguchi."""
import csv
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

from pathlib import Path

import pandas as pd

from FRASCI.diff_mols.report import (
    load_runs, best_per_group, plot_pes, plot_method_bar, compute_j_yamaguchi,
    enrich_lasscf_stage_columns,
)
from FRASCI.diff_mols.run_writer import RUNS_INDEX_COLUMNS


def _write_synthetic_runs(results_root: Path, mol: str):
    csv_path = results_root / mol / "runs_index.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # (method, partition, geom_tag, tag, e_tot, e_ref)
        ("plain_trimci", "full", "r1.00", "default", -1.10, -1.105),
        ("trimci_coo",   "full", "r1.00", "default", -1.103, -1.105),
        ("lasscf_cas",   "pair", "r1.00", "default", -1.08,  -1.105),
        ("lassis", "pair_on_lasscf_cas", "r1.00", "default", -1.104, -1.105),
        ("lasscf_trimci_coo", "pair", "r1.00", "default", -1.102, -1.105),
        ("plain_trimci", "full", "r1.50", "default", -1.05, -1.06),
        ("trimci_coo",   "full", "r1.50", "default", -1.058, -1.06),
    ]
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(RUNS_INDEX_COLUMNS))
        writer.writeheader()
        for (m, p, g, t, e, eref) in rows:
            row = {c: "" for c in RUNS_INDEX_COLUMNS}
            row.update({
                "timestamp": "20260622_140000", "molecule": mol, "method": m,
                "partition": p, "geom_tag": g, "tag": t,
                "e_tot": e, "e_ref": eref,
                "error_mha": (e - eref) * 1000.0,
                "converged": True, "n_dets_total": 50, "wall_s": 0.5,
                "run_dir": f"runs/{m}/{p}__{g}__{t}__20260622_140000",
            })
            writer.writerow(row)


def test_load_runs_single_mol(tmp_path: Path):
    _write_synthetic_runs(tmp_path, "syn")
    df = load_runs(tmp_path, "syn")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7
    assert {"method", "geom_tag", "e_tot", "error_mha"}.issubset(df.columns)
    assert {"stage", "analysis_method", "analysis_partition", "analysis_label"}.issubset(df.columns)


def test_load_runs_all(tmp_path: Path):
    _write_synthetic_runs(tmp_path, "a")
    _write_synthetic_runs(tmp_path, "b")
    df = load_runs(tmp_path, None)
    assert set(df["molecule"]) == {"a", "b"}


def test_best_per_group(tmp_path: Path):
    _write_synthetic_runs(tmp_path, "syn")
    df = load_runs(tmp_path, "syn")
    best = best_per_group(df, by=("method", "geom_tag"))
    # Synthetic rows include both raw LASSCF and LASSCF+LASSIS at r1.00.
    assert len(best) >= 5


def test_lassis_is_enriched_as_lasscf_stage():
    df = pd.DataFrame([{
        "method": "lassis",
        "partition": "h1diag_2_on_lasscf_trimci_coo",
    }])
    out = enrich_lasscf_stage_columns(df)
    row = out.iloc[0]
    assert row["stage"] == "lassis"
    assert row["parent_method"] == "lasscf_trimci_coo"
    assert row["base_partition"] == "h1diag_2"
    assert row["analysis_method"] == "lasscf_trimci_coo"
    assert row["analysis_partition"] == "h1diag_2"
    assert row["analysis_label"] == "lasscf_trimci_coo + LASSIS"


def test_plot_pes_returns_axes(tmp_path: Path):
    _write_synthetic_runs(tmp_path, "syn")
    df = load_runs(tmp_path, "syn")
    ax = plot_pes(df, "syn", methods=["plain_trimci"])
    assert ax is not None


def test_plot_method_bar_returns_axes(tmp_path: Path):
    _write_synthetic_runs(tmp_path, "syn")
    df = load_runs(tmp_path, "syn")
    ax = plot_method_bar(df, "syn", "r1.00")
    assert ax is not None


def test_yamaguchi_j():
    j = compute_j_yamaguchi(
        hs_row={"e_tot": -1.000, "s2_expectation": 6.0},
        bs_row={"e_tot": -1.001, "s2_expectation": 1.0},
    )
    # J = (E_HS - E_BS) / (<S^2>_HS - <S^2>_BS) = (-1.0 - -1.001)/(6-1) = 0.001/5 = 0.0002 Ha
    assert abs(j["J_ha"] - 0.0002) < 1e-12
    assert "J_cm-1" in j
