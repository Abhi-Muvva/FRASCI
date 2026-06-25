from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from FRASCI.diff_mols.report_plots import (
    _ensure_plot_dir,
    plot_single_point_protocol_sensitivity,
)


def test_single_point_protocol_plot_has_external_figure_legend(tmp_path: Path):
    rows = []
    for method, partition in (
        ("plain_trimci", "full"),
        ("trimci_coo", "full"),
        ("lasscf_trimci_coo", "chem"),
    ):
        for dets in (50, 500, 2000):
            for cycles in (1, 2, 4):
                for bfgs in (20, 40, 80):
                    protocol = f"cyc{cycles}_dets{dets}_bfgs{bfgs}"
                    rows.append({
                        "molecule": "demo",
                        "geom_tag": "eq",
                        "method": method,
                        "partition": partition,
                        "protocol": protocol,
                        "error_mha": 100.0 / dets + 1.0 / cycles + 1.0 / bfgs,
                        "e_tot": -1.0,
                        "e_ref": -1.1,
                        "wall_s": 1.0,
                    })

    save_to = tmp_path / "protocol.png"
    fig = plot_single_point_protocol_sensitivity(
        pd.DataFrame(rows), "demo", save_to=save_to
    )

    assert len(fig.axes) == 3
    assert len(fig.legends) == 1
    assert save_to.exists()


def test_single_point_plot_dir_omits_eq(tmp_path: Path):
    assert _ensure_plot_dir(tmp_path, "demo", "eq") == tmp_path / "demo" / "plots"
    assert _ensure_plot_dir(tmp_path, "demo", "r1.50") == (
        tmp_path / "demo" / "r1.50" / "plots"
    )
