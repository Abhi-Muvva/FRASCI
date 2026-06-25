from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from FRASCI.diff_mols.diagnostic_plots import (
    _protocol_axes,
    filter_target_threshold,
    primary_analysis_rows,
    plot_actual_determinants_vs_energy,
    plot_coo_knob_heatmaps,
    plot_determinant_convergence,
    plot_lasscf_vs_fullspace,
)


def _sample_df() -> pd.DataFrame:
    rows = []
    for method, partition in (
        ("plain_trimci", "full"),
        ("trimci_coo", "full"),
        ("lasscf_trimci_coo", "chem"),
    ):
        for dets in (50, 500, 2000):
            cycles = (1, 2, 4) if "coo" in method else (None,)
            bfgs_values = (20, 40, 80) if "coo" in method else (None,)
            for cycle in cycles:
                for bfgs in bfgs_values:
                    protocol = f"dets{dets}"
                    if cycle is not None:
                        protocol = f"cyc{cycle}_dets{dets}_bfgs{bfgs}"
                    rows.append({
                        "molecule": "demo",
                        "method": method,
                        "stage": "lasscf" if method.startswith("lasscf") else "base",
                        "partition": partition,
                        "base_partition": partition,
                        "geom_tag": "eq",
                        "protocol": protocol,
                        "e_tot": -10.0 + 1.0 / dets + (cycle or 1) * 1e-5,
                        "e_ref": -10.1,
                        "error_mha": 100.0 / dets + (cycle or 1) * 0.01,
                        "converged": True,
                        "wall_s": 1.0,
                        "n_dets_total": dets if not method.startswith("lasscf") else min(dets, 24),
                    })
    return pd.DataFrame(rows)


def test_protocol_axes_extracts_grid_values():
    parsed = _protocol_axes(_sample_df())
    coo = parsed[parsed["method"].eq("trimci_coo")].iloc[0]
    assert coo["det_limit"] == 50
    assert coo["coo_cycle"] == 1
    assert coo["bfgs_iter"] == 20


def test_primary_rows_use_lassis_for_lasscf_workflows():
    df = _sample_df()
    parent = df[df["method"].eq("lasscf_trimci_coo")].iloc[[0]].copy()
    lassis = parent.copy()
    lassis["method"] = "lassis"
    lassis["stage"] = "lassis"
    lassis["parent_method"] = "lasscf_trimci_coo"
    lassis["parent_protocol"] = parent["protocol"].iloc[0]
    lassis["base_partition"] = "chem"
    lassis["partition"] = "chem_on_lasscf_trimci_coo"
    combined = pd.concat([df, lassis], ignore_index=True)

    primary = primary_analysis_rows(combined)

    assert not primary["method"].eq("lasscf_trimci_coo").any()
    assert primary["method"].eq("lassis").sum() == 1
    assert primary.loc[primary["method"].eq("lassis"), "workflow"].iloc[0] == "lasscf_trimci_coo"
    assert primary.loc[primary["method"].eq("lassis"), "actual_n_dets"].iloc[0] == (
        parent["n_dets_total"].iloc[0]
    )


def test_target_threshold_filter_keeps_controls_and_matching_trimci():
    df = _sample_df()
    df["trimci_threshold"] = 0.01
    low = df[df["method"].eq("lasscf_trimci_coo")].iloc[[0]].copy()
    low["trimci_threshold"] = 1.0e-4
    combined = pd.concat([df, low], ignore_index=True)

    filtered = filter_target_threshold(combined, 1.0e-4)

    assert filtered["method"].isin(["plain_trimci", "trimci_coo"]).any()
    selected_lasscf = filtered[filtered["method"].eq("lasscf_trimci_coo")]
    assert len(selected_lasscf) == 1
    assert selected_lasscf["effective_trimci_threshold"].iloc[0] == 1.0e-4


def test_diagnostic_grid_plots_render(tmp_path: Path):
    df = _sample_df()
    det_path = tmp_path / "det.png"
    coo_path = tmp_path / "coo.png"
    actual_path = tmp_path / "actual.png"

    det_fig = plot_determinant_convergence(df, "demo", save_to=det_path)
    coo_fig = plot_coo_knob_heatmaps(df, "demo", save_to=coo_path)
    actual_fig = plot_actual_determinants_vs_energy(df, "demo", save_to=actual_path)

    assert len(det_fig.axes) >= 2
    assert len(coo_fig.axes) >= 2
    assert len(actual_fig.axes) >= 2
    assert det_path.exists()
    assert coo_path.exists()
    assert actual_path.exists()


def test_lasscf_vs_fullspace_matched_plot(tmp_path: Path):
    base = _sample_df()
    parents = base[base["method"].isin(["lasscf_trimci_coo"])].copy()
    lassis = parents.copy()
    lassis["method"] = "lassis"
    lassis["stage"] = "lassis"
    lassis["parent_method"] = "lasscf_trimci_coo"
    lassis["parent_protocol"] = lassis["protocol"]
    lassis["base_partition"] = lassis["partition"]
    lassis["partition"] = lassis["partition"] + "_on_lasscf_trimci_coo"
    lassis["e_tot"] -= 0.001
    df = pd.concat([base, lassis], ignore_index=True)

    path = tmp_path / "comparison.png"
    fig = plot_lasscf_vs_fullspace(df, "demo", save_to=path)

    assert len(fig.axes) == 2
    assert path.exists()
