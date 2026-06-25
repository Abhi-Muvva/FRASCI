"""Detailed diagnostics for completed single-point diff_mols benchmarks."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from FRASCI.diff_mols.report import (
    enrich_lasscf_stage_columns,
    load_runs,
)
from FRASCI.diff_mols.report_plots import METHOD_COLORS


METHOD_ORDER = (
    "plain_trimci",
    "trimci_coo",
    "lasscf_cas",
    "lasscf_trimci",
    "lasscf_trimci_coo",
)
PARTITION_MARKERS = {"full": "o", "chem": "s", "chem_bond": "P", "h1diag": "D", "h1diag_2": "D", "h1diag_4": "^"}


def _style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
    })


def _protocol_axes(df: pd.DataFrame) -> pd.DataFrame:
    out = enrich_lasscf_stage_columns(df).copy()
    own = out.get("protocol", pd.Series("", index=out.index)).fillna("").astype(str)
    parent = out.get("parent_protocol", pd.Series("", index=out.index)).fillna("").astype(str)
    if "parent_protocol" not in out.columns:
        out["parent_protocol"] = parent
    out["tuning_protocol"] = parent.where(parent.ne(""), own)
    for name, pattern in (
        ("det_limit", r"dets(\d+)"),
        ("coo_cycle", r"cyc(\d+)"),
        ("bfgs_iter", r"bfgs(\d+)"),
        ("nspin_value", r"nspin(\d+)"),
    ):
        out[name] = pd.to_numeric(
            out["tuning_protocol"].str.extract(pattern, expand=False),
            errors="coerce",
        )
    # In this benchmark a LASSCF workflow means orbital optimization followed
    # by LASSIS. Raw parent LASSCF rows are retained only for dedicated
    # correction/trajectory diagnostics.
    out["workflow"] = out["analysis_method"].astype(str)
    out["series"] = out["workflow"]
    fragmented = ~out["analysis_partition"].isin(["", "full"])
    out.loc[fragmented, "series"] += " / " + out.loc[fragmented, "analysis_partition"].astype(str)
    out["is_converged"] = out["converged"].astype(str).str.lower().isin(("true", "1"))
    out["actual_n_dets"] = pd.to_numeric(out.get("n_dets_total"), errors="coerce").astype(float)
    out["effective_trimci_threshold"] = pd.to_numeric(
        out.get("trimci_threshold", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).astype(float)
    out["effective_trimci_max_dets"] = pd.to_numeric(
        out.get("trimci_max_dets", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    ).astype(float)

    parent_rows = out[out["method"].isin(["lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo"])]
    parent_lookup = {
        (str(row.geom_tag), str(row.method), str(row.partition), str(row.protocol)): (
            row.actual_n_dets,
            row.effective_trimci_threshold,
            row.effective_trimci_max_dets,
        )
        for row in parent_rows.itertuples()
    }
    lassis_mask = out["method"].eq("lassis")
    if lassis_mask.any():
        inherited = []
        for row in out.loc[lassis_mask].itertuples():
            fallback = parent_lookup.get(
                (str(row.geom_tag), str(row.parent_method),
                 str(row.base_partition), str(row.parent_protocol)),
                (np.nan, np.nan, np.nan),
            )
            values = fallback
            # Repeated protocol labels can point to parents with different thresholds.
            # The LASSIS result records its exact checkpoint, so prefer that metadata.
            try:
                result_path = Path(str(getattr(row, "run_dir", ""))) / "result.json"
                result = json.loads(result_path.read_text())
                parent_dir = result.get("result", {}).get("runner_inner", {}).get(
                    "lasscf_checkpoint"
                )
                parent_result_path = Path(str(parent_dir)) / "result.json"
                parent_result = json.loads(parent_result_path.read_text())
                snapshot = parent_result.get("config_snapshot", {})
                values = (
                    parent_result.get("result", {}).get("n_dets_total", fallback[0]),
                    snapshot.get("trimci_threshold", fallback[1]),
                    snapshot.get("trimci_max_dets", fallback[2]),
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            inherited.append(values)
        inherited_array = np.asarray(inherited, dtype=float)
        out.loc[lassis_mask, "actual_n_dets"] = inherited_array[:, 0]
        out.loc[lassis_mask, "effective_trimci_threshold"] = inherited_array[:, 1]
        out.loc[lassis_mask, "effective_trimci_max_dets"] = inherited_array[:, 2]
    return out


def filter_target_threshold(
    df: pd.DataFrame, target_threshold: float | None
) -> pd.DataFrame:
    """Keep one strict TrimCI threshold while preserving threshold-free controls.

    LASSIS rows inherit the exact threshold from their parent checkpoint. This
    prevents an older parent with the same protocol label from entering filtered
    plots. Full-space controls and LASSCF/CAS rows have no fragment-TrimCI
    threshold and remain available as reference/control workflows.
    """
    if target_threshold is None or df.empty:
        return df.copy()
    out = _protocol_axes(df)
    threshold = pd.to_numeric(out["effective_trimci_threshold"], errors="coerce")
    threshold_match = np.isclose(
        threshold.to_numpy(dtype=float),
        float(target_threshold),
        rtol=1e-8,
        atol=max(abs(float(target_threshold)) * 1e-8, 1e-12),
        equal_nan=False,
    )
    threshold_independent = (
        out["method"].isin(["plain_trimci", "trimci_coo", "lasscf_cas"])
        | (out["method"].eq("lassis") & out["parent_method"].eq("lasscf_cas"))
    )
    return out.loc[threshold_independent.to_numpy() | threshold_match].copy()


def _primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used in headline comparisons.

    Full-space methods are represented directly. Every LASSCF family is
    represented only by its completed LASSIS-corrected row.
    """
    out = _protocol_axes(df)
    is_full_space = out["method"].isin(["plain_trimci", "trimci_coo"])
    is_completed_lasscf = out["method"].eq("lassis")
    return out[is_full_space | is_completed_lasscf].copy()


def primary_analysis_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Public notebook helper for completed-workflow tables."""
    return _primary_rows(df)


def _eq_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = _primary_rows(df)
    if out["geom_tag"].eq("eq").any():
        return out[out["geom_tag"].eq("eq")].copy()
    # Compatibility with the old diazene scan: use its equilibrium-like r=1.25 point.
    if out["geom_tag"].eq("r1.25").any():
        return out[out["geom_tag"].eq("r1.25")].copy()
    return out.copy()


def _metric(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if df["e_ref"].notna().any():
        return df["error_mha"].abs(), "|E - E_FCI| (mHa)"
    converged = df[df["is_converged"]]
    reference = converged["e_tot"].min() if not converged.empty else df["e_tot"].min()
    return (df["e_tot"] - reference) * 1000.0, "E - best converged E (mHa)"


def _save(fig: plt.Figure, save_to: Path | None) -> plt.Figure:
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


def plot_energy_landscape(df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None) -> plt.Figure:
    """Best energy per workflow/fragmentation, resolved in mHa."""
    _style()
    sub = _eq_rows(df)
    sub = sub[sub["is_converged"]].dropna(subset=["e_tot"])
    sub["metric"], ylabel = _metric(sub)
    best = sub.sort_values("metric").groupby(
        ["workflow", "analysis_method", "analysis_partition"], as_index=False
    ).first().sort_values("metric")

    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.46 * len(best) + 2.2)))
    y = np.arange(len(best))
    colors = [METHOD_COLORS.get(m, "#555555") for m in best["analysis_method"]]
    markers = [PARTITION_MARKERS.get(p, "x") for p in best["analysis_partition"]]
    max_metric = best["metric"].max() if not best.empty else 0.0
    for i, (_, row) in enumerate(best.iterrows()):
        ax.scatter(row["metric"], i, s=105, color=colors[i], marker=markers[i],
                   edgecolor="white", linewidth=1.1, zorder=3)
        metric_label = "reference" if abs(row["metric"]) < 5e-7 else f"{row['metric']:.3g} mHa"
        label_left = max_metric > 0 and row["metric"] > 0.72 * max_metric
        ax.annotate(
            metric_label, (row["metric"], i), xytext=((-7 if label_left else 7), 0),
            textcoords="offset points", va="center",
            ha=("right" if label_left else "left"),
            fontsize=8, color="0.35", clip_on=True,
        )
    labels = [
        f"{w}" if p == "full" else f"{w} / {p}"
        for w, p in zip(best["workflow"], best["analysis_partition"])
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(ylabel)
    ax.set_title(f"{mol_slug}: best converged energy by workflow", pad=76)
    positive = best.loc[best["metric"] > 0, "metric"]
    if (not positive.empty
            and best["metric"].max() / max(positive.min(), 1e-6) > 100):
        # Unlike a plain log axis, symlog keeps an exact-zero reference point
        # drawable. A zero metric plus savefig(bbox="tight") on a log axis can
        # otherwise produce an extremely wide, effectively blank PNG.
        ax.set_xscale("symlog", linthresh=max(positive.min() / 2.0, 1e-6))
        ax.set_xlim(0, best["metric"].max() * 1.45)
    elif not best.empty:
        ax.set_xlim(left=0, right=max(best["metric"].max() * 1.18, 1e-6))

    method_order = [m for m in METHOD_ORDER if m in set(best["analysis_method"])]
    partition_order = [
        p for p in ("full", "chem", "chem_bond", "h1diag", "h1diag_2", "h1diag_4")
        if p in set(best["analysis_partition"])
    ]
    method_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor=METHOD_COLORS.get(m, "#555555"),
               markeredgecolor="white", label=m)
        for m in method_order
    ]
    partition_handles = [
        Line2D([], [], marker=PARTITION_MARKERS.get(p, "x"), linestyle="none",
               markersize=8, color="#555555", label=p)
        for p in partition_order
    ]
    method_legend = ax.legend(
        handles=method_handles, title="Color = workflow", ncol=3,
        loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=8, title_fontsize=8,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=partition_handles, title="Marker = fragmentation", ncol=3,
        loc="lower right", bbox_to_anchor=(1, 1.01), fontsize=8, title_fontsize=8,
    )
    return _save(fig, save_to)


def plot_determinant_convergence(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Energy improvement with determinant budget for non-COO and COO workflows."""
    _style()
    sub = _eq_rows(df)
    sub = sub[sub["is_converged"]].dropna(subset=["det_limit", "e_tot"])
    sub["metric"], ylabel = _metric(sub)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), sharey=True)

    panels = [
        (axes[0], sub[sub["analysis_method"].isin(["plain_trimci", "lasscf_trimci"])],
         "TrimCI determinant convergence"),
        (axes[1], sub[sub["analysis_method"].isin(["trimci_coo", "lasscf_trimci_coo"])],
         "COO-TrimCI determinant convergence\n(best cycles/BFGS at each limit)"),
    ]
    handles: dict[str, object] = {}
    for ax, panel, title in panels:
        reduced = panel.groupby(
            ["series", "analysis_method", "analysis_partition", "stage", "det_limit"],
            as_index=False,
        )["metric"].min()
        for (series, method, partition, stage), grp in reduced.groupby(
            ["series", "analysis_method", "analysis_partition", "stage"]
        ):
            grp = grp.sort_values("det_limit")
            line, = ax.plot(
                grp["det_limit"], grp["metric"].clip(lower=1e-4),
                marker=PARTITION_MARKERS.get(partition, "x"),
                color=METHOD_COLORS.get(method, "#555555"),
                linestyle="-",
                linewidth=1.9,
                label=series,
            )
            handles.setdefault(series, line)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("determinant limit")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="center left",
                   bbox_to_anchor=(1.005, 0.5), fontsize=8.2, title="workflow / fragmentation")
    fig.suptitle(f"{mol_slug}: does increasing the determinant budget help?", y=1.03)
    return _save(fig, save_to)


def plot_actual_determinants_vs_energy(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Compare measured determinant totals with energy and requested caps."""
    _style()
    sub = _eq_rows(df)
    sub = sub[sub["is_converged"]].dropna(subset=["actual_n_dets", "det_limit", "e_tot"])
    sub = sub[sub["actual_n_dets"] > 0].copy()
    sub["metric"], ylabel = _metric(sub)
    if sub.empty:
        raise ValueError(f"no measured determinant counts for {mol_slug}")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3))
    reduced = sub.groupby(
        ["series", "analysis_method", "analysis_partition", "actual_n_dets", "det_limit"],
        as_index=False,
    ).agg(metric=("metric", "min"))

    handles: dict[str, object] = {}
    for (series, method, partition), grp in reduced.groupby(
        ["series", "analysis_method", "analysis_partition"]
    ):
        energy_grp = grp.groupby("actual_n_dets", as_index=False)["metric"].min().sort_values(
            "actual_n_dets"
        )
        line, = axes[0].plot(
            energy_grp["actual_n_dets"], energy_grp["metric"].clip(lower=1e-4),
            marker=PARTITION_MARKERS.get(partition, "x"),
            color=METHOD_COLORS.get(method, "#555555"),
            linewidth=1.8, label=series,
        )
        handles.setdefault(series, line)
        axes[1].scatter(
            grp["det_limit"], grp["actual_n_dets"],
            marker=PARTITION_MARKERS.get(partition, "x"),
            color=METHOD_COLORS.get(method, "#555555"),
            s=58, alpha=0.82,
        )

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("actual determinants retained (total across fragments)")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("Energy versus actual determinant count")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("requested max_dets (per fragment for LASSCF)")
    axes[1].set_ylabel("actual determinants retained (summed over fragments)")
    axes[1].set_title("Requested cap versus actual retained space")
    axes[1].text(
        0.02, 0.02,
        "LASSCF max_dets is a per-fragment cap; y is the fragment-total count.",
        transform=axes[1].transAxes, fontsize=8.5, color="0.35",
        bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9),
    )
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="center left",
                   bbox_to_anchor=(1.005, 0.5), fontsize=8.3,
                   title="completed workflow / fragmentation")
    fig.suptitle(f"{mol_slug}: actual selected-CI space versus requested budget", y=1.02)
    return _save(fig, save_to)


def plot_lasscf_vs_fullspace(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Matched comparison of completed LASSCF workflows against full-space TrimCI."""
    _style()
    sub = _protocol_axes(df)
    if sub["geom_tag"].eq("eq").any():
        sub = sub[sub["geom_tag"].eq("eq")].copy()
    elif sub["geom_tag"].eq("r1.25").any():
        sub = sub[sub["geom_tag"].eq("r1.25")].copy()
    sub = sub[sub["is_converged"]]

    comparisons = [
        ("lasscf_trimci", "plain_trimci", "LASSCF+TrimCI (with LASSIS) vs TrimCI"),
        ("lasscf_trimci_coo", "trimci_coo", "LASSCF+TrimCI+COO (with LASSIS) vs TrimCI+COO"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), sharey=True)
    handles: dict[str, object] = {}

    for ax, (parent_method, full_method, title) in zip(axes, comparisons):
        completed = sub[
            sub["method"].eq("lassis") & sub["parent_method"].eq(parent_method)
        ].copy()
        # Multiple nspin settings may exist. "Best" means the lowest converged
        # completed-workflow energy for each exact parent protocol.
        completed = completed.groupby(
            ["base_partition", "parent_protocol", "det_limit"], as_index=False
        )["e_tot"].min().rename(columns={"e_tot": "e_lasscf"})

        full = sub[sub["method"].eq(full_method)].copy()
        full = full.groupby(["protocol", "det_limit"], as_index=False)["e_tot"].min().rename(
            columns={"protocol": "parent_protocol", "e_tot": "e_full"}
        )
        matched = completed.merge(full, on=["parent_protocol", "det_limit"], how="inner")
        matched["delta_mha"] = (matched["e_lasscf"] - matched["e_full"]) * 1000.0

        for partition, grp in matched.groupby("base_partition"):
            summary = grp.groupby("det_limit", as_index=False)["delta_mha"].agg(
                ["min", "max", "median"]
            ).reset_index()
            summary = summary.sort_values("det_limit")
            color = METHOD_COLORS.get(parent_method, "#555555")
            ax.fill_between(
                summary["det_limit"], summary["min"], summary["max"],
                color=color, alpha=0.12,
            )
            line, = ax.plot(
                summary["det_limit"], summary["median"],
                marker=PARTITION_MARKERS.get(partition, "x"),
                color=color, linewidth=1.9, label=partition,
            )
            handles.setdefault(f"{parent_method} / {partition}", line)

        ax.axhline(0, color="0.25", linewidth=1.1)
        ax.set_xscale("log")
        ax.set_xlabel("matched determinant cap")
        ax.set_title(title, fontsize=11)
        ax.text(
            0.02, 0.03,
            "below 0: LASSCF+LASSIS lower energy\nabove 0: full-space method lower energy",
            transform=ax.transAxes, fontsize=8.5, color="0.35",
            bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9),
        )

    axes[0].set_ylabel("E(LASSCF workflow) - E(full-space workflow) (mHa)")
    if handles:
        fig.legend(
            handles.values(), handles.keys(), loc="center left",
            bbox_to_anchor=(1.005, 0.5), fontsize=8.5,
            title="completed workflow / fragmentation",
        )
    fig.suptitle(
        f"{mol_slug}: does fragmentation plus orbital optimization beat full-space TrimCI?",
        y=1.02,
    )
    return _save(fig, save_to)


def plot_fragmentation_heatmap(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Best metric for each LASSCF workflow and fragmentation."""
    _style()
    sub = _eq_rows(df)
    sub = sub[sub["is_converged"] & sub["analysis_method"].str.startswith("lasscf")].copy()
    sub["metric"], metric_label = _metric(sub)
    table = sub.pivot_table(
        index="workflow", columns="analysis_partition", values="metric", aggfunc="min"
    )
    columns = [p for p in ("chem", "chem_bond", "h1diag", "h1diag_2", "h1diag_4") if p in table.columns]
    table = table.reindex(columns=columns)

    fig, ax = plt.subplots(figsize=(1.65 * max(3, len(columns)) + 3.8,
                                    0.65 * max(3, len(table)) + 2.2))
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="viridis_r")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center",
                        color="white" if values[i, j] > np.nanmedian(values) else "black",
                        fontsize=9)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title(f"{mol_slug}: fragmentation sensitivity\n(best converged protocol)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric_label)
    return _save(fig, save_to)


def plot_coo_knob_heatmaps(
    df: pd.DataFrame, mol_slug: str, *, det_limit: int = 2000,
    save_to: Path | None = None,
) -> plt.Figure:
    """Cycle/BFGS sensitivity at a fixed determinant budget."""
    _style()
    sub = _eq_rows(df)
    sub = sub[
        sub["is_converged"]
        & sub["analysis_method"].isin(["trimci_coo", "lasscf_trimci_coo"])
    ].dropna(subset=["coo_cycle", "bfgs_iter"])
    if sub.empty:
        raise ValueError(f"no converged COO grid rows for {mol_slug}")
    available = sorted(sub["det_limit"].dropna().astype(int).unique())
    selected = det_limit if det_limit in available else max(available)
    sub = sub[sub["det_limit"].eq(selected)].copy()
    sub["metric"], metric_label = _metric(sub)
    series = list(sub.groupby("series")["metric"].min().sort_values().index[:6])

    ncols = min(3, len(series))
    nrows = int(np.ceil(len(series) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows),
        squeeze=False, constrained_layout=True,
    )
    panel_tables: dict[str, pd.DataFrame] = {}
    panel_max = 0.0
    for label in series:
        panel = sub[sub["series"].eq(label)].pivot_table(
            index="coo_cycle", columns="bfgs_iter", values="metric", aggfunc="min"
        ).sort_index().sort_index(axis=1)
        panel = panel - np.nanmin(panel.to_numpy())
        panel_tables[label] = panel
        panel_max = max(panel_max, float(np.nanmax(panel.to_numpy())))
    panel_max = max(panel_max, 1e-8)

    for ax, label in zip(axes.flat, series):
        panel = panel_tables[label]
        image = ax.imshow(
            panel.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=panel_max
        )
        ax.set_xticks(range(len(panel.columns)), [int(x) for x in panel.columns])
        ax.set_yticks(range(len(panel.index)), [int(x) for x in panel.index])
        ax.set_xlabel("BFGS max iterations")
        ax.set_ylabel("COO cycles")
        ax.set_title(label.replace(" / ", "\n"), fontsize=10)
        for i, row in enumerate(panel.to_numpy()):
            for j, value in enumerate(row):
                if np.isfinite(value):
                    color = "white" if value > panel_max * 0.45 else "black"
                    ax.text(j, i, f"{value:.3g}", ha="center", va="center",
                            color=color, fontsize=8)
    for ax in axes.flat[len(series):]:
        ax.set_axis_off()
    fig.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.82,
        label="mHa above best cycle/BFGS setting in each panel",
    )
    fig.suptitle(f"{mol_slug}: COO optimizer sensitivity at dets={selected}")
    return _save(fig, save_to)


def plot_lassis_gain(df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None) -> plt.Figure:
    """Matched parent-to-LASSIS energy lowering across determinant budgets."""
    _style()
    sub = _protocol_axes(df)
    if sub["geom_tag"].eq("eq").any():
        sub = sub[sub["geom_tag"].eq("eq")].copy()
    elif sub["geom_tag"].eq("r1.25").any():
        sub = sub[sub["geom_tag"].eq("r1.25")].copy()
    parents = sub[
        sub["is_converged"] & sub["method"].isin(["lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo"])
    ][["method", "partition", "protocol", "e_tot", "det_limit"]].rename(columns={
        "method": "parent_method",
        "partition": "base_partition",
        "protocol": "parent_protocol",
        "e_tot": "e_parent",
        "det_limit": "parent_det_limit",
    })
    lassis = sub[sub["method"].eq("lassis") & sub["is_converged"]].copy()
    merged = lassis.merge(
        parents,
        on=["parent_method", "base_partition", "parent_protocol"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"no matched LASSIS-parent rows for {mol_slug}")
    merged["gain_mha"] = (merged["e_parent"] - merged["e_tot"]) * 1000.0
    merged["det"] = merged["det_limit"].fillna(merged["parent_det_limit"])
    merged["label"] = merged["parent_method"] + " / " + merged["base_partition"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    with_det = merged.dropna(subset=["det"])
    reduced = with_det.groupby(
        ["label", "parent_method", "base_partition", "det"], as_index=False
    )["gain_mha"].agg(["min", "max", "median"]).reset_index()
    for (label, method, partition), grp in reduced.groupby(
        ["label", "parent_method", "base_partition"]
    ):
        grp = grp.sort_values("det")
        ax.fill_between(
            grp["det"], grp["min"], grp["max"],
            color=METHOD_COLORS.get(method, "#555555"), alpha=0.12,
        )
        ax.plot(grp["det"], grp["median"], marker=PARTITION_MARKERS.get(partition, "x"),
                color=METHOD_COLORS.get(method, "#555555"), label=label)
    ax.axhline(0, color="0.35", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("parent determinant limit")
    ax.set_ylabel("E(parent) - E(parent + LASSIS) (mHa)")
    ax.set_title(f"{mol_slug}: interfragment correlation recovered by LASSIS")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5)
    return _save(fig, save_to)


def plot_lassis_nspin_sensitivity(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Best LASSIS energy as the allowed spin-flip depth changes."""
    _style()
    sub = _protocol_axes(df)
    sub = sub[sub["method"].eq("lassis") & sub["is_converged"]].dropna(
        subset=["n_spin", "e_tot"]
    )
    spins = sorted(sub["n_spin"].astype(int).unique())
    if len(spins) < 2:
        raise ValueError(f"{mol_slug} has no multi-nspin LASSIS sweep")

    best = sub.groupby(
        ["analysis_method", "base_partition", "n_spin"], as_index=False
    )["e_tot"].min()
    best["series"] = best["analysis_method"] + " / " + best["base_partition"]
    best["delta_mha"] = best.groupby("series")["e_tot"].transform(
        lambda values: (values - values.min()) * 1000.0
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for (series, method, partition), grp in best.groupby(
        ["series", "analysis_method", "base_partition"]
    ):
        grp = grp.sort_values("n_spin")
        ax.plot(
            grp["n_spin"], grp["delta_mha"],
            marker=PARTITION_MARKERS.get(partition, "x"),
            color=METHOD_COLORS.get(method, "#555555"),
            linewidth=1.8, label=series,
        )
    ax.set_xticks(spins)
    ax.set_xlabel("LASSIS nspin")
    ax.set_ylabel("best E(nspin) - best across nspin (mHa)")
    ax.set_title(f"{mol_slug}: LASSIS spin-space sensitivity")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5)
    return _save(fig, save_to)


def plot_convergence_and_cost(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Convergence rate and wall-time distribution by workflow."""
    _style()
    sub = _eq_rows(df)
    order = [m for m in METHOD_ORDER if m in sub["analysis_method"].unique()]
    labels = []
    conv_rates = []
    wall_data = []
    colors = []
    for method in order:
        group = sub[sub["analysis_method"].eq(method)]
        labels.append(method)
        conv_rates.append(100.0 * group["is_converged"].mean())
        wall_data.append(group.loc[group["is_converged"], "wall_s"].dropna().to_numpy())
        colors.append(METHOD_COLORS.get(method, "#555555"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    axes[0].barh(labels, conv_rates, color=colors)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("converged runs (%)")
    axes[0].set_title("Convergence reliability")
    for i, value in enumerate(conv_rates):
        axes[0].text(min(value + 1, 97), i, f"{value:.1f}%", va="center", fontsize=9)

    positions = np.arange(1, len(labels) + 1)
    axes[1].boxplot(wall_data, positions=positions, vert=True, showfliers=False)
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("wall time (s, log)")
    axes[1].set_title("Cost distribution for converged runs")
    fig.suptitle(f"{mol_slug}: reliability and compute cost", y=1.02)
    return _save(fig, save_to)


_MACRO_RE = re.compile(r"LASSCF macro\s+(\d+)\s*:\s*E\s*=\s*(-?\d+(?:\.\d+)?)")
_TRIMCI_RE = re.compile(r"\[C\+\+\]\s+Iteration\s+(\d+)\s+energy:\s*(-?\d+(?:\.\d+)?)")


def _trajectory(run_dir: str | Path, method: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    log_path = Path(run_dir) / "log.txt"
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = _MACRO_RE.findall(text) if method.startswith("lasscf") else _TRIMCI_RE.findall(text)
    if not matches:
        return None
    steps = np.asarray([int(i) for i, _ in matches], dtype=int)
    energies = np.asarray([float(e) for _, e in matches], dtype=float)
    if not method.startswith("lasscf"):
        # COO/TrimCI logs can contain restarts; plot the cumulative recorded sequence.
        steps = np.arange(1, len(energies) + 1)
    delta = np.abs((energies - energies[-1]) * 1000.0)
    delta = np.clip(delta, 1e-7, None)
    return steps, delta, "macro iteration" if method.startswith("lasscf") else "recorded CI iteration"


def plot_iteration_trajectories(
    df: pd.DataFrame, mol_slug: str, *, save_to: Path | None = None
) -> plt.Figure:
    """Representative inner/macro convergence traces from saved logs."""
    _style()
    sub = _protocol_axes(df)
    if sub["geom_tag"].eq("eq").any():
        sub = sub[sub["geom_tag"].eq("eq")]
    elif sub["geom_tag"].eq("r1.25").any():
        sub = sub[sub["geom_tag"].eq("r1.25")]
    sub = sub[
        sub["is_converged"]
        & sub["method"].isin([
            "plain_trimci", "trimci_coo",
            "lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo",
        ])
    ].copy()
    sub["metric"], _ = _metric(sub)
    candidates = sub.sort_values("metric").groupby("method", as_index=False).first()

    trajectories = []
    for _, row in candidates.iterrows():
        trajectory = _trajectory(row["run_dir"], row["method"])
        if trajectory is None:
            continue
        steps, delta, step_label = trajectory
        if len(steps) > 600:
            keep = np.unique(np.linspace(0, len(steps) - 1, 600).astype(int))
            steps, delta = steps[keep], delta[keep]
        label = row["method"]
        if row["partition"] != "full":
            label += f" / {row['partition']}"
        trajectories.append((row["method"], label, steps, delta, step_label))
    if not trajectories:
        raise ValueError(f"no parseable iteration logs for {mol_slug}")

    ncols = min(2, len(trajectories))
    nrows = int(np.ceil(len(trajectories) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.0 * ncols, 4.2 * nrows),
        squeeze=False, constrained_layout=True,
    )
    for ax, (method, label, steps, delta, step_label) in zip(axes.flat, trajectories):
        ax.plot(steps, delta, color=METHOD_COLORS.get(method, "#555555"), linewidth=1.7)
        ax.set_yscale("log")
        ax.set_xlabel(step_label)
        ax.set_ylabel("|E(iter) - E(final recorded)| (mHa)")
        ax.set_title(label, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes.flat[len(trajectories):]:
        ax.set_axis_off()
    fig.suptitle(
        f"{mol_slug}: representative solver trajectories\n"
        "(raw LASSCF traces are the orbital-optimization stage feeding LASSIS)"
    )
    return _save(fig, save_to)


def plot_cross_molecule_summary(
    frames: dict[str, pd.DataFrame], *, save_to: Path | None = None
) -> plt.Figure:
    """Best converged workflow metric for every completed molecule."""
    _style()
    rows = []
    for slug, frame in frames.items():
        sub = _eq_rows(frame)
        sub = sub[sub["is_converged"]].copy()
        sub["metric"], _ = _metric(sub)
        best = sub.groupby("workflow", as_index=False)["metric"].min()
        best["molecule"] = slug
        rows.append(best)
    data = pd.concat(rows, ignore_index=True)
    table = data.pivot(index="molecule", columns="workflow", values="metric")

    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(table.columns) + 4),
                                    max(4.8, 0.75 * len(table) + 2.2)))
    values = np.log10(table.to_numpy(dtype=float).clip(min=1e-4))
    image = ax.imshow(values, aspect="auto", cmap="viridis_r")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center",
                        color="white" if values[i, j] > np.nanmedian(values) else "black",
                        fontsize=8)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title("Cross-molecule best converged result\n"
                 "(FCI error where available; otherwise mHa above best observed)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(metric in mHa)")
    return _save(fig, save_to)


PLOTTERS = (
    ("01_energy_landscape", plot_energy_landscape),
    ("02_determinant_convergence", plot_determinant_convergence),
    ("02b_actual_determinants_vs_energy", plot_actual_determinants_vs_energy),
    ("02c_lasscf_vs_fullspace", plot_lasscf_vs_fullspace),
    ("03_fragmentation_heatmap", plot_fragmentation_heatmap),
    ("04_coo_knob_heatmaps", plot_coo_knob_heatmaps),
    ("05_lassis_gain", plot_lassis_gain),
    ("06_convergence_and_cost", plot_convergence_and_cost),
    ("07_iteration_trajectories", plot_iteration_trajectories),
    ("08_lassis_nspin_sensitivity", plot_lassis_nspin_sensitivity),
)


def generate_detailed_diagnostics(
    results_root: str | Path,
    mol_slug: str,
    *,
    target_threshold: float | None = None,
) -> dict[str, Path | str]:
    """Generate all detailed diagnostics for one molecule."""
    root = Path(results_root)
    df = filter_target_threshold(load_runs(root, mol_slug), target_threshold)
    out_dir = root / mol_slug / "plots" / "detailed"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path | str] = {}
    for name, plotter in PLOTTERS:
        if name == "08_lassis_nspin_sensitivity":
            parsed = _protocol_axes(df)
            nspins = parsed.loc[
                parsed["method"].eq("lassis") & parsed["is_converged"], "n_spin"
            ].dropna().nunique()
            if nspins < 2:
                continue
        try:
            path = out_dir / f"{name}.png"
            fig = plotter(df, mol_slug, save_to=path)
            plt.close(fig)
            saved[name] = path
        except Exception as exc:
            saved[f"{name}_ERR"] = str(exc)
    return saved


def generate_all_detailed_diagnostics(
    results_root: str | Path,
    *,
    target_threshold: float | None = None,
) -> dict[str, dict[str, Path | str]]:
    """Generate per-molecule diagnostics plus a cross-molecule summary."""
    root = Path(results_root)
    slugs = sorted(p.parent.name for p in root.glob("*/runs_index.csv"))
    all_saved = {
        slug: generate_detailed_diagnostics(
            root, slug, target_threshold=target_threshold
        )
        for slug in slugs
    }
    frames = {
        slug: filter_target_threshold(load_runs(root, slug), target_threshold)
        for slug in slugs
    }
    cross_dir = root / "plots"
    cross_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = cross_dir / "cross_molecule_summary.png"
        fig = plot_cross_molecule_summary(frames, save_to=path)
        plt.close(fig)
        all_saved["_cross_molecule"] = {"summary": path}
    except Exception as exc:
        all_saved["_cross_molecule"] = {"summary_ERR": str(exc)}
    manifest = {
        "_filter": {"target_trimci_threshold": target_threshold},
        **{
            k: {n: str(v) for n, v in values.items()}
            for k, values in all_saved.items()
        },
    }
    (cross_dir / "detailed_plot_manifest.json").write_text(json.dumps(manifest, indent=2))
    return all_saved
