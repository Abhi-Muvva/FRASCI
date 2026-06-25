"""Publication-quality plotting for the diff_mols benchmark.

Every plot here is designed around a specific question the project is trying to answer
(see the comment at the top of each function). The point is:

  * one *story* per plot, no eye-chart legends;
  * consistent color/marker conventions across plots (method = color, partition = marker);
  * absolute energies AND error-vs-reference AND % correlation recovered, depending on
    what's most informative for the question;
  * everything saves to disk in <mol>/plots/ (and per-geom <mol>/<geom>/plots/ where
    applicable) at publication DPI, with matched figure widths so they tile nicely on a
    poster or in a report.

Reference is the inline FCI in the same active space (column ``e_ref`` in runs_index.csv,
populated by ``integrals_builder``). By definition no method beats FCI within the same AS,
so error_mha tells you "% of correlation NOT recovered within the AS".
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from FRASCI.diff_mols.report import (
    load_runs, best_per_group, enrich_lasscf_stage_columns,
)


# ---------------------------------------------------------------------------
# Style — consistent across every plot in this module
# ---------------------------------------------------------------------------

METHOD_COLORS: dict[str, str] = {
    "plain_trimci":      "#1f77b4",   # blue   — selected CI on full AS, no orbital opt
    "trimci_coo":        "#ff7f0e",   # orange — selected CI on full AS + COO orbital opt
    "lasscf_cas":        "#2ca02c",   # green  — LASSCF with mrh CSF solver
    "lasscf_trimci":     "#9467bd",   # purple — LASSCF with TrimCI per fragment
    "lasscf_trimci_coo": "#d62728",   # red    — LASSCF with COO-TrimCI per fragment (the best)
    "FCI (reference)":   "#000000",   # black  — exact in this AS
    "HF":                "#7f7f7f",   # gray   — uncorrelated reference
}

PARTITION_MARKERS: dict[str, str] = {
    "full":      "o",      # circle (non-fragmented methods)
    "chem":      "s",      # square (per-atom IAO)
    "chem_bond": "P",      # plus (literature-style bond-grouped IAO)
    "h1diag_2":  "D",      # diamond (2 frags by h1 diagonal)
    "h1diag_4":  "^",      # triangle (4 frags)
}

PARTITION_LINESTYLES: dict[str, str] = {
    "full":      "-",
    "chem":      "--",
    "chem_bond": (0, (3, 1, 1, 1)),   # dash-dot-dot
    "h1diag_2":  "-.",
    "h1diag_4":  ":",
}

# Method display order (left → right in bar charts, top → bottom in legends)
METHOD_ORDER = (
    "plain_trimci", "trimci_coo",
    "lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo",
)


STAGE_ALPHA: dict[str, float] = {
    "base": 0.78,
    "lasscf": 0.78,
    "lassis": 0.98,
}


def _apply_style():
    """Set matplotlib defaults for publication-quality figures."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "lines.linewidth": 1.7,
        "lines.markersize": 6.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.85",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_from_geom(geom_tag: str) -> float | None:
    """Parse the radius from a 'r1.25' style tag. Returns None for 'eq' / non-scan."""
    if not isinstance(geom_tag, str) or not geom_tag.startswith("r"):
        return None
    try:
        return float(geom_tag[1:])
    except ValueError:
        return None


def _method_label(method: str, partition: str, stage: str | None = None) -> str:
    """Compact human-friendly label combining method + partition."""
    if stage == "lassis":
        method = f"{method} + LASSIS"
    if partition in ("full", "", None):
        return method
    return f"{method} ({partition})"


def _line_style_for(method: str, partition: str, stage: str | None = None) -> dict:
    """Color from method, marker+linestyle from partition."""
    base = {
        "color":     METHOD_COLORS.get(method, "#444444"),
        "marker":    PARTITION_MARKERS.get(partition, "x"),
        "linestyle": PARTITION_LINESTYLES.get(partition, "-"),
        "alpha":     STAGE_ALPHA.get(str(stage), 0.95),
        "markeredgewidth": 1.2,
        "markeredgecolor": "white",
    }
    if stage == "lassis":
        base["linestyle"] = "-"
        base["linewidth"] = 2.25
    return base


def _analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    return enrich_lasscf_stage_columns(df)


def _ensure_plot_dir(results_root: Path, mol_slug: str, geom_tag: str | None = None) -> Path:
    if geom_tag and geom_tag != "eq":
        d = Path(results_root) / mol_slug / geom_tag / "plots"
    else:
        d = Path(results_root) / mol_slug / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _legend_outside(ax, *, title: str | None = None, ncols: int = 1):
    """Keep legends out of the data region so dense protocol lines remain readable."""
    return ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        title=title,
        ncols=ncols,
        fontsize=8.5,
    )


# ---------------------------------------------------------------------------
# Plot A — PES overlay with FCI reference
# ---------------------------------------------------------------------------

def plot_pes_with_reference(df: pd.DataFrame, mol_slug: str,
                            *, save_to: Path | None = None,
                            zoom_minimum: bool = False) -> plt.Figure:
    """Question: does every method correctly reproduce the PES shape, and how far above
    FCI does each one sit?

    One curve per (method, partition). FCI reference overlaid as a thick black line. The
    minimum geometry is annotated. If ``zoom_minimum=True``, a zoomed-in inset of the
    well region is added.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub["r"] = sub["geom_tag"].map(_r_from_geom)
    sub = sub.dropna(subset=["r"]).sort_values("r")
    if sub.empty:
        raise ValueError(f"no scan geometries for {mol_slug}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # FCI reference curve (one e_ref per geom; pick first row per geom_tag)
    ref = sub.drop_duplicates(subset="r").sort_values("r")
    ax.plot(ref["r"], ref["e_ref"], color=METHOD_COLORS["FCI (reference)"],
            linewidth=2.6, marker="None", label="FCI in active space (ref.)",
            zorder=10)

    # Method curves (best result per method+partition+geom)
    best = best_per_group(sub, by=("analysis_label", "analysis_method", "analysis_partition", "stage", "r"))
    for (label, method, partition, stage), grp in best.groupby(
        ["analysis_label", "analysis_method", "analysis_partition", "stage"]
    ):
        grp = grp.sort_values("r")
        style = _line_style_for(method, partition, stage)
        ax.plot(grp["r"], grp["e_tot"], label=_method_label(method, partition, stage),
                markersize=6.5, **style)

    # Annotate minimum
    ref_min = ref.loc[ref["e_ref"].idxmin()]
    ax.axvline(ref_min["r"], color="0.65", linestyle="--", linewidth=1.0,
               alpha=0.5, zorder=0)
    ax.annotate(f"FCI min: r={ref_min['r']:.2f} Å, E={ref_min['e_ref']:.4f} Ha",
                xy=(ref_min["r"], ref_min["e_ref"]),
                xytext=(8, 12), textcoords="offset points",
                fontsize=9, color="0.25",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.9))

    ax.set_xlabel("Bond length r (Å)")
    ax.set_ylabel("Total energy E (Ha)")
    ax.set_title(f"{mol_slug}: PES across methods (active-space FCI as reference)")
    _legend_outside(ax)
    ax.minorticks_on()

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot B — Error vs r, log scale
# ---------------------------------------------------------------------------

def plot_error_vs_r(df: pd.DataFrame, mol_slug: str,
                    *, save_to: Path | None = None) -> plt.Figure:
    """Question: quantitatively, how big is each method's gap to FCI at each geometry?

    Log-scale absolute error_mha vs r. Horizontal "chemical accuracy" line at 1.6 mHa
    (= 1 kcal/mol). Below the line → method is chemical-accuracy. Above → noticeable
    error in the chosen AS.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub["r"] = sub["geom_tag"].map(_r_from_geom)
    sub = sub.dropna(subset=["r", "error_mha"]).sort_values("r")
    sub["abs_err_mha"] = sub["error_mha"].abs().clip(lower=1e-3)   # avoid log(0)
    if sub.empty:
        raise ValueError(f"no error data for {mol_slug}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    best = best_per_group(sub, by=("analysis_label", "analysis_method", "analysis_partition", "stage", "r"))
    for (label, method, partition, stage), grp in best.groupby(
        ["analysis_label", "analysis_method", "analysis_partition", "stage"]
    ):
        grp = grp.sort_values("r")
        style = _line_style_for(method, partition, stage)
        ax.plot(grp["r"], grp["abs_err_mha"], label=_method_label(method, partition, stage),
                **style)

    # Chemical accuracy line (1 kcal/mol = 1.594 mHa)
    ax.axhline(1.594, color="0.45", linestyle="--", linewidth=1.2, alpha=0.7,
               zorder=0, label="chemical accuracy (1 kcal/mol)")

    ax.set_yscale("log")
    ax.set_xlabel("Bond length r (Å)")
    ax.set_ylabel("|E_method − E_FCI|   (mHa, log scale)")
    ax.set_title(f"{mol_slug}: method error vs FCI across the PES")
    _legend_outside(ax)
    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.25)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot C — Fragmentation sensitivity
# ---------------------------------------------------------------------------

def plot_fragmentation_sensitivity(df: pd.DataFrame, mol_slug: str,
                                   *, save_to: Path | None = None) -> plt.Figure:
    """Question: for each LASSCF-based method, how does the choice of partition
    (including chem_bond where configured) affect accuracy at each geometry?

    One panel per LASSCF-based method (lasscf_cas, lasscf_trimci_coo). x = r, y = error
    (log scale). Lines colored by partition. Lower = better.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub["r"] = sub["geom_tag"].map(_r_from_geom)
    sub = sub.dropna(subset=["r", "error_mha"]).sort_values("r")
    lasscf_methods = sorted(
        m for m in sub["analysis_method"].dropna().unique()
        if str(m).startswith("lasscf")
    )
    if not lasscf_methods:
        raise ValueError(f"no LASSCF runs for {mol_slug}")

    sub["abs_err_mha"] = sub["error_mha"].abs().clip(lower=1e-3)
    n = len(lasscf_methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]

    partition_colors = {
        "chem":      "#e41a1c",
        "chem_bond": "#ff7f00",
        "h1diag_2":  "#377eb8",
        "h1diag_4":  "#4daf4a",
    }

    for ax, method in zip(axes, lasscf_methods):
        mdat = sub[sub["analysis_method"] == method]
        for (partition, stage), grp in mdat.groupby(["analysis_partition", "stage"]):
            grp = grp.sort_values("r")
            ax.plot(grp["r"], grp["abs_err_mha"],
                    marker=PARTITION_MARKERS.get(partition, "x"),
                    linestyle="-" if stage == "lassis" else PARTITION_LINESTYLES.get(partition, "-"),
                    linewidth=2.25 if stage == "lassis" else 1.7,
                    alpha=STAGE_ALPHA.get(str(stage), 0.95),
                    color=partition_colors.get(partition, "#444"),
                    label=f"{partition} + LASSIS" if stage == "lassis" else partition,
                    markeredgecolor="white", markeredgewidth=1.0)
        ax.axhline(1.594, color="0.45", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="chem. accuracy" if ax is axes[0] else None)
        ax.set_yscale("log")
        ax.set_xlabel("r (Å)")
        ax.set_title(method, fontsize=12)
        _legend_outside(ax, title="partition")
        ax.minorticks_on()
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("|E_method − E_FCI|   (mHa, log scale)")
    fig.suptitle(f"{mol_slug}: how fragmentation choice affects LASSCF accuracy",
                 y=1.02, fontsize=13)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot D — Ingredient contribution (what does COO / LASSIS / fragmentation buy?)
# ---------------------------------------------------------------------------

def plot_ingredient_contributions(df: pd.DataFrame, mol_slug: str,
                                  *, save_to: Path | None = None) -> plt.Figure:
    """Three panels:
      1. COO contribution: ``error(plain_trimci) − error(trimci_coo)``  → mHa saved by
         adding COO orbital opt on the full active space.
      2. Fragmentation cost:  ``error(lasscf_cas/best_partition) − error(plain_trimci)``
         (positive = LASSCF loses this much vs full-AS TrimCI). Tells you the price you
         pay for fragmenting.
      3. LASSIS contribution: ``error(lasscf_*) − error(lassis on top of it)``  →
         multi-state correction recovered.

    All three: x = r (Å), y = Δerror (mHa). Positive y on the COO/LASSIS panels means
    the ingredient *helps*; positive y on the fragmentation panel means fragmentation
    *costs* you.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub["r"] = sub["geom_tag"].map(_r_from_geom)
    sub = sub.dropna(subset=["r", "error_mha"]).sort_values("r")

    # Pivot: best error per (method, partition) per r
    best = (best_per_group(sub, by=("method", "partition", "analysis_method",
                                    "analysis_partition", "stage", "r"))
            [["method", "partition", "analysis_method", "analysis_partition",
              "stage", "r", "error_mha"]])

    def _series(method, partition):
        s = best[(best.method == method) & (best.partition == partition)]
        return s.set_index("r")["error_mha"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # --- (1) COO contribution ---
    ax = axes[0]
    pt = _series("plain_trimci", "full")
    tc = _series("trimci_coo", "full")
    if not pt.empty and not tc.empty:
        delta = (pt - tc).reindex(sorted(set(pt.index) | set(tc.index)))
        ax.plot(delta.index, delta.values, "o-", color="#ff7f0e",
                label="full AS:  plain_trimci − trimci_coo", markeredgecolor="white")
    # Also compare best lasscf_cas vs best lasscf_trimci_coo (per geom)
    cas = best[best.method == "lasscf_cas"].groupby("r")["error_mha"].min()
    coo = best[best.method == "lasscf_trimci_coo"].groupby("r")["error_mha"].min()
    if not cas.empty and not coo.empty:
        idx = sorted(set(cas.index) & set(coo.index))
        delta = cas.reindex(idx) - coo.reindex(idx)
        ax.plot(delta.index, delta.values, "D--", color="#d62728",
                label="LASSCF: cas − trimci_coo (best partition)", markeredgecolor="white")
    ax.axhline(0, color="0.4", linewidth=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("Δerror_mHa  (positive = COO helps)")
    ax.set_title("(1) COO orbital-opt contribution")
    _legend_outside(ax)
    ax.minorticks_on()

    # --- (2) Fragmentation cost ---
    ax = axes[1]
    pt = _series("plain_trimci", "full")
    for method, color, marker in [("lasscf_cas", "#2ca02c", "s"),
                                  ("lasscf_trimci", "#9467bd", "^"),
                                  ("lasscf_trimci_coo", "#d62728", "D")]:
        s = best[best.method == method].groupby("r")["error_mha"].min()
        if pt.empty or s.empty:
            continue
        idx = sorted(set(pt.index) & set(s.index))
        cost = (s.reindex(idx) - pt.reindex(idx))      # positive = LASSCF worse than full-AS TrimCI
        ax.plot(cost.index, cost.values, marker=marker, linestyle="--",
                color=color, label=f"{method} − plain_trimci",
                markeredgecolor="white")
    ax.axhline(0, color="0.4", linewidth=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("Δerror_mHa  (positive = fragmentation costs)")
    ax.set_title("(2) Fragmentation cost vs full-AS TrimCI")
    _legend_outside(ax)
    ax.minorticks_on()

    # --- (3) LASSIS contribution on top of each LASSCF parent ---
    ax = axes[2]
    # LASSIS partition is "<part>_on_<parent_method>" — extract parent
    lassis = sub[sub.stage == "lassis"].copy()
    # For each parent method, compute LASSIS - LASSCF (best partition) per r
    for parent_method, color, marker in [("lasscf_cas", "#2ca02c", "s"),
                                          ("lasscf_trimci", "#9467bd", "^"),
                                          ("lasscf_trimci_coo", "#d62728", "D")]:
        las_best = (lassis[lassis.analysis_method == parent_method]
                    .groupby("r")["error_mha"].min())
        cas_best = best[best.method == parent_method].groupby("r")["error_mha"].min()
        if las_best.empty or cas_best.empty:
            continue
        idx = sorted(set(las_best.index) & set(cas_best.index))
        rec = cas_best.reindex(idx) - las_best.reindex(idx)   # positive = LASSIS recovered
        ax.plot(rec.index, rec.values, marker=marker, linestyle="-.",
                color=color, label=f"lassis − {parent_method}",
                markeredgecolor="white")
    ax.axhline(0, color="0.4", linewidth=1)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("Δerror_mHa  (positive = LASSIS recovers)")
    ax.set_title("(3) LASSIS multi-state contribution")
    _legend_outside(ax)
    ax.minorticks_on()

    fig.suptitle(f"{mol_slug}: ingredient contributions across the PES",
                 y=1.03, fontsize=13)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot E — Correlation recovery percentage
# ---------------------------------------------------------------------------

def plot_correlation_recovery(df: pd.DataFrame, mol_slug: str,
                              integrals_root: Path | None = None,
                              *, save_to: Path | None = None) -> plt.Figure:
    """Question: of the total electron-correlation energy (e_HF − e_FCI in the chosen AS),
    what fraction does each method recover at each geometry?

    Per row:  recovered = (e_HF − e_method) / (e_HF − e_FCI) × 100 %.

    e_HF is read from each geom's scf_summary.json.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub["r"] = sub["geom_tag"].map(_r_from_geom)
    sub = sub.dropna(subset=["r", "error_mha", "e_ref"])
    if sub.empty:
        raise ValueError(f"no error data for {mol_slug}")

    # Need e_hf per geom — read from scf_summary.json
    import json
    if integrals_root is None:
        # Infer: results_root / mol / <geom> / integrals / scf_summary.json
        sample_run = Path(sub["run_dir"].iloc[0])
        mol_root = sample_run.parent.parent
    else:
        mol_root = Path(integrals_root) / mol_slug
    e_hf_by_geom: dict[str, float] = {}
    for sub_dir in mol_root.iterdir():
        if not sub_dir.is_dir():
            continue
        sp = sub_dir / "integrals" / "scf_summary.json"
        if sp.exists():
            e_hf_by_geom[sub_dir.name] = json.loads(sp.read_text()).get("e_hf")

    sub["e_hf"] = sub["geom_tag"].map(e_hf_by_geom)
    sub = sub.dropna(subset=["e_hf"])
    sub["corr_total"] = sub["e_hf"] - sub["e_ref"]              # always positive
    sub["corr_recov"] = sub["e_hf"] - sub["e_tot"]
    sub["pct_recovered"] = 100.0 * sub["corr_recov"] / sub["corr_total"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    best = best_per_group(sub, by=("analysis_label", "analysis_method", "analysis_partition", "stage", "r"))
    for (label, method, partition, stage), grp in best.groupby(
        ["analysis_label", "analysis_method", "analysis_partition", "stage"]
    ):
        grp = grp.sort_values("r")
        style = _line_style_for(method, partition, stage)
        ax.plot(grp["r"], grp["pct_recovered"], label=_method_label(method, partition, stage),
                **style)

    ax.axhline(100, color="0.3", linestyle=":", linewidth=1.0,
               label="FCI (100% recovered)", alpha=0.8)
    ax.set_xlabel("Bond length r (Å)")
    ax.set_ylabel("% of (E_HF − E_FCI) recovered")
    ax.set_title(f"{mol_slug}: correlation energy recovery per method")
    _legend_outside(ax)
    ax.minorticks_on()
    ax.set_ylim(-5, 105)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot F — Pareto: compute cost vs accuracy
# ---------------------------------------------------------------------------

def plot_pareto_cost_vs_error(df: pd.DataFrame, mol_slug: str,
                              *, save_to: Path | None = None) -> plt.Figure:
    """Question: which methods deliver the best accuracy per second of compute?

    Scatter: x = wall_s (log), y = |error_mha| (log). One point per run. Method = color,
    partition = marker. Pareto-optimal points (lower-left frontier) shown as a connecting line.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub = sub.dropna(subset=["error_mha", "wall_s"])
    sub = sub[sub["wall_s"] > 0]
    sub["abs_err_mha"] = sub["error_mha"].abs().clip(lower=1e-4)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for (method, partition, stage), grp in sub.groupby(["analysis_method", "analysis_partition", "stage"]):
        style = _line_style_for(method, partition, stage)
        # Don't draw a line, just markers
        ax.scatter(grp["wall_s"], grp["abs_err_mha"],
                   color=style["color"], marker=style["marker"],
                   s=58, alpha=0.85,
                   edgecolor="white", linewidth=1.0,
                   label=_method_label(method, partition, stage))

    # Pareto frontier: sort by wall_s asc, keep points where err strictly decreases
    pareto = sub.sort_values("wall_s")
    pf = []
    best_err = np.inf
    for _, row in pareto.iterrows():
        if row["abs_err_mha"] < best_err:
            pf.append((row["wall_s"], row["abs_err_mha"]))
            best_err = row["abs_err_mha"]
    if len(pf) >= 2:
        xs, ys = zip(*pf)
        ax.plot(xs, ys, color="0.4", linestyle="-", linewidth=1.5, alpha=0.6,
                marker="None", zorder=1, label="Pareto frontier")

    ax.axhline(1.594, color="0.45", linestyle="--", linewidth=1.0, alpha=0.6,
               label="chemical accuracy")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (s, log)")
    ax.set_ylabel("|E_method − E_FCI|   (mHa, log)")
    ax.set_title(f"{mol_slug}: accuracy vs compute cost")
    _legend_outside(ax)
    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.25)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot G — Per-geom method ranking bar
# ---------------------------------------------------------------------------

def plot_method_ranking_at_geom(df: pd.DataFrame, mol_slug: str, geom_tag: str,
                                *, save_to: Path | None = None) -> plt.Figure:
    """Question: at a specific geometry, rank methods by accuracy (and show n_dets / wall
    on the same bar for context).

    One horizontal bar per (method × partition), sorted by error_mha. Bar color = method.
    Hatch pattern = partition. Text annotation on each bar showing (n_dets, wall_s).
    """
    _apply_style()
    sub = _analysis_df(df[(df.molecule == mol_slug) & (df.geom_tag == geom_tag)].copy())
    sub = sub.dropna(subset=["error_mha"])
    if sub.empty:
        raise ValueError(f"no rows for {mol_slug} @ {geom_tag}")
    best = best_per_group(sub, by=("analysis_label", "analysis_method", "analysis_partition", "stage"))
    best["abs_err_mha"] = best["error_mha"].abs()
    best = best.sort_values("abs_err_mha")
    best["label"] = best.apply(
        lambda r: _method_label(r["analysis_method"], r["analysis_partition"], r["stage"]),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(best) + 1.6))
    colors = [METHOD_COLORS.get(m, "#444") for m in best["analysis_method"]]
    bars = ax.barh(range(len(best)), best["abs_err_mha"], color=colors,
                   edgecolor="white", linewidth=1.0, alpha=0.92)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels(best["label"], fontsize=10)
    ax.invert_yaxis()                                     # best-method at top
    ax.set_xscale("log")
    ax.set_xlabel("|E_method − E_FCI|   (mHa, log)")
    ax.set_title(f"{mol_slug} @ {geom_tag}: method ranking vs FCI")
    ax.axvline(1.594, color="0.45", linestyle="--", linewidth=1, alpha=0.7,
               label="chemical accuracy (1 kcal/mol)")
    # Annotate bars with n_dets and wall_s
    for i, (_, row) in enumerate(best.iterrows()):
        n_dets = row.get("n_dets_total"); wall = row.get("wall_s")
        anno_bits = []
        if pd.notna(n_dets) and n_dets != "" and float(n_dets) > 0:
            anno_bits.append(f"{int(float(n_dets))} dets")
        if pd.notna(wall) and wall != "":
            anno_bits.append(f"{float(wall):.1f}s")
        if anno_bits:
            ax.text(row["abs_err_mha"] * 1.18, i, "  ".join(anno_bits),
                    va="center", fontsize=8.5, color="0.3")
    _legend_outside(ax)
    ax.minorticks_on()
    ax.grid(True, which="both", axis="x", alpha=0.25)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Plot H — Single-point protocol sensitivity
# ---------------------------------------------------------------------------

def plot_single_point_protocol_sensitivity(
    df: pd.DataFrame,
    mol_slug: str,
    *,
    save_to: Path | None = None,
) -> plt.Figure:
    """Summarize the tuning axes for a single representative geometry.

    Each panel reports the best observed absolute FCI error at a fixed value of
    one axis while minimizing over the remaining protocol knobs. This keeps the
    full det/cycle/BFGS grid readable without placing dozens of overlapping
    curves in one axes.
    """
    _apply_style()
    sub = _analysis_df(df[df.molecule == mol_slug].copy())
    sub = sub.dropna(subset=["error_mha"])
    if sub.empty:
        raise ValueError(f"no error data for {mol_slug}")

    protocol = sub.get("parent_protocol", pd.Series("", index=sub.index)).fillna("")
    own_protocol = sub.get("protocol", pd.Series("", index=sub.index)).fillna("")
    sub["tuning_protocol"] = protocol.where(protocol.ne(""), own_protocol).astype(str)
    sub["dets_axis"] = pd.to_numeric(
        sub["tuning_protocol"].str.extract(r"dets(\d+)", expand=False), errors="coerce"
    )
    sub["cycles_axis"] = pd.to_numeric(
        sub["tuning_protocol"].str.extract(r"cyc(\d+)", expand=False), errors="coerce"
    )
    sub["bfgs_axis"] = pd.to_numeric(
        sub["tuning_protocol"].str.extract(r"bfgs(\d+)", expand=False), errors="coerce"
    )
    sub["abs_err_mha"] = sub["error_mha"].abs().clip(lower=1e-4)
    sub["series_label"] = sub.apply(
        lambda r: _method_label(
            str(r["analysis_method"]), str(r["analysis_partition"]), str(r["stage"])
        ),
        axis=1,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    panels = [
        ("dets_axis", "determinant limit", "Determinant-limit convergence"),
        ("cycles_axis", "COO cycles", "COO cycle sensitivity"),
        ("bfgs_axis", "BFGS iteration limit", "BFGS iteration sensitivity"),
    ]

    for ax, (axis_col, xlabel, title) in zip(axes, panels):
        panel = sub.dropna(subset=[axis_col])
        if panel.empty:
            ax.text(0.5, 0.5, "No matching protocol axis", ha="center", va="center",
                    transform=ax.transAxes, color="0.4")
            ax.set_axis_off()
            continue

        reduced = (
            panel.groupby(
                ["series_label", "analysis_method", "analysis_partition", "stage", axis_col],
                as_index=False,
            )["abs_err_mha"]
            .min()
        )
        for (_, method, partition, stage), grp in reduced.groupby(
            ["series_label", "analysis_method", "analysis_partition", "stage"]
        ):
            grp = grp.sort_values(axis_col)
            ax.plot(
                grp[axis_col],
                grp["abs_err_mha"],
                label=_method_label(str(method), str(partition), str(stage)),
                **_line_style_for(str(method), str(partition), str(stage)),
            )

        ax.axhline(1.594, color="0.45", linestyle="--", linewidth=1.0, alpha=0.65)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.minorticks_on()
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("best observed |E − E_FCI| (mHa, log)")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            frameon=True,
            fontsize=8.5,
            title="workflow / fragmentation",
        )
    fig.suptitle(f"{mol_slug}: single-point protocol sensitivity", y=1.02)

    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to)
    return fig


# ---------------------------------------------------------------------------
# Convenience — generate-and-save the full plot panel for one molecule
# ---------------------------------------------------------------------------

def generate_all_plots(results_root: str | Path, mol_slug: str) -> dict:
    """Generate every plot in this module for one molecule, save to disk, and return
    a {plot_name: path} map. Per-mol plots in ``<mol>/plots/``; per-geom rankings in
    ``<mol>/<geom>/plots/``.
    """
    results_root = Path(results_root)
    df = load_runs(results_root, mol_slug)
    if df.empty:
        return {"warning": f"no data for {mol_slug}"}

    plot_dir = _ensure_plot_dir(results_root, mol_slug)
    saved: dict[str, Path] = {}

    # Per-mol plots (only if multi-geom — single-point molecules skip PES-ish plots)
    has_scan = df["geom_tag"].str.startswith("r").any()
    if has_scan:
        for name, fn in [
            ("01_pes_with_reference",      plot_pes_with_reference),
            ("02_error_vs_r",               plot_error_vs_r),
            ("03_fragmentation_sensitivity", plot_fragmentation_sensitivity),
            ("04_ingredient_contributions", plot_ingredient_contributions),
            ("05_correlation_recovery",     plot_correlation_recovery),
        ]:
            try:
                p = plot_dir / f"{name}.png"
                fig = fn(df, mol_slug, save_to=p)
                plt.close(fig)
                saved[name] = p
            except Exception as e:
                saved[f"{name}_ERR"] = str(e)
    if not has_scan:
        try:
            p = plot_dir / "01_protocol_sensitivity.png"
            fig = plot_single_point_protocol_sensitivity(df, mol_slug, save_to=p)
            plt.close(fig)
            saved["01_protocol_sensitivity"] = p
        except Exception as e:
            saved["01_protocol_sensitivity_ERR"] = str(e)

    # Pareto works for both single-point and scan.
    try:
        p = plot_dir / ("02_pareto_cost_vs_error.png" if not has_scan
                        else "06_pareto_cost_vs_error.png")
        fig = plot_pareto_cost_vs_error(df, mol_slug, save_to=p)
        plt.close(fig)
        saved[p.stem] = p
    except Exception as e:
        saved["06_pareto_cost_vs_error_ERR"] = str(e)

    # Per-geom method ranking. For single-point data this lives directly in
    # <mol>/plots/ rather than a redundant <mol>/eq/plots/ directory.
    for geom_tag in sorted(df["geom_tag"].dropna().unique()):
        gd = _ensure_plot_dir(results_root, mol_slug, geom_tag)
        try:
            p = gd / ("03_method_ranking.png" if geom_tag == "eq" else "method_ranking.png")
            fig = plot_method_ranking_at_geom(df, mol_slug, geom_tag, save_to=p)
            plt.close(fig)
            saved[f"geom_{geom_tag}_ranking"] = p
        except Exception as e:
            saved[f"geom_{geom_tag}_ranking_ERR"] = str(e)

    return saved
