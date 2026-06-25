"""Report helpers: data loaders, plot helpers, J coupling computation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HA_PER_CM = 4.5563352812e-6
CM_PER_HA = 1.0 / HA_PER_CM


def enrich_lasscf_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add analysis columns that treat LASSIS as a post-LASSCF stage.

    Raw rows keep their original ``method`` for backward compatibility. The added
    columns are what reports should use when the question is about the LASSCF family:

    * ``stage``: ``base`` | ``lasscf`` | ``lassis``
    * ``parent_method``: parent LASSCF method for LASSIS rows
    * ``base_partition``: partition before the ``_on_<parent>`` suffix
    * ``analysis_method``: parent method for LASSIS, otherwise raw method
    * ``analysis_label``: e.g. ``lasscf_trimci_coo + LASSIS``
    """
    if df.empty:
        return df
    out = df.copy()
    for col in ("stage", "parent_method", "base_partition"):
        if col not in out.columns:
            out[col] = ""
        else:
            out[col] = out[col].astype(object).where(out[col].notna(), "")

    lassis = out["method"].eq("lassis")
    out.loc[out["stage"].isna() | out["stage"].eq(""), "stage"] = "base"
    is_lasscf = out["method"].astype(str).str.startswith("lasscf")
    out.loc[is_lasscf, "stage"] = "lasscf"
    out.loc[lassis, "stage"] = "lassis"

    if lassis.any():
        idx = out.index[lassis]
        parsed = out.loc[lassis, "partition"].astype(str).str.extract(
            r"^(?P<base_partition>.+)_on_(?P<parent_method>lasscf(?:_.+)?)$"
        )
        for col in ("parent_method", "base_partition"):
            missing = out.loc[idx, col].isna() | out.loc[idx, col].eq("")
            fill_idx = idx[missing.to_numpy()]
            out.loc[fill_idx, col] = parsed.loc[fill_idx, col]

    missing_base = out["base_partition"].isna() | out["base_partition"].eq("")
    out.loc[missing_base, "base_partition"] = out.loc[missing_base, "partition"]

    out["analysis_method"] = out["method"]
    out.loc[lassis & out["parent_method"].notna() & out["parent_method"].ne(""),
            "analysis_method"] = out.loc[lassis, "parent_method"]
    out["analysis_partition"] = out["base_partition"]
    out["analysis_label"] = out["analysis_method"].astype(str)
    out.loc[lassis, "analysis_label"] = out.loc[lassis, "analysis_method"].astype(str) + " + LASSIS"
    return out


def load_runs(results_root: Path, mol_slug: str | None = None) -> pd.DataFrame:
    """Load runs_index.csv for one molecule (or concat for all if mol_slug is None)."""
    results_root = Path(results_root)
    if mol_slug is not None:
        csv_path = results_root / mol_slug / "runs_index.csv"
        if not csv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        df = enrich_lasscf_stage_columns(df)
        for col in ("e_tot", "e_ref", "error_mha", "wall_s",
                    "n_dets_total", "trimci_threshold", "trimci_max_dets",
                    "trimci_max_rounds", "coo_cycles",
                    "coo_bfgs_maxiter", "coo_bfgs_ftol", "coo_davidson_tol",
                    "parallel_workers", "process_workers", "omp_threads_per_frag",
                    "max_cycle_macro", "n_spin"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    frames = []
    for mol_dir in results_root.iterdir():
        if not mol_dir.is_dir():
            continue
        frames.append(load_runs(results_root, mol_dir.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_per_group(df: pd.DataFrame, by=("method", "partition", "geom_tag")) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values("e_tot").groupby(list(by), as_index=False).first()


def plot_pes(df: pd.DataFrame, mol_slug: str, methods: list[str] | None = None,
             ref_curve: dict | None = None, ax=None):
    sub = enrich_lasscf_stage_columns(df[df["molecule"] == mol_slug].copy())
    if "geom_tag" not in sub.columns or sub.empty:
        raise ValueError(f"no rows for {mol_slug}")
    # Extract numeric r from geom_tag like 'r1.25'
    sub["r"] = sub["geom_tag"].str.extract(r"r([\d.]+)").astype(float)
    sub = sub.dropna(subset=["r"]).sort_values("r")
    if methods:
        sub = sub[sub["analysis_method"].isin(methods) | sub["method"].isin(methods)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for method, grp in sub.groupby("analysis_label"):
        grp = grp.sort_values("r")
        ax.plot(grp["r"], grp["e_tot"], "o-", label=method)
    if ref_curve:
        ax.plot(ref_curve["r"], ref_curve["e"], "k--", label="reference", alpha=0.6)
    ax.set_xlabel("r (Å)"); ax.set_ylabel("E (Ha)")
    ax.set_title(f"PES — {mol_slug}"); ax.legend()
    return ax


def plot_method_bar(df: pd.DataFrame, mol_slug: str, geom_tag: str, ax=None):
    sub = enrich_lasscf_stage_columns(
        df[(df["molecule"] == mol_slug) & (df["geom_tag"] == geom_tag)].copy()
    )
    if sub.empty:
        raise ValueError(f"no rows for {mol_slug} / {geom_tag}")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    grouped = sub.groupby(["analysis_label", "analysis_partition"], as_index=False)["error_mha"].min()
    labels = grouped["analysis_label"] + "/" + grouped["analysis_partition"]
    ax.bar(labels, grouped["error_mha"])
    ax.set_ylabel("error (mHa)"); ax.set_title(f"{mol_slug} @ {geom_tag}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return ax


def plot_macro_trajectory(run_dir: Path, ax=None):
    """Parse log.txt for 'macro iter X: e=... |g_int|=...' lines and plot."""
    import re
    text = (Path(run_dir) / "log.txt").read_text()
    pat = re.compile(r"macro\s+(\d+).*?E\s*=\s*(-?\d+\.\d+).*?\|g[_ ]int\|\s*=\s*(\S+)", re.IGNORECASE)
    rows = [(int(m.group(1)), float(m.group(2)), float(m.group(3)))
            for m in pat.finditer(text)]
    if not rows:
        raise ValueError(f"no macro-iter lines parsed in {run_dir}/log.txt")
    iters, es, gs = zip(*rows)
    if ax is None:
        _, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax.plot(iters, es, "o-"); ax.set_ylabel("E (Ha)")
    if ax.figure.axes[-1] is not ax:
        ax2 = ax.figure.axes[-1]
        ax2.semilogy(iters, gs, "o-"); ax2.set_ylabel("|g_int|"); ax2.set_xlabel("macro iter")
    return ax


def compute_j_yamaguchi(hs_row: dict, bs_row: dict) -> dict:
    """Yamaguchi J coupling: J = (E_HS - E_BS) / (<S^2>_HS - <S^2>_BS)."""
    e_hs = float(hs_row["e_tot"]); e_bs = float(bs_row["e_tot"])
    s2_hs = float(hs_row["s2_expectation"]); s2_bs = float(bs_row["s2_expectation"])
    denom = s2_hs - s2_bs
    if abs(denom) < 1e-10:
        raise ValueError("Yamaguchi denominator <S²>_HS - <S²>_BS is zero")
    j_ha = (e_hs - e_bs) / denom
    return {
        "E_HS": e_hs, "E_BS": e_bs,
        "S2_HS": s2_hs, "S2_BS": s2_bs,
        "J_ha": j_ha, "J_cm-1": j_ha * CM_PER_HA,
    }
