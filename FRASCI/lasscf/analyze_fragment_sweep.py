from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REFERENCE_ENERGY = -327.1920


def _safe_float(value):
    if value is None:
        return math.nan
    return float(value)


def _run_name_parts(run_name: str) -> tuple[str, float]:
    candidate, thr = run_name.rsplit("__thr", 1)
    return candidate, float(thr)


def _load(root: Path):
    with open(root / "fragmentation_catalog.json", encoding="utf-8") as fp:
        catalog = {row["name"]: row for row in json.load(fp)}

    rows = []
    for result_path in sorted(root.glob("runs/*/result.json")):
        run_name = result_path.parent.name
        candidate, threshold = _run_name_parts(run_name)
        with open(result_path, encoding="utf-8") as fp:
            result = json.load(fp)
        meta = catalog.get(candidate, {})
        dets = [d for d in result.get("dets_per_frag_final", []) if isinstance(d, int)]
        row = {
            "run_name": run_name,
            "candidate": candidate,
            "threshold": threshold,
            "status": result.get("status"),
            "converged": bool(result.get("converged")),
            "e_tot": _safe_float(result.get("e_tot")),
            "error": _safe_float(result.get("e_tot")) - REFERENCE_ENERGY,
            "abs_error": abs(_safe_float(result.get("e_tot")) - REFERENCE_ENERGY),
            "total_dets": int(sum(dets)),
            "max_dets_frag": int(max(dets)) if dets else 0,
            "wall_time_total": _safe_float(result.get("wall_time_total")),
            "n_fragments": len(result.get("orbital_lists", [])),
            "max_fragment_size": max(result.get("ncas_sub", [0])),
            "min_fragment_size": min(result.get("ncas_sub", [0])),
            "size_pattern": "x".join(str(x) for x in result.get("ncas_sub", [])),
            "family": meta.get("family", "unknown"),
            "cut_strength": _safe_float(meta.get("cut_strength")),
            "mean_abs_spin_imbalance": _safe_float(meta.get("mean_abs_spin_imbalance")),
            "dets_per_frag_final": result.get("dets_per_frag_final", []),
            "nelec_per_frag": result.get("nelec_per_frag", []),
            "spin_sub": result.get("spin_sub", []),
            "output_dir": str(result_path.parent),
        }
        rows.append(row)

    failures = []
    for failure_path in sorted(root.glob("runs/*/failure.json")):
        run_name = failure_path.parent.name
        candidate, threshold = _run_name_parts(run_name)
        with open(failure_path, encoding="utf-8") as fp:
            failure = json.load(fp)
        meta = catalog.get(candidate, {})
        failures.append(
            {
                "run_name": run_name,
                "candidate": candidate,
                "threshold": threshold,
                "error_type": failure.get("error_type"),
                "error": failure.get("error"),
                "family": meta.get("family", "unknown"),
                "size_pattern": "x".join(str(x) for x in meta.get("size_pattern", [])),
                "max_fragment_size": meta.get("max_fragment_size"),
                "output_dir": str(failure_path.parent),
            }
        )
    return catalog, rows, failures


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )
    return plt


def _short_name(name: str) -> str:
    return name.replace("_roundrobin", "_rr").replace("coupling_strength", "cpl")


def _scatter_by_threshold(ax, rows, x_key, y_key, xlabel, ylabel, title):
    colors = {0.06: "#37659e", 0.01: "#b75d3c"}
    for thr in sorted({r["threshold"] for r in rows}):
        data = [r for r in rows if r["threshold"] == thr]
        ax.scatter(
            [r[x_key] for r in data],
            [r[y_key] for r in data],
            s=48,
            alpha=0.82,
            color=colors.get(thr, None),
            label=f"thr={thr:g}",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def _plot_rankings(plt, plot_dir: Path, rows: list[dict]) -> list[str]:
    written = []
    ranked = sorted(rows, key=lambda r: r["error"])
    labels = [_short_name(r["run_name"]) for r in ranked]
    y = [r["error"] for r in ranked]
    colors = ["#b75d3c" if r["threshold"] == 0.01 else "#37659e" for r in ranked]

    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    ax.bar(range(len(ranked)), y, color=colors)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("E - E_ref (Ha)")
    ax.set_title("Energy error ranking; lower is better")
    ax.set_xticks(range(len(ranked)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    path = plot_dir / "05_error_ranked_all_runs.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True, sharey=True)
    for ax, thr in zip(axes, [0.06, 0.01]):
        data = sorted([r for r in rows if r["threshold"] == thr], key=lambda r: r["error"])
        ax.bar(range(len(data)), [r["error"] for r in data], color="#37659e" if thr == 0.06 else "#b75d3c")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Energy error ranking, threshold={thr:g}")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([_short_name(r["candidate"]) for r in data], rotation=90, fontsize=6)
    axes[0].set_ylabel("E - E_ref (Ha)")
    path = plot_dir / "06_error_ranked_by_threshold.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))
    return written


def _plot_threshold_delta(plt, plot_dir: Path, rows: list[dict]) -> list[str]:
    by_candidate = defaultdict(dict)
    for row in rows:
        by_candidate[row["candidate"]][row["threshold"]] = row
    paired = []
    for candidate, by_thr in by_candidate.items():
        if 0.06 in by_thr and 0.01 in by_thr:
            paired.append(
                {
                    "candidate": candidate,
                    "delta_error_001_minus_006": by_thr[0.01]["error"] - by_thr[0.06]["error"],
                    "delta_abs_error": by_thr[0.01]["abs_error"] - by_thr[0.06]["abs_error"],
                    "delta_dets": by_thr[0.01]["total_dets"] - by_thr[0.06]["total_dets"],
                    "err_006": by_thr[0.06]["error"],
                    "err_001": by_thr[0.01]["error"],
                }
            )
    paired.sort(key=lambda r: r["delta_abs_error"])

    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    vals = [r["delta_abs_error"] for r in paired]
    colors = ["#4b8f6b" if v < 0 else "#b75d3c" for v in vals]
    ax.bar(range(len(paired)), vals, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("abs(error) change: thr 0.01 - thr 0.06 (Ha)")
    ax.set_title("Did tighter TrimCI improve the final LASSCF energy?")
    ax.set_xticks(range(len(paired)))
    ax.set_xticklabels([_short_name(r["candidate"]) for r in paired], rotation=90, fontsize=6)
    path = plot_dir / "07_threshold_delta_abs_error.png"
    fig.savefig(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.scatter([r["delta_dets"] for r in paired], [r["delta_abs_error"] for r in paired], s=52, color="#6750a4")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("extra final determinants at thr=0.01")
    ax.set_ylabel("abs(error) change (Ha)")
    ax.set_title("Tighter threshold: determinant cost vs error change")
    path2 = plot_dir / "08_threshold_delta_cost_vs_gain.png"
    fig.savefig(path2)
    plt.close(fig)
    return [str(path), str(path2)], paired


def _plot_structure_relations(plt, plot_dir: Path, rows: list[dict]) -> list[str]:
    written = []
    specs = [
        ("total_dets", "error", "final determinants", "E - E_ref (Ha)", "Cost vs signed error", "09_cost_vs_error.png"),
        ("total_dets", "abs_error", "final determinants", "|E - E_ref| (Ha)", "Cost vs absolute error", "10_cost_vs_abs_error.png"),
        ("max_fragment_size", "abs_error", "largest fragment size", "|E - E_ref| (Ha)", "Largest fragment vs error", "11_max_frag_size_vs_error.png"),
        ("n_fragments", "abs_error", "number of fragments", "|E - E_ref| (Ha)", "Fragment count vs error", "12_fragment_count_vs_error.png"),
        ("cut_strength", "abs_error", "integral graph cut strength", "|E - E_ref| (Ha)", "Inter-fragment coupling cut vs error", "13_cut_strength_vs_error.png"),
        ("mean_abs_spin_imbalance", "abs_error", "mean |n_alpha - n_beta|", "|E - E_ref| (Ha)", "Spin imbalance vs error", "14_spin_imbalance_vs_error.png"),
        ("wall_time_total", "abs_error", "wall time (s)", "|E - E_ref| (Ha)", "Wall time vs error", "15_walltime_vs_error.png"),
    ]
    for x_key, y_key, xlabel, ylabel, title, filename in specs:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        _scatter_by_threshold(ax, rows, x_key, y_key, xlabel, ylabel, title)
        path = plot_dir / filename
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    families = sorted({r["family"] for r in rows})
    data = [[r["abs_error"] for r in rows if r["family"] == fam] for fam in families]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.boxplot(data, labels=[fam.replace(" / ", "\n") for fam in families], showmeans=True)
    ax.set_ylabel("|E - E_ref| (Ha)")
    ax.set_title("Error distribution by fragmentation heuristic")
    ax.tick_params(axis="x", labelrotation=70, labelsize=7)
    path = plot_dir / "16_family_error_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    patterns = sorted({r["size_pattern"] for r in rows}, key=lambda s: (s.count("x"), s))
    data = [[r["abs_error"] for r in rows if r["size_pattern"] == pat] for pat in patterns]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.boxplot(data, labels=patterns, showmeans=True)
    ax.set_ylabel("|E - E_ref| (Ha)")
    ax.set_title("Error distribution by size pattern")
    ax.tick_params(axis="x", labelrotation=70, labelsize=7)
    path = plot_dir / "17_size_pattern_error_boxplot.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))
    return written


def _plot_heatmaps(plt, plot_dir: Path, rows: list[dict]) -> list[str]:
    candidates = sorted({r["candidate"] for r in rows})
    thresholds = sorted({r["threshold"] for r in rows})
    matrices = {
        "abs_error": np.full((len(candidates), len(thresholds)), np.nan),
        "total_dets": np.full((len(candidates), len(thresholds)), np.nan),
        "wall_time_total": np.full((len(candidates), len(thresholds)), np.nan),
    }
    lookup = {(r["candidate"], r["threshold"]): r for r in rows}
    for i, candidate in enumerate(candidates):
        for j, threshold in enumerate(thresholds):
            row = lookup.get((candidate, threshold))
            if row:
                for key in matrices:
                    matrices[key][i, j] = row[key]

    titles = {
        "abs_error": "|E - E_ref| heatmap",
        "total_dets": "Final determinant count heatmap",
        "wall_time_total": "Wall time heatmap",
    }
    files = {
        "abs_error": "18_heatmap_abs_error.png",
        "total_dets": "19_heatmap_total_dets.png",
        "wall_time_total": "20_heatmap_wall_time.png",
    }
    written = []
    for key, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(6, 11), constrained_layout=True)
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(candidates)))
        ax.set_yticklabels([_short_name(c) for c in candidates], fontsize=7)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{t:g}" for t in thresholds])
        ax.set_xlabel("TrimCI threshold")
        ax.set_title(titles[key])
        fig.colorbar(im, ax=ax)
        path = plot_dir / files[key]
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    return written


def _plot_fragments(plt, plot_dir: Path, rows: list[dict]) -> list[str]:
    written = []
    top = sorted(rows, key=lambda r: r["abs_error"])[:16]
    labels = [_short_name(r["run_name"]) for r in top]
    max_frag = max(len(r["dets_per_frag_final"]) for r in top)
    bottoms = np.zeros(len(top))
    colors = plt.cm.tab20(np.linspace(0, 1, max_frag))
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for frag_idx in range(max_frag):
        vals = []
        for row in top:
            dets = row["dets_per_frag_final"]
            vals.append(dets[frag_idx] if frag_idx < len(dets) and isinstance(dets[frag_idx], int) else 0)
        ax.bar(range(len(top)), vals, bottom=bottoms, color=colors[frag_idx], label=f"F{frag_idx}")
        bottoms += np.array(vals)
    ax.set_title("Fragment determinant load for the 16 best-energy runs")
    ax.set_ylabel("final determinants")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(ncol=5, fontsize=7)
    path = plot_dir / "21_top_runs_fragment_dets_stacked.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    all_rows = sorted(rows, key=lambda r: r["abs_error"])
    matrix = np.full((len(all_rows), max(len(r["dets_per_frag_final"]) for r in all_rows)), np.nan)
    for i, row in enumerate(all_rows):
        for j, value in enumerate(row["dets_per_frag_final"]):
            if isinstance(value, int):
                matrix[i, j] = value
    fig, ax = plt.subplots(figsize=(7, 12), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(all_rows)))
    ax.set_yticklabels([_short_name(r["run_name"]) for r in all_rows], fontsize=6)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels([f"F{i}" for i in range(matrix.shape[1])])
    ax.set_title("Per-fragment final determinant count")
    fig.colorbar(im, ax=ax, label="determinants")
    path = plot_dir / "22_fragment_dets_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))
    return written


def _load_kernel_calls(run_dir: Path) -> list[dict]:
    path = run_dir / "kernel_calls.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _plot_convergence(plt, root: Path, plot_dir: Path, rows: list[dict]) -> list[str]:
    written = []
    conv_dir = plot_dir / "convergence"
    conv_dir.mkdir(exist_ok=True)

    for row in sorted(rows, key=lambda r: r["abs_error"]):
        calls = _load_kernel_calls(Path(row["output_dir"]))
        if not calls:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
        for frag_idx in sorted({c["fragment_idx"] for c in calls}):
            frag_calls = [c for c in calls if c["fragment_idx"] == frag_idx]
            x = np.arange(1, len(frag_calls) + 1)
            axes[0].plot(x, [c["energy_electronic"] for c in frag_calls], marker=".", linewidth=1, label=f"F{frag_idx}")
            axes[1].plot(x, [c["n_dets"] for c in frag_calls], marker=".", linewidth=1, label=f"F{frag_idx}")
        axes[0].set_ylabel("fragment electronic E")
        axes[1].set_ylabel("selected dets")
        axes[1].set_xlabel("kernel call index for that fragment")
        axes[0].set_title(f"Fragment solver convergence: {_short_name(row['run_name'])}")
        axes[0].legend(ncol=4, fontsize=7)
        axes[1].legend(ncol=4, fontsize=7)
        path = conv_dir / f"{row['run_name']}_fragment_convergence.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for row in sorted(rows, key=lambda r: r["abs_error"])[:20]:
        calls = _load_kernel_calls(Path(row["output_dir"]))
        if not calls:
            continue
        by_round = defaultdict(list)
        for call in calls:
            by_round[call["fragment_idx"]].append(call)
        max_len = max(len(v) for v in by_round.values())
        final_by_step = []
        dets_by_step = []
        for step in range(max_len):
            energies = []
            dets = []
            for frag_calls in by_round.values():
                call = frag_calls[min(step, len(frag_calls) - 1)]
                energies.append(call["energy_electronic"])
                dets.append(call["n_dets"])
            final_by_step.append(sum(energies))
            dets_by_step.append(sum(dets))
        x = np.arange(1, len(final_by_step) + 1)
        axes[0].plot(x, np.array(final_by_step) - final_by_step[-1], alpha=0.65, linewidth=1)
        axes[1].plot(x, dets_by_step, alpha=0.65, linewidth=1)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Top-20 runs: fragment-energy sum settling")
    axes[0].set_xlabel("kernel call index")
    axes[0].set_ylabel("sum(fragment E) - final")
    axes[1].set_title("Top-20 runs: determinant growth")
    axes[1].set_xlabel("kernel call index")
    axes[1].set_ylabel("sum selected dets")
    path = plot_dir / "23_top20_convergence_spaghetti.png"
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    return written


def _plot_failures(plt, plot_dir: Path, failures: list[dict]) -> list[str]:
    if not failures:
        return []
    counts = Counter(f["size_pattern"] for f in failures)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    labels = list(counts)
    ax.bar(range(len(labels)), [counts[k] for k in labels], color="#8f3d52")
    ax.set_title("Failed runs by size pattern")
    ax.set_ylabel("failure count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    path = plot_dir / "24_failures_by_size_pattern.png"
    fig.savefig(path)
    plt.close(fig)
    return [str(path)]


def analyze(root: str | Path) -> dict:
    root = Path(root)
    analysis_dir = root / "analysis"
    plot_dir = analysis_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    catalog, rows, failures = _load(root)
    rows = sorted(rows, key=lambda r: r["abs_error"])
    fields = [
        "run_name",
        "candidate",
        "threshold",
        "status",
        "converged",
        "e_tot",
        "error",
        "abs_error",
        "total_dets",
        "max_dets_frag",
        "wall_time_total",
        "n_fragments",
        "size_pattern",
        "family",
        "cut_strength",
        "mean_abs_spin_imbalance",
        "output_dir",
    ]
    _write_csv(analysis_dir / "run_metrics.csv", rows, fields)
    _write_csv(
        analysis_dir / "failures.csv",
        failures,
        ["run_name", "candidate", "threshold", "error_type", "error", "family", "size_pattern", "max_fragment_size", "output_dir"],
    )

    plt = _setup_matplotlib()
    plots = []
    plots.extend(_plot_rankings(plt, plot_dir, rows))
    threshold_plots, paired = _plot_threshold_delta(plt, plot_dir, rows)
    plots.extend(threshold_plots)
    _write_csv(
        analysis_dir / "threshold_delta.csv",
        paired,
        ["candidate", "delta_error_001_minus_006", "delta_abs_error", "delta_dets", "err_006", "err_001"],
    )
    plots.extend(_plot_structure_relations(plt, plot_dir, rows))
    plots.extend(_plot_heatmaps(plt, plot_dir, rows))
    plots.extend(_plot_fragments(plt, plot_dir, rows))
    plots.extend(_plot_convergence(plt, root, plot_dir, rows))
    plots.extend(_plot_failures(plt, plot_dir, failures))

    best = rows[0] if rows else None
    best_by_threshold = {}
    for threshold in sorted({r["threshold"] for r in rows}):
        data = sorted([r for r in rows if r["threshold"] == threshold], key=lambda r: r["abs_error"])
        best_by_threshold[str(threshold)] = data[0] if data else None

    report_lines = [
        "# LASSCF Fragment Sweep Analysis",
        "",
        f"Reference energy: `{REFERENCE_ENERGY:.6f} Ha`",
        f"Completed runs: `{len(rows)}`",
        f"Failures: `{len(failures)}`",
        f"Formally converged LASSCF runs: `{sum(1 for r in rows if r['converged'])}`",
        "",
        "## Best Runs",
        "",
    ]
    for i, row in enumerate(rows[:12], start=1):
        report_lines.append(
            f"{i}. `{row['run_name']}`: E={row['e_tot']:.9f} Ha, "
            f"error={row['error']:+.6f} Ha, dets={row['total_dets']}, "
            f"sizes={row['size_pattern']}, family={row['family']}"
        )
    report_lines.extend(["", "## Best By Threshold", ""])
    for threshold, row in best_by_threshold.items():
        if row:
            report_lines.append(
                f"- threshold `{float(threshold):g}`: `{row['run_name']}`, "
                f"error={row['error']:+.6f} Ha, dets={row['total_dets']}"
            )
    report_lines.extend(["", "## Failed Runs", ""])
    for failure in failures:
        report_lines.append(
            f"- `{failure['run_name']}` ({failure['error_type']}): {failure['error']}"
        )
    report_lines.extend(["", "## Plot Files", ""])
    for plot in plots:
        report_lines.append(f"- `{Path(plot).relative_to(root)}`")
    (analysis_dir / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "root": str(root),
        "analysis_dir": str(analysis_dir),
        "reference_energy": REFERENCE_ENERGY,
        "n_completed": len(rows),
        "n_failures": len(failures),
        "n_converged": sum(1 for r in rows if r["converged"]),
        "best_run": best,
        "best_by_threshold": best_by_threshold,
        "plots": plots,
    }
    with open(analysis_dir / "analysis_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a LASSCF fragment sweep output directory.")
    parser.add_argument("root", help="Path to fragment_sweep_* directory")
    args = parser.parse_args()
    manifest = analyze(args.root)
    print(json.dumps({k: manifest[k] for k in ("analysis_dir", "n_completed", "n_failures", "n_converged")}, indent=2))
