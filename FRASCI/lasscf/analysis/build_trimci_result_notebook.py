"""Build notebook, tables, and plots for the TrimCI cold/warm vs CAS-LASSCF run."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "FRASCI" / "Outputs" / "lasscf" / "trimci_2000dets_cold_warm_20260614_161229"
FULL_CAS_REFERENCE_HA = -327.1920


def _run_label(run_dir: str) -> str:
    if run_dir.startswith("csf_control"):
        return "CAS-LASSCF"
    if "trimci_cold" in run_dir:
        return "TrimCI cold"
    if "trimci_warm" in run_dir:
        return "TrimCI warm"
    return run_dir


def _run_order(label: str) -> int:
    return {"CAS-LASSCF": 0, "TrimCI cold": 1, "TrimCI warm": 2}.get(label, 99)


def _wall_seconds(data: dict) -> float | None:
    value = data.get("wall_time_total", data.get("wall_clock_sec", data.get("wall_time_sec")))
    return float(value) if value is not None else None


def collect_rows(run_root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(run_root.glob("*/*/result.json")):
        data = json.loads(result_path.read_text())
        partition = result_path.parts[-3]
        run_dir = result_path.parts[-2]
        label = _run_label(run_dir)
        orbital_lists = data.get("orbital_lists") or []
        frag_sizes = [len(x) for x in orbital_lists]
        wall_sec = _wall_seconds(data)
        rows.append(
            {
                "partition": partition,
                "run_dir": run_dir,
                "run_label": label,
                "sort_key": (partition, _run_order(label)),
                "result_path": str(result_path),
                "energy_ha": float(data["e_tot"]),
                "converged": bool(data.get("converged")),
                "status": data.get("status", ""),
                "wall_sec": wall_sec,
                "wall_hours": None if wall_sec is None else wall_sec / 3600.0,
                "n_fragments": len(orbital_lists),
                "fragment_sizes": frag_sizes,
                "fragment_sizes_text": "x".join(str(x) for x in frag_sizes),
                "ncas_sub": data.get("ncas_sub"),
                "nelec_per_frag": data.get("nelec_per_frag"),
                "spin_sub": data.get("spin_sub"),
                "dets_per_frag_final": data.get("dets_per_frag_final"),
                "trimci_threshold": data.get("trimci_threshold"),
                "n_macro_iters": data.get("n_macro_iters", data.get("n_macro_iters_estimate", data.get("n_macro_iters_heuristic"))),
            }
        )

    cas_by_partition = {
        row["partition"]: row["energy_ha"]
        for row in rows
        if row["run_label"] == "CAS-LASSCF"
    }
    for row in rows:
        row["error_vs_full_ref_mha"] = (row["energy_ha"] - FULL_CAS_REFERENCE_HA) * 1000.0
        cas_energy = cas_by_partition.get(row["partition"])
        row["gap_vs_partition_cas_mha"] = None
        if cas_energy is not None and row["run_label"] != "CAS-LASSCF":
            row["gap_vs_partition_cas_mha"] = (row["energy_ha"] - cas_energy) * 1000.0
    return sorted(rows, key=lambda x: x["sort_key"])


def write_csv(rows: list[dict], out_dir: Path) -> Path:
    csv_path = out_dir / "final_results_summary.csv"
    fields = [
        "partition",
        "run_label",
        "energy_ha",
        "error_vs_full_ref_mha",
        "gap_vs_partition_cas_mha",
        "converged",
        "status",
        "wall_hours",
        "n_fragments",
        "fragment_sizes_text",
        "dets_per_frag_final",
        "trimci_threshold",
        "n_macro_iters",
        "result_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return csv_path


def write_markdown(rows: list[dict], out_dir: Path) -> Path:
    md_path = out_dir / "final_results_summary.md"
    lines = [
        "# LASSCF TrimCI Cold/Warm vs CAS-LASSCF Results",
        "",
        f"Run root: `{DEFAULT_RUN_ROOT}`",
        f"Full CAS reference used for error bars: `{FULL_CAS_REFERENCE_HA:.6f} Ha`",
        "",
        "| Partition | Run | Energy Ha | Error vs full ref mHa | Gap vs partition CAS mHa | Conv | Runtime h | Dets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gap = row["gap_vs_partition_cas_mha"]
        gap_text = "" if gap is None else f"{gap:.2f}"
        dets = row["dets_per_frag_final"]
        lines.append(
            "| {partition} | {run_label} | {energy_ha:.9f} | {error:.2f} | {gap} | {conv} | {wall:.2f} | {dets} |".format(
                partition=row["partition"],
                run_label=row["run_label"],
                energy_ha=row["energy_ha"],
                error=row["error_vs_full_ref_mha"],
                gap=gap_text,
                conv="yes" if row["converged"] else "no",
                wall=row["wall_hours"] or 0.0,
                dets="" if dets is None else dets,
            )
        )

    lines += [
        "",
        "## Quick Reads",
        "",
        "- `index_block_18x18` is the cleanest result: CAS converged and TrimCI is only about 25 mHa above same-partition CAS.",
        "- `index_roundrobin_6x12x18` is not reliable here: CAS did not converge and TrimCI got trapped far above CAS.",
        "- `h1diag_rev_block_6x8x10x12` CAS converged quickly, but TrimCI remains hundreds of mHa above CAS even after warm start.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return md_path


def _save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, *, symlog: bool = False, ylim: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5.2))
    colors = ["#4C78A8" if "CAS" in label else "#F58518" if "cold" in label else "#54A24B" for label in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if symlog:
        ax.set_yscale("symlog", linthresh=10)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_summary_plots(rows: list[dict], out_dir: Path) -> list[Path]:
    paths = []
    labels = [f"{r['partition']}\n{r['run_label']}" for r in rows]

    path = out_dir / "01_final_energies_absolute.png"
    _save_bar(path, labels, [r["energy_ha"] for r in rows], "Final energies", "Energy (Ha)")
    paths.append(path)

    path = out_dir / "02_error_vs_full_reference_mha_symlog.png"
    _save_bar(path, labels, [r["error_vs_full_ref_mha"] for r in rows], "Error vs full CAS reference", "E - E_ref (mHa)", symlog=True)
    paths.append(path)

    trim_rows = [r for r in rows if r["gap_vs_partition_cas_mha"] is not None]
    trim_labels = [f"{r['partition']}\n{r['run_label'].replace('TrimCI ', '')}" for r in trim_rows]
    trim_gaps = [r["gap_vs_partition_cas_mha"] for r in trim_rows]
    path = out_dir / "03_gap_vs_partition_cas_mha_symlog.png"
    _save_bar(path, trim_labels, trim_gaps, "TrimCI gap vs same-partition CAS-LASSCF", "TrimCI - CAS (mHa)", symlog=True)
    paths.append(path)

    path = out_dir / "04_gap_vs_partition_cas_mha_zoom_0_350.png"
    _save_bar(path, trim_labels, trim_gaps, "TrimCI gap vs CAS, zoomed to useful range", "TrimCI - CAS (mHa)", ylim=(0, 350))
    paths.append(path)

    path = out_dir / "05_runtime_hours.png"
    _save_bar(path, labels, [r["wall_hours"] or 0.0 for r in rows], "Wall time by run", "Hours")
    paths.append(path)

    by_part = defaultdict(dict)
    for row in rows:
        by_part[row["partition"]][row["run_label"]] = row
    warm_gain_labels = []
    warm_gain_values = []
    for partition, part_rows in sorted(by_part.items()):
        cold = part_rows.get("TrimCI cold")
        warm = part_rows.get("TrimCI warm")
        if cold and warm:
            warm_gain_labels.append(partition)
            warm_gain_values.append((cold["energy_ha"] - warm["energy_ha"]) * 1000.0)
    path = out_dir / "06_warm_start_energy_gain_mha.png"
    _save_bar(path, warm_gain_labels, warm_gain_values, "Warm-start energy gain over cold TrimCI", "E_cold - E_warm (mHa)")
    paths.append(path)

    frag_labels = [r["partition"] for r in rows if r["run_label"] == "CAS-LASSCF"]
    frag_counts = [r["n_fragments"] for r in rows if r["run_label"] == "CAS-LASSCF"]
    cas_errors = [r["error_vs_full_ref_mha"] for r in rows if r["run_label"] == "CAS-LASSCF"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(frag_counts, cas_errors, s=90, color="#4C78A8")
    for label, xval, yval in zip(frag_labels, frag_counts, cas_errors):
        ax.annotate(label, (xval, yval), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_title("CAS-LASSCF error vs number of fragments")
    ax.set_xlabel("Number of fragments")
    ax.set_ylabel("CAS-LASSCF - full reference (mHa)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "07_cas_error_vs_fragment_count.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _load_kernel_calls(path: Path) -> list[dict]:
    kernel_path = path / "kernel_calls.json"
    if not kernel_path.exists():
        return []
    calls = json.loads(kernel_path.read_text())
    counts = defaultdict(int)
    for call in calls:
        frag = int(call["fragment_idx"])
        counts[frag] += 1
        call["call_idx_for_fragment"] = counts[frag]
    return calls


def make_kernel_plots(run_root: Path, out_dir: Path) -> list[Path]:
    paths = []
    run_dirs = sorted(path for path in run_root.glob("*/*") if (path / "kernel_calls.json").exists())
    for metric, ylabel, suffix in [
        ("energy_electronic", "Fragment CI electronic energy (Ha)", "fragment_ci_energy_traces"),
        ("n_dets", "Determinants", "fragment_det_traces"),
    ]:
        n = len(run_dirs)
        fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(13, max(7, math.ceil(n / 2) * 3.1)), squeeze=False)
        for ax, run_dir in zip(axes.ravel(), run_dirs):
            calls = _load_kernel_calls(run_dir)
            by_frag = defaultdict(list)
            for call in calls:
                by_frag[int(call["fragment_idx"])].append(call)
            for frag, frag_calls in sorted(by_frag.items()):
                xs = [x["call_idx_for_fragment"] for x in frag_calls]
                ys = [x[metric] for x in frag_calls]
                ax.plot(xs, ys, label=f"frag {frag}", linewidth=1.3)
            ax.set_title(f"{run_dir.parent.name}\n{_run_label(run_dir.name)}", fontsize=9)
            ax.set_xlabel("Kernel call index per fragment")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.22)
            ax.legend(fontsize=7, ncol=2)
        for ax in axes.ravel()[len(run_dirs):]:
            ax.axis("off")
        fig.tight_layout()
        path = out_dir / f"08_{suffix}.png" if metric == "energy_electronic" else out_dir / f"09_{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def notebook_json(run_root: Path, out_dir: Path, plot_paths: list[Path]) -> dict:
    rel_plot_paths = [p.relative_to(run_root) for p in plot_paths]
    code = f"""
from pathlib import Path
import csv, json, math
from collections import defaultdict
import matplotlib.pyplot as plt

RUN_ROOT = Path(r"{run_root}")
ANALYSIS_DIR = RUN_ROOT / "analysis" / "trimci_cas_cold_warm_final"
FULL_CAS_REFERENCE_HA = {FULL_CAS_REFERENCE_HA!r}
SUMMARY_CSV = ANALYSIS_DIR / "final_results_summary.csv"

def load_summary():
    rows = []
    with SUMMARY_CSV.open() as handle:
        for row in csv.DictReader(handle):
            for key in ["energy_ha", "error_vs_full_ref_mha", "gap_vs_partition_cas_mha", "wall_hours"]:
                row[key] = None if row[key] in ("", "None") else float(row[key])
            row["converged"] = row["converged"] == "True"
            rows.append(row)
    return rows

rows = load_summary()
rows
"""
    table_code = """
from IPython.display import display, Markdown

def fmt(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"

lines = [
    "| Partition | Run | Energy Ha | Error vs full ref mHa | Gap vs same CAS mHa | Conv | Runtime h | Dets |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['partition']} | {r['run_label']} | {float(r['energy_ha']):.9f} | "
        f"{fmt(r['error_vs_full_ref_mha'], 2)} | {fmt(r['gap_vs_partition_cas_mha'], 2)} | "
        f"{'yes' if r['converged'] else 'no'} | {fmt(r['wall_hours'], 2)} | {r['dets_per_frag_final']} |"
    )
display(Markdown("\\n".join(lines)))
"""
    insight_code = """
by_part = defaultdict(dict)
for r in rows:
    by_part[r["partition"]][r["run_label"]] = r

insights = []
for part, part_rows in sorted(by_part.items()):
    cas = part_rows.get("CAS-LASSCF")
    cold = part_rows.get("TrimCI cold")
    warm = part_rows.get("TrimCI warm")
    if cold and warm:
        gain = (float(cold["energy_ha"]) - float(warm["energy_ha"])) * 1000
        insights.append(f"- `{part}` warm-start gain: `{gain:.2f} mHa`.")
    if cas and warm and warm["gap_vs_partition_cas_mha"] is not None:
        insights.append(f"- `{part}` warm TrimCI gap vs same CAS: `{warm['gap_vs_partition_cas_mha']:.2f} mHa`.")
display(Markdown("\\n".join(insights)))
"""
    display_plots = "\n".join(
        [
            "from IPython.display import Image, display, Markdown",
            "for path in [",
            *[f"    ANALYSIS_DIR / {str(path.name)!r}," for path in rel_plot_paths],
            "]:",
            "    display(Markdown(f'### {path.name}'))",
            "    display(Image(filename=str(path)))",
        ]
    )
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": "# LASSCF TrimCI cold/warm vs CAS-LASSCF final analysis\n\nThis notebook summarizes the completed 2000-det, threshold-0.01 cold/warm TrimCI tests against same-partition CAS-LASSCF controls."},
        {"cell_type": "markdown", "metadata": {}, "source": f"Run root: `{run_root}`\n\nFull CAS reference used for broad error plots: `{FULL_CAS_REFERENCE_HA:.6f} Ha`."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.strip()},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": table_code.strip()},
        {"cell_type": "markdown", "metadata": {}, "source": "## Quick comparison\n\nThe important diagnostic is the TrimCI gap to the same partition's CAS-LASSCF energy. That removes the fragment-product ansatz error and mostly isolates intra-fragment solver/orbital behavior."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": insight_code.strip()},
        {"cell_type": "markdown", "metadata": {}, "source": "## Figures\n\nThese plots are also exported as PNG files in the analysis folder."},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": display_plots},
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    run_root = DEFAULT_RUN_ROOT
    out_dir = run_root / "analysis" / "trimci_cas_cold_warm_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(run_root)
    write_csv(rows, out_dir)
    write_markdown(rows, out_dir)
    plot_paths = []
    plot_paths.extend(make_summary_plots(rows, out_dir))
    plot_paths.extend(make_kernel_plots(run_root, out_dir))
    notebook = notebook_json(run_root, out_dir, plot_paths)
    notebook_path = out_dir / "trimci_cas_cold_warm_final_analysis.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    print(notebook_path)
    print(out_dir)


if __name__ == "__main__":
    main()
