"""Build the H1 four-fragment LASSCF -> TrimCI -> LASSI/LASSIS notebook."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "FRASCIMain.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


cells = [
    markdown(
        """
# H1 four-fragment LASSCF -> TrimCI -> LASSI/LASSIS pipeline

This notebook runs one fixed four-fragment partition:

`h1diag_rev_block_6x8x10x12`

Pipeline:

1. CAS-LASSCF / CSF control
2. LASSCF + TrimCI cold start
3. LASSCF + TrimCI warm start
4. LASSI/LASSIS on top of the CAS checkpoint
5. LASSI/LASSIS on top of the cold TrimCI checkpoint
6. LASSI/LASSIS on top of the warm TrimCI checkpoint

The main knobs are collected in the next cell. The runner is resumable: if a result file already exists and `FORCE_RERUN = False`, the notebook loads it instead of rerunning.
        """
    ),
    code(
        r"""
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import time
from collections import defaultdict

import matplotlib.pyplot as plt
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name != "FRASCI":
    PROJECT_ROOT = next(
        (path / "FRASCI" for path in [PROJECT_ROOT, *PROJECT_ROOT.parents]
         if (path / "FRASCI").is_dir()),
        None,
    )
if PROJECT_ROOT is None:
    raise RuntimeError("Open this notebook from the FRASCI repository.")

PLAYGROUND = PROJECT_ROOT
PYTHON = PROJECT_ROOT.parent / "FRASCIenv" / "bin" / "python"
FCIDUMP = PROJECT_ROOT / "data" / "fcidump_cycle_6"
FRAGMENT_JSON = PROJECT_ROOT / "data" / "partitions" / "h1diag_rev_block_6x8x10x12.json"

PARTITION_NAME = "h1diag_rev_block_6x8x10x12"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = PLAYGROUND / "Outputs" / "lasscf" / f"h1_4frag_pipeline_{RUN_TAG}"

# Reuse an older run by uncommenting and pointing here:
# RUN_ROOT = PLAYGROUND / "Outputs" / "lasscf" / "h1_4frag_pipeline_YYYYMMDD_HHMMSS"

FORCE_RERUN = False

# LASSCF controls
CAS_MAX_CYCLE = 100
TRIMCI_MAX_CYCLE_MACRO = 100

# TrimCI controls
TRIMCI_THRESHOLD = 0.01
TRIMCI_MAX_DETS = 2000
TRIMCI_MAX_ROUNDS = 4

# Warm-start source: "cas" is the clean diagnostic; "cold" matches the previous cold -> warm batch.
WARM_START_FROM = "cas"

# LASSI/LASSIS controls
RUN_LASSI = True
RUN_LASSIS = True
LASSI_OPT = 1
LASSIS_NCHARGE = "s"
LASSIS_NSPIN = 0

# Optional run switches
RUN_CAS_CSF = True
RUN_TRIMCI_COLD = True
RUN_TRIMCI_WARM = True
RUN_INTERFRAG = True

RUN_ROOT.mkdir(parents=True, exist_ok=True)
print("RUN_ROOT:", RUN_ROOT)
print("FRAGMENT_JSON:", FRAGMENT_JSON)
print("PYTHON:", PYTHON)
        """
    ),
    markdown("## Helpers"),
    code(
        r"""
def run_cmd(cmd, out_dir, done_file, allow_nonzero=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done_file = Path(done_file)
    log_path = out_dir / "run.log"
    err_path = out_dir / "run.err"

    if done_file.exists() and not FORCE_RERUN:
        print(f"[load] {done_file}")
        return 0

    print("[run]", " ".join(map(str, cmd)))
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w") as out, err_path.open("w") as err:
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=out,
            stderr=err,
            text=True,
        )
    dt = time.time() - t0
    print(f"[done] rc={proc.returncode} wall={dt/3600:.2f} h")
    print(" log:", log_path)
    print(" err:", err_path)
    if proc.returncode != 0 and not allow_nonzero:
        raise RuntimeError(f"Command failed with return code {proc.returncode}. See {err_path}")
    if not done_file.exists():
        print(f"[warn] expected result missing: {done_file}")
    return proc.returncode


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def energy_from_result(result):
    if not result:
        return None
    for key in ("e_tot", "e_lassis", "e_lassi", "e_lasci", "e_lasscf"):
        value = result.get(key)
        if value is not None:
            return float(value)
    return None


def show_result(label, result):
    if result is None:
        display(Markdown(f"**{label}**: missing"))
        return
    lines = [f"### {label}"]
    for key in [
        "status",
        "converged",
        "e_tot",
        "e_lasscf",
        "e_lasci",
        "e_lassi",
        "delta_e_lassi",
        "e_lassis",
        "delta_e_lassis",
        "delta_e_lassi_vs_lassis",
        "wall_clock_sec",
        "wall_time_total",
        "dets_per_frag_final",
    ]:
        if key in result:
            lines.append(f"- `{key}`: `{result[key]}`")
    display(Markdown("\n".join(lines)))
        """
    ),
    markdown("## Output directories"),
    code(
        r"""
cas_dir = RUN_ROOT / "cas_csf_control"
cold_dir = RUN_ROOT / f"trimci_cold_thr{TRIMCI_THRESHOLD:g}_dets{TRIMCI_MAX_DETS}_rounds{TRIMCI_MAX_ROUNDS}"
warm_dir = RUN_ROOT / f"trimci_warm_from_{WARM_START_FROM}_thr{TRIMCI_THRESHOLD:g}_dets{TRIMCI_MAX_DETS}_rounds{TRIMCI_MAX_ROUNDS}"
interfrag_dirs = {
    "cas": RUN_ROOT / f"lassi_lassis_from_cas_ncharge{LASSIS_NCHARGE}_nspin{LASSIS_NSPIN}",
    "cold": RUN_ROOT / f"lassi_lassis_from_cold_ncharge{LASSIS_NCHARGE}_nspin{LASSIS_NSPIN}",
    "warm": RUN_ROOT / f"lassi_lassis_from_warm_ncharge{LASSIS_NCHARGE}_nspin{LASSIS_NSPIN}",
}

for path in [cas_dir, cold_dir, warm_dir, *interfrag_dirs.values()]:
    print(path)
        """
    ),
    markdown("## 1. CAS-LASSCF / CSF control"),
    code(
        r"""
if RUN_CAS_CSF:
    cmd = [
        PYTHON, "-m", "FRASCI.lasscf.runners.run_lasscf_csf",
        "--fcidump", FCIDUMP,
        "--partition", PARTITION_NAME,
        "--fragment-orbs-json", FRAGMENT_JSON,
        "--max-cycle", CAS_MAX_CYCLE,
        "--output-dir", cas_dir,
    ]
    run_cmd(cmd, cas_dir, cas_dir / "result.json", allow_nonzero=True)

cas_result = load_json(cas_dir / "result.json")
show_result("CAS-LASSCF / CSF control", cas_result)
        """
    ),
    markdown("## 2. LASSCF + TrimCI cold start"),
    code(
        r"""
if RUN_TRIMCI_COLD:
    cmd = [
        PYTHON, "-m", "FRASCI.lasscf.runners.run_lasscf_trimci",
        "--fcidump", FCIDUMP,
        "--partition", PARTITION_NAME,
        "--fragment-orbs-json", FRAGMENT_JSON,
        "--trimci-threshold", TRIMCI_THRESHOLD,
        "--trimci-max-dets", TRIMCI_MAX_DETS,
        "--trimci-max-rounds", TRIMCI_MAX_ROUNDS,
        "--max-cycle-macro", TRIMCI_MAX_CYCLE_MACRO,
        "--output-dir", cold_dir,
    ]
    run_cmd(cmd, cold_dir, cold_dir / "result.json", allow_nonzero=True)

cold_result = load_json(cold_dir / "result.json")
show_result("LASSCF + TrimCI cold", cold_result)
        """
    ),
    markdown("## 3. LASSCF + TrimCI warm start"),
    code(
        r"""
if WARM_START_FROM == "cas":
    warm_source = cas_dir
elif WARM_START_FROM == "cold":
    warm_source = cold_dir
else:
    raise ValueError("WARM_START_FROM must be 'cas' or 'cold'")

if RUN_TRIMCI_WARM:
    if not (warm_source / "checkpoint.npz").exists():
        raise FileNotFoundError(f"Missing warm-start checkpoint: {warm_source / 'checkpoint.npz'}")
    cmd = [
        PYTHON, "-m", "FRASCI.lasscf.runners.run_lasscf_trimci",
        "--fcidump", FCIDUMP,
        "--partition", PARTITION_NAME,
        "--fragment-orbs-json", FRAGMENT_JSON,
        "--init-from", warm_source,
        "--trimci-threshold", TRIMCI_THRESHOLD,
        "--trimci-max-dets", TRIMCI_MAX_DETS,
        "--trimci-max-rounds", TRIMCI_MAX_ROUNDS,
        "--max-cycle-macro", TRIMCI_MAX_CYCLE_MACRO,
        "--output-dir", warm_dir,
    ]
    run_cmd(cmd, warm_dir, warm_dir / "result.json", allow_nonzero=True)

warm_result = load_json(warm_dir / "result.json")
show_result(f"LASSCF + TrimCI warm from {WARM_START_FROM}", warm_result)
        """
    ),
    markdown("## 4. LASSI and LASSIS from CAS, cold, and warm checkpoints"),
    code(
        r"""
checkpoint_sources = {
    "cas": cas_dir,
    "cold": cold_dir,
    "warm": warm_dir,
}

interfrag_results = {}

if RUN_INTERFRAG:
    for source_label, checkpoint_dir in checkpoint_sources.items():
        out_dir = interfrag_dirs[source_label]
        if not (checkpoint_dir / "checkpoint.npz").exists():
            print(f"[skip] missing checkpoint for {source_label}: {checkpoint_dir / 'checkpoint.npz'}")
            continue

        cmd = [
            PYTHON, "-m", "FRASCI.lasscf.runners.run_lassi_lassis",
            "--lasscf-checkpoint-dir", checkpoint_dir,
            "--fcidump", FCIDUMP,
            "--output-dir", out_dir,
            "--opt", LASSI_OPT,
            "--lassis-ncharge", LASSIS_NCHARGE,
            "--lassis-nspin", LASSIS_NSPIN,
        ]
        if not RUN_LASSI:
            cmd.append("--skip-lassi")
        if not RUN_LASSIS:
            cmd.append("--skip-lassis")

        print(f"\n=== LASSI/LASSIS from {source_label} checkpoint ===")
        run_cmd(cmd, out_dir, out_dir / "summary.json", allow_nonzero=True)

for source_label, out_dir in interfrag_dirs.items():
    summary = load_json(out_dir / "summary.json")
    interfrag_results[source_label] = {
        "summary": summary,
        "lassi": load_json(out_dir / "lassi_result.json"),
        "lassis": load_json(out_dir / "lassis_result.json"),
        "out_dir": out_dir,
    }
    show_result(f"LASSI/LASSIS from {source_label}", summary)
        """
    ),
    markdown("## 5. Final comparison table"),
    code(
        r"""
rows = []
for label, result, path in [
    ("CAS-LASSCF", cas_result, cas_dir / "result.json"),
    ("TrimCI cold", cold_result, cold_dir / "result.json"),
    (f"TrimCI warm from {WARM_START_FROM}", warm_result, warm_dir / "result.json"),
]:
    if result:
        rows.append({
            "method": label,
            "energy": energy_from_result(result),
            "converged": result.get("converged"),
            "status": result.get("status"),
            "runtime_h": (result.get("wall_time_total") or result.get("wall_clock_sec") or 0) / 3600,
            "dets": result.get("dets_per_frag_final"),
            "path": str(path),
        })

for source_label, bundle in interfrag_results.items():
    summary = bundle.get("summary")
    if summary:
        for label, key in [("LASCI fixed-MO", "e_lasci"), ("LASSI", "e_lassi"), ("LASSIS", "e_lassis")]:
            if summary.get(key) is not None:
                rows.append({
                    "method": f"{label} from {source_label}",
                    "energy": float(summary[key]),
                    "converged": None,
                    "status": "OK",
                    "runtime_h": None,
                    "dets": None,
                    "path": str(bundle["out_dir"] / "summary.json"),
                })

cas_energy = next((r["energy"] for r in rows if r["method"] == "CAS-LASSCF"), None)
warm_energy = next((r["energy"] for r in rows if r["method"].startswith("TrimCI warm")), None)
for row in rows:
    row["gap_vs_cas_mha"] = None if cas_energy is None else (row["energy"] - cas_energy) * 1000
    row["delta_vs_warm_mha"] = None if warm_energy is None else (row["energy"] - warm_energy) * 1000

lines = [
    "| Method | Energy Ha | Gap vs CAS mHa | Delta vs warm mHa | Conv | Status | Runtime h | Dets |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    gap = "" if row["gap_vs_cas_mha"] is None else f"{row['gap_vs_cas_mha']:.2f}"
    dw = "" if row["delta_vs_warm_mha"] is None else f"{row['delta_vs_warm_mha']:.2f}"
    rt = "" if row["runtime_h"] is None else f"{row['runtime_h']:.2f}"
    lines.append(
        f"| {row['method']} | {row['energy']:.9f} | {gap} | {dw} | "
        f"{row['converged']} | {row['status']} | {rt} | {row['dets']} |"
    )
display(Markdown("\n".join(lines)))
        """
    ),
    markdown("## 6. Energy and inter-fragment correction plots"),
    code(
        r"""
if rows:
    labels = [r["method"] for r in rows]
    energies = [r["energy"] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(range(len(rows)), energies)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("Pipeline energies")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    gaps = [r["gap_vs_cas_mha"] for r in rows]
    ax.bar(range(len(rows)), gaps)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Energy - CAS-LASSCF (mHa)")
    ax.set_title("Gaps relative to same-fragment CAS-LASSCF")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.show()

names = []
lassi_vals = []
lassis_vals = []
for source_label, bundle in interfrag_results.items():
    summary = bundle.get("summary")
    if not summary:
        continue
    if summary.get("delta_e_lassi") is not None or summary.get("delta_e_lassis") is not None:
        names.append(source_label)
        lassi_vals.append(None if summary.get("delta_e_lassi") is None else summary["delta_e_lassi"] * 1000)
        lassis_vals.append(None if summary.get("delta_e_lassis") is None else summary["delta_e_lassis"] * 1000)

if names:
    x = range(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar([i - width/2 for i in x], [0 if v is None else v for v in lassi_vals], width=width, label="LASSI")
    ax.bar([i + width/2 for i in x], [0 if v is None else v for v in lassis_vals], width=width, label="LASSIS")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Energy lowering from LASCI (mHa)")
    ax.set_title("Inter-fragment correlation recovered by checkpoint source")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()
        """
    ),
    markdown("## 7. TrimCI fragment traces"),
    code(
        r"""
def load_kernel_calls(run_dir):
    path = Path(run_dir) / "kernel_calls.json"
    if not path.exists():
        return []
    calls = load_json(path)
    counts = defaultdict(int)
    for call in calls:
        frag = int(call["fragment_idx"])
        counts[frag] += 1
        call["idx"] = counts[frag]
    return calls


def plot_kernel_metric(run_dir, title, metric, ylabel):
    calls = load_kernel_calls(run_dir)
    if not calls:
        print("No kernel_calls.json for", run_dir)
        return
    by_frag = defaultdict(list)
    for call in calls:
        by_frag[int(call["fragment_idx"])].append(call)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for frag, frag_calls in sorted(by_frag.items()):
        ax.plot([c["idx"] for c in frag_calls], [c[metric] for c in frag_calls], label=f"frag {frag}")
    ax.set_title(title)
    ax.set_xlabel("Kernel call index per fragment")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    plt.show()


plot_kernel_metric(cold_dir, "Cold TrimCI fragment energies", "energy_electronic", "Fragment electronic energy (Ha)")
plot_kernel_metric(cold_dir, "Cold TrimCI determinant counts", "n_dets", "Determinants")
plot_kernel_metric(warm_dir, "Warm TrimCI fragment energies", "energy_electronic", "Fragment electronic energy (Ha)")
plot_kernel_metric(warm_dir, "Warm TrimCI determinant counts", "n_dets", "Determinants")
        """
    ),
    markdown("## Notes\n\n- Warm-start defaults to CAS because this isolates TrimCI solver/orbital behavior from a better orbital starting point.\n- LASSI/LASSIS are run separately from CAS, cold TrimCI, and warm TrimCI checkpoints.\n- LASSI uses neutral plus all single charge-transfer rootspaces from the runner.\n- LASSIS defaults to `ncharge='s'`, `nspin=0`, meaning automatic charge-hop singles only for the first pass."),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
