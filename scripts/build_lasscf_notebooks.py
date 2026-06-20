"""Build the portable FRASCI LASSCF pipeline and results notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(path: Path, cells) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "FRASCIenv",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    nbf.write(notebook, path)


main_cells = [
    md(
        """
# FRASCI Main: four-fragment LASSCF → TrimCI → LASSI/LASSIS

This is the reproducible H1-diagonal four-fragment workflow for
`h1diag_rev_block_6x8x10x12`.

It contains:

1. CAS-LASSCF/CSF control
2. TrimCI-LASSCF cold start
3. TrimCI-LASSCF warm start from the CAS orbitals
4. LASSI and LASSIS from CAS, cold, and warm checkpoints
5. LASSIS charge/spin expansions for `nspin = 0, 1, 2`

The notebook defaults to **reuse mode**, loading the checked-in June 16, 2026
results without launching the multi-hour calculations. Set `RUN_EXPENSIVE = True`
only when a fresh calculation is intentional.
        """
    ),
    code(
        """
from pathlib import Path
from datetime import datetime
import json
import os
import subprocess
import time

from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name != "FRASCI":
    candidate = next((p for p in [ROOT, *ROOT.parents] if (p / "FRASCI").is_dir()), None)
    if candidate is None:
        raise RuntimeError("Open this notebook from the FRASCI repository.")
    ROOT = candidate / "FRASCI"

PYTHON = ROOT.parent / "FRASCIenv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(os.sys.executable)

FCIDUMP = ROOT / "data" / "fcidump_cycle_6"
FRAGMENT_JSON = ROOT / "data" / "partitions" / "h1diag_rev_block_6x8x10x12.json"
PARTITION = "h1diag_rev_block_6x8x10x12"

EXISTING_RUN = ROOT / "Outputs" / "lasscf" / "h1_4frag_pipeline_20260616_122809"
RUN_EXPENSIVE = False
FORCE_RERUN = False
RUN_ROOT = (
    ROOT / "Outputs" / "lasscf" / f"h1_4frag_pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    if RUN_EXPENSIVE
    else EXISTING_RUN
)

CAS_MAX_CYCLE = 100
TRIMCI_MAX_CYCLE_MACRO = 100
TRIMCI_THRESHOLD = 0.01
TRIMCI_MAX_DETS = 2000
TRIMCI_MAX_ROUNDS = 4
LASSI_OPT = 1
LASSIS_NCHARGE = "s"
LASSIS_NSPINS = (0, 1, 2)

assert FCIDUMP.exists(), FCIDUMP
assert FRAGMENT_JSON.exists(), FRAGMENT_JSON
if not RUN_EXPENSIVE:
    assert RUN_ROOT.exists(), RUN_ROOT

print("Mode:", "RUN fresh calculations" if RUN_EXPENSIVE else "REUSE existing outputs")
print("Run root:", RUN_ROOT)
print("Python:", PYTHON)
        """
    ),
    md("## Helpers"),
    code(
        """
def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def run_cmd(args, out_dir, done_file):
    out_dir = Path(out_dir)
    done_file = Path(done_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    if done_file.exists() and not FORCE_RERUN:
        print("[reuse]", done_file)
        return 0
    if not RUN_EXPENSIVE:
        raise RuntimeError(f"Missing reusable result: {done_file}")

    log_path, err_path = out_dir / "run.log", out_dir / "run.err"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print("[run]", " ".join(map(str, args)))
    started = time.time()
    with log_path.open("w") as stdout, err_path.open("w") as stderr:
        result = subprocess.run(
            [str(x) for x in args],
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    print(f"[done] rc={result.returncode}; wall={(time.time() - started) / 3600:.2f} h")
    if not done_file.exists():
        raise RuntimeError(f"Expected output was not written: {done_file}. See {err_path}")
    return result.returncode


def show(label, result):
    if result is None:
        display(Markdown(f"**{label}: missing**"))
        return
    wanted = (
        "status", "converged", "e_tot", "e_lasscf", "e_lasci",
        "e_lassi", "e_lassis", "delta_e_lassis",
        "wall_clock_sec", "wall_time_total",
    )
    lines = [f"### {label}"] + [
        f"- `{key}`: `{result[key]}`" for key in wanted if key in result
    ]
    display(Markdown("\\n".join(lines)))
        """
    ),
    md("## Output layout"),
    code(
        """
cas_dir = RUN_ROOT / "cas_csf_control"
cold_dir = RUN_ROOT / "trimci_cold_thr0.01_dets2000_rounds4"
warm_dir = RUN_ROOT / "trimci_warm_from_cas_thr0.01_dets2000_rounds4"

sources = {"cas": cas_dir, "cold": cold_dir, "warm": warm_dir}
for label, path in sources.items():
    print(f"{label:>4}:", path)
        """
    ),
    md("## 1. CAS-LASSCF/CSF control"),
    code(
        """
run_cmd(
    [
        PYTHON, "-m", "FRASCI.lasscf.runners.run_lasscf_csf",
        "--fcidump", FCIDUMP,
        "--partition", PARTITION,
        "--fragment-orbs-json", FRAGMENT_JSON,
        "--max-cycle", CAS_MAX_CYCLE,
        "--output-dir", cas_dir,
    ],
    cas_dir,
    cas_dir / "result.json",
)
cas_result = load_json(cas_dir / "result.json")
show("CAS-LASSCF/CSF", cas_result)
        """
    ),
    md("## 2. TrimCI-LASSCF cold start"),
    code(
        """
run_cmd(
    [
        PYTHON, "-m", "FRASCI.lasscf.runners.run_lasscf_trimci",
        "--fcidump", FCIDUMP,
        "--partition", PARTITION,
        "--fragment-orbs-json", FRAGMENT_JSON,
        "--trimci-threshold", TRIMCI_THRESHOLD,
        "--trimci-max-dets", TRIMCI_MAX_DETS,
        "--trimci-max-rounds", TRIMCI_MAX_ROUNDS,
        "--max-cycle-macro", TRIMCI_MAX_CYCLE_MACRO,
        "--output-dir", cold_dir,
    ],
    cold_dir,
    cold_dir / "result.json",
)
cold_result = load_json(cold_dir / "result.json")
show("TrimCI-LASSCF cold", cold_result)
        """
    ),
    md("## 3. TrimCI-LASSCF warm start from CAS"),
    code(
        """
run_cmd(
    [
        PYTHON, "-m", "FRASCI.lasscf.runners.run_lasscf_trimci",
        "--fcidump", FCIDUMP,
        "--partition", PARTITION,
        "--fragment-orbs-json", FRAGMENT_JSON,
        "--init-from", cas_dir,
        "--trimci-threshold", TRIMCI_THRESHOLD,
        "--trimci-max-dets", TRIMCI_MAX_DETS,
        "--trimci-max-rounds", TRIMCI_MAX_ROUNDS,
        "--max-cycle-macro", TRIMCI_MAX_CYCLE_MACRO,
        "--output-dir", warm_dir,
    ],
    warm_dir,
    warm_dir / "result.json",
)
warm_result = load_json(warm_dir / "result.json")
show("TrimCI-LASSCF warm from CAS", warm_result)
        """
    ),
    md("## 4. LASSI/LASSIS from all starts and spin expansions"),
    code(
        """
interfragment = {}
for source, checkpoint_dir in sources.items():
    for nspin in LASSIS_NSPINS:
        out_dir = RUN_ROOT / f"lassi_lassis_from_{source}_ncharges_nspin{nspin}"
        args = [
            PYTHON, "-m", "FRASCI.lasscf.runners.run_lassi_lassis",
            "--lasscf-checkpoint-dir", checkpoint_dir,
            "--fcidump", FCIDUMP,
            "--output-dir", out_dir,
            "--opt", LASSI_OPT,
            "--lassis-ncharge", LASSIS_NCHARGE,
            "--lassis-nspin", nspin,
        ]
        # LASSI uses the same CT rootspace construction for every nspin.
        # Run it once (nspin=0); the nspin=1,2 branches add only LASSIS spaces.
        if nspin > 0:
            args.append("--skip-lassi")
        run_cmd(args, out_dir, out_dir / "summary.json")
        interfragment[(source, nspin)] = load_json(out_dir / "summary.json")

print(f"Loaded {sum(v is not None for v in interfragment.values())}/9 summaries")
        """
    ),
    md("## 5. Compact verification table"),
    code(
        """
lines = [
    "| Start | nspin | LASCI (Ha) | LASSI (Ha) | LASSIS (Ha) | LASSIS lowering (mHa) |",
    "|---|---:|---:|---:|---:|---:|",
]
for (source, nspin), result in interfragment.items():
    def fmt(key):
        value = result.get(key) if result else None
        return "—" if value is None else f"{value:.9f}"
    lowering = None if not result else result.get("delta_e_lassis")
    lowering_text = "—" if lowering is None else f"{1000 * lowering:.3f}"
    lines.append(
        f"| {source} | {nspin} | {fmt('e_lasci')} | {fmt('e_lassi')} | "
        f"{fmt('e_lassis')} | {lowering_text} |"
    )
display(Markdown("\\n".join(lines)))
        """
    ),
    md(
        """
## Interpretation guardrails

- The three LASSCF calculations reached the configured 100-cycle limit and are
  recorded as `NOT_CONVERGED`; their energies are useful checkpoint comparisons,
  not fully converged variational endpoints.
- LASSI is attempted only for `nspin=0`. Its saved result includes an error record
  if the state-average construction fails; missing energy is not plotted as zero.
- `FRASCI_Results.ipynb` is the presentation notebook. It reads these files and
  avoids expensive solver calls.
        """
    ),
]


results_cells = [
    md(
        """
# FRASCI LASSCF/LASSI/LASSIS Results

Presentation-only analysis of the saved June 16, 2026 H1 four-fragment run.
Running this notebook does **not** launch LASSCF, TrimCI, LASSI, or LASSIS.

The visuals focus on the scientifically useful small energy differences:

- start-dependent LASSCF/LASCI energies,
- LASSIS energy lowering relative to LASCI,
- final LASSIS energy as the spin expansion grows.
        """
    ),
    code(
        """
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name != "FRASCI":
    candidate = next((p for p in [ROOT, *ROOT.parents] if (p / "FRASCI").is_dir()), None)
    if candidate is None:
        raise RuntimeError("Open this notebook from the FRASCI repository.")
    ROOT = candidate / "FRASCI"

RUN_ROOT = ROOT / "Outputs" / "lasscf" / "h1_4frag_pipeline_20260616_122809"
STARTS = ("cas", "cold", "warm")
NSPINS = (0, 1, 2)
COLORS = {"cas": "#355C7D", "cold": "#C06C84", "warm": "#6C9A8B"}

def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)

base = {
    "cas": load_json(RUN_ROOT / "cas_csf_control" / "result.json"),
    "cold": load_json(RUN_ROOT / "trimci_cold_thr0.01_dets2000_rounds4" / "result.json"),
    "warm": load_json(RUN_ROOT / "trimci_warm_from_cas_thr0.01_dets2000_rounds4" / "result.json"),
}
summaries = {
    (start, nspin): load_json(
        RUN_ROOT / f"lassi_lassis_from_{start}_ncharges_nspin{nspin}" / "summary.json"
    )
    for start in STARTS for nspin in NSPINS
}
print("Loaded:", RUN_ROOT)
        """
    ),
    md("## Calculation status"),
    code(
        """
lines = [
    "| Start | Method | LASSCF energy (Ha) | Status | Runtime (h) |",
    "|---|---|---:|---|---:|",
]
labels = {"cas": "CAS/CSF", "cold": "TrimCI cold", "warm": "TrimCI warm from CAS"}
for start in STARTS:
    row = base[start]
    runtime = row.get("wall_time_total", row.get("wall_clock_sec", 0.0)) / 3600
    lines.append(
        f"| {start} | {labels[start]} | {row['e_tot']:.9f} | "
        f"{row['status']} | {runtime:.2f} |"
    )
display(Markdown("\\n".join(lines)))

display(Markdown(
    "> **Important:** all three orbital optimizations stopped at the configured "
    "100-cycle limit (`NOT_CONVERGED`). The comparisons below therefore describe "
    "the saved checkpoints, not fully converged LASSCF limits."
))
        """
    ),
    md("## Complete energy table"),
    code(
        """
lines = [
    "| Start | nspin | LASSCF (Ha) | LASCI fixed-MO (Ha) | LASSI (Ha) | "
    "LASSIS (Ha) | LASSIS lowering (mHa) |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for start in STARTS:
    for nspin in NSPINS:
        row = summaries[(start, nspin)]
        def val(key):
            value = row.get(key)
            return "—" if value is None else f"{value:.9f}"
        lines.append(
            f"| {start} | {nspin} | {row['e_lasscf']:.9f} | {row['e_lasci']:.9f} | "
            f"{val('e_lassi')} | {row['e_lassis']:.9f} | "
            f"{1000 * row['delta_e_lassis']:.3f} |"
        )
display(Markdown("\\n".join(lines)))
        """
    ),
    md("## Visual 1 — checkpoint energies on a millihartree scale"),
    code(
        """
fig, ax = plt.subplots(figsize=(8.4, 4.8))
reference = base["cas"]["e_tot"]
x = np.arange(len(STARTS))
lasscf_gap = [(base[s]["e_tot"] - reference) * 1000 for s in STARTS]
lasci_gap = [(summaries[(s, 0)]["e_lasci"] - reference) * 1000 for s in STARTS]
width = 0.34
ax.bar(x - width / 2, lasscf_gap, width, label="LASSCF checkpoint")
ax.bar(x + width / 2, lasci_gap, width, label="Fixed-MO LASCI")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x, ["CAS", "Cold", "Warm"])
ax.set_ylabel("Energy − CAS-LASSCF checkpoint (mHa)")
ax.set_title("Start dependence is visible only after removing the ~−327 Ha offset")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
plt.show()
        """
    ),
    md("## Visual 2 — correlation recovered by LASSIS"),
    code(
        """
fig, ax = plt.subplots(figsize=(8.4, 4.8))
x = np.arange(len(NSPINS))
width = 0.24
for offset, start in zip((-width, 0, width), STARTS):
    lowering = [1000 * summaries[(start, n)]["delta_e_lassis"] for n in NSPINS]
    ax.bar(x + offset, lowering, width, label=start, color=COLORS[start])
ax.set_xticks(x, [f"nspin={n}" for n in NSPINS])
ax.set_ylabel("LASCI − LASSIS energy (mHa)")
ax.set_title("Additional inter-fragment correlation from larger LASSIS spaces")
ax.legend(title="Checkpoint")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
plt.show()
        """
    ),
    md("## Visual 3 — final LASSIS energies and near-degeneracies"),
    code(
        """
fig, ax = plt.subplots(figsize=(8.4, 4.8))
global_min = min(row["e_lassis"] for row in summaries.values())
for start in STARTS:
    relative = [
        1000 * (summaries[(start, n)]["e_lassis"] - global_min)
        for n in NSPINS
    ]
    ax.plot(NSPINS, relative, marker="o", linewidth=2, label=start, color=COLORS[start])
ax.set_xticks(NSPINS)
ax.set_xlabel("LASSIS nspin")
ax.set_ylabel("Energy above best saved LASSIS result (mHa)")
ax.set_title("CAS and warm checkpoints become nearly degenerate at nspin=2")
ax.legend(title="Checkpoint")
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()
        """
    ),
    md("## Key observations"),
    code(
        """
best_key = min(summaries, key=lambda key: summaries[key]["e_lassis"])
best = summaries[best_key]["e_lassis"]
warm2 = summaries[("warm", 2)]["e_lassis"]
cas2 = summaries[("cas", 2)]["e_lassis"]
cold2 = summaries[("cold", 2)]["e_lassis"]

text = f'''
- **Best saved LASSIS energy:** `{best:.9f} Ha` from `{best_key[0]}`, `nspin={best_key[1]}`.
- At `nspin=2`, warm is only `{1000 * (warm2 - cas2):.3f} mHa` above CAS.
- At `nspin=2`, cold remains `{1000 * (cold2 - cas2):.3f} mHa` above CAS.
- Expanding `nspin` from 0 to 2 lowers LASSIS by
  `{1000 * (summaries[("cas", 0)]["e_lassis"] - cas2):.3f} mHa` (CAS),
  `{1000 * (summaries[("cold", 0)]["e_lassis"] - cold2):.3f} mHa` (cold), and
  `{1000 * (summaries[("warm", 0)]["e_lassis"] - warm2):.3f} mHa` (warm).
- LASSI energies are absent because the attempted LASSI state-average calculation
  failed; the files preserve that failure instead of substituting a numeric value.
'''
display(Markdown(text))
        """
    ),
]


write_notebook(ROOT / "FRASCIMain.ipynb", main_cells)
write_notebook(ROOT / "FRASCI_Results.ipynb", results_cells)
print("Wrote FRASCIMain.ipynb and FRASCI_Results.ipynb")
