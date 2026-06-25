"""Run-folder writer + atomic runs_index.csv append for diff_mols."""
from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path

import yaml


RUNS_INDEX_COLUMNS: tuple[str, ...] = (
    "timestamp", "molecule", "method", "stage",
    "parent_method", "base_partition", "partition", "geom_tag", "tag",
    "protocol", "parent_protocol", "lassis_protocol",
    "e_tot", "e_ref", "error_mha", "converged",
    "n_dets_total", "wall_s",
    "trimci_threshold", "trimci_max_dets",
    "trimci_max_rounds", "coo_cycles",
    "coo_bfgs_maxiter", "coo_bfgs_ftol", "coo_davidson_tol",
    "warm_start_kappa", "parallel_workers", "process_workers", "omp_threads_per_frag",
    "max_cycle_macro", "n_spin",
    "run_dir",
)


def protocol_path_parts(protocol: str) -> list[str]:
    """Return filesystem path parts for a protocol label.

    Compact labels stay in CSV/report rows, but directories are easier to browse
    when grid axes are nested as dets → cycles → BFGS → extra knobs.
    """
    protocol = str(protocol or "default")
    out: list[str] = []
    for segment in protocol.split("__"):
        tokens = [tok for tok in segment.split("_") if tok]
        if not any(tok.startswith(("dets", "cyc", "bfgs", "thr", "nspin", "opt"))
                   for tok in tokens):
            out.append(segment)
            continue
        ordered: list[str] = []
        for prefix in ("dets", "cyc", "bfgs", "thr", "nspin", "opt"):
            ordered.extend(tok for tok in tokens if tok.startswith(prefix))
        ordered.extend(tok for tok in tokens if tok not in ordered)
        out.extend(ordered or [segment])
    return out or ["default"]


def make_run_dir(
    results_root: Path,
    mol_slug: str,
    method: str,
    partition: str,
    geom_tag: str,
    tag: str,
    timestamp: str,
) -> Path:
    """Create a browsable run directory.

    Single-point benchmarks omit the redundant ``eq`` directory:
    ``results/<mol>/<method>/<partition>/<protocol-axes>/run_<ts>/``.
    Scan data retains ``results/<mol>/<geom>/...`` for backward compatibility.
    """
    mol_root = Path(results_root) / mol_slug
    data_root = mol_root if str(geom_tag) == "eq" else mol_root / str(geom_tag)
    p = (data_root / str(method) / str(partition)
         / Path(*protocol_path_parts(tag)) / f"run_{timestamp}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_run_outputs(
    run_dir: Path,
    *,
    result_json: dict,
    config_snapshot: dict,
    log_text: str | None,
    readme_summary: str,
) -> None:
    """Write the per-run artifacts.

    If ``log_text`` is None, log.txt is left untouched — the caller is responsible for
    writing log.txt itself (e.g., via streaming tee during the run).
    """
    (run_dir / "result.json").write_text(json.dumps(result_json, indent=2, default=str))
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False))
    if log_text is not None:
        (run_dir / "log.txt").write_text(log_text)
    (run_dir / "README.md").write_text(readme_summary)


def append_runs_index_row(results_root: Path, mol_slug: str, row: dict) -> None:
    """Append one row to results/<mol>/runs_index.csv under an exclusive flock."""
    csv_path = Path(results_root) / mol_slug / "runs_index.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Sanitize row → only keep known columns; fill missing with ""
    safe_row = {col: row.get(col, "") for col in RUNS_INDEX_COLUMNS}

    # Lock file separate from CSV so we never lock the same fd we're truncating
    lock_path = csv_path.with_suffix(".csv.lock")
    with open(lock_path, "w") as lockfp:
        fcntl.flock(lockfp.fileno(), fcntl.LOCK_EX)
        try:
            if csv_path.exists():
                with csv_path.open(newline="") as fp:
                    reader = csv.DictReader(fp)
                    old_rows = list(reader)
                    old_fields = reader.fieldnames or []
                if old_fields != list(RUNS_INDEX_COLUMNS):
                    with csv_path.open("w", newline="") as fp:
                        writer = csv.DictWriter(fp, fieldnames=list(RUNS_INDEX_COLUMNS))
                        writer.writeheader()
                        for old_row in old_rows:
                            writer.writerow({col: old_row.get(col, "") for col in RUNS_INDEX_COLUMNS})
                        fp.flush()
                        os.fsync(fp.fileno())

            write_header = not csv_path.exists()
            with csv_path.open("a", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(RUNS_INDEX_COLUMNS))
                if write_header:
                    writer.writeheader()
                writer.writerow(safe_row)
                fp.flush()
                os.fsync(fp.fileno())
        finally:
            fcntl.flock(lockfp.fileno(), fcntl.LOCK_UN)
