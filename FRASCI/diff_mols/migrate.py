"""One-shot helpers to migrate existing diff_mols results from older layouts to the
current per-geom layout AND backfill any newly-required runs_index.csv columns
(``e_ref``, ``error_mha``) from the inline FCI reference stored in scf_summary.json.

History of layouts:
  v1 (pre-2026-06-23): ``results/<mol>/runs/<method>/<partition>__<geom>__<tag>__<ts>/``
  v2 (post-2026-06-23): ``results/<mol>/<geom>/<method>__<partition>__<tag>__<ts>/``
  Per-geom dir also holds ``geometry.xyz`` and ``integrals/{fcidump, scf_summary.json, ...}``.

Usage::

    from FRASCI.diff_mols.migrate import migrate_mol_to_v2
    migrate_mol_to_v2('Outputs/diff_mols', 'diazene_trans')

The migration is idempotent — already-v2 layouts pass through unchanged.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

_RESERVED = {"integrals", "geometry", "runs", "report", "plots"}


def _is_geom_dir(p: Path) -> bool:
    return p.is_dir() and p.name not in _RESERVED


def ensure_e_fci_inline(results_root: str | Path, mol_slug: str) -> dict:
    """Compute and persist ``e_fci_inline`` in scf_summary.json for every integral bundle
    that lacks it. Skips bundles whose active space exceeds n_orb > 12 (too big for FCI).

    Returns ``{geom_tag: {"e_fci": float, "computed": bool}}``.
    """
    import trimci as _trimci
    from pyscf import fci as _fci

    mol_root = Path(results_root) / mol_slug
    report: dict = {}

    # Find all integral locations: either v2 (<geom>/integrals/) or v1 (integrals/<geom>/).
    candidates: list[tuple[str, Path]] = []
    for sub in mol_root.iterdir():
        if not sub.is_dir():
            continue
        if sub.name == "integrals":
            for g in sub.iterdir():
                if g.is_dir():
                    candidates.append((g.name, g))
        elif _is_geom_dir(sub):
            i = sub / "integrals"
            if i.is_dir():
                candidates.append((sub.name, i))

    for geom_tag, idir in candidates:
        scf_path = idir / "scf_summary.json"
        fcidump_path = idir / "fcidump"
        if not scf_path.exists() or not fcidump_path.exists():
            continue
        summary = json.loads(scf_path.read_text())
        if summary.get("e_fci_inline") is not None:
            report[geom_tag] = {"computed": False, "e_fci": summary["e_fci_inline"]}
            continue
        n_orb = int(summary.get("n_orb", 0))
        if n_orb > 12:
            report[geom_tag] = {"computed": False, "skipped": "n_orb > 12"}
            continue
        h1, eri, _ne, _no, _enuc, n_a, n_b, _psym = _trimci.read_fcidump(str(fcidump_path))
        e_const = summary.get("e_const_fcidump", summary.get("e_nuc", 0.0))
        e_active, _ = _fci.direct_spin1.kernel(
            h1, eri.reshape(n_orb, n_orb, n_orb, n_orb), n_orb, (n_a, n_b), ecore=e_const,
        )
        summary["e_fci_inline"] = float(e_active)
        scf_path.write_text(json.dumps(summary, indent=2))
        report[geom_tag] = {"computed": True, "e_fci": float(e_active)}

    return report


def migrate_mol_to_v2(results_root: str | Path, mol_slug: str, *, dry_run: bool = False) -> dict:
    """Move one molecule's results from the v1 flat layout to the v2 per-geom layout.

    Operations (each idempotent):
      - ``geometry/<geom>.xyz`` → ``<geom>/geometry.xyz``
      - ``integrals/<geom>/`` → ``<geom>/integrals/``
      - ``runs/<method>/<part>__<geom>__<tag>__<ts>/`` → ``<geom>/<method>__<part>__<tag>__<ts>/``
      - rewrite ``runs_index.csv``: update ``run_dir`` paths, backfill ``e_ref`` + ``error_mha``
        from ``<geom>/integrals/scf_summary.json::e_fci_inline``
      - update each ``run_dir/result.json``'s ``run_dir`` field to the new path

    Returns ``{actions: [...], csv_rows_patched: int, fci_report: {...}}``.
    """
    mol_root = Path(results_root) / mol_slug
    actions: list[str] = []
    if not mol_root.exists():
        return {"mol": mol_slug, "skipped": True, "reason": "no results dir"}

    # 1) geometry/<geom>.xyz → <geom>/geometry.xyz
    old_geom = mol_root / "geometry"
    if old_geom.is_dir():
        for xyz in list(old_geom.glob("*.xyz")):
            geom_tag = xyz.stem
            target = mol_root / geom_tag / "geometry.xyz"
            actions.append(f"mv geometry/{xyz.name} -> {geom_tag}/geometry.xyz")
            if dry_run:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                xyz.unlink()
            else:
                shutil.move(str(xyz), str(target))
        if not dry_run and old_geom.exists() and not any(old_geom.iterdir()):
            old_geom.rmdir()

    # 2) integrals/<geom>/ → <geom>/integrals/
    old_integrals = mol_root / "integrals"
    if old_integrals.is_dir():
        for geom_dir in list(old_integrals.iterdir()):
            if not geom_dir.is_dir():
                continue
            geom_tag = geom_dir.name
            target = mol_root / geom_tag / "integrals"
            actions.append(f"mv integrals/{geom_tag} -> {geom_tag}/integrals")
            if dry_run:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(geom_dir), str(target))
        if not dry_run and old_integrals.exists() and not any(old_integrals.iterdir()):
            old_integrals.rmdir()

    # 3) runs/<method>/<part>__<geom>__<tag>__<ts>/ → <geom>/<method>__<part>__<tag>__<ts>/
    move_map: dict[str, str] = {}
    old_runs = mol_root / "runs"
    if old_runs.is_dir():
        for method_dir in list(old_runs.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for run_dir in list(method_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                old_name = run_dir.name
                parts = old_name.split("__")
                if len(parts) < 4:
                    actions.append(f"SKIP malformed: runs/{method}/{old_name}")
                    continue
                partition = parts[0]
                geom_tag = parts[1]
                tag = parts[2]
                ts = "__".join(parts[3:])
                new_name = f"{method}__{partition}__{tag}__{ts}"
                new_path = mol_root / geom_tag / new_name
                actions.append(f"mv runs/{method}/{old_name} -> {geom_tag}/{new_name}")
                move_map[str(run_dir)] = str(new_path)
                if dry_run:
                    continue
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if new_path.exists():
                    shutil.rmtree(new_path)
                shutil.move(str(run_dir), str(new_path))
        if not dry_run:
            for d in list(old_runs.iterdir()):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            if old_runs.exists() and not any(old_runs.iterdir()):
                old_runs.rmdir()

    # 4) Update each result.json's `run_dir` field
    if not dry_run:
        for old_str, new_str in move_map.items():
            rj = Path(new_str) / "result.json"
            if rj.exists():
                try:
                    blob = json.loads(rj.read_text())
                    if blob.get("run_dir") == old_str:
                        blob["run_dir"] = new_str
                        rj.write_text(json.dumps(blob, indent=2, default=str))
                except json.JSONDecodeError:
                    pass

    # 5) Ensure e_fci_inline is populated for each geom (computes via PySCF FCI when missing)
    if not dry_run:
        fci_report = ensure_e_fci_inline(results_root, mol_slug)
    else:
        fci_report = {}

    # 6) Rewrite runs_index.csv with patched run_dir + e_ref + error_mha
    csv_path = mol_root / "runs_index.csv"
    csv_rows_patched = 0
    if csv_path.exists():
        from FRASCI.diff_mols.run_writer import RUNS_INDEX_COLUMNS

        with csv_path.open() as fp:
            rows = list(csv.DictReader(fp))

        e_ref_by_geom: dict = {}
        for sub in mol_root.iterdir():
            if not _is_geom_dir(sub):
                continue
            scf_path = sub / "integrals" / "scf_summary.json"
            if scf_path.exists():
                e_ref_by_geom[sub.name] = json.loads(scf_path.read_text()).get("e_fci_inline")

        for row in rows:
            old_run_dir = row.get("run_dir", "")
            if old_run_dir in move_map:
                row["run_dir"] = move_map[old_run_dir]

            geom_tag = row.get("geom_tag", "")
            e_ref = e_ref_by_geom.get(geom_tag)
            row["e_ref"] = "" if e_ref is None else f"{e_ref}"
            try:
                e_tot_str = row.get("e_tot", "") or "nan"
                e_tot = float(e_tot_str)
                if e_ref is not None and e_tot == e_tot:
                    row["error_mha"] = (e_tot - float(e_ref)) * 1000.0
                else:
                    row["error_mha"] = ""
            except (TypeError, ValueError):
                row["error_mha"] = ""
            csv_rows_patched += 1

        if not dry_run:
            with csv_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(RUNS_INDEX_COLUMNS))
                writer.writeheader()
                for row in rows:
                    safe = {col: row.get(col, "") for col in RUNS_INDEX_COLUMNS}
                    writer.writerow(safe)

    return {
        "mol": mol_slug,
        "actions": actions,
        "csv_rows_patched": csv_rows_patched,
        "fci_report": fci_report,
    }
