"""CLI entry: `python -m FRASCI.diff_mols.run ...`."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from FRASCI.diff_mols.benchmark import MoleculeBenchmark, METHOD_ORDER
from FRASCI.diff_mols.tuning import (
    LASSCF_TRIMCI_COO_PROTOCOLS, LASSCF_TRIMCI_PROTOCOLS, LASSIS_PROTOCOLS, TRIMCI_COO_PROTOCOLS,
    TRIMCI_PROTOCOLS, TUNING_PRESETS, expand_protocol_names, resolve_tuning_overrides,
)


def _parse_override(items: list[str]) -> dict:
    """`['key=val', ...]` → dict with literal-eval'd values."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got: {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def _resolve_mol_configs(mol_arg: str, configs_dir: Path) -> list[Path]:
    if mol_arg == "all":
        return sorted(configs_dir.glob("*.yaml"))
    p = configs_dir / f"{mol_arg}.yaml"
    if not p.exists():
        raise SystemExit(f"config not found: {p}")
    return [p]


def _parse_csv(value: str | None) -> list[str] | None:
    return [x.strip() for x in value.split(",") if x.strip()] if value else None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="diff_mols.run")
    p.add_argument("--mol", required=True, help="molecule slug or 'all'")
    p.add_argument("--methods", default=None,
                   help=f"comma-separated; default=all enabled. one of {','.join(METHOD_ORDER)}")
    p.add_argument("--partitions", default=None, help="comma-separated; default=yaml-listed")
    p.add_argument("--geom-tags", default=None, help="comma-separated; default=all")
    p.add_argument("--plain-trimci-protocols", default=None,
                   help=("comma-separated protocol sweep for plain_trimci; "
                         f"choices={','.join(sorted(TRIMCI_PROTOCOLS))}"))
    p.add_argument("--trimci-coo-protocols", default=None,
                   help=("comma-separated protocol sweep for trimci_coo; "
                         f"choices={','.join(sorted(TRIMCI_COO_PROTOCOLS))}"))
    p.add_argument("--lasscf-trimci-protocols", default=None,
                   help=("comma-separated protocol sweep for lasscf_trimci; "
                         f"choices={','.join(sorted(LASSCF_TRIMCI_PROTOCOLS))}"))
    p.add_argument("--lasscf-trimci-coo-protocols", default=None,
                   help=("comma-separated protocol sweep for lasscf_trimci_coo; "
                         f"choices={','.join(sorted(LASSCF_TRIMCI_COO_PROTOCOLS))}"))
    p.add_argument("--lassis-protocols", default=None,
                   help=("comma-separated LASSIS protocols; "
                         f"choices={','.join(sorted(LASSIS_PROTOCOLS))}"))
    p.add_argument("--lassis-nspin-values", default=None,
                   help="comma-separated LASSIS nspin sweep, e.g. 0,1,2,3")
    p.add_argument("--tag", default="default")
    p.add_argument("--tuning-preset", default="default",
                   choices=sorted(TUNING_PRESETS),
                   help="named quality/cost preset; manual --override values win")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--build-integrals-only", action="store_true")
    p.add_argument("--dry-run-matrix", action="store_true",
                   help="print the planned method/fragment/protocol matrix and exit")
    p.add_argument("--matrix-out", default=None,
                   help="optional CSV path for --dry-run-matrix")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--results-root", default="Outputs/diff_mols")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--configs-dir", default="configs/diff_mols")
    return p


def _filter_failed(results_root: Path, slug: str, tag: str) -> list[dict]:
    """Read runs_index.csv and return rows where converged is False."""
    import csv
    csv_path = results_root / slug / "runs_index.csv"
    if not csv_path.exists():
        return []
    with csv_path.open() as fp:
        return [r for r in csv.DictReader(fp)
                if r.get("converged", "").lower() in ("false", "0", "")
                and r.get("tag") == tag]


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    configs_dir = Path(args.configs_dir)
    results_root = Path(args.results_root)
    overrides = resolve_tuning_overrides(args.tuning_preset, _parse_override(args.override))
    methods = args.methods.split(",") if args.methods else None
    partitions = _parse_csv(args.partitions)
    geom_tags = _parse_csv(args.geom_tags)
    plain_trimci_protocols = expand_protocol_names("plain_trimci", _parse_csv(args.plain_trimci_protocols))
    trimci_coo_protocols = expand_protocol_names("trimci_coo", _parse_csv(args.trimci_coo_protocols))
    lasscf_trimci_protocols = expand_protocol_names("lasscf_trimci", _parse_csv(args.lasscf_trimci_protocols))
    lasscf_trimci_coo_protocols = expand_protocol_names("lasscf_trimci_coo", _parse_csv(args.lasscf_trimci_coo_protocols))
    lassis_protocols = _parse_csv(args.lassis_protocols)
    lassis_nspin_values = (
        [int(x) for x in args.lassis_nspin_values.split(",")]
        if args.lassis_nspin_values else None
    )
    skip_existing = not args.no_skip_existing

    rc = 0
    for cfg_path in _resolve_mol_configs(args.mol, configs_dir):
        bm = MoleculeBenchmark(cfg_path, results_root=results_root)
        if args.build_integrals_only:
            bundles = bm.build_all_geometries()
            print(f"[{bm.config.slug}] integrals built for tags: {sorted(bundles.keys())}")
            continue
        if args.dry_run_matrix:
            import csv
            rows = bm.experiment_matrix(
                methods=methods, partitions=partitions, geom_tags=geom_tags, tag=args.tag,
                plain_trimci_protocols=plain_trimci_protocols,
                trimci_coo_protocols=trimci_coo_protocols,
                lasscf_trimci_protocols=lasscf_trimci_protocols,
                lasscf_trimci_coo_protocols=lasscf_trimci_coo_protocols,
                lassis_protocols=lassis_protocols,
                lassis_nspin_values=lassis_nspin_values,
            )
            if args.matrix_out:
                out_path = Path(args.matrix_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fields = sorted({k for row in rows for k in row})
                with out_path.open("w", newline="") as fp:
                    writer = csv.DictWriter(fp, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[{bm.config.slug}] wrote {len(rows)} planned rows → {out_path}")
            else:
                for row in rows:
                    print(row)
                print(f"[{bm.config.slug}] planned rows: {len(rows)}")
            continue
        if args.retry_failed:
            failed = _filter_failed(results_root, bm.config.slug, args.tag)
            if not failed:
                print(f"[{bm.config.slug}] no failed runs to retry")
                continue
            for row in failed:
                method = row["method"]; partition = row["partition"]; gt = row["geom_tag"]
                print(f"[{bm.config.slug}] retrying {method} on {partition}/{gt}")
                if method in ("plain_trimci", "trimci_coo"):
                    getattr(bm, f"run_{method}")(gt, tag=args.tag, **overrides)
                elif method in ("lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo"):
                    getattr(bm, f"run_{method}")(gt, partition, tag=args.tag, **overrides)
                # lassis retry is rare — manual for now
            continue

        results = bm.run_all(
            methods=methods, partitions=partitions, geom_tags=geom_tags,
            tag=args.tag, skip_existing=skip_existing,
            plain_trimci_protocols=plain_trimci_protocols,
            trimci_coo_protocols=trimci_coo_protocols,
            lasscf_trimci_protocols=lasscf_trimci_protocols,
            lasscf_trimci_coo_protocols=lasscf_trimci_coo_protocols,
            lassis_protocols=lassis_protocols,
            lassis_nspin_values=lassis_nspin_values,
            **overrides,
        )

        n_ok = sum(1 for r in results if getattr(r, "converged", False))
        print(f"[{bm.config.slug}] {n_ok}/{len(results)} runs converged")
        if n_ok < len(results):
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
