"""MoleculeBenchmark — driver wiring benchmark methods + integrals + fragmentation."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from FRASCI.diff_mols.config import (
    MoleculeConfig, load_molecule_config,
    geom_tag_for_single, geom_tag_for_scan_point,
)
from FRASCI.diff_mols.integrals_builder import (
    IntegralBundle, build_integrals, build_all_geometries,
)
from FRASCI.diff_mols.fragmentation import resolve_fragmentation
from FRASCI.diff_mols.methods import (
    RunResult, run_plain_trimci, run_trimci_coo,
    run_lasscf_cas, run_lasscf_trimci, run_lasscf_trimci_coo, run_lassis,
)
from FRASCI.diff_mols.tuning import (
    LASSCF_TRIMCI_COO_PROTOCOLS,
    LASSCF_TRIMCI_PROTOCOLS,
    LASSIS_PROTOCOLS,
    TRIMCI_COO_PROTOCOLS,
    TRIMCI_PROTOCOLS,
    expand_protocol_names,
    resolve_lasscf_trimci_protocol,
    resolve_lasscf_trimci_coo_protocol,
    resolve_lassis_protocol,
    resolve_trimci_coo_protocol,
    resolve_trimci_protocol,
)


METHOD_ORDER = (
    "plain_trimci", "trimci_coo", "lasscf_cas",
    "lasscf_trimci", "lasscf_trimci_coo", "lassis",
)


def _default_results_root() -> Path:
    return Path("Outputs/diff_mols")


def _existing_run_for(results_root: Path, slug: str, method: str, partition: str,
                      geom_tag: str, tag: str) -> Path | None:
    """Return the run_dir of any existing run matching (method, partition, geom, tag)."""
    csv_path = results_root / slug / "runs_index.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as fp:
        for row in csv.DictReader(fp):
            if (row["method"] == method and row["partition"] == partition
                    and row["geom_tag"] == geom_tag and row["tag"] == tag):
                return Path(row["run_dir"])
    return None


def _csv_list(value: list[str] | None) -> list[str] | None:
    return list(value) if value else None


def _parent_run_identity(parent_dir: Path) -> tuple[str, str]:
    """Return the full parent protocol and partition from its result metadata."""
    parent_dir = Path(parent_dir)
    result_path = parent_dir / "result.json"
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
            protocol = data.get("protocol") or data.get("tag")
            partition = data.get("base_partition") or data.get("partition")
            if protocol and partition:
                return str(protocol), str(partition)
        except (OSError, ValueError, TypeError):
            pass

    # Compatibility with the original one-directory protocol layout.
    return parent_dir.parent.name, parent_dir.parent.parent.name


def _skipped_run_result(result_json: dict, run_dir: Path) -> RunResult:
    return RunResult(
        e_tot=None, n_dets=None, wall_s=0.0,
        run_dir=run_dir, result_json=result_json, converged=True,
    )


class MoleculeBenchmark:
    def __init__(self, config_path: str | Path, *, results_root: Path | None = None):
        self.config = load_molecule_config(config_path)
        self.results_root = Path(results_root) if results_root else _default_results_root()
        self._integrals_cache: dict[str, IntegralBundle] = {}

    # -- Integrals -------------------------------------------------------

    def build_integrals(self, geom_tag: str, *, force: bool = False) -> IntegralBundle:
        if (not force) and geom_tag in self._integrals_cache:
            return self._integrals_cache[geom_tag]
        bundle = build_integrals(self.config, geom_tag, self.results_root, force=force)
        self._integrals_cache[geom_tag] = bundle
        return bundle

    def build_all_geometries(self, *, force: bool = False) -> dict[str, IntegralBundle]:
        bundles = build_all_geometries(self.config, self.results_root, force=force)
        self._integrals_cache.update(bundles)
        return bundles

    # -- Methods ---------------------------------------------------------

    def run_plain_trimci(self, geom_tag: str, **kw) -> RunResult:
        return run_plain_trimci(self.config, self.build_integrals(geom_tag),
                                results_root=self.results_root, **kw)

    def run_trimci_coo(self, geom_tag: str, **kw) -> RunResult:
        return run_trimci_coo(self.config, self.build_integrals(geom_tag),
                              results_root=self.results_root, **kw)

    def run_lasscf_cas(self, geom_tag: str, partition: str, **kw) -> RunResult:
        return run_lasscf_cas(self.config, self.build_integrals(geom_tag),
                              partition_name=partition,
                              results_root=self.results_root, **kw)

    def run_lasscf_trimci(self, geom_tag: str, partition: str, **kw) -> RunResult:
        return run_lasscf_trimci(self.config, self.build_integrals(geom_tag),
                                 partition_name=partition,
                                 results_root=self.results_root, **kw)

    def run_lasscf_trimci_coo(self, geom_tag: str, partition: str, **kw) -> RunResult:
        return run_lasscf_trimci_coo(self.config, self.build_integrals(geom_tag),
                                     partition_name=partition,
                                     results_root=self.results_root, **kw)

    def run_lassis(self, lasscf_run_dir: Path, **kw) -> RunResult:
        """Run LASSIS on top of a LASSCF checkpoint; derives geom_tag from parent folder."""
        run_dir = Path(lasscf_run_dir)
        result_path = run_dir / "result.json"
        if result_path.exists():
            geom_tag = json.loads(result_path.read_text()).get("geom_tag")
        else:
            # Metadata-free fallback. Single-point runs have no geom directory.
            geom_tag = "eq" if self.config.geometry.kind == "single" else run_dir.parents[3].name
        return run_lassis(self.config, self.build_integrals(geom_tag),
                          lasscf_run_dir=run_dir,
                          results_root=self.results_root, **kw)

    # -- Batch -----------------------------------------------------------

    def _iter_geom_tags(self, geom_tags: list[str] | None) -> list[str]:
        if geom_tags is not None:
            return list(geom_tags)
        if self.config.geometry.kind == "single":
            return [geom_tag_for_single()]
        return [geom_tag_for_scan_point(p, self.config.geometry.scan_param)
                for p in self.config.geometry.scan_points]

    def experiment_matrix(
        self,
        *,
        methods: list[str] | None = None,
        partitions: list[str] | None = None,
        geom_tags: list[str] | None = None,
        tag: str = "default",
        plain_trimci_protocols: list[str] | None = None,
        trimci_coo_protocols: list[str] | None = None,
        lasscf_trimci_protocols: list[str] | None = None,
        lasscf_trimci_coo_protocols: list[str] | None = None,
        lassis_protocols: list[str] | None = None,
        lassis_nspin_values: list[int] | None = None,
    ) -> list[dict]:
        """Return the planned experiment rows without running any chemistry."""
        enabled_methods = list(methods) if methods else [
            m for m in METHOD_ORDER if getattr(self.config.methods, m).enabled
        ]
        ordered = [m for m in METHOD_ORDER if m in enabled_methods]
        geom_tags_to_run = self._iter_geom_tags(geom_tags)
        rows: list[dict] = []

        def method_protocols(method: str, explicit: list[str] | None) -> list[str]:
            spec = getattr(self.config.methods, method)
            selected = _csv_list(explicit) or list(spec.protocols)
            names = expand_protocol_names(method, selected) or [tag]
            registries = {
                "plain_trimci": TRIMCI_PROTOCOLS,
                "trimci_coo": TRIMCI_COO_PROTOCOLS,
                "lasscf_trimci": LASSCF_TRIMCI_PROTOCOLS,
                "lasscf_trimci_coo": LASSCF_TRIMCI_COO_PROTOCOLS,
            }
            registry = registries.get(method)
            if registry is not None:
                unknown = sorted(set(names) - set(registry))
                if unknown and names != [tag]:
                    raise ValueError(f"unknown {method} protocols: {unknown}")
            return names

        def lassis_protocol_names(spec) -> list[str]:
            names = _csv_list(lassis_protocols) or list(spec.protocols)
            if names:
                unknown = sorted(set(names) - set(LASSIS_PROTOCOLS))
                if unknown:
                    raise ValueError(f"unknown lassis protocols: {unknown}")
                return names
            spin_values = (
                list(map(int, lassis_nspin_values))
                if lassis_nspin_values
                else list(map(int, spec.n_spin_values)) if spec.n_spin_values
                else [int(spec.n_spin or 0)]
            )
            opt = int(spec.opt or 1)
            return [f"nspin{nspin}_opt{opt}" for nspin in spin_values]

        for gt in geom_tags_to_run:
            parent_plans: dict[str, list[dict]] = {}
            for method in ordered:
                spec = getattr(self.config.methods, method)
                if method == "plain_trimci":
                    for proto in method_protocols(method, plain_trimci_protocols):
                        rows.append({"molecule": self.config.slug, "geom_tag": gt,
                                     "method": method, "partition": "full",
                                     "protocol": proto})
                elif method == "trimci_coo":
                    for proto in method_protocols(method, trimci_coo_protocols):
                        rows.append({"molecule": self.config.slug, "geom_tag": gt,
                                     "method": method, "partition": "full",
                                     "protocol": proto})
                elif method in ("lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo"):
                    explicit = (
                        lasscf_trimci_protocols if method == "lasscf_trimci"
                        else lasscf_trimci_coo_protocols if method == "lasscf_trimci_coo"
                        else None
                    )
                    for part in (partitions or spec.partitions):
                        for proto in method_protocols(method, explicit):
                            row = {"molecule": self.config.slug, "geom_tag": gt,
                                   "method": method, "partition": part,
                                   "protocol": proto}
                            rows.append(row)
                            parent_plans.setdefault(method, []).append(row)
                elif method == "lassis":
                    for parent_method in spec.on_top_of:
                        for parent in parent_plans.get(parent_method, []):
                            for lproto in lassis_protocol_names(spec):
                                rows.append({
                                    "molecule": self.config.slug,
                                    "geom_tag": gt,
                                    "method": "lassis",
                                    "partition": f"{parent['partition']}_on_{parent_method}",
                                    "protocol": f"{parent['protocol']}__{lproto}",
                                    "parent_method": parent_method,
                                    "parent_protocol": parent["protocol"],
                                    "lassis_protocol": lproto,
                                })
        return rows

    def run_all(
        self,
        *,
        methods: list[str] | None = None,
        partitions: list[str] | None = None,
        geom_tags: list[str] | None = None,
        tag: str = "default",
        skip_existing: bool = True,
        verbose: bool = True,
        plain_trimci_protocols: list[str] | None = None,
        trimci_coo_protocols: list[str] | None = None,
        lasscf_trimci_protocols: list[str] | None = None,
        lasscf_trimci_coo_protocols: list[str] | None = None,
        lassis_protocols: list[str] | None = None,
        lassis_nspin_values: list[int] | None = None,
        **overrides,
    ) -> list[RunResult]:
        import time as _time
        enabled_methods = list(methods) if methods else [
            m for m in METHOD_ORDER if getattr(self.config.methods, m).enabled
        ]
        geom_tags_to_run = self._iter_geom_tags(geom_tags)

        if verbose:
            print(f"\n┌────────────────────────────────────────────────────────────────────────────")
            print(f"│ {self.config.slug}  ({self.config.name})")
            print(f"│   methods:  {', '.join(enabled_methods)}")
            print(f"│   geom_tags: {', '.join(geom_tags_to_run)}")
            print(f"│   tag={tag!r}  skip_existing={skip_existing}")
            print(f"└────────────────────────────────────────────────────────────────────────────", flush=True)

        # Build integrals up front for every geom
        for gt in geom_tags_to_run:
            if verbose:
                print(f"[build_integrals] {gt} ...", end=" ", flush=True)
            t0 = _time.time()
            self.build_integrals(gt)
            if verbose:
                print(f"done ({_time.time()-t0:.1f}s)", flush=True)

        out: list[RunResult] = []

        # Enforce method order: plain_trimci MUST run first (seeds dets.npz for LASSCF reads)
        ordered = [m for m in METHOD_ORDER if m in enabled_methods]

        def _run_tag(base_tag: str, protocol: str | None) -> str:
            return protocol if protocol else base_tag

        def _method_protocols(method: str, explicit: list[str] | None) -> list[str | None]:
            spec = getattr(self.config.methods, method)
            selected = _csv_list(explicit) or list(spec.protocols)
            names = expand_protocol_names(method, selected)
            return names or [None]

        def _lassis_protocol_items(spec) -> list[tuple[str, dict]]:
            names = _csv_list(lassis_protocols) or list(spec.protocols)
            if names:
                return [(name, resolve_lassis_protocol(name, overrides)) for name in names]
            spin_values = (
                list(map(int, lassis_nspin_values))
                if lassis_nspin_values
                else list(map(int, spec.n_spin_values)) if spec.n_spin_values
                else [int(spec.n_spin or 0)]
            )
            opt = int(spec.opt or 1)
            out = []
            for nspin in spin_values:
                label = f"nspin{nspin}_opt{opt}"
                proto = LASSIS_PROTOCOLS.get(label, {"lassis_nspin": nspin, "opt": opt})
                merged = {**overrides, **proto}
                out.append((label, merged))
            return out

        def _announce(method: str, partition: str, gt: str, run_tag: str = tag):
            if verbose:
                print(f"\n  ▶ {method:20s} | partition={partition:8s} | geom={gt:8s} | tag={run_tag}",
                      flush=True)

        def _done(method: str, res: RunResult, wall: float):
            if verbose:
                e = res.e_tot if res.e_tot is not None else float("nan")
                n = res.n_dets if res.n_dets is not None else 0
                conv = "" if res.converged else "  [NOT CONVERGED]"
                print(f"  ✓ {method:20s} → e_tot={e:.6f}  n_dets={n}  wall={wall:.1f}s{conv}",
                      flush=True)

        def _skipped(method: str, partition: str, gt: str, run_dir: Path):
            if verbose:
                print(f"  ⏭  {method:20s} | partition={partition:8s} | geom={gt:8s} "
                      f"| already exists → {run_dir.name}", flush=True)

        for gt in geom_tags_to_run:
            if verbose:
                print(f"\n══ geom = {gt} ══", flush=True)
            lasscf_dirs_by_method: dict[str, list[Path]] = {}

            for method in ordered:
                spec = getattr(self.config.methods, method)

                if method in ("lasscf_cas", "lasscf_trimci", "lasscf_trimci_coo"):
                    part_names = partitions or spec.partitions
                    protocol_names = (
                        _method_protocols(method, lasscf_trimci_protocols)
                        if method == "lasscf_trimci"
                        else
                        _method_protocols(method, lasscf_trimci_coo_protocols)
                        if method == "lasscf_trimci_coo"
                        else _method_protocols(method, None)
                    )
                    for p in part_names:
                        for protocol in protocol_names:
                            run_tag = _run_tag(tag, protocol)
                            run_overrides = (
                                resolve_lasscf_trimci_protocol(protocol, overrides)
                                if method == "lasscf_trimci" and protocol
                                else
                                resolve_lasscf_trimci_coo_protocol(protocol, overrides)
                                if method == "lasscf_trimci_coo" and protocol
                                else overrides
                            )
                            existing = _existing_run_for(
                                self.results_root, self.config.slug, method, p, gt, run_tag
                            )
                            if skip_existing and existing is not None:
                                _skipped(method, p, gt, existing)
                                out.append(_skipped_run_result(
                                    {"method": method, "skipped": True, "run_dir": str(existing)},
                                    existing,
                                ))
                                lasscf_dirs_by_method.setdefault(method, []).append(existing)
                                continue
                            _announce(method if protocol is None else f"{method}:{protocol}", p, gt, run_tag)
                            runner = (
                                self.run_lasscf_cas if method == "lasscf_cas"
                                else self.run_lasscf_trimci if method == "lasscf_trimci"
                                else self.run_lasscf_trimci_coo
                            )
                            t0 = _time.time()
                            res = runner(gt, p, tag=run_tag, **run_overrides)
                            _done(method if protocol is None else f"{method}:{protocol}", res, _time.time() - t0)
                            out.append(res)
                            lasscf_dirs_by_method.setdefault(method, []).append(res.run_dir)

                elif method == "lassis":
                    # one LASSIS per parent LASSCF run
                    lassis_items = _lassis_protocol_items(spec)
                    for parent_method in spec.on_top_of:
                        for parent_dir in lasscf_dirs_by_method.get(parent_method, []):
                            parent_dir = Path(parent_dir)
                            parent_tag, partition_name = _parent_run_identity(parent_dir)
                            partition_label = f"{partition_name}_on_{parent_method}"
                            for lassis_label, lassis_overrides in lassis_items:
                                run_tag = f"{parent_tag}__{lassis_label}"
                                existing = _existing_run_for(
                                    self.results_root, self.config.slug,
                                    "lassis", partition_label, gt, run_tag,
                                )
                                if skip_existing and existing is not None:
                                    _skipped("lassis", partition_label, gt, existing)
                                    out.append(_skipped_run_result(
                                        {"method": "lassis", "skipped": True,
                                         "run_dir": str(existing)},
                                        existing,
                                    ))
                                    continue
                                label = f"lassis:{lassis_label}"
                                _announce(label, partition_label, gt, run_tag)
                                t0 = _time.time()
                                res = self.run_lassis(
                                    parent_dir, tag=run_tag,
                                    **{**lassis_overrides, "lassis_protocol": lassis_label},
                                )
                                _done(label, res, _time.time() - t0)
                                out.append(res)

                else:
                    # plain_trimci or trimci_coo
                    protocol_names = _method_protocols(
                        method,
                        plain_trimci_protocols if method == "plain_trimci" else trimci_coo_protocols,
                    )
                    for protocol in protocol_names:
                        run_tag = _run_tag(tag, protocol)
                        existing = _existing_run_for(
                            self.results_root, self.config.slug, method, "full", gt, run_tag
                        )
                        if skip_existing and existing is not None:
                            _skipped(method, "full", gt, existing)
                            out.append(_skipped_run_result(
                                {"method": method, "skipped": True, "run_dir": str(existing)},
                                existing,
                            ))
                            continue
                        label = method if protocol is None else f"{method}:{protocol}"
                        _announce(label, "full", gt, run_tag)
                        runner = (self.run_plain_trimci if method == "plain_trimci"
                                  else self.run_trimci_coo)
                        run_overrides = (
                            resolve_trimci_protocol(protocol, overrides)
                            if method == "plain_trimci" and protocol
                            else resolve_trimci_coo_protocol(protocol, overrides)
                            if method == "trimci_coo" and protocol
                            else overrides
                        )
                        t0 = _time.time()
                        res = runner(gt, tag=run_tag, **run_overrides)
                        _done(label, res, _time.time() - t0)
                        out.append(res)

        if verbose:
            n_ran = sum(1 for r in out if not getattr(r, "result_json", {}).get("skipped", False))
            n_skip = len(out) - n_ran
            print(f"\n┌──── done: {n_ran} new + {n_skip} skipped = {len(out)} total ────", flush=True)

        return out
