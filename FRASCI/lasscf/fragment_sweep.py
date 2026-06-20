from __future__ import annotations

import json
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from FRASCI.lasscf.support import (
    fragment_electron_count,
    load_ref_det,
    orbital_coupling_matrix,
    validate_fragment_partition,
)


FULL_TRIMCI_REFERENCE_ENERGY = -327.1920


@dataclass(frozen=True)
class FragmentationCandidate:
    name: str
    family: str
    size_pattern: list[int]
    fragments: list[list[int]]
    description: str
    cut_strength: float
    max_fragment_size: int
    min_fragment_size: int
    mean_abs_spin_imbalance: float
    nelec_per_frag: list[tuple[int, int]]
    spin_sub: list[int]


def read_fcidump_problem(fcidump_path: str, dets_path: str):
    import trimci

    h1, eri, n_elec, n_orb, e_nuc, n_alpha, n_beta, psym = trimci.read_fcidump(
        fcidump_path
    )
    ref_alpha_bits, ref_beta_bits = load_ref_det(dets_path, row=0)
    return {
        "h1": h1,
        "eri": eri,
        "n_elec": int(n_elec),
        "n_orb": int(n_orb),
        "e_nuc": float(e_nuc),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "ref_alpha_bits": int(ref_alpha_bits),
        "ref_beta_bits": int(ref_beta_bits),
    }


def default_size_patterns(n_orb: int = 36) -> list[list[int]]:
    """Size recipes for deliberately different LASSCF decompositions."""
    if n_orb == 36:
        patterns = [
            [18, 18],
            [16, 20],
            [12, 24],
            [12, 12, 12],
            [6, 12, 18],
            [8, 12, 16],
            [9, 9, 18],
            [10, 12, 14],
            [6, 15, 15],
            [9, 9, 9, 9],
            [6, 8, 10, 12],
            [6, 6, 12, 12],
            [8, 7, 7, 7, 7],
            [4, 8, 8, 8, 8],
            [6, 6, 6, 9, 9],
            [6, 6, 6, 6, 6, 6],
            [4, 4, 7, 7, 7, 7],
            [3, 3, 6, 6, 9, 9],
        ]
    else:
        patterns = []
        for n_frag in range(2, min(6, n_orb) + 1):
            base = n_orb // n_frag
            rem = n_orb % n_frag
            equalish = [base + (1 if i < rem else 0) for i in range(n_frag)]
            patterns.append(equalish)
            if n_frag >= 3 and base > 1:
                skewed = equalish[:]
                skewed[0] = max(1, skewed[0] - 1)
                skewed[-1] += 1
                patterns.append(skewed)
    return [p for p in patterns if sum(p) == n_orb and all(s > 0 for s in p)]


def split_order_by_sizes(order: Iterable[int], sizes: list[int]) -> list[list[int]]:
    order = list(map(int, order))
    if sum(sizes) != len(order):
        raise ValueError(f"sizes sum to {sum(sizes)}, order has {len(order)} orbitals")
    fragments = []
    start = 0
    for size in sizes:
        fragments.append(sorted(order[start : start + size]))
        start += size
    return fragments


def round_robin_by_sizes(order: Iterable[int], sizes: list[int]) -> list[list[int]]:
    """Distribute an ordering across bins to deliberately mix orbital classes."""
    order = list(map(int, order))
    bins = [[] for _ in sizes]
    remaining = list(sizes)
    frag_idx = 0
    for orb in order:
        checked = 0
        while remaining[frag_idx] == 0 and checked <= len(sizes):
            frag_idx = (frag_idx + 1) % len(sizes)
            checked += 1
        bins[frag_idx].append(orb)
        remaining[frag_idx] -= 1
        frag_idx = (frag_idx + 1) % len(sizes)
    return [sorted(frag) for frag in bins]


def occupancy_interleaved_order(
    h1_diag: np.ndarray, ref_alpha_bits: int, ref_beta_bits: int
) -> list[int]:
    buckets: dict[str, list[int]] = {"docc": [], "alpha": [], "beta": [], "virt": []}
    for orb in range(len(h1_diag)):
        a = bool((ref_alpha_bits >> orb) & 1)
        b = bool((ref_beta_bits >> orb) & 1)
        if a and b:
            buckets["docc"].append(orb)
        elif a:
            buckets["alpha"].append(orb)
        elif b:
            buckets["beta"].append(orb)
        else:
            buckets["virt"].append(orb)

    for key in buckets:
        buckets[key].sort(key=lambda orb: (float(h1_diag[orb]), orb))

    order: list[int] = []
    keys = ["docc", "alpha", "beta", "virt"]
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                order.append(buckets[key].pop(0))
    return order


def spectral_order(weights: np.ndarray) -> list[int]:
    """Fiedler-vector ordering of the integral-coupling graph."""
    degree = np.diag(weights.sum(axis=1))
    laplacian = degree - weights
    evals, evecs = np.linalg.eigh(laplacian)
    vec_idx = 1 if len(evals) > 1 else 0
    fiedler = evecs[:, vec_idx]
    return np.argsort(fiedler, kind="stable").astype(int).tolist()


def stochastic_balanced_order(
    n_orb: int,
    h1_diag: np.ndarray,
    ref_alpha_bits: int,
    ref_beta_bits: int,
    seed: int,
) -> list[int]:
    """Deterministic pseudo-random order, stratified by occupancy class."""
    rng = np.random.default_rng(seed)
    base = occupancy_interleaved_order(h1_diag, ref_alpha_bits, ref_beta_bits)
    chunks = [base[i::4] for i in range(4)]
    for chunk in chunks:
        rng.shuffle(chunk)
    order = []
    while any(chunks):
        for chunk in chunks:
            if chunk:
                order.append(int(chunk.pop()))
    assert sorted(order) == list(range(n_orb))
    return order


def _cut_strength(weights: np.ndarray, fragments: list[list[int]]) -> float:
    labels = {}
    for frag_idx, frag in enumerate(fragments):
        for orb in frag:
            labels[orb] = frag_idx
    cut = 0.0
    n_orb = weights.shape[0]
    for p in range(n_orb):
        for q in range(p + 1, n_orb):
            if labels[p] != labels[q]:
                cut += float(weights[p, q])
    return cut


def _candidate_from_fragments(
    *,
    name: str,
    family: str,
    size_pattern: list[int],
    fragments: list[list[int]],
    description: str,
    weights: np.ndarray,
    n_orb: int,
    ref_alpha_bits: int,
    ref_beta_bits: int,
) -> FragmentationCandidate:
    fragments = [sorted(map(int, frag)) for frag in fragments]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Fragment .* has no virtual orbitals")
        validate_fragment_partition(fragments, n_orb, ref_alpha_bits, ref_beta_bits)
    nelec = [
        tuple(map(int, fragment_electron_count(ref_alpha_bits, ref_beta_bits, frag)))
        for frag in fragments
    ]
    spin_sub = [int(abs(na - nb)) + 1 for na, nb in nelec]
    return FragmentationCandidate(
        name=name,
        family=family,
        size_pattern=list(map(int, size_pattern)),
        fragments=fragments,
        description=description,
        cut_strength=round(_cut_strength(weights, fragments), 8),
        max_fragment_size=max(size_pattern),
        min_fragment_size=min(size_pattern),
        mean_abs_spin_imbalance=round(
            float(np.mean([abs(na - nb) for na, nb in nelec])), 6
        ),
        nelec_per_frag=nelec,
        spin_sub=spin_sub,
    )


def _select_diverse_candidates(
    candidates: list[FragmentationCandidate], target_count: int
) -> list[FragmentationCandidate]:
    if len(candidates) <= target_count:
        return candidates

    by_size: dict[tuple[int, ...], list[FragmentationCandidate]] = {}
    for candidate in candidates:
        by_size.setdefault(tuple(candidate.size_pattern), []).append(candidate)

    selected: list[FragmentationCandidate] = []
    selected_names: set[str] = set()
    family_counts: dict[str, int] = {}
    size_keys = list(by_size)

    while len(selected) < target_count:
        changed = False
        for size_key in size_keys:
            pool = [c for c in by_size[size_key] if c.name not in selected_names]
            if not pool:
                continue
            pool.sort(
                key=lambda c: (
                    family_counts.get(c.family, 0),
                    c.max_fragment_size,
                    c.cut_strength,
                    c.name,
                )
            )
            candidate = pool[0]
            selected.append(candidate)
            selected_names.add(candidate.name)
            family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1
            changed = True
            if len(selected) >= target_count:
                break
        if not changed:
            break

    return selected


def build_fragmentation_candidates(
    problem: dict,
    *,
    target_count: int = 24,
    max_fragment_size: int | None = None,
    random_seeds: Iterable[int] = (17, 29, 43),
) -> list[FragmentationCandidate]:
    h1 = problem["h1"]
    eri = problem["eri"]
    n_orb = problem["n_orb"]
    ref_alpha_bits = problem["ref_alpha_bits"]
    ref_beta_bits = problem["ref_beta_bits"]
    h1_diag = np.diag(h1)
    weights = orbital_coupling_matrix(h1, eri)
    strength = weights.sum(axis=1)

    orders: list[tuple[str, str, list[int], str]] = [
        (
            "index",
            "Index-contiguous",
            list(range(n_orb)),
            "keeps neighboring orbital numbers together; useful when orbitals preserve localization or atom ordering",
        ),
        (
            "h1diag",
            "h1 diagonal sorted",
            np.argsort(h1_diag, kind="stable").astype(int).tolist(),
            "groups orbitals by one-electron diagonal energy, the current strong baseline",
        ),
        (
            "h1diag_rev",
            "h1 diagonal reversed",
            np.argsort(-h1_diag, kind="stable").astype(int).tolist(),
            "same energy ordering as h1diag but inverted to stress fragment-boundary sensitivity",
        ),
        (
            "coupling_strength",
            "Integral strength sorted",
            np.argsort(-strength, kind="stable").astype(int).tolist(),
            "places globally most Coulomb/exchange-coupled orbitals early",
        ),
        (
            "spectral",
            "Integral graph spectral",
            spectral_order(weights),
            "orders orbitals by the Fiedler vector of the integral-coupling graph",
        ),
        (
            "occupancy_mix",
            "Occupancy interleaved",
            occupancy_interleaved_order(h1_diag, ref_alpha_bits, ref_beta_bits),
            "interleaves doubly occupied, alpha-only, beta-only, and virtual classes",
        ),
    ]
    for seed in random_seeds:
        orders.append(
            (
                f"stochastic_s{seed}",
                f"Seeded occupancy shuffle {seed}",
                stochastic_balanced_order(n_orb, h1_diag, ref_alpha_bits, ref_beta_bits, seed),
                "deterministic shuffled occupancy strata for boundary robustness tests",
            )
        )

    patterns = default_size_patterns(n_orb)
    if max_fragment_size is not None:
        patterns = [p for p in patterns if max(p) <= max_fragment_size]

    candidates: list[FragmentationCandidate] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for sizes in patterns:
        for order_key, family, order, family_desc in orders:
            for mode in ("block", "roundrobin"):
                if mode == "block":
                    fragments = split_order_by_sizes(order, sizes)
                    mode_desc = "contiguous blocks along that ordering"
                else:
                    fragments = round_robin_by_sizes(order, sizes)
                    mode_desc = "round-robin mixing along that ordering"
                canon = tuple(sorted(tuple(frag) for frag in fragments))
                if canon in seen:
                    continue
                seen.add(canon)
                sizes_label = "x".join(str(s) for s in sizes)
                name = f"{order_key}_{mode}_{sizes_label}"
                description = (
                    f"{family}; {mode_desc}; sizes={sizes}. This {family_desc}."
                )
                candidates.append(
                    _candidate_from_fragments(
                        name=name,
                        family=f"{family} / {mode}",
                        size_pattern=sizes,
                        fragments=fragments,
                        description=description,
                        weights=weights,
                        n_orb=n_orb,
                        ref_alpha_bits=ref_alpha_bits,
                        ref_beta_bits=ref_beta_bits,
                    )
                )
    return _select_diverse_candidates(candidates, target_count)


def write_candidate_files(candidates: list[FragmentationCandidate], output_dir: str) -> None:
    root = Path(output_dir)
    candidate_dir = root / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        with open(candidate_dir / f"{candidate.name}.json", "w", encoding="utf-8") as fp:
            json.dump(candidate.fragments, fp, indent=2)
            fp.write("\n")

    with open(root / "fragmentation_catalog.json", "w", encoding="utf-8") as fp:
        json.dump([asdict(c) for c in candidates], fp, indent=2)
        fp.write("\n")

    lines = [
        "# LASSCF Fragmentation Sweep Catalog",
        "",
        "Each entry is a non-overlapping partition of the full active space.",
        "",
    ]
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"## {idx}. {candidate.name}")
        lines.append("")
        lines.append(candidate.description)
        lines.append("")
        lines.append(
            f"- sizes: {candidate.size_pattern}; spin_sub: {candidate.spin_sub}; "
            f"cut_strength: {candidate.cut_strength:.6g}"
        )
        for frag_idx, (frag, nelec) in enumerate(
            zip(candidate.fragments, candidate.nelec_per_frag)
        ):
            lines.append(
                f"- F{frag_idx}: n_orb={len(frag)}, nelec={tuple(nelec)}, "
                f"orbs={frag}"
            )
        lines.append("")
    (root / "fragmentation_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def load_completed_result(path: Path) -> dict | None:
    result_path = path / "result.json"
    if not result_path.exists():
        return None
    with open(result_path, encoding="utf-8") as fp:
        return json.load(fp)


def plot_sweep(output_dir: str, reference_energy: float = FULL_TRIMCI_REFERENCE_ENERGY) -> list[str]:
    root = Path(output_dir)
    catalog_path = root / "fragmentation_catalog.json"
    if not catalog_path.exists():
        return []

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    with open(catalog_path, encoding="utf-8") as fp:
        candidates = json.load(fp)

    result_rows = []
    for candidate in candidates:
        candidate_runs = sorted((root / "runs").glob(f"{candidate['name']}__thr*"))
        for run_dir in candidate_runs:
            result = load_completed_result(run_dir)
            if result is None:
                continue
            row = {**candidate, **result}
            row["run_dir"] = str(run_dir)
            row["energy_error"] = (
                float(result["e_tot"]) - reference_energy
                if result.get("e_tot") is not None
                else math.nan
            )
            row["total_dets"] = sum(
                d for d in result.get("dets_per_frag_final", []) if isinstance(d, int)
            )
            result_rows.append(row)

    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    written: list[str] = []

    names = [c["name"] for c in candidates]
    max_size = [c["max_fragment_size"] for c in candidates]
    cut = [c["cut_strength"] for c in candidates]
    spin = [c["mean_abs_spin_imbalance"] for c in candidates]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].bar(range(len(names)), max_size, color="#3b6ea8")
    axes[0].set_title("Largest fragment size")
    axes[0].set_ylabel("orbitals")
    axes[1].bar(range(len(names)), cut, color="#7a9a43")
    axes[1].set_title("Inter-fragment integral cut")
    axes[2].bar(range(len(names)), spin, color="#b46b43")
    axes[2].set_title("Mean |n_alpha - n_beta|")
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
    path = plot_dir / "01_candidate_geometry.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(str(path))

    if result_rows:
        labels = [r["name"] for r in result_rows]
        energies = [float(r["e_tot"]) for r in result_rows]
        errors = [float(r["energy_error"]) for r in result_rows]
        dets = [int(r["total_dets"]) for r in result_rows]

        order = np.argsort(energies)
        fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        ax.bar(range(len(order)), [energies[i] for i in order], color="#375f5c")
        ax.axhline(reference_energy, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("E_tot (Ha)")
        ax.set_title("LASSCF+TrimCI energies by partition")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=7)
        path = plot_dir / "02_energy_ranked.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        scatter = ax.scatter(dets, errors, c=[r["max_fragment_size"] for r in result_rows], cmap="viridis", s=70)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("final selected determinants across fragments")
        ax.set_ylabel(f"E - E_ref (Ha), E_ref={reference_energy:.4f}")
        ax.set_title("Cost vs accuracy")
        fig.colorbar(scatter, ax=ax, label="largest fragment size")
        path = plot_dir / "03_cost_accuracy.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
        bottoms = np.zeros(len(result_rows))
        max_frags = max(len(r.get("dets_per_frag_final", [])) for r in result_rows)
        colors = plt.cm.tab20(np.linspace(0, 1, max_frags))
        for frag_idx in range(max_frags):
            vals = [
                (r.get("dets_per_frag_final", [])[frag_idx]
                 if frag_idx < len(r.get("dets_per_frag_final", []))
                 and isinstance(r.get("dets_per_frag_final", [])[frag_idx], int)
                 else 0)
                for r in result_rows
            ]
            ax.bar(range(len(result_rows)), vals, bottom=bottoms, color=colors[frag_idx], label=f"F{frag_idx}")
            bottoms += np.array(vals)
        ax.set_title("Final determinant load by fragment")
        ax.set_ylabel("selected determinants")
        ax.set_xticks(range(len(result_rows)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.legend(ncol=4, fontsize=7)
        path = plot_dir / "04_fragment_determinants.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(str(path))

    with open(root / "plot_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(written, fp, indent=2)
        fp.write("\n")
    return written


def run_lasscf_sweep(
    *,
    fcidump_path: str,
    dets_path: str,
    output_dir: str,
    target_count: int = 24,
    max_fragment_size: int | None = None,
    execute: bool = False,
    trimci_thresholds: Iterable[float] = (0.06,),
    max_cycle_macro: int = 20,
    trimci_max_dets="auto",
    trimci_max_rounds: int = 2,
    stop_on_error: bool = False,
) -> dict:
    from FRASCI.lasscf.runners.run_lasscf_trimci import run as run_one

    os.makedirs(output_dir, exist_ok=True)
    problem = read_fcidump_problem(fcidump_path, dets_path)
    candidates = build_fragmentation_candidates(
        problem,
        target_count=target_count,
        max_fragment_size=max_fragment_size,
    )
    write_candidate_files(candidates, output_dir)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fcidump": fcidump_path,
        "dets": dets_path,
        "execute": execute,
        "target_count": target_count,
        "n_candidates": len(candidates),
        "thresholds": list(trimci_thresholds),
        "runs": [],
    }

    if execute:
        runs_dir = Path(output_dir) / "runs"
        runs_dir.mkdir(exist_ok=True)
        for candidate in candidates:
            for threshold in trimci_thresholds:
                thr_label = f"{threshold:.6f}".rstrip("0").rstrip(".")
                run_dir = runs_dir / f"{candidate.name}__thr{thr_label}"
                existing = load_completed_result(run_dir)
                if existing is not None:
                    summary["runs"].append(
                        {
                            "candidate": candidate.name,
                            "threshold": threshold,
                            "status": "RESUMED_EXISTING",
                            "output_dir": str(run_dir),
                        }
                    )
                    continue
                try:
                    result = run_one(
                        fcidump_path=fcidump_path,
                        partition=candidate.name,
                        trimci_threshold=float(threshold),
                        max_cycle_macro=max_cycle_macro,
                        output_dir=str(run_dir),
                        trimci_max_dets=trimci_max_dets,
                        trimci_max_rounds=trimci_max_rounds,
                        explicit_orbital_lists=candidate.fragments,
                        partition_description=candidate.description,
                    )
                    summary["runs"].append(
                        {
                            "candidate": candidate.name,
                            "threshold": threshold,
                            "status": result.get("status"),
                            "e_tot": result.get("e_tot"),
                            "converged": result.get("converged"),
                            "output_dir": str(run_dir),
                        }
                    )
                except Exception as exc:
                    row = {
                        "candidate": candidate.name,
                        "threshold": threshold,
                        "status": "FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "output_dir": str(run_dir),
                    }
                    summary["runs"].append(row)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    with open(run_dir / "failure.json", "w", encoding="utf-8") as fp:
                        json.dump(row, fp, indent=2)
                        fp.write("\n")
                    if stop_on_error:
                        raise

    plot_paths = plot_sweep(output_dir)
    summary["plots"] = plot_paths
    with open(Path(output_dir) / "sweep_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
        fp.write("\n")
    return summary
