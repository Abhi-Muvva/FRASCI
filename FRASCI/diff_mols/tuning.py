"""Reusable tuning presets for diff_mols benchmark runs."""
from __future__ import annotations

from copy import deepcopy


TUNING_PRESETS: dict[str, dict] = {
    "default": {},
    "fast": {
        "threshold": 0.02,
        "max_dets": 1000,
        "num_runs": 2,
        "max_rounds": 1,
        "trimci_threshold": 0.02,
        "trimci_max_dets": 1000,
        "coo_cycles": 2,
        "bfgs_maxiter": 15,
        "max_cycle_macro": 30,
    },
    "tight": {
        "threshold": 0.005,
        "max_dets": 4000,
        "num_runs": 6,
        "max_rounds": 3,
        "trimci_threshold": 0.005,
        "trimci_max_dets": 4000,
        "coo_cycles": 4,
        "bfgs_maxiter": 50,
        "bfgs_ftol": 1.0e-9,
        "davidson_tol": 1.0e-8,
        "max_cycle_macro": 100,
    },
    "coo_cyc2_stable": {
        "trimci_threshold": 0.01,
        "trimci_max_dets": 2000,
        "trimci_max_rounds": 4,
        "coo_cycles": 2,
        "bfgs_maxiter": 40,
        "bfgs_ftol": 1.0e-8,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
        "max_cycle_macro": 100,
    },
    "aggressive": {
        "threshold": 0.002,
        "max_dets": 8000,
        "num_runs": 8,
        "max_rounds": 4,
        "trimci_threshold": 0.002,
        "trimci_max_dets": 8000,
        "coo_cycles": 6,
        "bfgs_maxiter": 80,
        "bfgs_ftol": 1.0e-10,
        "davidson_tol": 1.0e-9,
        "max_cycle_macro": 150,
    },
}


TRIMCI_PROTOCOLS: dict[str, dict] = {
    "dets50": {"max_dets": 50, "threshold": 0.01, "max_rounds": 2},
    "dets100": {"max_dets": 100, "threshold": 0.01, "max_rounds": 2},
    "dets250": {"max_dets": 250, "threshold": 0.01, "max_rounds": 2},
    "dets500": {"max_dets": 500, "threshold": 0.01, "max_rounds": 2},
    "dets1000": {"max_dets": 1000, "threshold": 0.01, "max_rounds": 2},
    "dets2000": {"max_dets": 2000, "threshold": 0.01, "max_rounds": 2},
    "thr005_dets1000": {"max_dets": 1000, "threshold": 0.005, "max_rounds": 3},
    "thr005_dets2000": {"max_dets": 2000, "threshold": 0.005, "max_rounds": 3},
}


LASSCF_TRIMCI_PROTOCOLS: dict[str, dict] = {
    name: {
        "trimci_threshold": cfg["threshold"],
        "trimci_max_dets": cfg["max_dets"],
        "trimci_max_rounds": cfg["max_rounds"],
    }
    for name, cfg in TRIMCI_PROTOCOLS.items()
}


COO_GRID_DETS = (50, 100, 250, 500, 1000, 2000)
COO_GRID_CYCLES = (1, 2, 4)
COO_GRID_BFGS = (20, 40, 80)


TRIMCI_COO_PROTOCOLS: dict[str, dict] = {
    f"cyc{cyc}_dets{dets}_bfgs{bfgs}": {
        "max_dets": dets,
        "threshold": 0.01,
        "coo_cycles": cyc,
        "bfgs_maxiter": bfgs,
    }
    for dets in COO_GRID_DETS
    for cyc in COO_GRID_CYCLES
    for bfgs in COO_GRID_BFGS
}
TRIMCI_COO_PROTOCOLS.update({
    "thr005_cyc2_dets1000_bfgs40": {
        "max_dets": 1000, "threshold": 0.005, "coo_cycles": 2, "bfgs_maxiter": 40,
    },
    "thr005_cyc2_dets2000_bfgs40": {
        "max_dets": 2000, "threshold": 0.005, "coo_cycles": 2, "bfgs_maxiter": 40,
    },
})


LASSCF_TRIMCI_COO_PROTOCOLS: dict[str, dict] = {
    f"cyc{cyc}_dets{dets}_bfgs{bfgs}": {
        "coo_cycles": cyc,
        "trimci_max_rounds": 4 if dets >= 1000 or cyc > 1 else 2,
        "trimci_max_dets": dets,
        "bfgs_maxiter": bfgs,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    }
    for dets in COO_GRID_DETS
    for cyc in COO_GRID_CYCLES
    for bfgs in COO_GRID_BFGS
}
LASSCF_TRIMCI_COO_PROTOCOLS.update({
    "cyc2_stable": {
        "coo_cycles": 2,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 2000,
        "bfgs_maxiter": 40,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    },
    "cyc2_dets4k": {
        "coo_cycles": 2,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 4000,
        "bfgs_maxiter": 40,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    },
    "cyc2_thr005": {
        "coo_cycles": 2,
        "trimci_threshold": 0.005,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 4000,
        "bfgs_maxiter": 40,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    },
    "cyc2_bfgs80": {
        "coo_cycles": 2,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 4000,
        "bfgs_maxiter": 80,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    },
    "cyc2_warmkappa": {
        "coo_cycles": 2,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 4000,
        "bfgs_maxiter": 40,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": True,
    },
    "thr005_cyc2_dets1000_bfgs40": {
        "coo_cycles": 2,
        "trimci_threshold": 0.005,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 1000,
        "bfgs_maxiter": 40,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    },
    "thr005_cyc2_dets2000_bfgs40": {
        "coo_cycles": 2,
        "trimci_threshold": 0.005,
        "trimci_max_rounds": 4,
        "trimci_max_dets": 2000,
        "bfgs_maxiter": 40,
        "davidson_tol": 1.0e-9,
        "warm_start_kappa": False,
    },
})


LASSIS_PROTOCOLS: dict[str, dict] = {
    "default": {},
    "nspin0_opt1": {"lassis_nspin": 0, "opt": 1},
    "nspin1_opt1": {"lassis_nspin": 1, "opt": 1},
    "nspin2_opt1": {"lassis_nspin": 2, "opt": 1},
    "nspin3_opt1": {"lassis_nspin": 3, "opt": 1},
}


PROTOCOL_GROUPS: dict[str, dict[str, list[str]]] = {
    "plain_trimci": {
        "@dets_grid": ["dets50", "dets100", "dets250", "dets500", "dets1000", "dets2000"],
        "@dets_threshold_grid": [
            "dets50", "dets100", "dets250", "dets500", "dets1000", "dets2000",
            "thr005_dets1000", "thr005_dets2000",
        ],
    },
    "trimci_coo": {
        "@coo_grid": [
            f"cyc{cyc}_dets{dets}_bfgs{bfgs}"
            for dets in COO_GRID_DETS
            for cyc in COO_GRID_CYCLES
            for bfgs in COO_GRID_BFGS
        ],
        "@coo_grid_plus_thr005": [
            f"cyc{cyc}_dets{dets}_bfgs{bfgs}"
            for dets in COO_GRID_DETS
            for cyc in COO_GRID_CYCLES
            for bfgs in COO_GRID_BFGS
        ] + ["thr005_cyc2_dets1000_bfgs40", "thr005_cyc2_dets2000_bfgs40"],
    },
    "lasscf_trimci_coo": {
        "@lasscf_coo_grid": [
            f"cyc{cyc}_dets{dets}_bfgs{bfgs}"
            for dets in COO_GRID_DETS
            for cyc in COO_GRID_CYCLES
            for bfgs in COO_GRID_BFGS
        ],
        "@lasscf_coo_grid_plus_thr005": [
            f"cyc{cyc}_dets{dets}_bfgs{bfgs}"
            for dets in COO_GRID_DETS
            for cyc in COO_GRID_CYCLES
            for bfgs in COO_GRID_BFGS
        ] + ["thr005_cyc2_dets1000_bfgs40", "thr005_cyc2_dets2000_bfgs40"],
    },
    "lasscf_trimci": {
        "@lasscf_trimci_grid": ["dets50", "dets100", "dets250", "dets500", "dets1000", "dets2000"],
        "@lasscf_trimci_threshold_grid": [
            "dets50", "dets100", "dets250", "dets500", "dets1000", "dets2000",
            "thr005_dets1000", "thr005_dets2000",
        ],
    },
}


def expand_protocol_names(method: str, names: list[str] | None) -> list[str]:
    """Expand YAML/CLI protocol group names like ``@coo_grid`` into protocol names."""
    out: list[str] = []
    groups = PROTOCOL_GROUPS.get(method, {})
    for name in names or []:
        if name in groups:
            out.extend(groups[name])
        else:
            out.append(name)
    return out


def resolve_tuning_overrides(preset: str | None, manual: dict | None = None) -> dict:
    """Return overrides for a named preset, with manual values taking precedence."""
    name = preset or "default"
    if name not in TUNING_PRESETS:
        choices = ", ".join(sorted(TUNING_PRESETS))
        raise ValueError(f"unknown tuning preset {name!r}; choose one of: {choices}")
    merged = deepcopy(TUNING_PRESETS[name])
    if manual:
        merged.update(manual)
    return merged


def resolve_lasscf_trimci_coo_protocol(protocol: str, base: dict | None = None) -> dict:
    """Return overrides for one named LASSCF+TrimCI+COO protocol.

    ``base`` is applied first so notebook/CLI overrides can set shared knobs
    such as process-worker counts, while the protocol itself controls the
    physics/optimizer choices being compared.
    """
    if protocol not in LASSCF_TRIMCI_COO_PROTOCOLS:
        choices = ", ".join(sorted(LASSCF_TRIMCI_COO_PROTOCOLS))
        raise ValueError(f"unknown LASSCF+TrimCI+COO protocol {protocol!r}; choose one of: {choices}")
    merged = deepcopy(base or {})
    merged.update(deepcopy(LASSCF_TRIMCI_COO_PROTOCOLS[protocol]))
    return merged


def resolve_protocol(registry: dict[str, dict], registry_name: str,
                     protocol: str, base: dict | None = None) -> dict:
    """Return merged overrides for one named protocol from a registry."""
    if protocol not in registry:
        choices = ", ".join(sorted(registry))
        raise ValueError(f"unknown {registry_name} protocol {protocol!r}; choose one of: {choices}")
    merged = deepcopy(base or {})
    merged.update(deepcopy(registry[protocol]))
    return merged


def resolve_trimci_protocol(protocol: str, base: dict | None = None) -> dict:
    return resolve_protocol(TRIMCI_PROTOCOLS, "TrimCI", protocol, base)


def resolve_lasscf_trimci_protocol(protocol: str, base: dict | None = None) -> dict:
    return resolve_protocol(LASSCF_TRIMCI_PROTOCOLS, "LASSCF+TrimCI", protocol, base)


def resolve_trimci_coo_protocol(protocol: str, base: dict | None = None) -> dict:
    return resolve_protocol(TRIMCI_COO_PROTOCOLS, "TrimCI+COO", protocol, base)


def resolve_lassis_protocol(protocol: str, base: dict | None = None) -> dict:
    return resolve_protocol(LASSIS_PROTOCOLS, "LASSIS", protocol, base)
