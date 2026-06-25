"""Tuning preset and LASSCF+TrimCI+COO protocol helpers."""

from FRASCI.diff_mols.tuning import (
    LASSCF_TRIMCI_COO_PROTOCOLS, LASSCF_TRIMCI_PROTOCOLS, LASSIS_PROTOCOLS, TRIMCI_COO_PROTOCOLS,
    TRIMCI_PROTOCOLS,
    expand_protocol_names,
    resolve_lasscf_trimci_coo_protocol,
    resolve_lasscf_trimci_protocol,
)


def test_lasscf_trimci_coo_protocols_cover_planned_sweep():
    assert {
        "cyc2_stable",
        "cyc2_dets4k",
        "cyc2_thr005",
        "cyc2_bfgs80",
        "cyc2_warmkappa",
    }.issubset(LASSCF_TRIMCI_COO_PROTOCOLS)


def test_protocol_registries_cover_broad_matrix_axes():
    assert {"dets50", "dets500", "dets2000"}.issubset(TRIMCI_PROTOCOLS)
    assert {
        "cyc1_dets50_bfgs20",
        "cyc1_dets2000_bfgs80",
        "cyc2_dets2000_bfgs40",
        "cyc4_dets50_bfgs20",
        "cyc4_dets2000_bfgs80",
    }.issubset(TRIMCI_COO_PROTOCOLS)
    assert len(expand_protocol_names("trimci_coo", ["@coo_grid"])) == 54
    assert len(expand_protocol_names("lasscf_trimci", ["@lasscf_trimci_grid"])) == 6
    assert len(expand_protocol_names("lasscf_trimci_coo", ["@lasscf_coo_grid"])) == 54
    assert {"nspin0_opt1", "nspin1_opt1", "nspin2_opt1", "nspin3_opt1"}.issubset(LASSIS_PROTOCOLS)
    assert {"dets50", "dets1000", "dets2000"}.issubset(LASSCF_TRIMCI_PROTOCOLS)


def test_lasscf_trimci_protocol_maps_trimci_knobs():
    out = resolve_lasscf_trimci_protocol("dets500", {"max_cycle_macro": 20})
    assert out["max_cycle_macro"] == 20
    assert out["trimci_threshold"] == 0.01
    assert out["trimci_max_dets"] == 500
    assert out["trimci_max_rounds"] == 2


def test_lasscf_trimci_coo_protocol_overlays_shared_overrides():
    out = resolve_lasscf_trimci_coo_protocol(
        "cyc2_dets4k",
        {"process_workers": 4, "parallel_workers": 4, "trimci_max_dets": 123},
    )
    assert out["process_workers"] == 4
    assert out["parallel_workers"] == 4
    assert out["coo_cycles"] == 2
    assert out["trimci_max_rounds"] == 4
    assert out["trimci_max_dets"] == 4000
