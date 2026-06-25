"""PES bump-trigger logic."""
import pandas as pd
import pytest

from FRASCI.diff_mols.bump_trigger import check_pes_bump_trigger


_COARSE = [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00]
_MEDIUM = [0.9 + 0.1*i for i in range(21)]


def _make_df(method_results: dict[float, tuple[float, float, bool]]) -> pd.DataFrame:
    """method_results: r -> (e_tot, error_mha, converged) for lasscf_trimci_coo."""
    rows = []
    for r, (e, err, conv) in method_results.items():
        rows.append({
            "molecule": "diazene_trans", "method": "lasscf_trimci_coo",
            "partition": "chem", "geom_tag": f"r{r:.2f}", "tag": "default",
            "e_tot": e, "e_ref": e - err / 1000.0, "error_mha": err,
            "converged": conv,
        })
    return pd.DataFrame(rows)


def test_ready_when_all_geoms_converged_and_error_under_5mha():
    df = _make_df({r: (-1.0 - 0.001*r, 3.0, True) for r in _COARSE})
    res = check_pes_bump_trigger(df)
    assert res["ready"] is True
    assert res["medium_grid"] == [round(x, 2) for x in _MEDIUM]


def test_not_ready_when_missing_geometry():
    df = _make_df({r: (-1.0, 3.0, True) for r in _COARSE[:-1]})  # drop last
    res = check_pes_bump_trigger(df)
    assert res["ready"] is False
    assert any("missing geometry" in r.lower() for r in res["reasons"])


def test_not_ready_when_error_too_high():
    df = _make_df({r: (-1.0, 10.0 if r == 1.50 else 2.0, True) for r in _COARSE})
    res = check_pes_bump_trigger(df)
    assert res["ready"] is False
    assert any("error" in r.lower() for r in res["reasons"])


def test_not_ready_when_unconverged():
    df = _make_df({r: (-1.0, 2.0, r != 2.00) for r in _COARSE})
    res = check_pes_bump_trigger(df)
    assert res["ready"] is False
    assert any("converge" in r.lower() for r in res["reasons"])
