"""PES bump-trigger: evaluate when a coarse scan is ready for a 21-point bump."""
from __future__ import annotations

import pandas as pd


COARSE_GRID = (1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00)
MEDIUM_GRID = tuple(round(0.9 + 0.1*i, 2) for i in range(21))


def check_pes_bump_trigger(
    df: pd.DataFrame,
    *,
    error_threshold_mha: float = 5.0,
    monotonicity: bool = True,
    method: str = "lasscf_trimci_coo",
) -> dict:
    """Evaluate the 3-condition trigger from spec §5.3.

    Returns `{"ready": bool, "reasons": list[str], "medium_grid": list[float]}`.
    """
    reasons: list[str] = []
    if df.empty:
        return {"ready": False, "reasons": ["no runs found"], "medium_grid": list(MEDIUM_GRID)}

    sub = df[df["method"] == method].copy()
    if sub.empty:
        return {"ready": False, "reasons": [f"no {method} runs found"],
                "medium_grid": list(MEDIUM_GRID)}

    expected_tags = [f"r{r:.2f}" for r in COARSE_GRID]
    present_tags = set(sub["geom_tag"])
    missing = [t for t in expected_tags if t not in present_tags]
    if missing:
        reasons.append(f"missing geometry: {missing}")

    unconverged = sub[sub["converged"].astype(str).str.lower().isin(("false", "0"))]["geom_tag"].tolist()
    if unconverged:
        reasons.append(f"did not converge: {unconverged}")

    error_max = pd.to_numeric(sub["error_mha"], errors="coerce").abs().max()
    if pd.notna(error_max) and error_max > error_threshold_mha:
        reasons.append(f"max error {error_max:.2f} mHa > {error_threshold_mha} mHa")

    if monotonicity and not missing:
        ordered = sub.sort_values("geom_tag")
        es = pd.to_numeric(ordered["e_tot"], errors="coerce").tolist()
        # Allow one descent → ascent flip (avoided crossing); flag two or more
        sign_changes = sum(
            1 for i in range(1, len(es) - 1)
            if (es[i] - es[i-1]) * (es[i+1] - es[i]) < 0
        )
        if sign_changes >= 2:
            reasons.append(f"non-monotone PES: {sign_changes} sign changes in dE/dr")

    return {"ready": not reasons, "reasons": reasons, "medium_grid": list(MEDIUM_GRID)}
