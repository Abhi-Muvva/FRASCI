from __future__ import annotations

from typing import Any

__all__ = ["run_cross_coupled_solver"]


def __getattr__(name: str) -> Any:
    if name == "run_cross_coupled_solver":
        from FRASCI.crossflow.solver import run_cross_coupled_solver

        return run_cross_coupled_solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
