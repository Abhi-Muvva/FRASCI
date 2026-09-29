"""Public, FRASCI-native input metadata for FCIDUMP workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from frasci._mock_scf import build_mock_scf_from_fcidump


@dataclass(frozen=True)
class FCIDUMPInput:
    """The fixed Hamiltonian metadata consumed by ``frasci.recovery.solve``.

    ``from_file`` reads the FCIDUMP header and integrals through PySCF.  Direct
    construction remains available when a caller already has validated metadata.
    """

    fcidump_path: Path
    n_orb: int
    n_alpha: int
    n_beta: int

    def __post_init__(self):
        path = Path(self.fcidump_path)
        if self.n_orb < 1:
            raise ValueError(f"n_orb must be positive, got {self.n_orb}")
        if min(self.n_alpha, self.n_beta) < 0:
            raise ValueError("electron counts must be nonnegative")
        if max(self.n_alpha, self.n_beta) > self.n_orb:
            raise ValueError("spin-resolved electron counts cannot exceed n_orb")
        object.__setattr__(self, "fcidump_path", path)

    @classmethod
    def from_file(
        cls, fcidump_path: str | Path, *, max_memory_mb: int | None = None
    ) -> "FCIDUMPInput":
        """Build validated recovery metadata from an existing FCIDUMP file."""
        path = Path(fcidump_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        mf = build_mock_scf_from_fcidump(str(path), max_memory_mb=max_memory_mb)
        n_orb = int(getattr(mf.mol, "_nao", 0) or mf.get_hcore().shape[0])
        n_alpha, n_beta = map(int, mf.mol.nelec)
        return cls(path, n_orb, n_alpha, n_beta)
