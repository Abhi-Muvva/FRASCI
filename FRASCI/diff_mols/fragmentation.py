"""Fragmentation resolution for diff_mols.

Supports:
  - explicit list[list[int]] (passthrough)
  - "auto_per_atom" — IAO populations group MOs by their dominant atom
  - "auto_per_metal" — like auto_per_atom but only metal atoms get their own fragment;
    non-metal MOs append to the nearest metal's fragment
  - "chem_bond" — IAO populations summed over user-specified atom groups; each group
    becomes one fragment (keeps multiple bonds like N=N intact inside one fragment,
    matching the Hermes & Gagliardi 2019 LASSCF fragmentation for azomethane / diazene)
  - "h1diag" — delegate to FRASCI.lasscf.fragments.h1diag_fragments
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from FRASCI.diff_mols.config import MoleculeConfig
from FRASCI.diff_mols.integrals_builder import IntegralBundle


_METALS = set(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "La Hf Ta W Re Os Ir Pt Au Hg".split()
)


@dataclass
class FragmentPartition:
    name: str
    orbital_lists: list[list[int]]
    nelec_per_frag: list[tuple[int, int]]
    spin_sub: list[int]                    # multiplicities (2S+1)
    description: str = ""


def _load_ref_det_or_aufbau(bundle: IntegralBundle) -> tuple[int, int]:
    """Return (alpha_bits, beta_bits) of the reference det.

    Reads dets.npz row 0 if present; else falls back to Aufbau.
    """
    dets_path = bundle.fcidump_path.parent / "dets.npz"
    if dets_path.exists():
        data = np.load(dets_path)
        return int(data["dets"][0, 0]), int(data["dets"][0, 1])
    a = (1 << bundle.n_alpha) - 1
    b = (1 << bundle.n_beta) - 1
    return a, b


def _fragment_electron_count(alpha_bits: int, beta_bits: int, frag: list[int]) -> tuple[int, int]:
    na = sum(1 for p in frag if (alpha_bits >> p) & 1)
    nb = sum(1 for p in frag if (beta_bits >> p) & 1)
    return na, nb


def _build_pyscf_mol_from_bundle(config: MoleculeConfig, bundle: IntegralBundle) -> "gto.Mole":
    from pyscf import gto
    xyz = bundle.xyz_path.read_text()
    return gto.M(
        atom=xyz, basis=config.electronic_structure.basis,
        charge=config.electronic_structure.charge,
        spin=config.electronic_structure.spin,
        unit="Angstrom", verbose=0,
    )


def _iao_populations(mol, mo_coeff_active: np.ndarray) -> np.ndarray:
    """Return populations[atom, mo_active] using IAO basis projection."""
    from pyscf import lo
    iao = lo.iao.iao(mol, mo_coeff_active)        # AO × IAO
    # Project active MOs onto IAOs
    s = mol.intor_symmetric("int1e_ovlp")
    proj = iao.T @ s @ mo_coeff_active             # IAO × MO
    pop_per_iao = proj ** 2                        # contribution of each IAO to each MO
    # IAO labels (mol.ao_labels-style); fall back to using IAO grouping by atom of associated AO
    # iao columns correspond to minimal-basis IAOs of each atom; lo.iao.iao retains atom block structure
    ao_atom_idx = []
    iao_labels = mol.ao_labels(fmt=False)          # list[(atom_id, sym, l, m)]
    n_iao = iao.shape[1]
    # IAO basis is the minimal AO set; map by ao_labels of the minimal basis
    # As a robust fallback: assign each IAO column to atom by argmax of |iao|^2 per AO×IAO column
    iao2atom = np.zeros(n_iao, dtype=int)
    for j in range(n_iao):
        col = iao[:, j] ** 2
        # majority atom
        per_atom = np.zeros(mol.natm)
        for ao_idx, (atm_id, _, _, _) in enumerate(iao_labels[:iao.shape[0]]):
            per_atom[atm_id] += col[ao_idx]
        iao2atom[j] = int(np.argmax(per_atom))

    n_atoms = mol.natm
    n_mo = mo_coeff_active.shape[1]
    pop = np.zeros((n_atoms, n_mo))
    for j in range(n_iao):
        pop[iao2atom[j], :] += pop_per_iao[j, :]
    return pop


def _auto_per_atom(config: MoleculeConfig, bundle: IntegralBundle, metals_only: bool) -> list[list[int]]:
    mol = _build_pyscf_mol_from_bundle(config, bundle)
    mo_full = np.load(bundle.mo_coeff_path)["mo_coeff"]
    mo_active = mo_full[:, : bundle.n_orb]
    # MOs are already Boys-localized by build_integrals when auto fragmentation is
    # requested; no second localization step here.
    pop = _iao_populations(mol, mo_active)             # (n_atom, n_mo_active)

    atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    if metals_only:
        metal_atoms = [i for i, s in enumerate(atom_symbols) if s in _METALS]
        if not metal_atoms:
            raise ValueError("auto_per_metal: no metal atoms found in molecule")
        # Each active MO → nearest metal atom by population (restricting pop matrix to metals)
        assignment = np.argmax(pop[metal_atoms, :], axis=0)
        # Build orbital_lists indexed by metal_atoms order
        frag_for_mo = [metal_atoms[a] for a in assignment]
    else:
        assignment = np.argmax(pop, axis=0)
        frag_for_mo = list(assignment)

    # Group MOs by atom, preserving atom order
    atom_to_mos: dict[int, list[int]] = {}
    for mo_idx, atm in enumerate(frag_for_mo):
        atom_to_mos.setdefault(int(atm), []).append(mo_idx)
    # Order fragments by atom index for determinism
    return [sorted(atom_to_mos[a]) for a in sorted(atom_to_mos.keys()) if atom_to_mos[a]]


def _validate_atom_groups(atom_groups: list[list[int]], n_atoms: int) -> None:
    if not atom_groups:
        raise ValueError("chem_bond: atom_groups must be a non-empty list of lists")
    seen: set[int] = set()
    for g_idx, group in enumerate(atom_groups):
        if not group:
            raise ValueError(f"chem_bond: atom_groups[{g_idx}] is empty")
        for a in group:
            if not (0 <= a < n_atoms):
                raise ValueError(
                    f"chem_bond: atom_groups[{g_idx}] contains out-of-range atom index {a} "
                    f"(molecule has {n_atoms} atoms, 0-indexed)"
                )
            if a in seen:
                raise ValueError(
                    f"chem_bond: atom {a} appears in multiple groups (groups must be disjoint)"
                )
            seen.add(a)
    missing = sorted(set(range(n_atoms)) - seen)
    if missing:
        raise ValueError(
            f"chem_bond: atoms {missing} are not covered by any group; "
            f"every atom must be assigned to exactly one group"
        )


def _chem_bond_groups(config: MoleculeConfig, bundle: IntegralBundle,
                      atom_groups: list[list[int]]) -> list[list[int]]:
    """Assign each active MO to the atom-group with the largest summed IAO population.

    Returns one fragment per group, in the order groups were given. Empty fragments
    are kept as [] in the output to preserve group order (downstream callers can
    filter if needed); a warning is emitted via description text by the caller.
    """
    mol = _build_pyscf_mol_from_bundle(config, bundle)
    _validate_atom_groups(atom_groups, mol.natm)

    mo_full = np.load(bundle.mo_coeff_path)["mo_coeff"]
    mo_active = mo_full[:, : bundle.n_orb]
    pop = _iao_populations(mol, mo_active)             # (n_atom, n_mo_active)

    # Sum populations over each group → (n_groups, n_mo_active)
    group_pop = np.stack([pop[g, :].sum(axis=0) for g in atom_groups], axis=0)
    assignment = np.argmax(group_pop, axis=0)          # (n_mo_active,) → group index

    orbital_lists: list[list[int]] = [[] for _ in atom_groups]
    for mo_idx, g_idx in enumerate(assignment):
        orbital_lists[int(g_idx)].append(int(mo_idx))
    return [sorted(f) for f in orbital_lists]


def _h1diag_partition(bundle: IntegralBundle, n_frags: int = 2):
    """h1diag-style partition local to diff_mols (independent of mfa/solver's hardcoded 3).

    Sorts active orbitals by h1 diagonal energy, splits into ``n_frags`` equal-sized groups.
    Defaults to ``n_frags=2`` because all seven diff_mols molecules are dimer-like.

    For Fe4S4 (n_orb=36, n_frags=3) the existing ``mfa.solver.make_nonoverlapping_partition``
    is the right tool — but it's hardcoded to 3 frags and lives in a read-only module, so
    diff_mols re-implements the trivial sort+split here.

    Returns ``(orbital_lists, nelec_per_frag, spin_sub)`` — same shape as the existing
    ``lasscf.fragments.h1diag_fragments``.
    """
    import trimci

    if bundle.n_orb % n_frags != 0:
        raise ValueError(
            f"h1diag partition: n_orb={bundle.n_orb} not divisible by n_frags={n_frags}. "
            f"For odd or prime active-space sizes, use 'auto_per_atom' instead."
        )

    h1, _eri, _ne, n_orb, _e_nuc, _na, _nb, _psym = trimci.read_fcidump(str(bundle.fcidump_path))
    order = np.argsort(np.diag(h1))
    per_frag = n_orb // n_frags
    orbital_lists = [sorted(int(x) for x in order[i*per_frag:(i+1)*per_frag])
                     for i in range(n_frags)]

    alpha_bits, beta_bits = _load_ref_det_or_aufbau(bundle)
    nelec_per_frag = [_fragment_electron_count(alpha_bits, beta_bits, frag)
                      for frag in orbital_lists]
    spin_sub = _spin_sub_from_nelec(nelec_per_frag)
    return orbital_lists, nelec_per_frag, spin_sub


def _spin_sub_from_nelec(nelec_per_frag: list[tuple[int, int]]) -> list[int]:
    return [abs(na - nb) + 1 for na, nb in nelec_per_frag]


def resolve_fragmentation(config: MoleculeConfig, bundle: IntegralBundle) -> dict[str, FragmentPartition]:
    """Resolve every config.fragmentation entry into a FragmentPartition."""
    alpha_bits, beta_bits = _load_ref_det_or_aufbau(bundle)
    out: dict[str, FragmentPartition] = {}

    for fspec in config.fragmentation:
        ol = fspec.orbital_lists

        if isinstance(ol, list):                     # explicit
            orbital_lists = [list(map(int, frag)) for frag in ol]
            nelec_per_frag = [_fragment_electron_count(alpha_bits, beta_bits, frag) for frag in orbital_lists]
            spin_sub = _spin_sub_from_nelec(nelec_per_frag)
            out[fspec.name] = FragmentPartition(
                name=fspec.name, orbital_lists=orbital_lists,
                nelec_per_frag=nelec_per_frag, spin_sub=spin_sub,
                description="explicit",
            )

        elif ol == "auto_per_atom":
            orbital_lists = _auto_per_atom(config, bundle, metals_only=False)
            nelec_per_frag = [_fragment_electron_count(alpha_bits, beta_bits, frag) for frag in orbital_lists]
            spin_sub = _spin_sub_from_nelec(nelec_per_frag)
            out[fspec.name] = FragmentPartition(
                name=fspec.name, orbital_lists=orbital_lists,
                nelec_per_frag=nelec_per_frag, spin_sub=spin_sub,
                description="auto_per_atom (IAO populations)",
            )

        elif ol == "auto_per_metal":
            orbital_lists = _auto_per_atom(config, bundle, metals_only=True)
            nelec_per_frag = [_fragment_electron_count(alpha_bits, beta_bits, frag) for frag in orbital_lists]
            spin_sub = _spin_sub_from_nelec(nelec_per_frag)
            out[fspec.name] = FragmentPartition(
                name=fspec.name, orbital_lists=orbital_lists,
                nelec_per_frag=nelec_per_frag, spin_sub=spin_sub,
                description="auto_per_metal (IAO populations, metals only)",
            )

        elif ol == "chem_bond":
            if fspec.atom_groups is None:
                raise ValueError(
                    f"fragmentation '{fspec.name}': orbital_lists='chem_bond' requires "
                    f"atom_groups (list of 0-indexed atom-index groups)"
                )
            raw_lists = _chem_bond_groups(config, bundle, fspec.atom_groups)
            # Drop empty groups but record which groups produced no MOs in description
            empty_idx = [i for i, f in enumerate(raw_lists) if not f]
            orbital_lists = [f for f in raw_lists if f]
            nelec_per_frag = [_fragment_electron_count(alpha_bits, beta_bits, frag) for frag in orbital_lists]
            spin_sub = _spin_sub_from_nelec(nelec_per_frag)
            desc = "chem_bond (IAO populations summed over atom_groups)"
            if empty_idx:
                desc += f"; groups {empty_idx} produced no MOs and were dropped"
            out[fspec.name] = FragmentPartition(
                name=fspec.name, orbital_lists=orbital_lists,
                nelec_per_frag=nelec_per_frag, spin_sub=spin_sub,
                description=desc,
            )

        elif ol == "h1diag":
            n_frags = fspec.n_frags if fspec.n_frags is not None else 2
            orbital_lists, nelec_per_frag, spin_sub = _h1diag_partition(bundle, n_frags=n_frags)
            out[fspec.name] = FragmentPartition(
                name=fspec.name,
                orbital_lists=[list(map(int, f)) for f in orbital_lists],
                nelec_per_frag=[tuple(map(int, ne)) for ne in nelec_per_frag],
                spin_sub=list(map(int, spin_sub)),
                description=f"h1diag ({n_frags}-fragment split by h1 diagonal, diff_mols-local)",
            )

        else:
            raise ValueError(f"Unknown orbital_lists value: {ol!r}")

    return out
