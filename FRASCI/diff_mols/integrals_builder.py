"""PySCF → FCIDUMP integrals builder for diff_mols.

Single-point + scan geometries; explicit / AVAS / window active-space; optional CASSCF.
Writes fcidump, mo_coeff.npz, scf_summary.json, xyz, dets.npz (Aufbau seed), inputs_hash.txt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from jinja2 import Template
from pyscf import gto, scf, ao2mo, tools

from FRASCI.diff_mols.config import MoleculeConfig, geom_tag_for_single, geom_tag_for_scan_point


@dataclass
class IntegralBundle:
    fcidump_path: Path
    mo_coeff_path: Path
    scf_summary_path: Path
    xyz_path: Path
    n_orb: int
    n_elec: int
    n_alpha: int
    n_beta: int
    e_nuc: float
    e_hf: float
    inputs_hash: str
    output_dir: Path
    geom_tag: str = ""        # the per-geom label ("eq", "r1.00", ...); set by build_integrals


# ---------------------------------------------------------------------------
# Geometry rendering (single only in this task; scan added in Task 5)
# ---------------------------------------------------------------------------

def _render_geometry(config: MoleculeConfig, geom_tag: str) -> str:
    g = config.geometry
    if g.kind == "single":
        if g.xyz_inline:
            xyz_text = g.xyz_inline
        elif g.xyz_file:
            base = config.source_path.parent if config.source_path else Path.cwd()
            xyz_text = (base / g.xyz_file).read_text()
        else:
            raise ValueError(f"{config.slug}: single geometry needs xyz_inline or xyz_file")

        # Parse XYZ format: strip count header line if present
        lines = xyz_text.strip().split('\n')
        if lines and lines[0].isdigit():
            # First line is atom count — skip it and the optional comment line
            lines = lines[2:] if len(lines) > 2 else lines[1:]
        return '\n'.join(lines).strip()
    if g.kind == "scan":
        if g.template_engine != "jinja2":
            raise ValueError(f"{config.slug}: only template_engine=jinja2 supported, got {g.template_engine!r}")
        # Find the scan point matching this geom_tag
        for pt in g.scan_points:
            if geom_tag_for_scan_point(pt, g.scan_param) == geom_tag:
                tmpl = Template(g.template)
                xyz_text = tmpl.render(**pt)
                # Parse XYZ format: strip count header line if present
                lines = xyz_text.strip().split('\n')
                if lines and lines[0].isdigit():
                    # First line is atom count — skip it and the optional comment line
                    lines = lines[2:] if len(lines) > 2 else lines[1:]
                return '\n'.join(lines).strip()
        raise ValueError(f"{config.slug}: geom_tag {geom_tag!r} not found in scan_points")
    raise ValueError(f"Unknown geometry.kind: {g.kind!r}")


# ---------------------------------------------------------------------------
# Hash for skip-when-unchanged
# ---------------------------------------------------------------------------

_INTEGRALS_BUILDER_VERSION = "v3_inline_fci_ref_2026-06-23"   # bump when the builder semantics change


def _inputs_hash(config: MoleculeConfig, geom_tag: str, xyz: str) -> str:
    h = hashlib.sha256()
    h.update(xyz.encode())
    h.update(_INTEGRALS_BUILDER_VERSION.encode())          # cache-bust on builder fixes
    h.update(json.dumps({
        "charge": config.electronic_structure.charge,
        "spin": config.electronic_structure.spin,
        "basis": config.electronic_structure.basis,
        "scf": config.electronic_structure.scf,
        "active_space": asdict(config.active_space),
        "geom_tag": geom_tag,
    }, sort_keys=True).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Aufbau dets.npz seed (closed-shell only — multi-ref needs plain_trimci first)
# ---------------------------------------------------------------------------

def _write_aufbau_dets(dets_path: Path, n_alpha: int, n_beta: int) -> None:
    alpha_bits = (1 << n_alpha) - 1
    beta_bits = (1 << n_beta) - 1
    arr = np.array([[alpha_bits, beta_bits]], dtype=np.uint64)
    np.savez_compressed(dets_path, dets=arr)


# ---------------------------------------------------------------------------
# SCF + active-space selection
# ---------------------------------------------------------------------------

def _build_mol(config: MoleculeConfig, xyz: str) -> "gto.Mole":
    return gto.M(
        atom=xyz,
        basis=config.electronic_structure.basis,
        charge=config.electronic_structure.charge,
        spin=config.electronic_structure.spin,
        unit="Angstrom",
        verbose=0,
    )


def _run_scf(mol: "gto.Mole", scf_kind: str):
    if scf_kind == "RHF":
        mf = scf.RHF(mol)
    elif scf_kind == "ROHF":
        mf = scf.ROHF(mol)
    elif scf_kind in ("UHF→ROHF", "UHF->ROHF"):
        mf_u = scf.UHF(mol); mf_u.verbose = 0; mf_u.kernel()
        # Project back to ROHF by averaging alpha/beta occupations
        mf = scf.ROHF(mol); mf.verbose = 0
        # Use UHF MOs as initial guess
        mf.kernel(dm0=mf_u.make_rdm1())
        return mf
    else:
        raise ValueError(f"Unknown scf kind: {scf_kind!r}")
    mf.verbose = 0
    mf.kernel()
    return mf


def _select_active_space_explicit(mf, config: MoleculeConfig):
    spec = config.active_space
    norb = len(spec.orb_indices)
    nelec = tuple(spec.nelec)
    # Reorder MOs so the active block is contiguous at the front
    mo = mf.mo_coeff.copy()
    cols_active = spec.orb_indices
    cols_other = [i for i in range(mo.shape[1]) if i not in cols_active]
    mo_reordered = np.concatenate([mo[:, cols_active], mo[:, cols_other]], axis=1)
    return norb, nelec, mo_reordered


def _select_active_space_avas(mf, config: MoleculeConfig):
    from pyscf.mcscf import avas
    norb, nelec_active, mo = avas.kernel(mf, config.active_space.avas_patterns, verbose=0)
    n_alpha = nelec_active // 2
    n_beta = nelec_active - n_alpha
    if config.electronic_structure.spin:
        n_alpha = (nelec_active + config.electronic_structure.spin) // 2
        n_beta = nelec_active - n_alpha
    return int(norb), (int(n_alpha), int(n_beta)), mo


def _select_active_space_window(mf, config: MoleculeConfig):
    spec = config.active_space
    norb = spec.norb
    nelec_active = spec.n_active_elec
    homo_idx = mf.mol.nelectron // 2 - 1
    lumo_idx = homo_idx + 1
    # Symmetric window around HOMO/LUMO
    n_below = (norb - 1) // 2
    n_above = norb - 1 - n_below
    start = homo_idx - n_below
    end = homo_idx + n_above + 1
    if start < 0:
        end -= start; start = 0
    if end > mf.mo_coeff.shape[1]:
        start -= (end - mf.mo_coeff.shape[1]); end = mf.mo_coeff.shape[1]
    cols_active = list(range(start, end))
    cols_other = [i for i in range(mf.mo_coeff.shape[1]) if i not in cols_active]
    mo_reordered = np.concatenate(
        [mf.mo_coeff[:, cols_active], mf.mo_coeff[:, cols_other]], axis=1
    )
    n_alpha = (nelec_active + config.electronic_structure.spin) // 2
    n_beta = nelec_active - n_alpha
    return int(norb), (int(n_alpha), int(n_beta)), mo_reordered


def _run_casscf(mf, n_orb: int, nelec: tuple, mo_active: np.ndarray):
    from pyscf import mcscf as _mcscf
    cas = _mcscf.CASSCF(mf, n_orb, nelec)
    cas.verbose = 0
    cas.max_cycle_macro = 50
    cas.kernel(mo_active)
    return cas.mo_coeff, bool(cas.converged), float(cas.e_tot)


def _reorder_casscf_output(mo_opt: np.ndarray, n_orb: int, n_elec_active: int,
                            n_elec_total: int) -> np.ndarray:
    """Pyscf CASSCF returns mo_opt in [core | active | virtual] order. Re-permute to
    diff_mols' [active | other] convention so downstream code sees the active block
    at columns 0..n_orb-1."""
    n_core = (n_elec_total - n_elec_active) // 2
    cols_active = list(range(n_core, n_core + n_orb))
    cols_other = ([i for i in range(n_core)]
                  + [i for i in range(n_core + n_orb, mo_opt.shape[1])])
    return np.concatenate([mo_opt[:, cols_active], mo_opt[:, cols_other]], axis=1)


def _compute_active_space_integrals(mol, mf, mo_reordered: np.ndarray,
                                    n_orb: int, n_elec_active: int):
    """Build the FCIDUMP integrals for an active space embedded in a larger molecular system.

    For ``n_elec_active < mol.nelectron`` (typical for any window/AVAS active space in a real
    basis), the FCIDUMP must encode the frozen-core HF contribution:

        h1_eff[p,q] = ⟨p| h_core + 2J(D_core) - K(D_core) |q⟩    (effective 1-body)
        e_const    = e_nuc + e_core_HF                            (FCIDUMP constant)

    where the core consists of the n_core lowest-energy doubly-occupied MOs that are NOT in
    the active block. ``mo_reordered[:, :n_orb]`` is the active subspace (possibly Boys-rotated).
    Cores are identified from ``mo_reordered[:, n_orb:]`` by their (post-rotation) orbital energy.

    Returns ``(h1, eri_active, e_const, e_core)``.
    """
    n_core = (mol.nelectron - n_elec_active) // 2
    mo_active = mo_reordered[:, :n_orb]

    hcore_ao = mf.get_hcore()
    eri_active = ao2mo.full(mol, mo_active, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)

    if n_core == 0:
        # All electrons live in the active space; no frozen-core term to fold in.
        h1 = mo_active.T @ hcore_ao @ mo_active
        return h1, eri_active, float(mol.energy_nuc()), 0.0

    # Core MOs = lowest-energy non-active MOs (by their fock-diagonal energy in
    # mo_reordered's basis, which is robust against AVAS/window/Boys rotations of the
    # active block — the core sub-block is left untouched by any of those).
    mo_nonactive = mo_reordered[:, n_orb:]
    fock_ao = mf.get_fock()
    energies_nonactive = np.diag(mo_nonactive.T @ fock_ao @ mo_nonactive)
    order = np.argsort(energies_nonactive)
    mo_core = mo_nonactive[:, order[:n_core]]

    # Effective 1-body operator (core's Coulomb + exchange contribution to the active block).
    # We use scf.hf.get_jk directly rather than mf.get_veff so the result is always (n_AO, n_AO)
    # — for ROHF/UHF mf.get_veff would return a (2, n_AO, n_AO) alpha/beta-channeled array.
    from pyscf import scf as _scf
    dm_core = 2.0 * mo_core @ mo_core.T            # restricted density (closed-shell core)
    j_core, k_core = _scf.hf.get_jk(mol, dm_core)
    veff_core_ao = j_core - 0.5 * k_core           # RHF convention: J - K/2
    h1_eff_ao = hcore_ao + veff_core_ao
    h1 = mo_active.T @ h1_eff_ao @ mo_active

    # Core HF energy: e_core = Tr(D h) + 1/2 Tr(D V_eff(D))
    e_core_1body = float(np.einsum("ij,ji->", dm_core, hcore_ao))
    e_core_2body = 0.5 * float(np.einsum("ij,ji->", dm_core, veff_core_ao))
    e_core = e_core_1body + e_core_2body
    e_const = float(mol.energy_nuc()) + e_core
    return h1, eri_active, e_const, e_core


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def build_integrals(
    config: MoleculeConfig,
    geom_tag: str,
    output_root: Path,
    *,
    force: bool = False,
) -> IntegralBundle:
    """Build (or load cached) FCIDUMP + mo_coeff + scf_summary for one geometry."""
    output_root = Path(output_root)
    mol_root = output_root / config.slug
    geom_root = mol_root if geom_tag == geom_tag_for_single() else mol_root / geom_tag
    out_dir = geom_root / "integrals"
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz = _render_geometry(config, geom_tag)
    xyz_path = geom_root / "geometry.xyz"
    xyz_path.write_text(xyz)

    hash_path = out_dir / "inputs_hash.txt"
    new_hash = _inputs_hash(config, geom_tag, xyz)
    fcidump_path = out_dir / "fcidump"

    if (not force) and fcidump_path.exists() and hash_path.exists() and hash_path.read_text().strip() == new_hash:
        # Cache hit — load summary and return
        summary = json.loads((out_dir / "scf_summary.json").read_text())
        return IntegralBundle(
            fcidump_path=fcidump_path,
            mo_coeff_path=out_dir / "mo_coeff.npz",
            scf_summary_path=out_dir / "scf_summary.json",
            xyz_path=xyz_path,
            n_orb=summary["n_orb"], n_elec=summary["n_elec"],
            n_alpha=summary["n_alpha"], n_beta=summary["n_beta"],
            e_nuc=summary["e_nuc"], e_hf=summary["e_hf"],
            inputs_hash=new_hash, output_dir=out_dir,
            geom_tag=geom_tag,
        )

    # Build
    mol = _build_mol(config, xyz)
    mf = _run_scf(mol, config.electronic_structure.scf)

    # Active space selection
    spec = config.active_space
    if spec.kind == "explicit":
        n_orb, nelec, mo_reordered = _select_active_space_explicit(mf, config)
    elif spec.kind == "avas":
        n_orb, nelec, mo_reordered = _select_active_space_avas(mf, config)
    elif spec.kind == "window":
        n_orb, nelec, mo_reordered = _select_active_space_window(mf, config)
    else:
        raise ValueError(f"Unknown active_space.kind: {spec.kind!r}")

    n_alpha, n_beta = nelec
    n_elec = n_alpha + n_beta
    mo_active = mo_reordered[:, :n_orb]

    casscf_converged = None
    e_casscf = None
    if spec.casscf:
        mo_active_full = mo_reordered.copy()
        mo_opt, casscf_converged, e_casscf = _run_casscf(mf, n_orb, nelec, mo_active_full)
        # PySCF CASSCF returns mo_opt in [core | active | virtual] order. Re-permute
        # back to diff_mols' [active | other] convention so downstream code (Boys,
        # frozen-core integrals, mo_coeff.npz) sees the active block at columns 0..n_orb-1.
        mo_reordered = _reorder_casscf_output(mo_opt, n_orb, n_elec, mol.nelectron)
        mo_active = mo_reordered[:, :n_orb]

    # Boys localization: apply to active block when auto fragmentation is requested,
    # or when active_space.boys=true is set explicitly.
    # This ensures FCIDUMP integrals, mo_coeff.npz, and IAO population analysis
    # all live in the same Boys-localized frame, avoiding basis mismatch downstream.
    _AUTO_FRAG_KINDS = {"auto_per_atom", "auto_per_metal", "chem_bond"}
    needs_boys = spec.boys or any(
        isinstance(fspec.orbital_lists, str) and fspec.orbital_lists in _AUTO_FRAG_KINDS
        for fspec in config.fragmentation
    )
    boys_localized = False
    if needs_boys:
        from pyscf import lo
        mo_active = lo.Boys(mol, mo_active).kernel()
        # Patch the reordered coefficient matrix with the localized active block
        mo_reordered = mo_reordered.copy()
        mo_reordered[:, :n_orb] = mo_active
        boys_localized = True

    # Compute frozen-core effective Hamiltonian for the active space.
    # When the molecule has more electrons than the active space holds, the remaining
    # doubly-occupied "core" MOs must be folded into the FCIDUMP as:
    #   - an effective 1-body operator (h1 + 2J_core - K_core projected onto active)
    #   - a constant energy term (e_nuc + e_core_HF) so that TrimCI's diagonalization
    #     plus the constant gives the correct total molecular energy.
    # Without this, TrimCI/LASSCF would diagonalize the active-only piece and produce
    # numerically meaningless total energies (often positive).
    h1, eri, e_const, e_core = _compute_active_space_integrals(
        mol, mf, mo_reordered, n_orb, n_elec,
    )
    e_nuc = float(mol.energy_nuc())
    e_hf = float(mf.e_tot)

    # Write FCIDUMP: nuc=e_const so that <D|H|D> + e_const = total molecular energy.
    tools.fcidump.from_integrals(
        str(fcidump_path), h1, eri, n_orb, n_elec,
        nuc=e_const, ms=(n_alpha - n_beta), orbsym=None, tol=1e-15,
    )

    # Inline FCI reference (if config asks for it AND the active space is small enough).
    # This is the EXACT energy in the chosen active space — the natural upper-bound benchmark
    # for any method's "% of correlation recovered". For active spaces up to ~(10,10), PySCF's
    # direct FCI solver returns in well under a second per geometry; bigger AS we skip.
    e_fci_inline = None
    fci_threshold_norb = 12      # ~ (12, 12) FCI is still tractable (924 × 924 alpha × beta dets)
    if config.reference.computed_inline and n_orb <= fci_threshold_norb:
        from pyscf import fci as _fci
        e_active, _civec = _fci.direct_spin1.kernel(
            h1, eri, n_orb, (n_alpha, n_beta), ecore=e_const,
        )
        e_fci_inline = float(e_active)

    # Write mo_coeff
    np.savez_compressed(out_dir / "mo_coeff.npz", mo_coeff=mo_reordered)

    # Aufbau dets.npz seed
    # When Boys localization is applied, MO ordering may not be energy-sorted,
    # so we compute Aufbau by filling the n_alpha/n_beta lowest-energy Boys MOs.
    if boys_localized:
        h1_diag = np.diag(h1)
        alpha_orbs = sorted(np.argsort(h1_diag)[:n_alpha].tolist())
        beta_orbs = sorted(np.argsort(h1_diag)[:n_beta].tolist())
        alpha_bits = sum(1 << o for o in alpha_orbs)
        beta_bits = sum(1 << o for o in beta_orbs)
        arr = np.array([[alpha_bits, beta_bits]], dtype=np.uint64)
        np.savez_compressed(out_dir / "dets.npz", dets=arr)
    else:
        _write_aufbau_dets(out_dir / "dets.npz", n_alpha, n_beta)

    # SCF summary — mo_energy_active is computed in the FINAL (post-CASSCF, post-Boys)
    # mo_active basis as diag(mo_active.T @ Fock @ mo_active). This is what downstream
    # tools actually see (vs. the original mf.mo_energy which is wrong if the active
    # block isn't aligned with the lowest-energy MOs).
    _fock_ao = mf.get_fock()
    mo_energy_active = np.diag(mo_active.T @ _fock_ao @ mo_active).tolist()
    n_core = (mol.nelectron - n_elec) // 2

    summary = {
        "n_orb": int(n_orb), "n_elec": int(n_elec),
        "n_alpha": int(n_alpha), "n_beta": int(n_beta),
        "e_nuc": e_nuc, "e_hf": e_hf,
        "e_core": float(e_core),                # frozen-core HF energy (active-space-only FCIDUMP folds this in)
        "e_const_fcidump": float(e_const),      # e_nuc + e_core; written as the FCIDUMP nuc constant
        "n_core_frozen": int(n_core),           # number of doubly-occupied frozen-core MOs
        "e_fci_inline": e_fci_inline,           # exact FCI in this AS (or None if skipped); the reference for error_mha
        "active_space_kind": spec.kind,
        "active_space_avas_patterns": spec.avas_patterns,
        "mo_energy_active": mo_energy_active,
        "casscf_converged": casscf_converged,
        "e_casscf": e_casscf,
        "boys_localized": boys_localized,
        "inputs_hash": new_hash,
    }
    (out_dir / "scf_summary.json").write_text(json.dumps(summary, indent=2))
    hash_path.write_text(new_hash)

    return IntegralBundle(
        fcidump_path=fcidump_path,
        mo_coeff_path=out_dir / "mo_coeff.npz",
        scf_summary_path=out_dir / "scf_summary.json",
        xyz_path=xyz_path,
        n_orb=n_orb, n_elec=n_elec,
        n_alpha=n_alpha, n_beta=n_beta,
        e_nuc=e_nuc, e_hf=e_hf,
        inputs_hash=new_hash, output_dir=out_dir,
        geom_tag=geom_tag,
    )


def build_all_geometries(
    config: MoleculeConfig,
    output_root: Path,
    *,
    force: bool = False,
) -> dict[str, IntegralBundle]:
    """Build integrals for every geometry in the config — single → {'eq': ...} or scan → {tag: ...}."""
    if config.geometry.kind == "single":
        return {geom_tag_for_single(): build_integrals(config, geom_tag_for_single(), output_root, force=force)}
    if config.geometry.kind == "scan":
        out = {}
        for pt in config.geometry.scan_points:
            tag = geom_tag_for_scan_point(pt, config.geometry.scan_param)
            out[tag] = build_integrals(config, tag, output_root, force=force)
        return out
    raise ValueError(f"Unknown geometry.kind: {config.geometry.kind!r}")
