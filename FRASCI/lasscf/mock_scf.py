"""
mock_scf.py
===========
Build a PySCF SCF object from an FCIDUMP file so that LASSCF can be
initialized from it.  The FCIDUMP at data/fcidump_cycle_6
represents 36 orbitals, 54 electrons (27α+27β), and E_nuc=0.

The MOs from the FCIDUMP are already the canonical orbitals (identity
transform), so mo_coeff is set to the 36×36 identity matrix.

Shims applied vs bare to_scf() output
--------------------------------------
1. mol.symmetry = False
   Reason: the mol parsed from FCIDUMP has C1 symmetry labels but the AO
   basis is a minimal proxy (each orbital = 1 fake AO on 1 fake atom). When
   symmetry is True, mrh's localize_init_guess and LASSCF's orbital-symmetry
   tagging attempt to look up mo_orbsym arrays that don't make physical sense
   for this mock system. Disabling symmetry sidesteps the issue without
   affecting the Hamiltonian (the FCIDUMP integrals carry no symmetry tags).

2. mf.mo_coeff = np.eye(n_orb)
   Reason: to_scf() leaves mo_coeff=None; LASSCF requires it to be set.

3. mf.mo_occ = [2,...,2, 0,...,0] (n_alpha+n_beta doubly-occupied orbitals)
   Reason: for a closed-shell reference with 27α+27β in 36 orbs, the first 27
   orbitals are doubly occupied.  LASSCF uses mo_occ to separate core/active/
   virtual sectors via ncore.  With our all-active setup (ncore=0, ncas=36),
   mo_occ only affects initial Fock construction; setting it to 0 everywhere
   also works but 2.0 for the occupied block is more semantically correct.

4. mf.mo_energy = np.diag(mf.get_hcore())
   Reason: to_scf() leaves mo_energy=None.  LASSCF/LASCI calls
   las._eig(fock[i:j, i:j], ...) during recanonicalisation; providing
   diagonal Fock approximation (= h1 diagonal) avoids an AttributeError when
   fock is needed before the first SCF cycle.

Note: get_h2eff is NOT shimmed.  LASSCF (lasscf_sync_o0) uses get_h2eff
internally but acquires it via its own ao2mo transformation from mf._eri,
which to_scf() populates correctly as a packed 1D array (222111 entries for
36 orbs in chemist notation).  No additional shim is needed.
"""

from __future__ import annotations

import numpy as np
from pyscf.tools.fcidump import to_scf


def build_mock_scf_from_fcidump(fcidump_path: str):
    """
    Build a mock PySCF SCF object from an FCIDUMP file for LASSCF bootstrap.

    Parameters
    ----------
    fcidump_path : str
        Path to the FCIDUMP file (e.g. 'data/fcidump_cycle_6').

    Returns
    -------
    mf : pyscf.scf.hf.SCF
        Mock SCF object with:
        - mf.mol.nelectron == 54, mf.mol.spin == 0
        - mf.mol.energy_nuc() == 0.0   (from FCIDUMP ECORE=0)
        - mf.mo_coeff == eye(n_orb)    (FCIDUMP MOs are canonical identity)
        - mf.mo_occ   set to 2 for first n_occ orbs, 0 for remainder
        - mf.mo_energy == diag(h1)     (minimal Fock approximation)
        - mf._eri populated by to_scf  (packed chemist-notation ERIs)
        - mf.mol.symmetry == False     (avoids mock-AO-basis symmetry issues)
    """
    mf = to_scf(fcidump_path)

    # Shim 1: disable symmetry (see module docstring)
    mf.mol.symmetry = False

    # Note: mf.mol.nao_nr() returns 0 for the FCIDUMP mock mol because natm=0.
    # Use mol._nao (set by pyscf's FCIDUMP parser) or fall back to h1 shape.
    n_orb = int(mf.mol._nao) if hasattr(mf.mol, "_nao") and mf.mol._nao else mf.get_hcore().shape[0]
    n_elec = mf.mol.nelectron  # 54
    # For closed-shell: n_alpha = n_beta = n_elec // 2
    n_occ = n_elec // 2  # 27 doubly-occupied orbitals

    # Shim 2: identity MO coefficients
    mf.mo_coeff = np.eye(n_orb)

    # Shim 3: MO occupations
    mf.mo_occ = np.zeros(n_orb)
    mf.mo_occ[:n_occ] = 2.0

    # Shim 4: MO energies (diagonal of h1 as first approximation)
    mf.mo_energy = np.diag(mf.get_hcore())

    return mf
