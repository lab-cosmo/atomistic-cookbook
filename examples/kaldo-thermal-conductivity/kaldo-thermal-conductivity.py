r"""
Thermal conductivity with kaldo and UPET
=========================================

:Authors: Giuseppe Barbalinardo `@gbarbalinardo <https://github.com/gbarbalinardo/>`_

This recipe shows how to compute lattice thermal conductivity from first
principles using `kaldo <https://nanotheorygroup.github.io/kaldo/>`_ and
the `PET-MAD <https://arxiv.org/abs/2603.02089>`_ universal machine-learning
potential via the `UPET <https://github.com/lab-cosmo/pet>`_ calculator.

**kaldo** solves the linearized Boltzmann transport equation (BTE) for phonons
using anharmonic (third-order) force constants.  It supports several solution
methods (direct inversion, relaxation time approximation, self-consistent
iteration).
See `Barbalinardo et al., J. Appl. Phys. 128, 135104 (2020)
<https://doi.org/10.1063/5.0020443>`_.

**PET-MAD** is a pre-trained universal interatomic potential built on the
Point Edge Transformer (PET) architecture.  Here we use the extra-small (XS)
variant for speed; the small (S) model gives more accurate forces for
production runs.

We use silicon (diamond) as a test system.  The experimental thermal
conductivity of natural Si at 300 K is ~150 W/(m·K).

.. note::

   The supercell and k-point grids used here are deliberately small so the
   recipe runs in CI in a few minutes.  See the convergence discussion at the
   end for production-quality settings.
"""

# %%
# We use the extra-small (XS) variant of
# `PET-MAD v1.5.0 <https://arxiv.org/abs/2603.02089>`_. For production
# calculations, the S model (``pet-mad-s``) is recommended.

import numpy as np
from ase.build import bulk
from ase.constraints import FixSymmetry
from ase.filters import StrainFilter
from ase.optimize import BFGS
from upet.calculator import UPETCalculator
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import kaldo.controllers.plotter as plotter

DEVICE = "cpu"

calc = UPETCalculator(
    model="pet-mad-xs",
    device=DEVICE,
    dtype="float32",
    version="1.5.0",
)

# %%
# We build a 2-atom Si primitive cell (diamond) and relax the lattice
# parameter while preserving symmetry.

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = calc

atoms.set_constraint(FixSymmetry(atoms))
sf = StrainFilter(atoms)
opt = BFGS(sf, logfile=None)
opt.run(fmax=1e-4)

atoms.set_constraint(None)
a_opt = atoms.cell.cellpar()[0]
print(f"Optimized lattice parameter: {a_opt:.3f} Å")

# %%
# We compute second-order (harmonic) and third-order (anharmonic) interatomic
# force constants using finite differences.  Both use 3×3×3 supercells
# to keep CI runtime short; see the convergence section below for production
# settings.

supercell = np.array([3, 3, 3])

forceconstants = ForceConstants(
    atoms=atoms,
    supercell=supercell,
    third_supercell=supercell,
    folder="fd_si/",
)

forceconstants.second.calculate(calc, delta_shift=3e-2)
forceconstants.third.calculate(calc, delta_shift=3e-2)

# %%
# We create a ``Phonons`` object with a 5×5×5 k-point mesh and use
# kaldo's ``plot_crystal`` to produce a summary figure with the phonon
# dispersion, density of states, and thermal conductivity.

kpts = np.array([5, 5, 5])
temperature = 300  # K

phonons = Phonons(
    forceconstants=forceconstants,
    kpts=kpts,
    is_classic=False,
    temperature=temperature,
    folder="ald_si/",
    storage="numpy",
)

plotter.plot_crystal(phonons, is_showing=False)

# %%
# The 3×3×3 supercells and 5×5×5 k-point mesh used above are chosen to
# keep CI runtime short and are **not fully converged**.  To systematically
# converge results, increase the 2nd-order supercell until the phonon
# dispersion (especially acoustic branches near Γ) is stable, then increase
# the 3rd-order supercell and k-point mesh until the conductivity converges.
# The ``pet-mad-xs`` model used here is faster but less accurate than
# ``pet-mad-s``; use the S model for production calculations.
