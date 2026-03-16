r"""
Thermal conductivity with kaldo and UPET
=========================================

:Authors: Giuseppe Barbalinardo `@gbarbalinardo <https://github.com/gbarbalinardo/>`_,
          Paolo Pegolo `@ppegolo <https://github.com/ppegolo/>`_

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

We use NaCl (rocksalt) as a test system, following the study by
`Barbalinardo et al., Phys. Rev. B 103, 024204 (2021)
<https://doi.org/10.1103/PhysRevB.103.024204>`_.

.. note::

   The supercell and k-point grids used here are deliberately small so the
   recipe runs in CI in a few minutes.  See the convergence discussion at the
   end for production-quality settings.
"""

# %%
# Setup
# -----
#
# We use the extra-small (XS) variant of
# `PET-MAD v1.5.0 <https://arxiv.org/abs/2603.02089>`_. For production
# calculations, the S model (``pet-mad-s``) is recommended.

import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.constraints import FixSymmetry
from ase.filters import StrainFilter
from ase.optimize import BFGS
from upet.calculator import UPETCalculator
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import kaldo.controllers.plotter as plotter

DEVICE = "cpu"

calc = UPETCalculator(
    model="pet-mad-xs",
    device=DEVICE,
    dtype="float32",
    version="1.5.0",
)

# sphinx_gallery_thumbnail_number = 2

# %%
# Structure setup and relaxation
# ------------------------------
#
# We build a 2-atom NaCl primitive cell (rocksalt) and relax the lattice
# parameter while preserving symmetry.

atoms = bulk("NaCl", "rocksalt", a=5.59)
atoms.calc = calc

atoms.set_constraint(FixSymmetry(atoms))
sf = StrainFilter(atoms)
opt = BFGS(sf, logfile=None)
opt.run(fmax=1e-4)

atoms.set_constraint(None)
a_opt = atoms.cell.cellpar()[0] * np.sqrt(2)
print(f"Optimized lattice parameter: {a_opt:.3f} Å")

# %%
# Force constants
# ---------------
#
# We compute second-order (harmonic) and third-order (anharmonic) interatomic
# force constants using finite differences.  The supercell sizes are kept small
# for CI; see the convergence section for production settings.

second_supercell = np.array([3, 3, 3])
third_supercell = np.array([2, 2, 2])

forceconstants = ForceConstants(
    atoms=atoms,
    supercell=second_supercell,
    third_supercell=third_supercell,
    folder="fd_nacl/",
)

forceconstants.second.calculate(calc, delta_shift=3e-2)
forceconstants.third.calculate(calc, delta_shift=3e-2)

# %%
# Phonon dispersion and density of states
# ----------------------------------------
#
# We create a ``Phonons`` object with a 5×5×5 k-point mesh (CI-friendly).
# Production calculations should use at least 12×12×12.

kpts = np.array([5, 5, 5])
temperature = 300  # K

phonons = Phonons(
    forceconstants=forceconstants,
    kpts=kpts,
    is_classic=False,
    temperature=temperature,
    folder="ald_nacl/",
    storage="numpy",
)

# %%
# Dispersion relation along high-symmetry directions:

plotter.plot_dispersion(phonons, n_k_points=300, is_showing=False)
plt.tight_layout()
plt.show()

# %%
# Density of states with partial contributions from Na and Cl:

fig, ax = plt.subplots(figsize=(6, 4))
plotter.plot_dos(phonons, p_atoms=None, bandwidth=0.05, is_showing=False)
plt.tight_layout()
plt.show()

# %%

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

plt.sca(axes[0])
plotter.plot_dos(phonons, p_atoms=[0], bandwidth=0.05, is_showing=False)
axes[0].set_title("Na partial DOS")

plt.sca(axes[1])
plotter.plot_dos(phonons, p_atoms=[1], bandwidth=0.05, is_showing=False)
axes[1].set_title("Cl partial DOS")

plt.tight_layout()
plt.show()

# %%
# Thermal conductivity
# --------------------
#
# We solve the BTE using the direct inversion method.

inv_cond = Conductivity(phonons=phonons, method="inverse")
kappa = inv_cond.conductivity.sum(axis=0)
kappa_scalar = np.mean(np.diag(kappa))

print("Thermal conductivity:")
print(f"  Scalar average: {kappa_scalar:.2f} W/(m·K)")
print(f"  Tensor:\n{kappa}")

# %%
# Convergence and reference values
# ---------------------------------
#
# The small supercell and k-point grids used above (3×3×3 / 2×2×2 / 5×5×5)
# are chosen to keep CI runtime short and are **not converged**.
#
# Converged results with the PET-MAD S model use:
#
# - 2nd-order supercell: 12×12×12
# - 3rd-order supercell: 4×4×4
# - k-point mesh: 12×12×12
#
# These give ~8.0 W/(m·K) for the thermal conductivity at 300 K.
#
# To systematically converge results:
#
# 1. Increase the 2nd-order supercell until the phonon dispersion
#    (especially acoustic branches near Γ) is stable.
# 2. Increase the 3rd-order supercell until the conductivity is stable.
# 3. Increase the k-point mesh until the conductivity is converged.
#
# Also note that the ``pet-mad-xs`` model used here is faster but less
# accurate than ``pet-mad-s``.  For production calculations, use the S model.
