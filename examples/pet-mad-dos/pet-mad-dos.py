"""
PET-MAD-DOS: A universal model for the Density of States
=========================================================

:Authors: Pol Febrer `@pfebrer <https://github.com/pfebrer>`_,
           How Wei Bin `@HowWeiBin <https://github.com/HowWeiBin>`_,

This example demonstrates how to use a universal model for the Density of States (DOS)
to compute the electronic heat capacity of a material using only universal ML models.

In the example, we will:

- Use the universal PET-MAD force field to run MD for our material of interest.
- Use PET-MAD-DOS to compute the DOS for snapshots of the MD trajectory.
- Compute the electronic heat capacity from the predicted DOS.
"""

# %%
#
# Load PET-MAD
# ------------
#

import numpy as np
import torch
from pet_mad.calculator import PETMADCalculator


petmad = PETMADCalculator(version="latest", device="cpu")

# %%
#
# Create a system and run some MD steps
# -------------------------------------
#

import ase.md
from ase.build import bulk


atoms = bulk("Au", "diamond", a=5.43, cubic=True)
atoms.calc = petmad

integrator = ase.md.VelocityVerlet(atoms, timestep=0.5 * ase.units.fs)

n_steps = 100

traj_atoms = []

for _ in range(n_steps):
    # run a single simulation step
    integrator.run(1)

    traj_atoms.append(atoms.copy())

# %%
#
# Load the DOS model
# ------------------
#

from pet_mad.calculator import PETMADDOSCalculator


dos_model = PETMADDOSCalculator()

# %%
#
# Run the DOS model
# -----------------
#

# Run the model to compute the DOS on the snapshots of the trajectory.
# We only evaluate every 4th system to speed up the computation
E, all_DOS = dos_model.calculate_dos(traj_atoms[::4])

# %%
#
# Plot ensemble DOS
# -----------------
#

import matplotlib.pyplot as plt


# Compute the ensemble DOS by averaging over all systems
# and ensure that there are no negative values
ensemble_DOS = torch.mean(all_DOS, dim=0)
ensemble_DOS[ensemble_DOS < 0] = 0

# Plot the ensemble DOS
plt.plot(E, ensemble_DOS)
plt.xlabel("Energy (eV)")
plt.ylabel("Density of States")
plt.title("Ensemble Density of States from PET-MAD")
plt.show()

# %%
#
# Compute electronic heat capacity
# --------------------------------
#
# First some helper functions to:
#
# - Compute the Fermi-Dirac distribution
# - Compute the Fermi level for a given density of states and number of electrons
# - Compute the electronic heat capacity given a DOS.
#

from collections.abc import Sequence

from ase.units import kB


def fd_distribution(E: Sequence, mu: float, T: float) -> np.array:
    r"""Compute the Fermi-Dirac distribution.

    .. math::
        f(E) = \frac{1}{1 + e^{(E - \mu) / (k_B T)}}

    Parameters
    ----------
    E:
        Values of the energy axis for which to compute the Fermi-Dirac distribution (eV)
    mu:
        Fermi level / chemical potential (eV)
    T:
        Temperature (K)
    """

    y = (E - mu) / (kB * T)
    # np.exp(y) can lead to overflow if y is too large, so we use a trick to avoid it
    # We compute exp(-|y|) and then treat positive and negative values separately
    ey = np.exp(-np.abs(y))

    negs = y < 0
    pos = y >= 0
    y[negs] = 1 / (1 + ey[negs])
    y[pos] = ey[pos] / (1 + ey[pos])

    return y


def compute_heat_capacity(
    dos: Sequence, E: Sequence, T: float, dT: float = 1.0
):
    """Compute the electronic heat capacity.

    It uses the finite difference method to compute the heat capacity
    from the change in internal energy with temperature.

    Parameters
    ----------
    dos:
        Density of states.
    E:
        Energy axis corresponding to the DOS (eV)
    T:
        Temperature (K).
    dT:
        Temperature step for finite difference (K).
    """
    # The calculations are more numerically stable if we shift the energy so that
    # near the fermi level energies are close to zero. We compute the Fermi level
    # at T to shift the energy axis.
    Ef_T = dos_model.calculate_efermi(
        traj_atoms[0], dos=dos.reshape(1, -1), temperature=T
    )
    E = E - Ef_T

    # Compute the internal energy at T-dT and T+dT
    U = []
    for T_side in (T - dT, T + dT):

        Ef_side = dos_model.calculate_efermi(
            traj_atoms[0], dos=dos.reshape(1, -1), temperature=T_side
        )

        U_side = torch.trapezoid(E * dos * fd_distribution(E, Ef_side, T_side), E)
        U.append(U_side)

    # Compute the heat capacity as the gradient of the internal energy
    # with respect to temperature
    heat_capacity = (U[1] - U[0]) / (2 * dT) / kB

    return heat_capacity


# %%
#
# Compute the heat capacity at different temperatures.
#
# TODO: Here we need to compute the ensemble DOS at each temperature by doing MD
# at each temperature.
#

# Compute the heat capacity at different temperatures
heat_capacities = []
Ts = np.linspace(200, 1000, 10)
for T in Ts:
    heat_capacity = compute_heat_capacity(ensemble_DOS, E, T, dT=1)

    heat_capacities.append(heat_capacity.item() / len(traj_atoms[0]))

# Plot them
plt.plot(Ts, heat_capacities)
plt.xlabel("Temperature (K)")
plt.ylabel("Heat Capacity [eV/(K*atom)] (divided by kB)")
plt.title("Electronic Heat Capacity from PET-MAD")
plt.show()
