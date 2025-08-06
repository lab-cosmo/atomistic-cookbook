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

import torch
import numpy as np

from pet_mad.calculator import PETMADCalculator 

petmad = PETMADCalculator(version="latest", device="cpu")

# %%
#
# Create a system and run some MD steps
# -------------------------------------
#

from ase.build import bulk
import ase.md

atoms = bulk("Au", "diamond", a=5.43, cubic=True)
atoms.calc = petmad

integrator = ase.md.VelocityVerlet(atoms, timestep=0.5 * ase.units.fs)

n_steps = 100

traj_atoms = []

for step in range(n_steps):
    # run a single simulation step
    integrator.run(1)

    traj_atoms.append(atoms.copy())

# %%
#
# Load the DOS model
# ------------------
#

from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from metatensor.torch.atomistic import load_atomistic_model, systems_to_torch

model = load_atomistic_model("dos_model.pt")

# %%
#
# Prepare trajectory for running the DOS model
# --------------------------------------------
#

# Convert from ase atoms to System objects and add the neighbor lists.
eval_systems = systems_to_torch(traj_atoms)

eval_systems = [
    get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model)).to(dtype=torch.float32)
    for system in eval_systems
]

# %%
#
# Run the DOS model
# -----------------
#

from metatomic.torch import ModelEvaluationOptions

# Define the evaluation options for the model (we only need the DOS output)
DOS_output = model.capabilities().outputs["mtt::dos"]
DOS_output.per_atom = False
options = ModelEvaluationOptions(
    outputs={"mtt::dos": DOS_output},
)

# Run the model to compute the DOS on the snapshots of the trajectory.
# We only evaluate every 4th system to speed up the computation
out = model(eval_systems[::4], options=options, check_consistency=False)

# %%
#
# Plot ensemble DOS
# -----------------
#

import matplotlib.pyplot as plt

# Get the DOS values from the output of the model (this will be shape [n_systems, n_E])
all_DOS = out["mtt::dos"].block(0).values
# Get the energy axis (the bounds are always the same for PET-MAD-DOS)
E = torch.linspace(-148.1456 - 1.5 - 10, 79.1528 + 1.5, all_DOS.shape[1])

# Compute the ensemble DOS by averaging over all systems
# and ensure that there are no negative values
ensemble_DOS = torch.mean(all_DOS, dim = 0)
ensemble_DOS[ensemble_DOS < 0] = 0

# Plot the ensemble DOS
plt.plot(E, ensemble_DOS)
plt.xlabel('Energy (eV)')
plt.ylabel('Density of States')
plt.title('Ensemble Density of States from PET-MAD')
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
from scipy.interpolate import interp1d

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

    y = (E-mu)/ (kB * T)
    # np.exp(y) can lead to overflow if y is too large, so we use a trick to avoid it
    # We compute exp(-|y|) and then treat positive and negative values separately
    ey = np.exp(-np.abs(y)) 

    negs = (y<0)
    pos = (y>=0)
    y[negs] = 1 / (1+ey[negs])        
    y[pos] = ey[pos] / (1+ey[pos])

    return y

def get_fermi(dos: Sequence, E: Sequence, n_elec: float, T: float = 0.) -> float:
    """Compute the Fermi level for a given density of states and number of electrons.
    
    Parameters
    ----------
    dos:
        Density of states.
    E:
        Energy axis corresponding to the DOS (eV)
    n_elec:
        Total number of electrons in the system.
    T:
        Temperature (K).
    """
    # First compute the Fermi level at T=0 by finding the energy where the 
    # cumulative DOS equals the number of electrons
    cumulative_dos = torch.cumulative_trapezoid(dos, E)
    Efdos = interp1d(cumulative_dos, E[1:])
    Ef_0 = Efdos(n_elec)

    if T == 0:
        return Ef_0
    
    # For finite temperatures, test a range of Fermi levels around Ef_0
    # and find the one that gives the correct number of electrons.
    integrated_doses = []
    trial_fermis = np.linspace(Ef_0 - 0.5, Ef_0 + 0.5, 100)
    for trial_fermi in trial_fermis:
        fd = fd_distribution(E, trial_fermi, T)
        integrated_dos = torch.trapezoid(dos * fd, E)
        integrated_doses.append(integrated_dos)

    n_elecs_Ef = interp1d(integrated_doses, trial_fermis)
    Ef = n_elecs_Ef(n_elec)
    return Ef

def compute_heat_capacity(dos: Sequence, E: Sequence, T: float, n_elec: float, dT: float = 1.):
    """Compute the electronic heat capacity.
    
    It uses the finite difference method to compute the heat capacity
    from the change in internal energy with temperature.

    Parameters
    ----------
    dos:
        Density of states.
    E:
        Energy axis corresponding to the DOS (eV)
    n_elec:
        Total number of electrons in the system.
    T:
        Temperature (K).
    dT:
        Temperature step for finite difference (K).
    """
    # The calculations are more numerically stable if we shift the energy so that
    # near the fermi level energies are close to zero. We compute the Fermi level
    # at T to shift the energy axis. 
    Ef_T = get_fermi(dos, E, n_elec=n_elec, T=T)
    E = E - Ef_T

    # Compute the internal energy at T-dT and T+dT
    U = []
    for T_side in (T - dT, T + dT):
        
        Ef_side = get_fermi(dos, E, n_elec=n_elec, T=T_side)

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

# Total number of electrons in the system
total_elec = 19 * len(traj_atoms[0])

# Compute the heat capacity at different temperatures
heat_capacities = []
Ts = np.linspace(200, 1000, 10)
for T in Ts:
    heat_capacity = compute_heat_capacity(ensemble_DOS, E, T, n_elec=total_elec, dT=1)

    heat_capacities.append(heat_capacity.item() / len(traj_atoms[0]))

# Plot them
plt.plot(Ts, heat_capacities)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity [eV/(K*atom)] (divided by kB)')
plt.title('Electronic Heat Capacity from PET-MAD')
plt.show()


