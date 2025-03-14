"""
PET-MAD Example
===============

:Authors: Philip Loche `@PicoCentauri <https://github.com/picocentauri>`_,

This example demonstrates how to use the PET-MAD model with ASE,
`i-PI <https://ipi-code.org>` and `LAMMPS <https://lammps.org>`_. 
PET-MAD is a "universal" machine-learning forcefield trained on 
a dataset that aims to incorporate a very high degree of 
structural diversity.

We will first create a Si crystal and then calculate the energy and forces using the PET-MAD
model in ASE. Finally, we will run a molecular dynamics simulation using ASE and LAMMPS.
"""

# %%
#
# Start by importing the required libraries.

import subprocess

import ase.units
import chemiscope
import numpy as np
from ase.build import bulk
from ase.md.langevin import Langevin
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatrain.utils.io import load_model

# %%
#
# Load the model
# ^^^^^^^^^^^^^^
#
# To start the simulation we first have to load the model. We will use the latest
# version of the PET-MAD model from the Huggingface model hub and export to be able to
# make inference with it.

mad_huggingface = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"
)
model = load_model(mad_huggingface).export()


# %%
#
# Initialize the system
# ^^^^^^^^^^^^^^^^^^^^^
#
# Create silicon crystal and rattle structures slightly to create non zero forces.

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
atoms.rattle(stdev=0.1)


# %%
#
# To speed the loading process we can also download the model by yourself and load it
# from disk by providing the path to the :py:func:`load_model
# <metatensor.utils.io.load_model>` function.
#
# Single point energy and forces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# PET-MAD is compatible with the metatensor atomistic model interface which allows us to
# run it with ASE and many other MD engines. For more details see the `metatensor
# documentation
# <https://docs.metatensor.org/latest/atomistic/engines/index.html#atomistic-models-engines>`_.
#
# We now wrap model in ASE compatible calculator and calculate energy and forces.

calculator = MetatensorCalculator(model, device="cpu")

# %%
#
# Here, we run the computation on the CPU.. If you have a CUDA GPU you can also set
# ``device="cuda"`` to speed up the computation.

atoms.set_calculator(calculator)

energy = atoms.get_potential_energy()
print(f"Energy is {energy} eV")

forces = atoms.get_forces()
print(f"Force on first atom is {forces[0]} eV/Å")

# %%
#
# Molecular dynamics with ``ASE``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We run Langevin molecular dynamics simulation with ASE. The system is being simulated
# at 310 Kelvin with a timestep of 0.5 femtoseconds and for 200 steps (total simulation
# time of 100 femtoseconds).

integrator = Langevin(atoms, 0.5 * ase.units.fs, temperature_K=310, friction=5e-3)

# %%
#
# We save the energies and atoms object in every step to visualize the trajectory with
# chemiscope.

n_steps = 100

potential_energy = np.zeros(n_steps)
kinetic_energy = np.zeros(n_steps)
trajectory = []

for step in range(n_steps):
    # run a single simulation step
    integrator.run(1)

    trajectory.append(atoms.copy())
    potential_energy[step] = atoms.get_potential_energy()
    kinetic_energy[step] = atoms.get_kinetic_energy()


# %%
#
# Visualize the trajectory with chemiscope

potential_energy -= potential_energy.mean()
total_energy = potential_energy + kinetic_energy


properties = {
    "time": np.arange(n_steps) * 0.5,
    "potential_energy": potential_energy,
    "kinetic_energy": kinetic_energy,
    "total_energy": total_energy,
}

chemiscope.show(frames=trajectory, properties=properties)


# %%
#
# Molecular dynamics with ``LAMMPS``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now run the same MD with `LAMMPS <https://lammps.org>`_.
#
# To do so, it is only required defining a ``pair_style`` that defines the model and a
# single ``pair_coeff`` command should be used with the metatensor style, specifying the
# mapping from LAMMPS types to the atomic types the model can handle. The first 2
# arguments must be ``* *`` so as to span all LAMMPS atom types. This is followed by a
# list of N arguments that specify the mapping of metatensor atomic types to LAMMPS
# types, where N is the number of LAMMPS atom types.

with open("data/pet-mad-si.in", "r") as f:
    print(f.read())

# %%
#
# To run the model with LAMMPS we first have to save the model to disk

model.save("pet-mad-latest.pt", collect_extensions="extensions")

# We use to the ``collect_extensions`` argument to save the compiled extensions to disk.
# These extensions ensure that the model remains self-contained and can be executed
# without requiring the original Python or C++ source code. This is necessary for the
# LAMMPS interface to work because it has no access to the Python code.
#
# .. warning::
#
#   Be aware that the extensions are compiled files and depend on your operating system.
#   Usually you have re-export the extensions for different systems!
#
# We also save to geometry to a LAMMPS data file and finally run the simulation.

ase.io.write("silicon.data", atoms, format="lammps-data", masses=True)

# subprocess.check_call(["lmp_serial", "-in", "data/pet-mad-si.in"])


# %%
#
# The LAMMPS input files looks like an usual input file but uses our metatensor
# interface.
#
