"""
I-PI via python interface: the case of the ML model nanopet
===========================================================

:Authors: Davide Tisi `@DavideTisi <https://github.com/DavideTisi>`_ and
    Michele Ceriotti `@ceriottm <https://github.com/ceriottm>`_


This notebook provides an introduction to the use of both 
`metatrain` and the `i-PI <http://ipi-code.org>`_
Python interface. We will use the `nanopet` model, a machine-learning
model based on the 
`PET <https://shorturl.at/tlJtv>`_
architecture. The model is trained on a dataset of Zundel molecules.
The model is used to compute the potential energy and forces.
I-Pi is used to run a molecular dynamics simulation in the NVT ensamble.
"""

import time

import chemiscope
import ipi
import matplotlib.pyplot as plt
import numpy as np
from ipi.utils.scripting import (
    simulation_xml,
    forcefield_xml,
    motion_nvt_xml,
    InteractiveSimulation,
)
import ase, ase.io


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821

# sphinx_gallery_thumbnail_number = 2

# %%
# Look at the dataset
# -------------------
#
# The model is trained on a dataset of Zundel molecules, and the pot
# Let's look the structure on the dataset
# The dataset is stored in the file `zundel_dataset.xyz`

frames = ase.io.read("inputs/zundel_dataset.xyz", index=":",format="extxyz")


chemiscope.show(
    frames=frames,
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True, structure_settings={"unitCell": True}
    ),
)


# %%
# Load the model
# --------------
#
# Load the model and check the properties of the model.
# We use the scripting feature of i-pi.
# There are utilities to quickly set up XML inputs for commonly-used simulations

data = ase.io.read("inputs/h5o2+.extxyz", format="extxyz")
input_xml = simulation_xml(
    structures=data,
    forcefield=forcefield_xml(
        name="pet-mtt",
        mode="direct",
        pes="metatensor",
        parameters={"model": "inputs/model-zundel.pt",
                    "template": "inputs/h5o2+.extxyz","device":"cpu"},
    ),
    motion=motion_nvt_xml(timestep=2 * ase.units.fs),
    temperature=300,
    prefix="script",
)


# %%
# Let's have a look at the generated input file

print("Running with XML input:\n\n", input_xml)


# %%
# The main object for scripting is `InteractiveSimulation`, that is initialized from
# and XML input and acts as a wrapper around an i-PI simulation object

sim = InteractiveSimulation(input_xml)

# %% 
# `properties` accesses the properties of the simulation object

print(
    sim.properties("time") / ase.units.fs,
    sim.properties("potential"),
    sim.properties("temperature"),
)


#%% 
# `run` advances the interactive simulation by one (or the prescribed number) of steps
# let's run the simulation for 5000 steps

sim.run(5000)


# %%
# Plot the results
# ----------------
#
# Let's plot the time, potential energy, and temperature

properties,_tmp = ipi.read_output("script.out")
time = properties["time"] / ase.units.fs
potential = properties["potential"]
temperature = properties["temperature"]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

ax1.plot(time, potential, label="Potential Energy")
ax1.set_ylabel("Potential Energy (eV)")
ax1.legend()
ax1.grid()

ax2.plot(time, temperature, label="Temperature", color="r")
ax2.set_xlabel("Time (fs)")
ax2.set_ylabel("Temperature (K)")
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# %%
# `get_structures` dumps the state of the system as ASE Atoms objects, possibly listing
# all systems and beads

ase.io.write("final_positions.xyz", sim.get_structures())