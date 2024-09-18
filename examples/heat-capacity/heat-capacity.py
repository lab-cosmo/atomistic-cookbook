"""
Path integral molecular dynamics
================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This example shows how to estimate the heat capacity of liquid water
from a path integral molecular dynamics simulation. The dynamics are
run with `i-PI <http://ipi-code.org>`_, and 
`LAMMPS <http://lammps.org>`_ is used
as the driver to simulate the `q-TIP4P/f water
model <http://doi.org/10.1063/1.3167790>`_.
"""

import subprocess
import time

import ipi
import matplotlib.pyplot as plt
import numpy as np


# %%
# Quantum heat capacity of water
# ------------------------------
#
# As introduced in the ``path-integrals`` example, path-integral estimators
# for observables that depend on momenta are generally not trivial to compute.
#
# In this example, we will focus on the heat capacity, which is one such
# observable, and we will calculate it for liquid water at room temperature.


# %%
# Running the PIMD calculation
# ----------------------------
#
# This follows the same steps as the ``path-integrals`` example. One important
# difference is that we will request the ``scaledcoords`` output, which
# contains the estimators for the total energy and heat capacity as defined
# in this `paper <https://arxiv.org/abs/physics/0505109>`_. In order to do this,
# the ``scaledcoords`` output is added to the relevant section of the ``i-PI``
# input XML file.

# Open and read the XML file
with open("data/input.xml", "r") as file:
    xml_content = file.read()
print(xml_content)

# %%
# NB1: In a realistic simulation you will want to increase the field
# ``total_steps``, to simulate at least a few 100s of picoseconds.
#
# NB2: To converge a simulation of water at room temperature, you
# typically need at least 32 beads.

# %%
# We launch the i-PI and LAMMPS processes, exactly as in the
# ``path-integrals`` example.

ipi_process = subprocess.Popen(["i-pi", "data/input.xml"])
time.sleep(2)  # wait for i-PI to start
lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(2)]

ipi_process.wait()
lmp_process[0].wait()
lmp_process[1].wait()

output_data, output_desc = ipi.read_output("simulation.out")


# %%
# Let's plot the potential and conserved energy as a function of time,
# just to check that the simulation ran sensibly.

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["potential"] - output_data["potential"][0],
    "b-",
    label="Potential, $V$",
)
ax.plot(
    output_data["time"],
    output_data["conserved"] - output_data["conserved"][0],
    "r-",
    label="Conserved, $H$",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"energy / eV")
ax.legend()

# %%
# We will now plot the values of the energy and heat capacity estimators
# as a function of time.
# First the energy:

scaledcoords_energy = np.loadtxt("simulation.out")[:, 6]

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    scaledcoords_energy,
    "b",
    label="Total energy$",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$E / a.u.$")
ax.legend()

# %%
# And, finally, the heat capacity:

scaledcoords_Cv = np.loadtxt("simulation.out")[:, 7]

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    scaledcoords_Cv,
    "b",
    label=f"Constant-volume heat capacity (mean={scaledcoords_Cv.mean()})$",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$C_{V} / a.u.$")
ax.legend()
