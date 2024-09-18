"""
Path integral molecular dynamics
================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This example shows how to run a path integral molecular dynamics
simulation using ``i-PI``, analyze the output and visualize the
trajectory in ``chemiscope``. It uses `LAMMPS <http://lammps.org>`_
as the driver to simulate the `q-TIP4P/f water
model <http://doi.org/10.1063/1.3167790>`_.
"""

import subprocess
import time

import ipi
import matplotlib.pyplot as plt
import numpy as np


# %%
# Quantum nuclear effects and path integral methods
# -------------------------------------------------
#
# The Born-Oppenheimer approximation separates the joint quantum mechanical
# problem for electrons and nuclei into two independent problems. Even though
# often one makes the additional approximation of treating nuclei as classical
# particles, this is not necessary, and in some cases (typically when H atoms are
# present) can add considerable error.
#
#
# .. figure:: pimd-slices-round.png
#    :align: center
#    :width: 600px
#
#    A representation of ther ring-polymer Hamiltonian for a water molecule.
#
# In order to describe the quantum mechanical nature of light nuclei
# (nuclear quantum effects) one of the most widely-applicable methods uses
# the *path integral formalism*  to map the quantum partition function of a
# set of distinguishable particles onto the classical partition function of
# *ring polymers* composed by multiple beads (replicas) with
# corresponding atoms in adjacent replicas being connected by harmonic
# springs.
# `The textbook by Tuckerman <https://tinyurl.com/bdfhk2tx>`_
# contains a pedagogic introduction to the topic, while
# `this paper <https://doi.org/10.1063/1.3489925>`_ outlines the implementation
# used in ``i-PI``.
#
# The classical partition function of the path converges to quantum statistics
# in the limit of a large number of replicas. In this example, we will use a
# technique based on generalized Langevin dynamics, known as
# `PIGLET <http://doi.org/10.1103/PhysRevLett.109.100604>`_ to accelerate the
# convergence.


# %%
# Running PIMD calculations with ``i-PI``
# ---------------------------------------
#
# `i-PI <http://ipi-code.org>`_ is based on a client-server model, with ``i-PI``
# controlling the nuclear dynamics (in this case sampling the path Hamiltonian using
# molecular dynamics) while the calculation of energies and forces is delegated to
# an external client program, in this example ``LAMMPS``.
#
# An i-PI calculation is specified by an XML file.

# Open and read the XML file
with open("data/input.xml", "r") as file:
    xml_content = file.read()
print(xml_content)

# %%
# NB1: In a realistic simulation you may want to increase the field
# ``total_steps``, to simulate at least a few 100s of picoseconds.
#
# NB2: To converge a simulation of water at room temperature, you
# typically need at least 32 beads. We will see later how to accelerate
# convergence using a colored-noise thermostat, but you can try to
# modify the input to check convergence with conventional PIMD

# %%
# i-PI and lammps should be run separately, and it is possible to
# launch separate lammps processes to parallelize the evaluation over
# the beads. On the the command line, this amounts to launching
#
# .. code-block:: bash
#
#    i-pi data/input.xml > log &
#    sleep 2
#    lmp -in data/in.lmp &
#    lmp -in data/in.lmp &
#
# Note how ``i-PI`` and ``LAMMPS`` are completely independent, and
# therefore need a separate set of input files. The client-side communication
# in ``LAMMPS`` is described in the ``fix_ipi`` section, that matches the socket
# name and mode defined in the ``ffsocket`` field in the ``i-PI`` file.
#
# We can launch the external processes from a Python script as follows

ipi_process = subprocess.Popen(["i-pi", "data/input.xml"])
time.sleep(2)  # wait for i-PI to start
lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(2)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

ipi_process.wait()
lmp_process[0].wait()
lmp_process[1].wait()


# %%
# Analyzing the simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# After the simulation has run, you can visualize and post-process the trajectory data.
# Note that i-PI prints a separate trajectory for each bead, as structural properties
# can be computed averaging over the configurations of any of the beads.

# drops first frame where all atoms overlap
output_data, output_desc = ipi.read_output("simulation.out")
traj_data = [ipi.read_trajectory(f"simulation.pos_{i}.xyz")[1:] for i in range(8)]


# %%
# The simulation parameters are pushed at the limits: with the aggressive stochastic
# thermostatting and the high-frequency normal modes of the ring polymer, there are
# fairly large fluctuations of the conserved quantity. This is usually not affecting
# physical observables, but if you see this level of drift in a production run, check
# carefully for convergence and stability with a reduced time step.

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
# Comment

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
# Comment

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
