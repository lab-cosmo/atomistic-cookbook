"""
Path integral metadynamics
==========================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This example shows how to run a free-energy sampling calculation that 
combines path integral molecular dynamics to model nuclear quantum effects
and metadynamics to accelerate sampling of the high-free-energy regions. 

The rather complicated setup combines ``i-PI`` to perform path integra
MD, the built-in driver to compute energy and forces for the Zundel
H5O2+ cation, and `PLUMED <http://plumed.org/>` to perform metadynamics.
If you want to see an example in a more realistic scenario, you can look
at `this paper <http://doi.org/10.1021/acs.jctc.0c00362>`, in which this
methodology is used to simulate the decomposition of methanesulphonic
acid in a solution of phenol and hydrogen peroxide. 
"""

import subprocess
import time

import chemiscope
import ipi
import matplotlib.pyplot as plt
import numpy as np

# %%
# Install the Python driver
# -------------------------
#
# i-PI comes with a FORTRAN driver, which however has to be installed
# from source. We use a utility function to compile it. Note that this requires
# a functioning build system with `gfortran` and `make`.

ipi.install_driver()

# %%
# Metadynamics for the Zundel cation
# ----------------------------------
#
# Metadynamics is a metnod to accelerate sampling of rare events - microscopic processes
# that are too infrequent to be observed over the time scale (ns-µs) accessible to 
# molecular dynamics simulations. You can read one of the many excellent reviews
# on metadynamics (see e.g. 
# `Bussi and Branduardi (2015) <https://doi.org/10.1002/9781118889886.ch1>`_)
# In short, during a metadynamics simulation an adaptive biasing potential is 
# built as a superimposition of Gaussians centered over configurations that have
# been previously visited by the trajectory. This discourages the system from remaining
# in high-probability configurations and accelerates sampling of free-energy barriers.
#
# Crucially, the bias is *not* built relative to the Cartesian coordinates of the atoms,
# but relative to a lower-dimensional description of the system (so-called collective 
# variables) that are suited to describe the processes being studied. 
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
with open("data/input-md.xml", "r") as file:
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
#    i-pi data/input_pimd.xml > log &
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

ipi_process = subprocess.Popen(["i-pi", "data/input-md.xml"])
time.sleep(2)  # wait for i-PI to start
driver_process = [subprocess.Popen(["i-pi-driver", "-u", "-a", "zundel", "-m", "zundel"], cwd="data/") for i in range(1)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files _before_ i-PI and lammps have finished running, by
# skipping this cell

ipi_process.wait()
for process in driver_process:
    process.wait()


# We can launch the external processes from a Python script as follows

ipi_process = subprocess.Popen(["i-pi", "data/input-pimd.xml"])
time.sleep(2)  # wait for i-PI to start
driver_process = [subprocess.Popen(["i-pi-driver", "-u", "-a", "zundel", "-m", "zundel"], cwd="data/") for i in range(4)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files _before_ i-PI and lammps have finished running, by
# skipping this cell

ipi_process.wait()
for process in driver_process:
    process.wait()

# %%
# After the simulation has run, you can visualize and post-process the trajectory data.
# Note that i-PI prints a separate trajectory for each bead, as structural properties
# can be computed averaging over the configurations of any of the beads.

# drops first frame where all atoms overlap
output_data, output_desc = ipi.read_output("metad-md.out")
traj_data = [ipi.read_trajectory(f"metad-md.pos_{i}.xyz")[1:] for i in range(1)]


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
# While the potential energy is simply the mean over the beads of the
# energy of individual replicas, computing the kinetic energy requires
# averaging special quantities that involve also the correlations between beads.
# Here we compare two of these *estimators*: the 'thermodynamic' estimator becomes
# statistically inefficient when increasing the number of beads, whereas the
# 'centroid virial' estimator remains well-behaved. Note how quickly these estimators
# equilibrate to roughly their stationary value, much faster than the equilibration
# of the potential energy above. This is thanks to the ``pile_g`` thermostat
# (see `DOI:10.1063/1.3489925 <http://doi.org/10.1063/1.3489925>`_) that is
# optimally coupled to the normal modes of the ring polymer.

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["kinetic_cv"],
    "b-",
    label="Centroid virial, $K_{CV}$",
)
ax.plot(
    output_data["time"],
    output_data["kinetic_td"],
    "r-",
    label="Thermodynamic, $K_{TD}$",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"energy / eV")
ax.legend()

# %%
# You can also visualize the (very short) trajectory in a way that highlights the
# fast spreading out of the beads of the ring polymer. ``chemiscope`` provides a
# utility function to interleave the trajectories of the beads, forming a trajectory
# that shows the connecttions between the replicas of each atom. Each atom and its
# connections are color-coded.

traj_pimd = chemiscope.ase_merge_pi_frames(traj_data)
# we also tweak the visualization options, and then show the viewer
traj_pimd["shapes"]["paths"]["parameters"]["global"]["radius"] = 0.05
traj_pimd["settings"]["structure"][0].update(
    dict(
        atoms=False,
        keepOrientation=True,
        color={"property": "bead_id", "palette": "hsv (periodic)"},
    )
)

chemiscope.show(**traj_pimd, mode="structure")

# %%
# Accelerating PIMD with a PIGLET thermostat
# ------------------------------------------
#
# The simulations in the previous sections are very far from converged -- typically
# one would need approximately 32 replicas to converge a simulation of
# room-temperature water. To address this problem we will use a method based on
# generalized Langevin equations, called
# `PIGLET <http://doi.org/10.1103/PhysRevLett.109.100604>`_
#
# The input file is ``input_piglet.xml``, that only differs by the definition of
# the thermostat, that uses a ``nm_gle`` mode in which each normal mode
# of the ring polymer is attached to a different colored-noise Generalized Langevin
# equation. This makes it possible to converge exactly the simulation results with
# a small number of replicas, and to accelerate greatly convergence for realistic
# systems such as this. The thermostat parameters can be generated on
# `the GLE4MD website <https://tinyurl.com/4y2e45jx>`_
#

ipi_process = subprocess.Popen(["i-pi", "data/input_piglet.xml"])
time.sleep(2)  # wait for i-PI to start
lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(2)]

ipi_process.wait()
lmp_process[0].wait()
lmp_process[1].wait()

# %%
# The mean potential energy from the PIGLET trajectory is higher than that for the
# PIMD one, because it is closer to the converged value (try to run a PIMD trajectory
# with 64 beads for comparison)

# drops first frame
output_gle, desc_gle = ipi.read_output("simulation_piglet.out")
traj_gle = [ipi.read_trajectory(f"simulation_piglet.pos_{i}.xyz")[1:] for i in range(8)]

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["potential"] - output_data["potential"][0],
    "b--",
    label="PIMD",
)
ax.plot(
    output_gle["time"],
    output_gle["potential"] - output_gle["potential"][0],
    "b-",
    label="PIGLET",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"energy / eV")
ax.legend()

# %%
# However, you should be somewhat careful: PIGLET converges *some* but not all the
# correlations within a path. For instance, it is designed to converge the
# centroid-virial estimator for the kinetic energy, but not the thermodynamic
# estimator. For the same reason, don't try to look at equilibration in terms of
# the mean temperature: it won't match the target value, because PIGLET uses a
# Langevin equation that breaks the classical fluctuation-dissipation theorem, and
# generates a steady-state distribution that mimics quantum fluctuations.

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(output_data["time"], output_data["kinetic_cv"], "b--", label="PIMD, $K_{CV}$")
ax.plot(output_gle["time"], output_gle["kinetic_cv"], "b", label="PIGLET, $K_{CV}$")
ax.plot(output_data["time"], output_data["kinetic_td"], "r--", label="PIMD, $K_{TD}$")
ax.plot(output_gle["time"], output_gle["kinetic_td"], "r", label="PIGLET, $K_{TD}$")
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"energy / eV")
ax.legend()

# %%
# Kinetic energy tensors
# ~~~~~~~~~~~~~~~~~~~~~~
#
# While we're at it, let's do something more complicated (and instructive).
# Classically, the momentum distribution of any atom is isotropic, so the
# kinetic energy tensor (KET) :math:`\mathbf{p}\mathbf{p}^T/2m` is a constant
# times the identity matrix. Quantum mechanically, the kinetic energy tensor
# has more structure, that reflects the higher kinetic energy of particles
# along directions with stiff bonds. We can compute a moving average of the
# centroid virial estimator of the KET, and plot it to show the direction
# of anisotropy. Note that there are some subtleties connected with the
# evaluation of the moving average, see e.g.
# `DOI:10.1103/PhysRevLett.109.100604 <http://doi.org/10.1103/PhysRevLett.109.100604>`_

# %%
# We first need to postprocess the components of the kinetic energy tensors
# (that i-PI prints out separating the diagonal and off-diagonal bits), averaging
# them over the last 10 frames and combining them with the centroid configuration
# from the last frame in the trajectory.

kinetic_cv = ipi.read_trajectory("simulation_piglet.kin.xyz")[1:]
kinetic_od = ipi.read_trajectory("simulation_piglet.kod.xyz")[1:]
kinetic_tens = np.hstack(
    [
        np.asarray([k.positions for k in kinetic_cv[-10:]]).mean(axis=0),
        np.asarray([k.positions for k in kinetic_od[-10:]]).mean(axis=0),
    ]
)

centroid = traj_gle[-1][-1].copy()
centroid.positions = np.asarray([t[-1].positions for t in traj_gle]).mean(axis=0)
centroid.arrays["kinetic_cv"] = kinetic_tens

# %%
# We can then view these in ``chemiscope``, setting the proper parameters to
# visualize the ellipsoids associated with the KET. Note that some KETs have
# negative eigenvalues, because we are averaging over a few frames, which is
# insufficient to converge the estimator fully.

ellipsoids = chemiscope.ase_tensors_to_ellipsoids(
    [centroid], "kinetic_cv", scale=15, force_positive=True
)

chemiscope.show(
    [centroid],
    shapes={"kinetic_cv": ellipsoids},
    mode="structure",
    settings=chemiscope.quick_settings(
        structure_settings={
            "shape": ["kinetic_cv"],
            "unitCell": True,
        }
    ),
)
