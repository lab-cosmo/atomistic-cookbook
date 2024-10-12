"""
Constant-temperature MD and thermostats
=======================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This recipe gives a practical introduction to finite-temperature 
molecular dynamics simulations, and provide  a guide to choose the
most appropriate thermostat for the simulation at hand. 

As for other examples in the cookbook, a small simulation of liquid
water is used as an archetypal example. This 
`seminal paper by H.C.Andersen <https://doi.org/10.1063/1.439486>`_
provides a good historical introduction to the problem of 
thermostatting.
"""

# %%
import subprocess
import time
import os
import xml.etree.ElementTree as ET

import chemiscope
import ipi
from ipi.utils.tools.acf_xyz import compute_acf_xyz
import matplotlib.pyplot as plt
import numpy as np

# %%
%matplotlib widget

# %%
# Constant-temperature sampling of (thermo)dynamics
# -------------------------------------------------
#
# Even though Hamilton's equations in classical mechanics conserve the total
# energy of the group of atoms in a simulation, experimental boundary conditions
# usually involve exchange of heat with the surroundings, especially when considering
# the relatively small supercells that are often used in simulations.
# 
# The goal of a constant-temperature MD simulation is to compute efficiently thermal
# averages of the form :math`\langle A(q,p)\rangle>\beta`, where the average
# of the observable :math:`A(q,p)` is 
# evaluated over the Boltzmann distribution at inverse temperature 
# :math:`\beta=1/k_\mathrm{B}T`, 
# :math:`P(q,p)=Q^{-1} \exp(-\beta(p^2/2m + V(q)))`.
# In all these scenarios, optimizing the simulation involves reducing as much as
# possible the *autocorrelation time* of the observable. 
# 
# Constant-temperature sampling is also important when one wants to compute 
# *dynamical* properties. In principle these would require 
# constant-energy trajectories, as any thermostatting procedure modifies
# the dynamics of the system. However, the initial conditions 
# should usually be determined from constant-temperature conditions,
# averaging over multiple constant-energy trajectories. 
# As we shall see, this protocol can often be simplified greatly, by choosing 
# thermostats that don't interfere with the natural microscopic dynamics. 

# %%
# Running simulations
# ~~~~~~~~~~~~~~~~~~~
#
# We use `i-PI <http://ipi-code.org>`_ together with a ``LAMMPS`` driver to run
# all the simulations in this recipe. The two codees need to be ran separately,
# and communicate atomic positions, energy and forces through a socket interface.
#
# The LAMMPS input defines the parameters of the 
# `q-TIP4P/f water model <http://doi.org/10.1063/1.3167790>`_, 
# while the XML-formatted input of i-PI describes the setup of the 
# MD simulation.
#
# We begin running a constant-energy calculation, that 
# we will use to illustrate the metrics that can be applied to 
# assess the performance of a thermostatting scheme.

# Open and read the XML file
with open("data/input_nve.xml", "r") as file:
    xml_content = file.read()
print(xml_content)

# %%
# Note that this -- and other runs in this example -- are too short to 
# provide quantitative results, and you may wat to increase the 
# ``<total_steps>`` parameter so that the simulation runs for at least
# a few tens of ps. The time step of 1 fs is also at the limit of what 
# is acceptable for running simulations of water. 0.5 fs would be a 
# safer, stabler value.


# %%
# To launch i-PI and LAMMPS from the command line you can jus
# execute the following commands
#
# .. code-block:: bash
#
#    i-pi data/input_nve.xml > log &
#    sleep 2
#    lmp -in data/in.lmp &
#
# To launch the external processes from a Python script 
# qw proeed as follows


ipi_process = None
if not os.path.exists("simulation_nve.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input_nve.xml"])
    time.sleep(4)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()


# %%
# Analyzing the simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# After the simulation is finished, we can look at the outputs

output_data, output_desc = ipi.read_output("simulation_nve.out")
traj_data = ipi.read_trajectory(f"simulation_nve.pos_0.xyz")

# %%
# The trajectory shows mostly local vibrations on this short time scale,
# but if you re-run with a longer ``<total_steps>`` settings you should be 
# able to observe diffusing molecules in the liquid.

chemiscope.show(traj_data, mode="structure",
                    settings=chemiscope.quick_settings(trajectory=True,
                                                   structure_settings={"unitCell":True})
)

# %%
# Potential and kinetic energy fluctuate, but the total energy is 
# (almost) constant, the small fluctuations being due to integration
# errors, that are quite large with the long time step used for this
# example. 

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["potential"] - output_data["potential"][0],
    "b-",
    label="Potential, $V$",
)
ax.plot(
    output_data["time"],
    output_data["kinetic_md"],
    "r-",
    label="Centroid virial, $K_{CV}$",
)
ax.plot(
    output_data["time"],
    output_data["conserved"] - output_data["conserved"][0],
    "k-",
    label="Conserved, $H$",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"energy / eV")
ax.legend()
plt.show()

# %% DRAFT
# Temperature  - comment on how this is off the target, 
# but explain it's just the kinetic energy and doesn't mean much here

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["temperature"],
    "k-",
    label="All atoms"
)
ax.plot(
    output_data["time"],
    output_data["temperature(O)"],
    "r-",
    label="O atoms",
)
ax.plot(
    output_data["time"],
    output_data["temperature(H)"],
    "c-",
    label="H atoms",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$\tilde{T}$ / K")
ax.legend()
plt.show()


# %%
# DRAFT - compute v-v acf 
acf_nve = compute_acf_xyz("simulation_nve.vel_0.xyz",
                      maximum_lag=300, length_zeropadding=2000,
                      spectral_windowing="cosine-blackman", 
                      timestep=1, time_units="femtosecond",
                      skip=100)

# %%
# DRAFT - plot ACF, comment on shape of peaks (very short trajectory!)
# and what they mean (omega->0 ~ diffusion)

ha2cm1 = 219474.63
fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    acf_nve[3][:1000]*ha2cm1, 
    acf_nve[4][:1000]*1e5,
    "r-",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$c_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
# Langevin thermostatting
# -----------------------
# DRAFT - intro to Langevin, equations, etc

xmlroot = ET.parse("data/input_higamma.xml").getroot()

# %%
# The `<ffsocket>` block describe the way communication will occur with the
# driver code

print("      " + ET.tostring(xmlroot.find(".//dynamics"), encoding="unicode"))

# %%


ipi_process = None
if not os.path.exists("simulation_higamma.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input_higamma.xml"])
    time.sleep(4)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()

# %%   

output_data, output_desc = ipi.read_output("simulation_higamma.out")
traj_data = ipi.read_trajectory(f"simulation_higamma.pos_0.xyz")

# %% DRAFT
# Temperature  - now this is 100% on top of the target, and 
# O and H are perfectly equipartitioned

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["temperature"],
    "k-",
    label="All atoms"
)
ax.plot(
    output_data["time"],
    output_data["temperature(O)"],
    "r-",
    label="O atoms",
)
ax.plot(
    output_data["time"],
    output_data["temperature(H)"],
    "c-",
    label="H atoms",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$\tilde{T}$ / K")
ax.legend()
plt.show()


# %%
# DRAFT - compute v-v acf 
acf_higamma = compute_acf_xyz("simulation_higamma.vel_0.xyz",
                      maximum_lag=300, length_zeropadding=2000,
                      spectral_windowing="cosine-blackman", 
                      timestep=1, time_units="femtosecond",
                      skip=100)

# %%
# DRAFT - plot ACF, comment on peak broadening, and
# the reduction of the diffusion coefficient

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    acf_nve[3][:1000]*ha2cm1, 
    acf_nve[4][:1000]*1e5,
    "r-",
    label="NVE",
)
ax.plot(
    acf_higamma[3][:1000]*ha2cm1, 
    acf_higamma[4][:1000]*1e5,
    "b-",
    label=r"Langevin, $\tau=10$fs",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$c_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
# R-L purification
# ~~~~~~~~~~~~~~~~
# maybe include the R-L purification (requires moving 
# it to tools)

# %%
# Global thermostats: stochastic velocity rescaling
# -------------------------------------------------

# %%
# The `<ffsocket>` block describe the way communication will occur with the
# driver code

xmlroot = ET.parse("data/input_svr.xml").getroot()
print("        " + ET.tostring(xmlroot.find(".//thermostat"), encoding="unicode"))

# %%


ipi_process = None
if not os.path.exists("simulation_svr.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input_svr.xml"])
    time.sleep(4)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()

# %%   

output_data, output_desc = ipi.read_output("simulation_svr.out")
traj_data = ipi.read_trajectory(f"simulation_svr.pos_0.xyz")

# %% DRAFT
# Temperature  - now this is 100% on top of the target, and 
# O and H are perfectly equipartitioned

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["temperature"],
    "k-",
    label="All atoms"
)
ax.plot(
    output_data["time"],
    output_data["temperature(O)"],
    "r-",
    label="O atoms",
)
ax.plot(
    output_data["time"],
    output_data["temperature(H)"],
    "c-",
    label="H atoms",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$\tilde{T}$ / K")
ax.legend()
plt.show()


# %%
# DRAFT - compute v-v acf 
acf_svr = compute_acf_xyz("simulation_svr.vel_0.xyz",
                      maximum_lag=300, length_zeropadding=2000,
                      spectral_windowing="cosine-blackman", 
                      timestep=1, time_units="femtosecond",
                      skip=100)

# %%
# DRAFT - plot ACF, note this is too short, and statistically equivalent

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    acf_nve[3][:1000]*ha2cm1, 
    acf_nve[4][:1000]*1e5,
    "r-",
    label="NVE",
)
ax.plot(
    acf_svr[3][:1000]*ha2cm1, 
    acf_svr[4][:1000]*1e5,
    "b-",
    label=r"SVR, $\tau=10$fs",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$c_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
# Generalized Langevin Equation thermostat
# ----------------------------------------

# %%
# The `<ffsocket>` block describe the way communication will occur with the
# driver code

xmlroot = ET.parse("data/input_gle.xml").getroot()
print("  " + ET.tostring(xmlroot.find(".//thermostat"), encoding="unicode"))

# %%


ipi_process = None
if not os.path.exists("simulation_svr.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input_svr.xml"])
    time.sleep(4)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()

# %%   

output_data, output_desc = ipi.read_output("simulation_svr.out")
traj_data = ipi.read_trajectory(f"simulation_svr.pos_0.xyz")

# %% DRAFT
# Temperature  - now this is 100% on top of the target, and 
# O and H are perfectly equipartitioned

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    output_data["temperature"],
    "k-",
    label="All atoms"
)
ax.plot(
    output_data["time"],
    output_data["temperature(O)"],
    "r-",
    label="O atoms",
)
ax.plot(
    output_data["time"],
    output_data["temperature(H)"],
    "c-",
    label="H atoms",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$\tilde{T}$ / K")
ax.legend()
plt.show()


# %%
# DRAFT - compute v-v acf 
acf_svr = compute_acf_xyz("simulation_svr.vel_0.xyz",
                      maximum_lag=300, length_zeropadding=2000,
                      spectral_windowing="cosine-blackman", 
                      timestep=1, time_units="femtosecond",
                      skip=100)

# %%
# DRAFT - plot ACF, note this is too short, and statistically equivalent

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    acf_nve[3][:1000]*ha2cm1, 
    acf_nve[4][:1000]*1e5,
    "r-",
    label="NVE",
)
ax.plot(
    acf_svr[3][:1000]*ha2cm1, 
    acf_svr[4][:1000]*1e5,
    "b-",
    label=r"SVR, $\tau=10$fs",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$c_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
