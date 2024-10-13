"""
Constant-temperature MD and thermostats
=======================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This recipe gives a practical introduction to finite-temperature
molecular dynamics simulations, and provide  a guide to choose the
most appropriate thermostat for the simulation at hand.

As for other examples in the cookbook, a small simulation of liquid
water is used as an archetypal example. Molecular dynamics, sampling,
and constant-temperature simulations are discussed in much detail in
the book "Understanding Molecular Simulations" by Daan Frenkel and Berend Smit.
This
`seminal paper by H.C.Andersen <https://doi.org/10.1063/1.439486>`_
provides a good historical introduction to the problem of
thermostatting, and this
`PhD thesis
<https://www.research-collection.ethz.ch/handle/20.500.11850/152344>`_
provides a more detailed background to several of the techniques
discussed in this recipe.
"""

import os

# %%
import subprocess
import time
import xml.etree.ElementTree as ET

import chemiscope
import ipi
import matplotlib.pyplot as plt
import numpy as np
from ipi.utils.tools.acf_xyz import compute_acf_xyz
from ipi.utils.tools.gle import get_gle_matrices, gle_frequency_kernel, isra_deconvolute


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
# assess the performance of a thermostatting scheme. If it is the
# first time you see an ``i-PI`` input, you may want to look at
# this file while checking the
# `input reference <https://docs.ipi-code.org/input-tags.html>`_.

# Open and read the XML file
with open("data/input_nve.xml", "r") as file:
    xml_content = file.read()
print(xml_content)

# %%
# The part of the input that describes the molecular dynamics integrator
# is the ``motion`` class. For this run, it specifies and *NVE* ensemble, and
# a ``timestep`` of 1 fs for the integrator.

xmlroot = ET.parse("data/input_nve.xml").getroot()
print("    " + ET.tostring(xmlroot.find(".//motion"), encoding="unicode"))

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
# proceed as follows:

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
# After the simulation is finished, we can look at the outputs.
# The outputs include the trajectory of positions, the velocities
# and a number of energetic observables

output_data, output_desc = ipi.read_output("simulation_nve.out")
traj_data = ipi.read_trajectory("simulation_nve.pos_0.xyz")

# %%
# The trajectory shows mostly local vibrations on this short time scale,
# but if you re-run with a longer ``<total_steps>`` settings you should be
# able to observe diffusing molecules in the liquid.

chemiscope.show(
    traj_data,
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True, structure_settings={"unitCell": True}
    ),
)

# %%
# Potential and kinetic energy fluctuate, but the total energy is
# (almost) constant, the small fluctuations being due to integration
# errors, that are quite large with the long time step used for this
# example. If you run with smaller ``<timestep>`` values, you should
# see that the energy conservation condition is fulfilled with higher
# accuracy.

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
    label="Kinetic, $K$",
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

# %%
# I a classical MD simulation, based on the momentum :math:`\mathbf{p}`
# of each atom, it is possible to evaluate its *kinetic temperature estimator*
# :math:`T=\langle \mathbf{p}^2/m \rangle /3k_B` the average is to be intended
# over a converged trajectory. Keep in mind that
#
# 1. The *instantaneous* value of this estimator is meaningless
# 2. It is only well-defined in a constant-temperature simulation, so here
#    it only gives a sense of whether atomic momenta are close to what one
#    would expect at 300 K.
#
# With these caveat in mind, we can observe that the simulation has higher
# velocities than expected at 300 K, and that there is no equipartition, the
# O atoms having on average a higher energy than the H atoms.

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(output_data["time"], output_data["temperature"], "k-", label="All atoms")
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
# In order to investigate the dynamics more carefully, we
# can compute the velocity-velocity autocorrelation function
# :math:`c_{vv}(t)=\sum_i \langle \mathbf{v}_i(t) \cdot \mathbf{v}_i(0) \rangle`.
# We use a utility function that reads the outputs of ``i-PI``
# and computes both the autocorrelation function and its Fourier
# transform.
# :math:`c_{vv}(t)` contains information on the time scale and amplitude
# of molecular motion, and is closely related to the vibrational density
# of states and to spectroscopic observables such as IR and Raman spectra.

acf_nve = compute_acf_xyz(
    "simulation_nve.vel_0.xyz",
    maximum_lag=600,
    length_zeropadding=2000,
    spectral_windowing="cosine-blackman",
    timestep=1,
    time_units="femtosecond",
    skip=100,
)

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    acf_nve[0][:1200] * 2.4188843e-05,  # atomic time to ps
    acf_nve[1][:1200] * 1e5,
    "r-",
)
ax.set_xlabel(r"$t$ / ps$")
ax.set_ylabel(r"$c_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
# The power spectrum (that can be computed as the Fourier transform of
# :math:`c_{vv}`, reveals the frequencies of stretching, bending and libration
# modes of water; the :math:`\omega\rightarrow 0` limit is proportional
# to the diffusion coefficient.
# We also load the results from a reference calculation (average of 8
# trajectories initiated from NVT-equilibrated samples, shown as the
# confidence interval).
# The differences are due to the short trajectory, and to the fact that the
# NVE trajectory is not equilibrated at 300 K.

ha2cm1 = 219474.63

# Loads reference trajectory
acf_ref = np.loadtxt("data/traj-all_facf.data")

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

ax.fill_between(
    acf_ref[:1200, 0] * ha2cm1,
    (acf_ref[:1200, 1] - acf_ref[:1200, 2]) * 1e5,
    (acf_ref[:1200, 1] + acf_ref[:1200, 2]) * 1e5,
    color="gray",
    label="reference",
)

ax.loglog(acf_nve[3][:1200] * ha2cm1, acf_nve[4][:1200] * 1e5, "r-", label="NVE")
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\hat{c}_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
# Langevin thermostatting
# -----------------------
#
# In order to perform a simulations that samples configurations
# consistent with a Boltzmann distribution :math:`e^{-V(x)/k_B T}`
# one needs to modify the equations of motion. There are many
# different approaches to do this, some of which lead to deterministic
# dynamics; the two more widely used deterministic thermostats
# are the
# `Berendsen thermostat <https://doi.org/10.1063/1.448118>`_
# which does not sample the Boltzmann distribution exactly and
# should never be used given the many more rigorous alternatives,
# and the Nosé-Hoover thermostat, that requires a
# `"chain" implementation <https://doi.org/10.1063/1.463940>`_
# to be ergodic, which amounts essentially to a complicated way
# to generate poor-quality pseudo-random numbers.
#
# Given the limitations of deterministic thermostats, in this
# recipe we focus on stochastic thermostats, that model the
# coupling to the chaotic dynamics of an external bath through
# explicit random numbers. Langevin dynamics amounts to adding
# to Hamiltonian, for each degree of freedom, a term of the form
#
# .. math::
#
#    \dot{p} =  -\gamma p + \sqrt{2\gamma m k_B T} \, \xi(t)
#
# where :math:`\gamma` is a friction coefficient, and :math:`\xi`
# uncorrelated random numbers that mimic collisions with the bath
# particles. The friction can be seen as the inverse of a
# characteristic *coupling time scale*
# :math:`\tau=1/\gamma` that describes how strongly the bath
# interacts with the system.

# %%
# Setting up a thermostat in ``i-PI``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In order to set up a thermostat in ``i-PI``, one simply needs
# to adjust the ``<dynamics>`` block, to perform ``nvt`` dynamics
# and include an appropriate ``<thermostat>`` section.
# Here we use a very-strongly coupled Langevin thermostat,
# with :math:`\tau=10~fs`.

xmlroot = ET.parse("data/input_higamma.xml").getroot()
print("      " + ET.tostring(xmlroot.find(".//dynamics"), encoding="unicode"))

# %%
# ``i-PI`` and ``lammps`` are launched as above ...

ipi_process = None
if not os.path.exists("simulation_higamma.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input_higamma.xml"])
    time.sleep(4)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]

# %%
# ... and you should probably wait until they're done, will be fast.

if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()

# %%
# Analysis of the trajectory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The temperature converges very quickly to the target value
# (fluctuations are to be expected, given that as discussed above
# the temperature estimator is just the instantaneous kinetic energy,
# that is not constant). There is also equipartition between O and H.

output_data, output_desc = ipi.read_output("simulation_higamma.out")
traj_data = ipi.read_trajectory("simulation_higamma.pos_0.xyz")

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(output_data["time"], output_data["temperature"], "k-", label="All atoms")
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
# The velocity-velocity correlation function
# shows how much this thermostat affects the
# system dynamics The high-frequency peaks,
# corresponding to stretches and bending, are
# greatly broadened, and the :math:`\omega\rightarrow 0`
# limit of :math:`\hat{c}_{vv}`, corresponding to the
# diffusion coefficient, is reduced by almost a factor of 5.
# This last observation highlights that a too-aggressive
# thermostat is not only disrupting the dynamics:
# it also slows down diffusion through phase space,
# making the dynamics less efficient at sampling slow,
# collective motions. We shall see further down various
# methods to counteract this effect, but in general one shold
# use s weaker coupling, that improves the sampling of configuration
# space even though it slows down the convergence of the
# kinetic energy.

# compute the v-v acf
acf_higamma = compute_acf_xyz(
    "simulation_higamma.vel_0.xyz",
    maximum_lag=600,
    length_zeropadding=2000,
    spectral_windowing="cosine-blackman",
    timestep=1,
    time_units="femtosecond",
    skip=100,
)

# and plot
fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.fill_between(
    acf_ref[:1200, 0] * ha2cm1,
    (acf_ref[:1200, 1] - acf_ref[:1200, 2]) * 1e5,
    (acf_ref[:1200, 1] + acf_ref[:1200, 2]) * 1e5,
    color="gray",
    label="reference",
)
ax.loglog(
    acf_higamma[3][:1200] * ha2cm1,
    acf_higamma[4][:1200] * 1e5,
    "b-",
    label=r"Langevin, $\tau=10$fs",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\hat{c}_{vv}$ / arb. units")
ax.legend()
plt.show()

# %%
# Global thermostats: stochastic velocity rescaling
# -------------------------------------------------
# TODO work in progress....

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
traj_data = ipi.read_trajectory("simulation_svr.pos_0.xyz")

# %% DRAFT
# Temperature  - now this is 100% on top of the target, and
# O and H are perfectly equipartitioned

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(output_data["time"], output_data["temperature"], "k-", label="All atoms")
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
acf_svr = compute_acf_xyz(
    "simulation_svr.vel_0.xyz",
    maximum_lag=600,
    length_zeropadding=2000,
    spectral_windowing="cosine-blackman",
    timestep=1,
    time_units="femtosecond",
    skip=100,
)

# %%
# DRAFT - plot ACF, note this is too short, and statistically equivalent

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.fill_between(
    acf_ref[:1200, 0] * ha2cm1,
    (acf_ref[:1200, 1] - acf_ref[:1200, 2]) * 1e5,
    (acf_ref[:1200, 1] + acf_ref[:1200, 2]) * 1e5,
    color="gray",
    label="reference",
)
ax.loglog(
    acf_svr[3][:1200] * ha2cm1,
    acf_svr[4][:1200] * 1e5,
    "b-",
    label=r"SVR, $\tau=10$fs",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\hat{c}_{vv}$ / arb. units")
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
if not os.path.exists("simulation_gle.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input_gle.xml"])
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

output_data, output_desc = ipi.read_output("simulation_gle.out")
traj_data = ipi.read_trajectory("simulation_gle.pos_0.xyz")

# %% DRAFT
# Temperature  - now this is 100% on top of the target, and
# O and H are perfectly equipartitioned

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(output_data["time"], output_data["temperature"], "k-", label="All atoms")
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
acf_gle = compute_acf_xyz(
    "simulation_gle.vel_0.xyz",
    maximum_lag=600,
    length_zeropadding=2000,
    spectral_windowing="cosine-blackman",
    timestep=1,
    time_units="femtosecond",
    skip=100,
)

# %%
# DRAFT - plot ACF, note this is too short, and statistically equivalent

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.fill_between(
    acf_ref[:1200, 0] * ha2cm1,
    (acf_ref[:1200, 1] - acf_ref[:1200, 2]) * 1e5,
    (acf_ref[:1200, 1] + acf_ref[:1200, 2]) * 1e5,
    color="gray",
    label="reference",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    acf_gle[4][:1200] * 1e5,
    "b-",
    label=r"GLE",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\hat{c}_{vv}$ / arb. units")
ax.legend()
plt.show()


# %%
len(acf_gle[3])
acf_gle[3]
# %%
# R-L purification
# ~~~~~~~~~~~~~~~~
# maybe include the R-L purification (requires moving
# it to tools)

n_omega = 1200
Ap, Cp, Dp = get_gle_matrices("data/input_gle.xml")
gle_kernel = gle_frequency_kernel(acf_gle[3][:n_omega], Ap, Dp)

# %%
#

isra_acf, history, errors, laplace = isra_deconvolute(
    acf_gle[3][:n_omega], acf_gle[4][:n_omega], gle_kernel, 10, 1
)

# %%
fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.fill_between(
    acf_ref[:1200, 0] * ha2cm1,
    (acf_ref[:1200, 1] - acf_ref[:1200, 2]) * 1e5,
    (acf_ref[:1200, 1] + acf_ref[:1200, 2]) * 1e5,
    color="gray",
    label="reference",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    acf_gle[4][:1200] * 1e5,
    "b-",
    label=r"GLE",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    isra_acf * 1e5,
    "r--",
    label=r"GLE$\rightarrow$ NVE",
)
ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\hat{c}_{vv}$ / arb. units")
ax.legend()
plt.show()


# %%
# demonstrate the iterations of ISRE
fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.fill_between(
    acf_ref[:1200, 0] * ha2cm1,
    (acf_ref[:1200, 1] - acf_ref[:1200, 2]) * 1e5,
    (acf_ref[:1200, 1] + acf_ref[:1200, 2]) * 1e5,
    color="gray",
    label="reference",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    acf_gle[4][:1200] * 1e5,
    "b-",
    label=r"GLE",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    history[0] * 1e5,
    ":",
    color="#4000FF",
    label=r"iter[0]",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    history[4] * 1e5,
    ":",
    color="#A000A0",
    label=r"iter[4]",
)
ax.loglog(
    acf_gle[3][:1200] * ha2cm1,
    history[8] * 1e5,
    ":",
    color="#FF0040",
    label=r"iter[8]",
)

ax.set_xlabel(r"$\omega$ / cm$^{-1}$")
ax.set_ylabel(r"$\hat{c}_{vv}$ / arb. units")
ax.legend()
plt.show()


# %%
# Running with LAMMPS
# -------------------
