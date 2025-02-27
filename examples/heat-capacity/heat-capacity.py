"""
Quantum heat capacity of water
==============================

:Authors:
    Filippo Bigi `@frostedoyster <https://github.com/frostedoyster/>`_;
    Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This example shows how to estimate the heat capacity of liquid water
from a path integral molecular dynamics simulation. The dynamics are
run with `i-PI <http://ipi-code.org>`_, and
`LAMMPS <http://lammps.org>`_ is used
as the driver to simulate the `q-TIP4P/f water
model <http://doi.org/10.1063/1.3167790>`_.
"""

# %%

import os
import subprocess
import time
import xml.etree.ElementTree as ET

import ipi
import matplotlib.pyplot as plt
import numpy as np


# %%
# A non-trivial energy estimator
# ------------------------------
#
# As introduced in the
# `path-integrals example
# <http://lab-cosmo.github.io/atomistic-cookbook/latest/examples/path-integrals>`_,
# path-integral estimators
# for observables that depend on momenta are generally not trivial to compute.
#
# In this example, we will focus on the constant-volume heat capacity,
# :math:`c_V`, which is one such
# observable, and we will calculate it for liquid water at room temperature.
# Because of the presence of high-frequency vibrations, many of the nuclear
# degrees of freedom are trapped in the vibrational ground state, which reduces
# substantially the heat capacity from the value that would be obtained
# in the classical limit. See `this review
# <http://doi.org/10.1021/acs.chemrev.5b00674>`_
# for an overview of the impact of quantum nuclei on the properties of water.
# From a computational perspective, this means that it is necessary to use
# specialized simulations and estimators to evaluate the correct value.


# %%
# Running the PIMD calculation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This follows the same steps as the ``path-integrals`` example. One important
# difference is that we will request the ``scaledcoords`` output to the relevant
# section of the ``i-PI`` input XML file, which
# contains estimators that can be used to calculate the total energy and
# heat capacity as following
# `Yamamoto, J. Chem. Phys. (2005) <https://arxiv.org/abs/physics/0505109>`_.
#
# The input file is shown below. It should be noted that ``scaledcoords``
# is given a finite differences displacement as a parameter. This is necessary
# as the estimators require higher order derivatives of the potential energy,
# which are calculated using finite differences. This also means that
# evaluating the estimator adds substantial overhead (so it is wise to only
# compute it every few simulation steps, to eliminate correlations between
# snapshots) and that one should be careful to use well-converged simulation
# parameters to avoid discontinuities and noise (for instance, we increase
# the accuracy of the particle-mesh electrostatic calculation, and use a
# shifted Lennard-Jones potential to avoid a discontinuity at the cutoff).

# Open and show the relevant part of the input
xmlroot = ET.parse("data/input.xml").getroot()
print("      " + ET.tostring(xmlroot.find(".//properties"), encoding="unicode"))

# %%
# We launch the i-PI and LAMMPS processes, exactly as in the
# ``path-integrals`` example.

# don't rerun if the outputs already exist
ipi_process = None
if not os.path.exists("water-cv.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input.xml"])
    time.sleep(2)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(2)]

# %%
# Skip this cell if you want to run in the background
if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()
    lmp_process[1].wait()

# %%
# Analyzing the results
# ~~~~~~~~~~~~~~~~~~~~~
# Let's plot the potential and conserved energy as a function of time,
# just to check that the simulation ran sensibly.

output_data, output_desc = ipi.read_output("water-cv.out")
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
ax.set_ylabel(r"energy / a.u.")
ax.legend()
plt.show()

# %%
# As described in the `i-PI documentation
# <https://ipi-code.org/i-pi/output-tags.html>`_,
# the two quantities returned by the ``scaledcoords`` output are ``eps_v``
# and ``eps_v'``, defined in the aforementioned
# `paper <https://arxiv.org/abs/physics/0505109>`_.
#
# These estimators (:math:`\epsilon_v` and :math:`\epsilon_v'`) are derived in the
# "scaled coordinates" formalism, which is a useful trick to avoid the
# growth of the error in the instantaneous values of the estimators with
# the number of beads used in the path integral simulation.
#
# The same paper contains the formulas to calculate the total energy and
# heat capacity from these estimators:
#
# .. math::
#   E = \langle \epsilon_v \rangle \quad
#   C_V = k_B \beta^2 \left( \langle \epsilon_v^2 \rangle - \langle
#       \epsilon_v \rangle^2 - \langle \epsilon_v' \rangle \right)
#
# First the energy, whose estimator will be compared to the total energy
# calculated as the sum of the potential and kinetic energy estimators.
# Since the kinetic energy is itself calculated from a scaled-coordinates
# estimator (the "centroid virial" estimator), the two total energies are
# the same.

eps_v = output_data["scaledcoords(fd_delta=5e-3)"][:, 0]
eps_v_prime = output_data["scaledcoords(fd_delta=5e-3)"][:, 1]

energy_estimator = eps_v  # first formula

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"],
    energy_estimator - energy_estimator[0],
    "b-",
    label="scaled coordinates estimator",
)
ax.plot(
    output_data["time"][:],
    (
        output_data["potential"]
        - output_data["potential"][0]
        + output_data["kinetic_cv"]
        - output_data["kinetic_cv"][0]
    ),
    "r.",
    label="potential + virial kinetic",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"total energy / a.u.")
ax.legend()
plt.show()


# %%
# And, finally, the heat capacity. Note that normally the simulation
# requires a few ps for equilibration. Here we discard a few dozen steps
# to eliminate the initial jump, which is due to the relaxation of the
# ring polymers starting from a single atomic configuration.

# i-PI scaledcoords outputs are in atomic units (see docs)
kB = 3.16681e-6  # Boltzmann constant in atomic units
T = 298.0  # temperature in K, as defined in the input file
beta = 1.0 / (kB * T)

skip = 20
heat_capacity = (  # second formula
    kB
    * (beta**2)
    * (
        np.mean(eps_v[skip:] ** 2)
        - np.mean(eps_v[skip:]) ** 2
        - np.mean(eps_v_prime[skip:])
    )
)
heat_capacity_per_molecule = heat_capacity / 32  # 32 molecules in the simulation
print(f"Heat capacity (per water molecule): {(heat_capacity_per_molecule/kB):.2f} kB")

# %%
# You may recognize that the first part of the estimator is reminiscent
# of the classical estimator for the heat capacity as the fluctuations of the
# (quantum) total energy, which in this case however requires a correction given
# by the mean of the second part of the scaled-coordinates estimator.

# %%
# Estimating the statistical error
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Especially with such an underconverged simulation, it is important to
# estimate the statistical error in the heat capacity.
#
# Generally, errors on measurements are computed
# as "standard errors", i.e. the standard deviation of a series of data points
# divided by the square root of the number of data points. In our case,
# however, this is made more complicated by the correlation between
# close steps in the molecular dynamics trajectory, which would lead to an
# overestimation of the number of independent samples. To fix this, we can
# calculate the autocorrelation time of the estimators whose errors we
# want to estimate, and apply a correction factor to the number of samples.


def autocorrelate(x):
    n = len(x)
    xo = x - x.mean()  # remove mean
    acov = (np.correlate(xo, xo, "full"))[n - 1 :]
    return acov[: len(acov) // 2]


def autocorrelation_time(x):
    acov = autocorrelate(x)
    return 1.0 + np.sum(acov) / acov[0]


# %%
# Using these helper functions, we can now calculate the error on the various
# parts of the heat capacity estimator. Note also the autocorrelation times, that
# are just a little larger than one, indicating that the stride used to print out
# the estimators is appropriate (as there is little correlation between the samples).

# Autocorrelation times (i.e. number of steps needed to have independent samples)
autocorr_time_error_delta_eps_v = autocorrelation_time(
    (eps_v[skip:] - eps_v[skip:].mean()) ** 2
)
autocorr_time_error_eps_v_prime = autocorrelation_time(eps_v_prime[skip:])

print(
    f"""
Autocorrelation times (in number of samples):
(eps-<eps>)^2: {autocorr_time_error_delta_eps_v:.2f}
eps':          {autocorr_time_error_eps_v_prime:.2f}
"""
)

# Effective number of samples
effective_samples_delta_eps_v = len(eps_v[skip:]) / autocorr_time_error_delta_eps_v
effective_samples_eps_v_prime = len(eps_v[skip:]) / autocorr_time_error_eps_v_prime

# Standard errors using the effective number of samples
error_delta_eps_v = np.std((eps_v[skip:] - eps_v[skip:].mean()) ** 2) / np.sqrt(
    effective_samples_delta_eps_v
)
error_eps_v_prime = np.std(eps_v_prime[skip:]) / np.sqrt(effective_samples_eps_v_prime)

# Error on the heat capacity (assuming quadrature sum)
error_heat_capacity = (
    kB * (beta**2) * np.sqrt(error_delta_eps_v**2 + error_eps_v_prime**2)
)

error_heat_capacity_per_molecule = (
    error_heat_capacity / 32
)  # 32 molecules in the simulation

print(
    "Error on the heat capacity (per water molecule): "
    f"{(error_heat_capacity_per_molecule/kB):.2f} kB"
)


# %%
# The obtained heat capacity is consistent with the values from the literature
# (see e.g. `Ceriotti et al., J. Chem. Phys. (2011)
# <http://doi.org/10.1063/1.3556661>`_
# where the convergence of the heat capacity with number of beads is shown for the same
# water model used in this example).
# However, the error is quite large, which is expected given the short simulation time.
# To reduce the error, one would need to run a longer simulation. Other important error
# sources, which are not accounted for in the error estimate, are the finite size of the
# system and number of beads. Both of these are too small in this example to give
# reliable results.
#
# In a realistic simulation, up to a few 100s of picoseconds might be needed to reduce
# the sampling error to a small value (1-2% of the heat capacity). For water at room
# temperature, you will need 32 beads at the very least (8 were used in this example).
# It is more difficult to give a general rule for the system size: (quantum) energy
# fluctuations are usually localized, but to guarantee accurate sampling of the
# liquid structure, a few hundred water molecules would be a reasonable guess
# (32 were used in this example).
