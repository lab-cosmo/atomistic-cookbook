"""
Quantum heat capacity of water
==============================

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
# A non-trivial estimator
# -----------------------
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
# contains estimators that can be used to calculate the total energy and
# heat capacity as defined in this `paper <https://arxiv.org/abs/physics/0505109>`_.
# In order to do this, the ``scaledcoords`` output is added to the relevant section
# of the ``i-PI`` input XML file.

# Open and read the XML file
with open("data/input.xml", "r") as file:
    xml_content = file.read()
print(xml_content)

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
plt.show()
plt.close()

# %%
# As described in the ``i-PI``
# <documentation https://ipi-code.org/i-pi/output-tags.html>,
# the two quantities returned by the ``scaledcoords`` output are ``eps_v``
# and ``eps_v'``, defined in the aforementioned
# `paper <https://arxiv.org/abs/physics/0505109>`_. The same paper contains
# the formulas to calculate the total energy and heat capacity from these
# estimators.
#
# First the energy:

eps_v = np.loadtxt("simulation.out")[100:, 6]
eps_v_prime = np.loadtxt("simulation.out")[100:, 7]
# discarding the first 100 steps which are highly non-equilibrated

energy_estimator = eps_v  # from the paper

fix, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.plot(
    output_data["time"][100:],
    energy_estimator,
    "b",
    label="Total energy$",
)
ax.set_xlabel(r"$t$ / ps")
ax.set_ylabel(r"$E / a.u.$")
ax.legend()
plt.show()
plt.close()

# %%
# And, finally, the heat capacity:

# i-PI scaledcoords outputs are in atomic units (see docs)
kB = 3.16681e-6  # Boltzmann constant in atomic units
T = 298.0  # temperature in K, as defined in the input file
beta = 1.0 / (kB * T)

heat_capacity = (
    kB * (beta**2) * (np.mean(eps_v**2) - np.mean(eps_v) ** 2 - np.mean(eps_v_prime))
)
heat_capacity_per_molecule = heat_capacity / 32  # 32 molecules in the simulation
print(f"Heat capacity (per water molecule): {(heat_capacity_per_molecule/kB):.2f} kB")


# %%
# Finding an error estimate
# -------------------------
#
# Especially with such an underconverged simulation, it is important to
# estimate the error in the heat capacity.
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
# We can now calculate the error on the heat capacity estimate.

# Autocorrelation times (i.e. number of steps needed to have independent samples)
autocorr_time_error_eps_v = autocorrelation_time(eps_v)
autocorr_time_error_eps_v_squared = autocorrelation_time(eps_v**2)
autocorr_time_error_eps_v_prime = autocorrelation_time(eps_v_prime)

# Effective number of samples
effective_samples_eps_v = len(eps_v) / autocorr_time_error_eps_v
effective_samples_eps_v_squared = len(eps_v) / autocorr_time_error_eps_v_squared
effective_samples_eps_v_prime = len(eps_v) / autocorr_time_error_eps_v_prime

# Standard errors using the effective number of samples
error_eps_v = np.std(eps_v) / np.sqrt(effective_samples_eps_v)
error_eps_v_squared = np.std(eps_v**2) / np.sqrt(effective_samples_eps_v_squared)
error_eps_v_prime = np.std(eps_v_prime) / np.sqrt(effective_samples_eps_v_prime)

# Error on the heat capacity
error_heat_capacity = (
    kB
    * (beta**2)
    * np.sqrt(  # error propagation in the sum of terms
        error_eps_v_squared**2
        + (2.0 * np.mean(eps_v) * error_eps_v) ** 2  # error of <eps_v>**2
        + error_eps_v_prime**2
    )
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
# (see ...).
# However, the error is quite large, which is expected given the short simulation time.
# To reduce the error, one would need to run a longer simulation. Other important error
# sources, which are not accounted for in the error estimate, are the finite size of the
# system and number of beads. Both of these are too small in this example to give
# reliable results.
#
# In a realistic simulation, up to a few 100s of picoseconds might be needed to reduce
# the sampling error to a small value (1-10% of the heat capacity). For water at room
# temperature, you will need 32 beads at the very least (8 were used in this example).
# It is more difficult to give a general rule for the system size, but a few hundred
# water molecules would be a reasonable guess (32 were used in this example).
