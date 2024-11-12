"""
Multiple time stepping and ring-polymer contraction
===================================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm>` and
           Davide Tisi `@DavideTisi <https://github.com/DavideTisi>`_

This notebook provides an introduction to two closely-related techniques,
that are geared towards reducing the cost of calculations by separating
slowly-varying (and computationally-expensive) components of the potential
energy from the fast-varying (and hopefully cheaper) ones.

The first is named `multiple time stepping`, and is a well-established technique
to avoid evaluating the slowly-varying components at every time step of a MD simulation.
It was first introduced in `LAMMPS <https://lammps.org>`_.
`M. Tuckerman, B. J. Berne, and G. J. Martyna,
JCP 97(3), 1990 (1992) <https://doi.org/10.1063/1.463137>`_
and can be applied to classical simulations,
typically to avoid the evaluation of long-range electrostatics in classical potentials.

The second is named `ring polymer contraction`, first introduced in
`T. E. Markland and D. E. Manolopoulos, JCP 129(2),
024105 (2008) <https://doi.org/10.1063/1.2953308>`_
can be seen as performing a similar simplification `in imaginary time`,
evaluating the expensive part of the potential on a smaller number of PI replicas.

The techniques can be combined, which reduces even further the computational effort.
This dual approach, which was introduced in
`V. Kapil, J. VandeVondele, and M. Ceriotti, JCP 144(5),
054111 (2016) <(https://doi.org/10.1063/1.4941091>`_
and `O. Marsalek and T. E. Markland, JCP 144(5),
(2016) <https://doi.org/10.1063/1.4941093>`_,
is the one that we will discuss here, allowing us to showcase two advanced features of i-PI.
It is worth stressing that MTS and/or RPC can be used very conveniently together with
machine-learning potentials
(see e.g. `V. Kapil, J. Behler, and M. Ceriotti, JCP 145(23), 234103 (2016 <https://doi.org/10.1063/1.4971438>`_
for an early application).
"""

import subprocess
import time
import warnings

import ase
import ase.io
import chemiscope
import ipi
import ipi.utils.parsing as pimdmooc
import matplotlib.pyplot as plt
import numpy as np


# pimdmooc.add_ipi_paths()


# %%
# some utility functions that will be usefull
def correlate(x, y, xbar=None, ybar=None, normalize=True):
    """Computes the correlation function of two quantities.
    It can be given the exact averages as parameters."""
    if xbar is None:
        xbar = x.mean()
    if ybar is None:
        ybar = y.mean()

    cf = np.correlate(x - xbar, y - ybar, mode="same")
    return cf[len(x) // 2 :] / (((x - xbar) * (y - ybar)).sum() if normalize else 1)


def autocorrelate(x, xbar=None, normalize=True):
    """Computes the autocorrelation function of a trajectory.
    It can be given the exact average as a parameter"""

    if xbar is None:
        xbar = x.mean()
    acf = np.correlate(x - xbar, x - xbar, mode="same")
    return acf[len(x) // 2 :] / (((x - xbar) * (x - xbar)).sum() if normalize else 1)


# %%
# Multiple time stepping in real and imaginary time
# -------------------------------------------------
#
# The core underlying assumption in these techniques is that the potential
# can be decomposed into a short-range/fast-varying/computationally-cheap
# part :math:`V_\mathrm{sr}` and a long-range/slow-varying/computationally-expensive
# part :math:`V_\mathrm{lr}`.  This is usually written as
# :math:`V=V_\mathrm{sr} +V_\mathrm{lr}`, although in many cases $V_\mathrm{sr}$
# is a cheap approximation of the potential, and :math:`V_\mathrm{lr}`
# is taken to be the difference between this potential and the full one.
#
# .. figure:: pimd-mts-pots.png
#    :align: center
#    :width: 600px
#
#    A smooth and rough potential components combine to form the total potential
#    energy function used in a simulation.

# %%
# way this is realized in practice is by splitting the propagation of
# Hamilton's equations into an inner loop that uses the fast/cheap force,
# and an outer loop that applies the slow force, using a larger time step
# (and therefore giving a larger "kick").
#
# .. figure:: pimd-mts-integrator.png
#    :align: center
#    :width: 600px
#
#    Schematic representation of the application of slow and fast
#    forces in a multiple time step molecular dynamics algorithm
#
# .. math::
#
#   \begin{split}
#   &p \leftarrow p + f_\mathrm{lr} \, dt/2 \\
#   &\left.
#   \begin{split}
#   &p \leftarrow p + f_\mathrm{sr} \, dt/2M \\
#   &q \leftarrow q + p \, dt/M \\
#   &p \leftarrow p + f_\mathrm{sr} \, dt/2M \\
#   \end{split}
#   \right\} M\ \mathrm{times}\\
#   &p \leftarrow p + f_\mathrm{lr} \, dt/2 \\
#   \end{split}
#
# This approach can (and usually is) complemented by aggressive
# thermostatting, which helps stabilize the dynamics in the
# limit of large :math:`M`.
#
# For a detailed discussion on how thermostatting aids
# in this context, see:
# `J. A. Morrone, T. E. Markland, M. Ceriotti, and B. J. Berne, JCP 134(1), 14103 (2011) <https://doi.org/10.1063/1.3518369>`_

# %%
# The idea behind ring-polymer contraction is very similar:
# it is unnecessary to evaluate on a very fine-grained discretization
# of the path integral components of the potential that are slowly-varying.
#
# .. image:: pimd-mts-rpc.png
#    :align: center
#    :width: 600px
#
# As shown in the right-hand panel above, ring-polymer contraction
# is realized by computing a Fourier interpolation of the bead positions,
# :math:`\tilde{\mathbf{q}}^{(k)}`, and then evaluating the total potential
# that enters the ring-polymer Hamiltonian as
#
# .. math::
#
#    V(\mathbf{q}) = \sum_{k=1}^P V_\mathrm{sr}(\mathbf{q}^{(k)}) + \frac{P}{P'}
#      \sum_{k=1}^{P'} V_\mathrm{lr}(\tilde{\mathbf{q}}^{(k)})
#
# where :math:`P` and :math:`P'` indicate the full and contracted discretizations of the path.

# %%
# A reference calculation using PIGLET
# ------------------------------------
#
# First, let's run a reference calculation without RPC or MTS.
# These calculations will be done for the q-TIP4P/f water model,
# `S. Habershon, T. E. Markland, and D. E. Manolopoulos, JCP 131(2), 24501 (2009) <https://doi.org/10.1063/1.3167790>`_
# , that contains a Morse-potential anharmonic intra-molecular potential,
# and an inter-molecular potential based on a Lennard-Jones term and a 4-point
# electrostatic model (the venerable TIP4P idea).
# It is fitted to reproduce experimental properties of water `when performing PIMD calculations`
# and it captures nicely several subtle effects while being cheap and easy-to-implement.
# Easy enough to have it in the built-in driver distributed with i-PI.
# The input for this run is `h2o_pimd.xml`, and we will use the
# `-m qtip4pf` option of `i-pi-driver` to compute the appropriate potential.
# The simulation involves a respectable box containing 216 water molecules, and is run with
# 8 beads and a `PIGLET` thermostat (cf.
# `M. Ceriotti and D. E. Manolopoulos, Phys. Rev. Lett. 109(10), 100604 (2012) <https://doi.org/10.1103/PhysRevLett.109.100604>`_
# . For simplicity, we use the constant-volume `NVT` ensemble, but you can easily
# modify the input to perform constant-pressure simulations.
#
# The important parts of the simulation
# - which we will modify to run a RPC/MTS simulation -
# are the definition of the forcefield socket
#

# Open and read the XML file
with open("data/h2o_pimd.xml", "r") as file:
    lines = file.readlines()

for line in lines[7:10]:
    print(line, end="")

# %%
#
# The important parts of the simulation
# - which we will modify to run a RPC/MTS simulation -
# are the definition of the forcefield socket,
# with the corresponding force definition


for line in lines[7:10]:
    print(line, end="")

print("\n[...]\n")

for line in lines[15:18]:
    print(line, end="")

# %%
# ... the definition of the number of beads

print(lines[11], end="")

# %%
# ... and the time step

print(lines[23], end="")

# %%
# The `<thermostat>` section contains PIGLET parameters generated using the
# `GLE4MD website <https://gle4md.org/index.html?page=matrix&kind=piglet&centroid=kh_8-4&cw0=4000&ucw0=cm1&nbeads=8&temp=300&utemp=k&parset=20_8_t&outmode=ipi&aunits=ps&cunits=k>`_
# .

# %%
# Installing the Python driver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# i-PI comes with a FORTRAN driver, which however has to be installed
# from source. We use a utility function to compile it. Note that this requires
# a functioning build system with `gfortran` and `make`.

ipi.install_driver()

# %%
# Launch i-PI simulation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We are going to launch i-PI from here, and put it in background and detach the processes from the
# jupyter instance, so we can continue with the notebook.
# On the the command line, this amounts to launching
#
# .. code-block:: bash
#
#    PYTHONUNBUFFERED=1 i-pi data/h2o_pimd.xml &> log.pimd &
#    sleep 5
#    for i in `seq 1 4`; do
#        i-pi-driver -u -a qtip4pf -m qtip4pf -v &> log.driver.$i &
#    done
#

ipi_process = subprocess.Popen(["i-pi", "data/h2o_pimd.xml"])
time.sleep(5)  # wait for i-PI to start
lmp_process = [
    subprocess.Popen(["i-pi-driver", "-u", "-a", "qtip4pf", "-m", "qtip4pf", "-v"])
    for i in range(4)
]

# %%
# Multiple time stepping
# ----------------------
#
# Let's now run a classical MD simulation, with and without multiple time stepping.
# We use very conservative parameters and a weak thermostat, to be able to see the
# difference in time scales between the full and short-range parts of the potential.
# Let's first run a classical MD for reference. The input file is `data/h2o_md.xml`,
# nothing exciting to see there.
# The bash command this time would be:
# .. code-block:: bash
#
#    PYTHONUNBUFFERED=1 i-pi data/h2o_md.xml &> log.md &
#    sleep 5
#    i-pi-driver -u -a qtip4pf-md -m qtip4pf -v &> log.driver.$i &
#

ipi_process = subprocess.Popen(["i-pi", "data/h2o_md.xml"])
time.sleep(5)  # wait for i-PI to start
lmp_process = subprocess.Popen(
    ["i-pi-driver", "-u", "-a", "qtip4pf-md", "-m", "qtip4pf", "-v"]
)

# %%
# Let's have a look at `h2o_mts.xml`, that provides the parameters
# of the MTS calculation. We define two `ffsocket` sections:
# one will be used with the `qtip4pf` driver and the other with `qtip4pf-sr`.
# Note the different names of the sockets (that have to match the `-a`
# option in the invocation of the driver) and the internal labels that
# will be referred to in the `<forces>` section.

with open("data/h2o_mts.xml", "r") as file:
    lines = file.readlines()

for line in lines[7:13]:
    print(line, end="")

# %%
# Each `<force>` block contains a `<mts_weights>` section.
# This provides a list of weights that determine which force
# components are active at each level of the MTS hierarchy.
# These weights indicate that the smooth part (full minus short-range)
# is active in the outer loop, and the short-range part is active in the inner
# loop. Note that the implementation is smart enough to re-use the short-range
# potential computed in the inner loop, multiplying it with a weight of $-1$ to
# compute :math:`V_\mathrm{lr}=V-V_\mathrm{sr}`.
