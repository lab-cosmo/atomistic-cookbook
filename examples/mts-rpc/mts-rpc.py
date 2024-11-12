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
`M. Tuckerman, B. J. Berne, and G. J. Martyna, JCP 97(3), 1990 (1992) <https://doi.org/10.1063/1.463137>`_
and can be applied to classical simulations,
typically to avoid the evaluation of long-range electrostatics in classical potentials. 

The second is named `ring polymer contraction`, first introduced in 
`T. E. Markland and D. E. Manolopoulos, JCP 129(2), 024105 (2008) <https://doi.org/10.1063/1.2953308>`_
can be seen as performing a similar simplification `in imaginary time`,
evaluating the expensive part of the potential on a smaller number of PI replicas. 

The techniques can be combined, which reduces even further the computational effort.
This dual approach, which was introduced in
`V. Kapil, J. VandeVondele, and M. Ceriotti, JCP 144(5), 054111 (2016) <(https://doi.org/10.1063/1.4941091>`_
and `O. Marsalek and T. E. Markland, JCP 144(5), (2016) <https://doi.org/10.1063/1.4941093>`_,
is the one that we will discuss here, allowing us to showcase two advanced features of i-PI.
It is worth stressing that MTS and/or RPC can be used very conveniently together with
machine-learning potentials
(see e.g. `V. Kapil, J. Behler, and M. Ceriotti, JCP 145(23), 234103 (2016 <https://doi.org/10.1063/1.4971438>`_ 
for an early application). 
"""

import numpy as np
import matplotlib.pyplot as plt
import ase, ase.io
import chemiscope

import ipi.utils.parsing as pimdmooc
import warnings

# pimdmooc.add_ipi_paths()


# %%
# Multiple time stepping in real and imaginary time
# -------------------------------------------------
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
