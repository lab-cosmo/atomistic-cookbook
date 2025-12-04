# -*- coding: utf-8 -*-
"""
ML collective variables in PLUMED with metatomic
================================================

:Authors: Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_;
          Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_;
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example shows how to build a `metatomic model
<https://docs.metatensor.org/metatomic/latest/overview.html>`_ that computes order
parameters for a Lennard-Jones cluster, and how to use it with the `PLUMED
<https://www.plumed.org/>`_ package to run a metadynamics simulation.

.. note::
    This is currently disabled due to persistent dependency resolution issues.

The LJ38 cluster is a classic benchmark system because its global minimum energy
structure is a truncated octahedron with :math:`O_h` symmetry, which is difficult to
find with simple optimization methods. The PES has a multi-funnel landscape, meaning the
system can easily get trapped in other local minima. Our goal is to explore the PES,
moving from a random initial configuration to the low-energy structures. To do this, we
will:

1.  Define a set of **collective variables (CVs)** that can distinguish between the
    disordered (liquid-like) and ordered (solid-like) states of the cluster. We will use
    two sets of CVs: histograms of the coordination number of atoms, and two CVs derived
    from SOAP descriptors that are analogous to the **Steinhardt order parameters**
    :math:`Q_4` and :math:`Q_6` (a.k.a the bond-order parameters).
2.  Implement these custom CV using ``featomic``, ``metatensor``, and ``metatomic`` to
    create a portable ``metatomic`` model.
3.  Run metadynamics trajectories with LAMMPS, and visualize the system as it explores
    different configurations.
4.  Show an example of integration with `i-PI <https://ipi-code.org/>`_, that uses
    multiple time stepping to reduce the cost of computing complicated CVs.

As usual for these examples, the simulation is run on a small system and for a short
time, so that results will be fast but inaccurate. If you want to use this example as a
template, you should set more appropriate parameters.
"""

# %%
