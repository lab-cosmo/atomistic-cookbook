"""
Using non-conservative forces with PET-MAD
===========================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm>`_,
          Filippo Bigi `@frostedoyster <https://github.com/frostedoyster>`_

PET-MAD is introduced, and benchmarked for several challenging modeling tasks,
in `this preprint <https://arxiv.org/abs/2503.14118>`_. To get a first taste
of PET-MAD, for basic tasks such as geometry optimization and conservative
MD, see also
`this recipe <https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html>`_.
"""

# %%
#
# If you don't want to use the conda environment for this recipe,
# you can get all dependencies installing
# the `PET-MAD package <https://github.com/lab-cosmo/pet-mad>`_:
#
# .. code-block:: bash
#
#     pip install git+https://github.com/lab-cosmo/pet-mad.git
#

import os
import subprocess
from copy import copy, deepcopy

# i-PI scripting utilities
import ase.units
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import requests
from ipi.utils.parsing import read_output, read_trajectory
from ipi.utils.scripting import (
    InteractiveSimulation,
    forcefield_xml,
    motion_nvt_xml,
    simulation_xml,
)


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821


# %%
# Fetch PET-MAD and export the model
# ----------------------------------

## TODO

subprocess.run("cp data/*pt .", shell=True, check=True)

# %%
#
# Non-conservative forces
# -----------------------
#
# Interatomic potentials are typically used to compute the forces acting
# on the atoms by differentiating with respect to atomic positions, i.e. if
#
# .. math ::
#
#    V(\mathbf{r}_1, \mathbf{r}_2, \ldots \mathbf{r}_n)
#
# is the potential for an atomic configuration then the force acting on
# atom :math:`i` is
#
# .. math ::
#
#    \mathbf{f}_i = \partial V/\partial \mathbf{r}_i
#
# Even though the early ML-based interatomic potentials followed this route,
# .... blah blah and intro to be filled. Reference to our paper.

# %%
#
# Constant-energy molecular dynamics without energy conservation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Some more blah blah. Explain the system (melting of Ni)

# %%
# We use an `XML` input file that instructs ``i-PI`` to perform a constant-
# energy dynamics. The ``non_conservative:True`` option requires the model
# to predict forces using a dedicated direct head, rather than backpropagation.

with open("data/input-nc-nve.xml", "r") as file:
    input_nve = file.read()
print(input_nve)

# %%
# This input file uses a ``<ffdirect>`` block to run the metatensor PES
# as a library -- it is also possible to use a socket interface that is useful
# for parallelizing over multiple evaluators, see e.g.
# `this recipe
# <https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html>`_.
# ``i-PI`` can be run as a stand-alone command
#
# .. code-block:: bash
#
#    i-pi data/input-nc-nve.xml > log
#
# but here we use the Python API to integrate it in the notebook.

sim = InteractiveSimulation(input_nve)
sim.run(800)

# %%
#
# The simulation generates output files that can be parsed and visualized from Python.

data, info = read_output("nve-nc.out")
trj = read_trajectory("nve-nc.pos_0.extxyz")

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

ax.set_facecolor("white")
ax.plot(data["time"], data["potential"], "b-", label="potential")
ax.plot(data["time"], data["conserved"] - 20, "k-", label="conserved")
ax.set_xlabel("t / ps")
ax.set_ylabel("energy / ev")
ax.legend()

# %%
#
# The trajectory (which is started from oxygen molecules placed on top of the surface)
# shows quick relaxation to an oxide layer. If you look carefully, you'll also see that
# Mg and Si atoms tend to cluster together, and accumulate at the surface.

chemiscope.show(
    frames=trj,
    properties={
        "time": data["time"][::5],
        "potential": data["potential"][::5],
        "temperature": data["temperature"][::5],
    },
    mode="default",
    settings=chemiscope.quick_settings(
        map_settings={
            "x": {"property": "time", "scale": "linear"},
            "y": {"property": "potential", "scale": "linear"},
        },
        structure_settings={
            "unitCell": True,
        },
        trajectory=True,
    ),
)


# %%
#
# Energy conservation at low-cost with multiple time stepping
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Some more blah blah.

# %%
# We use an `XML` input file that instructs ``i-PI`` to perform a constant-
# energy dynamics. The ``non_conservative:True`` option requires the model
# to predict forces using a dedicated direct head, rather than backpropagation.

with open("data/input-nc-nve-mts.xml", "r") as file:
    input_nve_mts = file.read()
print(input_nve_mts)

# %%
# This input file uses a ``<ffdirect>`` block to run the metatensor PES
# as a library -- it is also possible to use a socket interface that is useful
# for parallelizing over multiple evaluators, see e.g.
# `this recipe
# <https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html>`_.
# ``i-PI`` can be run as a stand-alone command
#
# .. code-block:: bash
#
#    i-pi data/input-nc-nve-mts.xml > log
#
# but here we use the Python API to integrate it in the notebook.

sim = InteractiveSimulation(input_nve_mts)
sim.run(100)

# %%
