"""
Long-stride trajectories with a universal FlashMD model
=======================================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm>`_

This example demonstrates how to run long-stride molecular dynamics using the
universal FlashMD model. FlashMD predicts directly positions and momenta of atoms
at a later time based on the current positions and momenta.
It is trained on MD trajectories obtained with the
`PET-MAD universal potential  <https://arxiv.org/abs/2503.14118>`_.
You can read more about the model and its limitations in
`this preprint <>`_.
"""

# %%
#
# Start by importing the required libraries. You will need the
# `PET-MAD potential <https://github.com/lab-cosmo/pet-mad>`_,
# as well as `FlashMD <https://github.com/lab-cosmo/flashmd>`_
# and a recent version of `i-PI <http://ipi-code.org>`_.
#
# .. code-block:: bash
#
#     pip install pet-mad flashmd ipi
#

import os

import chemiscope
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from flashmd import get_universal_model
from flashmd.ipi import get_npt_stepper, get_nvt_stepper
from ipi.utils.parsing import read_output, read_trajectory
from ipi.utils.scripting import InteractiveSimulation
from metatensor.torch.atomistic import load_atomistic_model
from pet_mad.calculator import PETMADCalculator


# %%
# A rough schematic of the architecture of FlashMD is shown below.
# Each model is trained for a specific stride length, aiming to
# reproduce the trajectories obtained with a traditional velocity
# Verlet integrator.

# use matplotlib to display the image so it also display as a thumbnail
fig, ax = plt.subplots(figsize=(5728 / 300, 2598 / 300), dpi=300)
img = mpimg.imread("flashmd-scheme.png")
ax.imshow(img)
ax.axis("off")
fig.tight_layout()
plt.show()


# %%
# We start by pulling FlashMD and PET-MAD models from the repository.

if not os.path.exists("flashmd-universal-16fs.pt"):
    flashmd_model = get_universal_model(16)
    flashmd_model.save("flashmd-universal-16fs.pt")
if not os.path.exists("flashmd-universal-64fs.pt"):
    flashmd_model = get_universal_model(64)
    flashmd_model.save("flashmd-universal-64fs.pt")
if not os.path.exists("et-mad-latest.pt"):
    calculator = PETMADCalculator(version="latest", device="cpu")
    calculator._model.save("pet-mad-latest.pt")


# %%
# Al(110) surface dynamics
# ------------------------
#
# The (110) surface of aluminum exhibits `an interesting dynamical behavior
# <https://doi.org/10.1103/PhysRevLett.82.3296>`_ well below the bulk melting
# temperature. This manifests itself in the spontaneous formation of surface
# defects, with mobile adatoms emerging at the surface.
#
# We run a FlashMD simulation with 64 fs strides (as opposed to 1 or 2 fs) at
# 600 K, observing the motion of the adatom at the surface. We use the
# ```i-PI``<https://docs.ipi-code.org/>`_ scripting API to set up the
# simulation and run it interactively.

# %%
# The starting point is a "base" XML file that contains the setup for a traditional
# MD simulation in i-PI. It contains PET-MAD as the potential energy calculator
# (needed for the optional energy rescaling filter), and the only difference is
# the use of a much larger large time step than conventional MD.

with open("data/input-al110-base.xml", "r") as input_xml:
    sim = InteractiveSimulation(input_xml)

# %%
# To run FlashMD, we set up a custom step, using the ``get_nvt_stepper``
# utility function from the `flashmd.ipi` module. Note the filters
# ``rescale_energy=True`` and ``random_rotation=True``. The first one
# ensures that the total energy of the system is conserved, while the second one
# allows for random rotations of the system, which is useful to correct for the
# fact that the model is not exactly equivariant with respect to rotations.

model = load_atomistic_model("flashmd-universal-64fs.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

sim.set_motion_step(
    get_nvt_stepper(sim, model, device, rescale_energy=True, random_rotation=True)
)

# run for a few steps - this is a large box, and is rather slow on CPU
sim.run(20)

# %%
# The trajectory is stable, and one can check that the mean fluctuations
# of the adatom are qualitatively correct, by comparing with a (much slower)
# PET-MAD simulation.

data, info = read_output("al110-nvt-flashmd.out")
trj = read_trajectory("al110-nvt-flashmd.pos_0.extxyz")

chemiscope.show(
    frames=trj,
    properties={
        "time": data["time"],
        "potential": data["potential"],
        "temperature": data["temperature"],
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
# Solvated alanine dipeptide
# --------------------------
#
# As a second example, we run a constant-pressure simulation of explicitly
# solvated alanine dipeptide, using the FlashMD universal model with 16 fs
# time steps (as opposed to 0.5 fs). The setup is very similar to the previous
# example, but we use an input template that contains a NpT setup, and use
# the ``get_npt_stepper`` utility function to set up a stepper that
# combine the FlashMD velocity-Verlet step with cell updates.

with open("data/input-ala2-base.xml", "r") as input_xml:
    sim = InteractiveSimulation(input_xml)

model = load_atomistic_model("flashmd-universal-16fs.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

sim.set_motion_step(
    get_npt_stepper(sim, model, device, rescale_energy=True, random_rotation=True)
)
sim.run(10)  # only run 10 steps, again, pretty slow on CPU

# %%
# The cell fluctuates around the equilibrium volume, in a way that
# is consistent with the correct NpT ensemble. The trajectory is stable
# and the alanine molecule explores the different conformations
# (obviously when running for a reasonably long time).

data, info = read_output("ala2-npt-flashmd.out")
trj = read_trajectory("ala2-npt-flashmd.pos_0.extxyz")

chemiscope.show(
    frames=trj,
    properties={
        "time": data["time"],
        "volume": data["volume"],
        "potential": data["potential"],
        "pressure": data["pressure_md"],
        "temperature": data["temperature"],
    },
    mode="default",
    settings=chemiscope.quick_settings(
        map_settings={
            "x": {"property": "time", "scale": "linear"},
            "y": {"property": "volume", "scale": "linear"},
        },
        structure_settings={
            "unitCell": True,
        },
        trajectory=True,
    ),
)
# %%
