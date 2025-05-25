"""
Long-stride trajectories with a universal FlashMD model
=======================================================

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm>`_,

This example demonstrates how to run long-stride molecular dynamics using the
universal FlahsMD model. FlahsMD predicts directly positions and momenta of atoms
at a later time based on the current positions and momenta.
It is trained on MD trajectories obtained with the
`PET-MAD universal potential  <https://arxiv.org/abs/2503.14118>`_.
You can read more about the model and its limitations in
`this prepring <>`_.
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

import torch
from flashmd import get_universal_model
from flashmd.ipi import get_npt_stepper, get_nvt_stepper
from ipi.utils.scripting import InteractiveSimulation
from ipi.utils.parsing import read_output, read_trajectory
import chemiscope
from metatensor.torch.atomistic import load_atomistic_model
from pet_mad.calculator import PETMADCalculator


# %%
# Pulls flashmd and pet-mad models from the repository.

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
# ========================
#
# The (110) surface of aluminum exhibits `an interesting dynamical behavior 
# <https://doi.org/10.1103/PhysRevLett.82.3296>`_ well below the bulk melting
# temperature. This manifests itself in the spontaneous formation of surface
# defects, with mobile adatoms emerging at the surface.
# 
# We run a FlashMD simulation with 64 fs strides at 600 K, observing the motion
# of the adatom at the surface. We use the `i-PI` scripting API to set up the 
# simulation and run it interactively.

# %%
# The starting point is a "base" XML file that contains the setup for a traditional
# MD simulation in i-PI. It contains PET-MAD as the potential energy calculator,
# and the only difference is the use of an anomalously large time step. 

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

# run for a afes steps - this is a large box, and is rather slow on CPU
sim.run(20)

# %%
# The trajectory is stable, and one can check that the mean fluctuations
# of the adatom are qualitatively correct, by comparing with a (much slower)
# PET-MAD simulation with a ~2 fs time step. 

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
# ==========================


with open("data/input-ala2-base.xml", "r") as input_xml:
    sim = InteractiveSimulation(input_xml)

model = load_atomistic_model("flashmd-universal-16fs.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

sim.set_motion_step(
    get_npt_stepper(sim, model, device, rescale_energy=True, random_rotation=True)
)
sim.run(10)

# %%
