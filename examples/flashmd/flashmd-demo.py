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

import torch
import os
from ipi.utils.scripting import InteractiveSimulation
from metatensor.torch.atomistic import load_atomistic_model
from flashmd import get_universal_model
from flashmd.ipi import get_npt_stepper, get_nvt_stepper
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

#%%
# Al(110) surface dynamics
# ========================
#
#

with open("data/input-al110-base.xml", "r") as input_xml:
    sim = InteractiveSimulation(input_xml)

model = load_atomistic_model("flashmd-universal-64fs.pt")
device = ("cuda" if torch.cuda.is_available() else "cpu")

sim.set_motion_step(get_nvt_stepper(sim, model, device, rescale_energy=True, random_rotation=True))
sim.run(20)


# %%
# Solvated alanine dipeptide
# ==========================


with open("data/input-ala2-base.xml", "r") as input_xml:
    sim = InteractiveSimulation(input_xml)

model = load_atomistic_model("flashmd-universal-16fs.pt")
device = ("cuda" if torch.cuda.is_available() else "cpu")

sim.set_motion_step(get_npt_stepper(sim, model, device, rescale_energy=True, random_rotation=True))
sim.run(20)

# %%
