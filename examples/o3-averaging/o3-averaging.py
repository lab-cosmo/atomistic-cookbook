r"""
Rotational averaging of non-equivariant models
==============================================

:Authors:
Marcel Langer `@sirmarcel <https://github.com/sirmarcel/>`_;
Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

Molecular dynamics (MD) simulates the movement of atoms on the potential energy surface
(PES) :math:`U(R)` where :math:`R` stands for all :math:`i` atomic positions. Evaluating
this function makes up the dominating cost in running MD. Machine learning interatomic
potentials (MLPs) are often used to approximate some expensive first-principles PES at
high accuracy and reasonably high speed.

It is well known that the PES has underlying physical symmetries: It doesn't change under
rotations of the coordinates, overall translations in space, and it doesn't depend on the
order in which we list all the atoms in the system. It has long been believed that it is
essential for MLPs to exactly fulfil these symmetries. However, this places severe
constraints on possible model architectures, and so there has been a lot of interest in
so-called *unconstrained* models that do not respect symmetries exactly, by construction,
but rather learn them approximately from the training data. In the domain of MLPs, such
models have largely focused on dropping rotational invariance.

This tutorial, based on a `recent paper <https://doi.org/10.1088/2632-2153/ad86a0>`_,
explains how to test and control the impact of approximate rotational invariance in
MLPs for MD simulations.
"""

# %%
# Setting up
# ----------
#
# First, we import things

import os
import subprocess
import time
import xml.etree.ElementTree as ET

import ase
import ase.io, ase.build
import chemiscope
import ipi
import numpy as np
from ipi.utils.mathtools import (
    get_rotation_quadrature_lebedev,
    get_rotation_quadrature_legendre,
)


import matplotlib.pyplot as plt
import numpy as np


# %%
# show rotations

# Gets quadrature grid (regular spacing in alpha,
# lebedev grids for beta, gamma)
quad_grid = get_rotation_quadrature_lebedev(3)
quad = {
    "rotation_matrices": np.array([q[0] for q in quad_grid]),
    "weights": np.array([q[1] for q in quad_grid]),
    "angles": np.array([q[2] for q in quad_grid]),
}

# Display the rotations of an atom with a "force" vector
# The "reference atom" is marked as an O, and its copies as H
vs = [
    np.asarray([[2, 0, 0], [2, 1, 0]]),
    np.asarray([[0, 2, 0], [0, 2, 1]]),
    np.asarray([[0, 0, 2], [1, 0, 2]]),
]

frames = []
for v in vs:
    rl = v @ quad["rotation_matrices"]
    ats = ase.Atoms("O" + ("H" * (len(rl) - 1)), positions=rl[:, 0])
    ats.arrays["forces"] = rl[:, 1] - rl[:, 0]
    ats.arrays["weight"] = quad["weights"]
    ats.arrays["alpha"] = quad["angles"][:, 0]
    ats.arrays["beta"] = quad["angles"][:, 1]
    ats.arrays["gamma"] = quad["angles"][:, 2]
    frames.append(ats)

# Display with chemiscope. Three frames for three
# initial positions. You can also color by grid weight
# and by Euler angles

chemiscope.show(
    frames=frames,
    mode="structure",
    shapes={"forces": chemiscope.ase_vectors_to_arrows(frames)},
    properties=chemiscope.extract_properties(frames),
    settings={
        "structure": [
            dict(
                bonds=False,
                atoms=True,
                shape="forces",
                environments={"activated": False},
                color={
                    "property": "element",
                    "transform": "linear",
                    "palette": "viridis",
                },
            ),
        ]
    },
    environments=chemiscope.all_atomic_environments(frames),
)

# %%
# Next, we set up the interface with LAMMPS, which has to be compiled.
# We use a utility function to compile it. Note that this requires
# a functioning build system with `gfortran` and `make`.

ipi.install_driver()


# %%
# Non-invariant water potential
# -----------------------------
# 
# Instead of a complex MLP, we'll be using an artificial example in this
# tutorial: We use the classical "tip4p" forcefield for water, and add an
# orientation-dependent additional term to simulate missing rotational invariance.
#
# Let's have a look at that potential. We'll take a water, and rotate it
# around 180 degrees in steps of two degrees, and look at the energy:

atoms = ase.build.molecule("H2O")
atoms.set_cell(np.eye(3) * 5)
atoms.positions += 2.5

frames = []
for i in range(0, 180, 2):
    a = atoms.copy()
    a.rotate(i, "z", center="COM")
    frames.append(a)
print(len(frames))
ase.io.write("single/replay.xyz", frames)
# todo: actually do the work, get the results, make plot

# %%
# As we can see, the potential energy changes under rotations. We have a spurious
# angular dependence in our potential!

# %%
# Conservation of angular momentum
# --------------------------------
# 
# The most drastic consequence of a lack of rotational invariance is that angular
# momentum is no longer conserved. We can easily see this by simulating a single
# water molecule, initialised with zero total angular momentum. With our non-
# invariant potential, it starts rotating after just a few timesteps:

# todo: actually do the things here.

# %%
# In the manuscript linked at the beginning, we show that for molecules, this
# leads to a spurious preferred orientation of the molecules, even for NVT
# simulations. Interestingly, we find that for an approximately invariant MLP
# that was trained with data augmentation, there is no impact on observables for
# *liquid* water, which the model was trained on.
# However, this tutorial is focused on something else: What to do to control
# the lack of invariance after the fact.


# %%
# Controlling non-invariance with averaging
# -----------------------------------------
#
# A not rotationally invariant potential can be made more invariant by *averaging*
# over rotations. Rotations are continuous, so we can't recover exact invariance,
# but we can get close enough for many practical cases, as we show in the paper above.
#
# I-Pi has *builtin* support for different symmetrisation schemes: Two different types
# of grids over rotations (Legendre, Lebedev) and a single randomly chosen rotation.
# They are invoked by using `ffrotations` in place of `ffsocket` to interface with the driver:

# Open and show the relevant part of the input
xmlroot = ET.parse("data/input-noo3.xml").getroot()
print("      " + ET.tostring(xmlroot.find(".//ffrotations"), encoding="unicode"))

# %%
# So, let's now run the same dynamics from above *with* this rotational augmentation.

# todo: run, get results, make animation

# %%
#
# With this grid, we've managed to successfully control the non-invariance: There is
# only a very slow remaining precession. In a less "strict" setting, for instance NVT,
# this would be more than sufficient to mitigate the impacts of the non-invariance.

# scratch space below

# # %%
# # We launch the i-PI and LAMMPS processes, exactly as in the
# # ``path-integrals`` example.

# # don't rerun if the outputs already exist
# ipi_process = None
# if not os.path.exists("water-noo3.out"):
#     ipi_process = subprocess.Popen(["i-pi", "data/input-noo3.xml"])
#     time.sleep(2)  # wait for i-PI to start
#     lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]
#     driver_process = [
#         subprocess.Popen(
#             ["i-pi-driver", "-u", "-a", "h2o-noo3", "-m", "noo3-h2o"], cwd="data/"
#         )
#         for i in range(1)
#     ]

# # %%
# # Skip this cell if you want to run in the background
# if ipi_process is not None:
#     ipi_process.wait()
#     lmp_process[0].wait()
#     driver_process[0].wait()


# # %%
# #
# # Discuss how molecules get oriented

# traj_data = ase.io.read("water-noo3.pos_0.extxyz", ":")

# # %%
# # then, assemble a visualization
# chemiscope.show(
#     frames=traj_data,
#     settings=chemiscope.quick_settings(trajectory=True),
#     mode="structure",
# )
