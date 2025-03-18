"""
The PET-MAD universal potential
===============================

:Authors: Philip Loche `@PicoCentauri <https://github.com/picocentauri>`_,
          Michele Ceriotti `@ceriottm <https://github.com/ceriottm>`_,
          Arslan Mazitov `@abmazitov <https://github.com/abmazitov>`_

This example demonstrates how to use the PET-MAD model with ASE, `i-PI
<https://ipi-code.org>`_ and `LAMMPS <https://lammps.org>`_. PET-MAD is a "universal"
machine-learning forcefield trained on a dataset that aims to incorporate a very high
degree of structural diversity.

The point-edge transformer (PET) is an unconstrained architecture that achieves a high
degree of symmetry compliance through data augmentation during training (see the `PET
paper
<https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_
for more details). The unconstrained nature of the model simplifies its implementation
and structure, making it computationally efficient and very expressive.

The MAD dataset combines "stable" inorganic structures from the
`MC3D dataset <https://mc3d.materialscloud.org/>`_,
2D structures from the
`MC2D dataset <https://mc2d.materialscloud.org/>`_,
and molecular crystals from the
`ShiftML dataset <https://archive.materialscloud.org/record/2022.147>`_
with "Maximum Atomic Diversity" configurations, generated by distorting the composition
and structure of these stable templates. By doing so, PET-MAD achieves state-of-the-art
accuracy despite the MAD dataset containing fewer than 100k structures. The reference
DFT settings are highly converged, but limited to a PBEsol functional, so the accuracy
against experimental data depends on how good this level of theory is for a given
system. PET-MAD is introduced, and benchmarked for several challenging modeling tasks,
in `this preprint <http://arxiv.org/TBD>`_.
"""

# %%
#
# Start by importing the required libraries.

import os
import subprocess
from copy import copy, deepcopy

# ASE and i-PI scripting utilities
import ase.units
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import requests
from ase.optimize import LBFGS
from ipi.utils.mathtools import get_rotation_quadrature_lebedev
from ipi.utils.parsing import read_output, read_trajectory
from ipi.utils.scripting import (
    InteractiveSimulation,
    forcefield_xml,
    motion_nvt_xml,
    simulation_xml,
)

# pet-mad ASE calculator
from pet_mad.calculator import PETMADCalculator


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821


# %%
# Inference on the MAD test set
# -----------------------------
#
# We begin by using the ``ase``-compatible calculator to evaluate energy and forces for
# a test dataset that contains both hold-out structures from the MAD dataset, and a few
# structures from popular datasets (`MPtrj
# <https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375>`,
# `Alexandria <https://alexandria.icams.rub.de/>`_,
# `OC2020 <https://paperswithcode.com/dataset/oc20>`_,
# `SPICE <https://www.nature.com/articles/s41597-022-01882-6>`_,
# `MD22 <https://www.science.org/doi/10.1126/sciadv.adf0873>`_) re-computed
# with consistent DFT settings.
#
# Load the dataset
# ^^^^^^^^^^^^^^^^
#
# We fetch the dataset, and load only some of the structures, to speed up the example
# runtime on CPU. The model can also run (much faster) on GPUs if you have some at hand.

filename = "data/mad-test-mad-settings.xyz"
if not os.path.exists(filename):
    url = (
        "https://huggingface.co/lab-cosmo/pet-mad/resolve/"
        "main/benchmarks/mad-test-mad-settings-v1.0.xyz"
    )
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)

test_structures = ase.io.read(filename, "::16")

# also extract reference energetics and metadata
test_energy = []
test_forces = []
test_natoms = []
test_origin = []
subsets = []

for s in test_structures:
    test_energy.append(s.get_potential_energy())
    test_natoms.append(len(s))
    test_forces.append(s.get_forces())
    test_origin.append(s.info["origin"])
    if s.info["origin"] not in subsets:
        subsets.append(s.info["origin"])

test_natoms = np.array(test_natoms)
test_origin = np.array(test_origin)
test_energy = np.array(test_energy)
test_forces = np.array(test_forces, dtype=object)

# %%
#
# Install PET-MAD
# ^^^^^^^^^^^^^^^
#
# To start using PET-MAD, we first have to install the PET-MAD package.
# If you have not done so, you can install it with pip:
# ``pip install git+https://github.com/lab-cosmo/pet-mad.git``.

# # %%
# #
# # Load the model
# # ^^^^^^^^^^^^^^
# #
# # We will use the latest version of the model, available as an
# # ASE calculator.

# calculator = PETMADCalculator(version="latest", device="cpu")

# # %%
# #
# # The model can also be downloaded separately, and loaded from disk by providing the
# # path to the :py:func:`load_model <metatensor.utils.io.load_model>` function.
# #
# # This model can be used "as is" in Python - and in this form one can modify it, e.g. to
# # continue training, or to fine-tune on a new dataset. However, to run with external
# # codes, it can/should be saved to disk.

# model.save("pet-mad-latest.pt", collect_extensions="extensions")

# # %%
# # We use the ``collect_extensions`` argument to save the compiled extensions to disk.
# # These extensions ensure that the model remains self-contained and can be executed
# # without requiring the original Python or C++ source code. In particular,
# # this is necessary for the LAMMPS interface to work because it has no access to
# # the Python code.

# %%
#
# Single point energy and forces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# PET-MAD is compatible with the metatensor atomistic models interface which allows us
# to run it with ASE and many other MD engines. For more details see the `metatensor
# documentation
# <https://docs.metatensor.org/latest/atomistic/engines/index.html#atomistic-models-engines>`_.
#
# We now load the PET-MAD ASE calculator and calculate energy and forces.

calculator = PETMADCalculator(version="latest", device="cpu")

# # %%
# #
# # Note also that exporting the model compiles it with ``torchscript`` which
# # is also usually beneficial in terms of execution speed. For this reason,
# # we recommend loading the model from the exported (``.pt``) version also
# # before using it for inference in ASE.

# calculator = MetatensorCalculator(
#     "pet-mad-latest.pt", device="cpu", extensions_directory="extensions"
# )


# %%
#
# Here, we run the computation on the CPU. If you have a CUDA GPU you can also set
# ``device="cuda"`` to speed up the computation.

mad_energy = []
mad_forces = []
mad_structures = []
for structure in test_structures:
    tmp = deepcopy(structure)
    tmp.calc = copy(calculator)  # avoids ovewriting results.
    mad_energy.append(tmp.get_potential_energy())
    mad_forces.append(tmp.get_forces())
    mad_structures.append(tmp)

mad_energy = np.array(mad_energy)
mad_forces = np.array(mad_forces, dtype=object)


# %%
#
# A parity plot with the model predictions

tab10 = plt.get_cmap("tab10")
fig, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

ax[0].plot([0, 1], [0, 1], "b:", transform=ax[0].transAxes)
ax[1].plot([0, 1], [0, 1], "b:", transform=ax[1].transAxes)

for i, sub in enumerate(subsets):
    sel = np.where(test_origin == sub)[0]
    ax[0].plot(
        mad_energy[sel] / test_natoms[sel],
        test_energy[sel] / test_natoms[sel],
        ".",
        c=tab10(i),
        label=sub,
    )
    ax[1].plot(
        np.concatenate(mad_forces[sel]).flatten(),
        np.concatenate(test_forces[sel]).flatten(),
        ".",
        c=tab10(i),
    )

ax[0].set_xlabel("MAD energy / eV/atom")
ax[0].set_ylabel("Reference energy / eV/atom")
ax[1].set_xlabel("MAD forces / eV/Å")
ax[1].set_ylabel("Refrerence forces / eV/Å")

fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.20), ncol=3)


# %%
# Explore the dataset using
# `chemiscope <http://chemiscope.org>`_

chemiscope.show(
    test_structures,
    mode="default",
    properties={
        "origin": test_origin,
        "energy_ref": test_energy / test_natoms,
        "energy_mad": mad_energy / test_natoms,
        "energy_error": np.abs((test_energy - mad_energy) / test_natoms),
        "force_error": [
            np.linalg.norm(f1 - f2) / n
            for (f1, f2, n) in zip(mad_forces, test_forces, test_natoms)
        ],
    },
    shapes={
        "forces_ref": chemiscope.ase_vectors_to_arrows(
            mad_structures, "forces", scale=1.0
        ),
        "forces_mad": chemiscope.ase_vectors_to_arrows(
            test_structures, "forces", scale=1.0
        ),
    },
    settings=chemiscope.quick_settings(
        x="energy_mad",
        y="energy_ref",
        symbol="origin",
        structure_settings={"unitCell": True, "shape": ["forces_ref"]},
    ),
)

# %%
# How about equivariance‽
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The PET architecture does not provide "intrinsically" invariant
# energy predictions, but learns symmetry from data augmentation.
# Should you worry? The authors of PET-MAD certainly do, and they
# have `studied extensively
# <http://doi.org/10.1088/2632-2153/ad86a0>`_ whether
# the symmetry breaking can cause serious artefacts. You can check by
# yourself following the procedure below, that evaluates a structures
# over a grid of rotations, estimating the variability in energy
# (which is around 1meV/atom, much smaller than the test error).

rotations = get_rotation_quadrature_lebedev(3)

rot_test = test_structures[100]
rot_structures = []
rot_weights = []
rot_energies = []
rot_forces = []
rot_angles = []

for rot, w, angles in rotations:
    tmp = rot_test.copy()
    tmp.positions = tmp.positions @ rot.T
    tmp.cell = tmp.cell @ rot.T
    tmp.calc = copy(calculator)
    rot_weights.append(w)
    rot_energies.append(tmp.get_potential_energy() / len(tmp))
    rot_forces.append(tmp.get_forces())
    rot_structures.append(tmp)
    rot_angles.append(angles)

rot_energies = np.array(rot_energies)
rot_weights = np.array(rot_weights)
rot_angles = np.array(rot_angles)
erot_rms = 1e3 * np.sqrt(
    np.sum(rot_energies**2 * rot_weights) / np.sum(rot_weights)
    - (np.sum(rot_energies * rot_weights) / np.sum(rot_weights)) ** 2
)
erot_max = 1e3 * np.abs(rot_energies.max() - rot_energies.min())
print(
    f"""
Symmetry breaking, energy:
RMS: {erot_rms:.3f} meV/at.
Max: {erot_max:.3f} meV/at.
"""
)

# %%
# You can also inspect the rotational behavior visually

chemiscope.show(
    rot_structures,
    mode="default",
    properties={
        "delta_energy": 1e3 * (rot_energies - rot_energies.mean()),
        "euler_angles": rot_angles,
    },
    shapes={
        "forces": chemiscope.ase_vectors_to_arrows(rot_structures, "forces", scale=4.0),
    },
    settings=chemiscope.quick_settings(
        x="euler_angles[1]",
        y="euler_angles[2]",
        z="euler_angles[3]",
        color="delta_energy",
        structure_settings={"unitCell": True, "shape": ["forces"]},
    ),
)

# %%
#
# Note also that `i-PI <http://ipi-code.org>`_ provides functionalities to do this
# automatically to obtain MD trajectories with a even higher degree of
# symmetry-compliance.


# %%
#
# Simulating a complex surface
# ----------------------------
#
# PET-MAD is designed to be robust and stable when executing sophisticated
# modeling workflows. As an example, we consider a slab of an Al-6xxx alloy
# (aluminum with a few percent Mg and Si) with some oxygen molecules adsorbed
# at the (111) surface.
#
# .. warning::
#
#   The overall Si+Mg concentration in an Al6xxx alloy is far lower than what
#   depicted here. This is just a demonstrative example and should not be taken
#   as the starting point of a serious study of this system.
#

al_surface = ase.io.read("data/al6xxx-o2.xyz")

# %%
#
# Geometry optimization with ``ASE``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As a first example, we use the ``ase`` geometry `LBFGS` optimizer to relax the initial
# positions. This leads to the rapid decomposition of the oxygen molecules and the
# formation of an oxide layer.

atoms = al_surface.copy()
atoms.calc = calculator

opt = LBFGS(atoms)

traj_atoms = []
traj_energy = []
opt.attach(lambda: traj_atoms.append(atoms.copy()))
opt.attach(lambda: traj_energy.append(atoms.get_potential_energy()))

# stop the optimization early to speed up the example
opt.run(fmax=0.001, steps=30)

# %%
#
# Even if the optimization is cut short and far from converged,
# the decomposition of the oxygen molecules is apparent, and
# leads to a large energetic stabilization

chemiscope.show(
    frames=traj_atoms,
    properties={
        "index": np.arange(0, len(traj_atoms)),
        "energy": traj_energy,
    },
    mode="default",
    settings=chemiscope.quick_settings(trajectory=True),
)

# %%
#
# Molecular dynamics with atoms exchange with ``i-PI``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The geometry optimization shows the high reactivity of
# this surface, but does not properly account for finite
# temperature and does not sample the diffusion of solute
# atoms in the alloy (which is mediated by vacancies).
#
# We use `i-PI <http://ipi-code.org>`_ to perform a
# molecular dynamics trajectory at 800K, combined with
# Monte Carlo steps that swap the nature of atoms, allowing the simulation
# to reach equilibrium in the solute-atoms distributions
# without having to introduce vacancies or wait for the
# very long time scale needed for diffusion.

# %%
#
# Before starting the simulations with MD engines, it is important
# to export the model to a format that can be used by the engine.
# This is done by saving the model to a file, which includes the
# model weights and the compiled extensions.

calculator.model.save("pet-mad-latest.pt", collect_extensions="extensions")


# %%
#
# The behavior of i-PI is controlled by an XML input file.
# The ``utils.scripting`` module contains several helper
# functions to generate the basic components.
#
# Here we use a ``<motion mode="multi">`` block to combine
# a MD run with a ``<motion mode="atomswap">`` block that
# attemts swapping atoms, with a Monte Carlo acceptance.

motion_xml = f"""
<motion mode="multi">
    {motion_nvt_xml(timestep=5.0 * ase.units.fs)}
    <motion mode="atomswap">
        <atomswap>
            <names> [ Al, Si, Mg, O]  </names>
        </atomswap>
    </motion>
</motion>
"""

input_xml = simulation_xml(
    structures=[al_surface],
    forcefield=forcefield_xml(
        name="pet-mad",
        mode="direct",
        pes="metatensor",
        parameters={"model": "pet-mad-latest.pt", "template": "data/al6xxx-o2.xyz"},
    ),
    motion=motion_xml,
    temperature=800,
    verbosity="low",
    prefix="nvt_atomxc",
)

print(input_xml)

# %%
#
# The simulation can be run from a Python script or the command line. By changing the
# forcefield interface from ``direct`` to the use of a socket, it is also possible to
# execute separately ``i-PI`` and the ``metatensor`` driver.

sim = InteractiveSimulation(input_xml)
sim.run(100)

# %%
#
# The simulation generates output files that can be parsed and visualized from Python.

data, info = read_output("nvt_atomxc.out")
trj = read_trajectory("nvt_atomxc.pos_0.xyz")

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

ax.plot(data["time"], data["potential"], "b-", label="potential")
ax.plot(data["time"], data["conserved"] - 4, "k-", label="conserved")
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
        "time": data["time"][::10],
        "potential": data["potential"][::10],
        "temperature": data["temperature"][::10],
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
# Molecular dynamics with ``LAMMPS``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now run the same MD with `LAMMPS <https://lammps.org>`_. To run a LAMMPS
# calculation with a ``metatomic`` potential, one needs a LAMMPS build that contains an
# appropriate pair style. You can compile it from `source
# <https://github.com/metatensor/lammps>`_, or fetch it from the `metatensor` channel on
# conda. One can then just include in the input a ``pair_style metatensor`` that points
# to the exported model and a single ``pair_coeff`` command that specifies the mapping
# from LAMMPS types to the atomic types the model can handle. The first two arguments
# must be ``* *`` so as to span all LAMMPS atom types. This is followed by a list of N
# arguments that specify the mapping of metatensor atomic types to LAMMPS types, where N
# is the number of LAMMPS atom types.

with open("data/al6xxx-o2.in", "r") as f:
    print(f.read())

# %%
#
# .. warning::
#
#   Be aware that the extensions are compiled files and depend on your operating system.
#   Usually you have re-export the extensions for different systems! You can do this
#   by running the appropriate parts of this file, or using the ``mtt export``
#   command-line utility.
#
# We also save the geometry to a LAMMPS data file and finally run the simulation.

ase.io.write("al6xxx-o2.data", al_surface, format="lammps-data", masses=True)

subprocess.check_call(["lmp_serial", "-in", "data/al6xxx-o2.in"])

# %%
#
# The resulting trajectory is qualitatively consistent with what
# we observed with ``i-PI``.

lmp_trj = ase.io.read("trajectory.xyz", ":")

chemiscope.show(frames=lmp_trj, mode="structure")
