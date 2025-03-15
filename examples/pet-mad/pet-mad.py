"""
PET-MAD Example
===============

:Authors: Philip Loche `@PicoCentauri <https://github.com/picocentauri>`_,

This example demonstrates how to use the PET-MAD model with ASE,
`i-PI <https://ipi-code.org>` and `LAMMPS <https://lammps.org>`_. 
PET-MAD is a "universal" machine-learning forcefield trained on 
a dataset that aims to incorporate a very high degree of 
structural diversity.

*SHORT OVERVIEW OF PET AND MAD HERE, LINK TO THE ARXIV WHEN AVAILABLE*
"""

# %%
#
# Start by importing the required libraries.

import subprocess
from copy import deepcopy, copy

import ase.units
import chemiscope
import numpy as np
import matplotlib.pyplot as plt

from ase.optimize import LBFGS
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatrain.utils.io import load_model

if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821


# %%
#
# Load the model
# ^^^^^^^^^^^^^^
#
# To start using PET-MAD, we first have to load the model. We will use the latest
# version of the model. PET-MAD is distributed as a check-point file, that
# allows re-training and fine-tuning, and needs to be *exported* to be used in 
# calculators and to make inference.

mad_huggingface = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"
)
model = load_model(mad_huggingface).export()


# %%
# The model can also be downloaded separately, and loaded
# from disk by providing the path to the :py:func:`load_model
# <metatensor.utils.io.load_model>` function.

# %%
# Inference on the MAD test set
# =============================
# 
# We begin by using the ``ase``-compatible calculator to evaluate
# energy and forces for a test dataset that contains both hold-out
# structures from the MAD dataset, and a few structures from popular
# datasets (MPtrj, Alexandria, OC2020, SPICE, MD22) re-computed with 
# consistent DFT settings.
#
# Load the dataset
# ^^^^^^^^^^^^^^^^
#
# We fetch the dataset, and load only some of the structures, to 
# speed up the example runtime on CPU. The model can also run (much faster)
# on GPUs if you have some at hand. 

# TODO: FETCH THESE ONLINE AND REMOVE THE FILE FROM THE REPO
test_structures = ase.io.read('data/mad-test_1.0.xyz', "::8")

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
    if not s.info["origin"] in subsets:
        subsets.append(s.info["origin"])
test_natoms = np.array(test_natoms)    
test_origin = np.array(test_origin)    
test_energy = np.array(test_energy)
test_forces = np.array(test_forces, dtype=object)


# %%
#
# Single point energy and forces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# PET-MAD is compatible with the metatomic interface which allows us to
# run it with ASE and many other MD engines. For more details see the `metatensor
# documentation
# <https://docs.metatensor.org/latest/atomistic/engines/index.html#atomistic-models-engines>`_.
#
# We now wrap model in an ASE compatible calculator and calculate energy and forces.

calculator = MetatensorCalculator(model, device="cpu")

# %%
#
# Here, we run the computation on the CPU. If you have a CUDA GPU you can also set
# ``device="cuda"`` to speed up the computation.

mad_energy = []
mad_forces = []
mad_structures = []
for structure in test_structures:
    tmp = deepcopy(structure)
    tmp.calc = copy(calculator) # avoids ovewriting results. thanks ase 3.23!
    mad_energy.append(tmp.get_potential_energy())
    mad_forces.append(tmp.get_forces())
    mad_structures.append(tmp)

mad_energy = np.array(mad_energy)
mad_forces = np.array(mad_forces, dtype=object)


# %%
aa = chemiscope.ase_vectors_to_arrows(mad_structures, "forces", scale=1.0)
len(aa['parameters']['atom'])

# %%
# A parity plot with the model predictions

tab10 = plt.get_cmap("tab10")
fig, ax = plt.subplots(1,2,figsize=(6,3), constrained_layout=True)
for i, sub in enumerate(subsets):
    sel = np.where(test_origin==sub)[0]
    ax[0].plot(
    mad_energy[sel]/test_natoms[sel], 
    test_energy[sel]/test_natoms[sel], 
    '.', c=tab10(i), label=sub)
    ax[1].plot(
    np.concatenate(
        mad_forces[sel]
    ).flatten(), 
    np.concatenate(
    test_forces[sel]
    ).flatten(),
    '.', c=tab10(i))
ax[0].set_xlabel("MAD energy / eV/at.")
ax[0].set_ylabel("Ref. energy / eV/at.")
ax[1].set_xlabel("MAD forces / eV/Å")
ax[1].set_ylabel("Ref. forces / eV/Å")
fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.20),ncol=3)


# %%
# Explore the dataset using 
# `chemiscope <http://chemiscope.org>`_

chemiscope.show(test_structures, mode="default", 
properties={
'origin' : test_origin,
'energy_ref': test_energy/test_natoms,
'energy_mad': mad_energy/test_natoms,
'energy_error': np.abs((test_energy-mad_energy)/test_natoms),
'force_error': [np.linalg.norm(f1-f2)/n for (f1,f2,n) in zip(mad_forces,test_forces,test_natoms) ],
},
shapes={
    'forces_ref': chemiscope.ase_vectors_to_arrows(mad_structures, "forces", scale=1.0),
    'forces_mad': chemiscope.ase_vectors_to_arrows(test_structures, "forces", scale=1.0)
    },
settings=chemiscope.quick_settings(
    x='energy_mad',
    y='energy_ref',
    symbol='origin',
    structure_settings={'unitCell':True, 'shape':['forces_ref']})
    )

# %%
# Uncertainty estimation
# ^^^^^^^^^^^^^^^^^^^^^^
# 
# TODO - Filippo pls add something here

# %%
# Simulating a complex surface 
# ============================
# 
# PET-MAD is designed to be robust and stable when executing sophisticated
# modeling workflows. As an example, we consider a slab of an Al-6xxx alloy
# (aluminum with a few percent Mg and Si) with some oxygen molecules adsorbed 
# (111) surface. 

al_surface = ase.io.read("data/al6xxx-o2.xyz", "0")

# %%
#
# Geometry optimization with ``ASE``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

class LBFGSLogger(LBFGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._traj_atoms = []
        self._traj_energy = []

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)  # Call original step function
        
        self._traj_atoms.append(self.atoms.copy())
        self._traj_energy.append(self.atoms.get_potential_energy())


atoms = al_surface.copy()
atoms.calc = calculator

opt = LBFGSLogger(atoms)
opt.run(fmax=0.001, steps=40)

# %%
plt.plot(opt._traj_energy)
chemiscope.show(frames=opt._traj_atoms, mode="structure")
# %%
# We run Langevin molecular dynamics simulation with ASE. The system is being simulated
# at 310 Kelvin with a timestep of 0.5 femtoseconds and for 200 steps (total simulation
# time of 100 femtoseconds).

integrator = Langevin(atoms, 0.5 * ase.units.fs, temperature_K=310, friction=5e-3)

# %%
#
# We save the energies and atoms object in every step to visualize the trajectory with
# chemiscope.

n_steps = 100

potential_energy = np.zeros(n_steps)
kinetic_energy = np.zeros(n_steps)
trajectory = []

for step in range(n_steps):
    # run a single simulation step
    integrator.run(1)

    trajectory.append(atoms.copy())
    potential_energy[step] = atoms.get_potential_energy()
    kinetic_energy[step] = atoms.get_kinetic_energy()


# %%
#
# Visualize the trajectory with chemiscope

potential_energy -= potential_energy.mean()
total_energy = potential_energy + kinetic_energy


properties = {
    "time": np.arange(n_steps) * 0.5,
    "potential_energy": potential_energy,
    "kinetic_energy": kinetic_energy,
    "total_energy": total_energy,
}

chemiscope.show(frames=trajectory, properties=properties)


# %%
#
# Molecular dynamics with ``LAMMPS``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We now run the same MD with `LAMMPS <https://lammps.org>`_.
#
# To do so, it is only required defining a ``pair_style`` that defines the model and a
# single ``pair_coeff`` command should be used with the metatensor style, specifying the
# mapping from LAMMPS types to the atomic types the model can handle. The first 2
# arguments must be ``* *`` so as to span all LAMMPS atom types. This is followed by a
# list of N arguments that specify the mapping of metatensor atomic types to LAMMPS
# types, where N is the number of LAMMPS atom types.

with open("data/pet-mad-si.in", "r") as f:
    print(f.read())

# %%
#
# To run the model with LAMMPS we first have to save the model to disk

model.save("pet-mad-latest.pt", collect_extensions="extensions")

# We use to the ``collect_extensions`` argument to save the compiled extensions to disk.
# These extensions ensure that the model remains self-contained and can be executed
# without requiring the original Python or C++ source code. This is necessary for the
# LAMMPS interface to work because it has no access to the Python code.
#
# .. warning::
#
#   Be aware that the extensions are compiled files and depend on your operating system.
#   Usually you have re-export the extensions for different systems!
#
# We also save to geometry to a LAMMPS data file and finally run the simulation.

ase.io.write("silicon.data", atoms, format="lammps-data", masses=True)

# subprocess.check_call(["lmp_serial", "-in", "data/pet-mad-si.in"])


# %%
#
# The LAMMPS input files looks like an usual input file but uses our metatensor
# interface.
#
