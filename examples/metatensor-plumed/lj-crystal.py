# -*- coding: utf-8 -*-
"""
Exploring the Lennard-Jones 38 Cluster with Metadynamics
=========================================================

:Authors: Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_;
          Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_;
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

We shall demonstrate the usage of the metatensor ecosystem within enhanced
sampling techniques, specifically by running **metadynamics**, to explore the
complex potential energy surface (PES) of a 38-atom Lennard-Jones (LJ) cluster.
The LJ38 cluster is a classic benchmark system because its global minimum energy
structure is a truncated octahedron with :math:`O_h` symmetry, which is
difficult to find with simple optimization methods. The PES has a multi-funnel
landscape, meaning the system can easily get trapped in local minima.

Our goal is progressively transition from a random configuration to the low-energy
structures. To do this, we will:

1.  Define a set of **collective variables (CVs)** that can distinguish between
    the disordered (liquid-like) and ordered (solid-like) states of the
    cluster. We will use a custom CV based on **Steinhardt order parameters**
    (:math:`Q_4` and :math:`Q_6`, a.k.a the bond-order parameters).
2.  Implement this custom CV using `featomic`, `metatensor`, and `metatomic` to
    create a portable `metatomic` model.
3.  Use the `PLUMED <https://www.plumed.org/>`_ package, integrated with the
    `Atomic Simulation Environment (ASE) <https://wiki.fysik.dtu.dk/ase/>`_, to
    run a metadynamics simulation.
4.  Analyze the results to visualize the free energy surface and run
    trajectories with LAMMPS of the system as it explores different
    configurations.
"""

import os

from typing import Dict, List, Optional

import ase.build
import ase.calculators.lj
import ase.calculators.plumed
import ase.io
import ase.md.langevin
import ase.optimize
import ase.units

#
import chemiscope
import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
import featomic.torch
import torch


# %%
# Simulation Starting Point: Random Structure w/ Energy Minimized
# ---------------------------------------------------------------
#
# We start with a random cloud of 38 Argon atoms. To get a more reasonable
# starting point for our MD simulation, we'll perform a quick energy
# minimization using the FIRE optimizer. This relaxes the structure into a
# nearby local minimum, mitigating artificial effects from an unphysical
# randomized structure.

np.random.seed(0xDEADBEEF)
atoms = ase.Atoms(
    symbols=["Ar"] * 38,
    positions=2 * np.random.rand(38, 3),
    # Set a cell without pbc to silence a warning from PLUMED
    cell=10 * np.eye(3),
    pbc=False,
)

lj_potential = ase.calculators.lj.LennardJones(rc=2.5)
atoms.calc = lj_potential
optimizer = ase.optimize.FIRE(atoms)
optimizer.run(fmax=0.8)

# %%
# The Target Structures
# ---------------------
#
# The two most famous structures for LJ38 are the global minimum (a perfect
# truncated octahedron) and a lower-symmetry icosahedral structure which is a
# deep local minimum. Let's load them and visualize all three (our starting
# structure and the two targets) using `chemiscope`.

minimal = ase.io.read("lj-oct-0k.xyz")
# FIXME: this does not look like the "other" stable structure
other = ase.io.read("lj38.xyz")

settings = {"structure": [{"playbackDelay": 50, "unitCell": True, "bonds": False}]}
chemiscope.show([minimal, other, atoms], mode="structure", settings=settings)


# %%
# Defining our custom collective variable
# ---------------------------------------
#
# To distinguish between the liquid-like state and the highly ordered
# face-centered cubic (FCC) packing of the global minimum, we use **Steinhardt
# order parameters**, specifically :math:`Q_4` and :math:`Q_6`. These parameters
# are rotationally invariant and measure the local orientational symmetry around
# each atom.
#
# - :math:`Q_6` is often high for both icosahedral and FCC-like structures,
#   making it a good measure of general "solidness".
# - :math:`Q_4` helps to distinguish between different crystal packing types. It
#   is close to zero for icosahedral structures but has a distinct non-zero value
#   for FCC structures.
#
# We will build a model that calculates global, system-averaged versions of these
# parameters.


# %%
#
# Encapsulating the Logic in a ``torch.nn.Module``
# '''''''''''''''''''''''''''''''''''''''''''''''''
#
# To make this CV usable by PLUMED via the ``METATOMIC`` interface, we must wrap
# our calculation logic in a ``torch.nn.Module``. This class takes a list of
# atomic systems and returns a `metatensor.TensorMap` containing the calculated
# CV values. The interface is defined in [TODO link].


class CollectiveVariable(torch.nn.Module):
    def __init__(self, cutoff, angular_list):
        super().__init__()

        self.max_angular = max(angular_list)
        # initialize and store the featomic calculator inside the class
        self.spex = featomic.torch.SphericalExpansion(
            **{
                "cutoff": {
                    "radius": 1.3,
                    "smoothing": {"type": "ShiftedCosine", "width": 0.5},
                },
                "density": {"type": "Gaussian", "width": 0.3},
                "basis": {
                    "type": "TensorProduct",
                    "max_angular": 6,
                    "radial": {"type": "Gto", "max_radial": 3},
                },
            }
        )
        self.selected_keys = mts.Labels(
            "o3_lambda", torch.tensor(angular_list).reshape(-1, 1)
        )

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:

        # execute the same code as above
        spex = self.spex(
            systems, selected_samples=selected_atoms, selected_keys=self.selected_keys
        )

        if len(spex) == 0:
            # PLUMED will first call the model with 0 atoms to get the size of the
            # output, so we need to handle this case first
            keys = mts.Labels("_", torch.tensor([[0]]))
            block = mts.TensorBlock(
                torch.zeros((0, len(self.selected_keys)), dtype=torch.float64),
                samples=mts.Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=self.selected_keys,
            )
            return {"features": mts.TensorMap(keys, [block])}

        spex = mts.remove_dimension(spex, axis="keys", name="o3_sigma")
        spex = spex.keys_to_properties("neighbor_type")
        spex = spex.keys_to_samples("center_type")

        spex = mts.sum_over_samples(spex, sample_names=["atom", "center_type"])

        blocks: List[mts.TensorBlock] = []
        for block in spex.blocks():
            new_block = mts.TensorBlock(
                (block.values**2).sum(dim=(1, 2)).reshape(-1, 1),
                samples=block.samples,
                components=[],
                properties=mts.Labels("n", torch.tensor([[0]])),
            )
            blocks.append(new_block)

        summed_q = mts.TensorMap(spex.keys, blocks)
        summed_q = summed_q.keys_to_properties("o3_lambda")

        # This model has a single output, named "features". This can be used by multiple
        # tools, including PLUMED where it defines a custom collective variable.
        return {"features": summed_q}


# %%
#
# Exporting the Model
# '''''''''''''''''''
#
# Once we have defined our custom model, we can now annotate it with multiple metadata
# and export it to the disk. The resulting model file and extensions directory can then
# be loaded by PLUMED and other, without requiring a Python installation (for example on
# HPC systems).
#
# See [TODO link] for more information about exporting metatensor models.

# initialize the model
cutoff = 1.3
module = CollectiveVariable(cutoff, angular_list=[4, 6])

# metatdata about the model itself
metadata = mta.ModelMetadata(name="TODO", description="TODO")

# metatdata about what the model can do
outputs = {"features": mta.ModelOutput(per_atom=False)}
capabilities = mta.ModelCapabilities(
    outputs=outputs,
    atomic_types=[18],
    interaction_range=cutoff,
    supported_devices=["cpu"],
    dtype="float64",
)

model = mta.AtomisticModel(
    module=module.eval(),
    metadata=metadata,
    capabilities=capabilities,
)

# finally, save the model to a standalone file
model.save("custom-cv.pt", collect_extensions="./extensions/")

# %%
# Optional: Test the Model in Python
# ''''''''''''''''''''''''''''''''''
#
# Before running the full simulation, we can use ``chemiscope``'s
# ``metatomic_featurizer`` to quickly check the output of our model on our
# initial structures. This is a great way to verify that the CVs produce
# different values for the different structures.
featurizer = chemiscope.metatomic_featurizer(model)
# TODO: add settings once https://github.com/lab-cosmo/chemiscope/pull/378 is released
chemiscope.explore([minimal, other, atoms], featurize=featurizer)

# %%
#
# Using the model to run metadynamics with PLUMED
# -----------------------------------------------
#
# With our model saved, we can now write the PLUMED input file. This file
# instructs PLUMED on what to do during the simulation.
# The input file consists of the following sections:
# - `UNITS` : Specifies the energy and length units
# - `METATOMIC` : Defines a collective variable which is essentially an exported metatomic model
# - `SELECT_COMPONENTS` : Splits the model output :math:`Q_4` and :math:`Q_6` parameters to scalars
# - `METAD` : sets up the metadynamics algorithm. It will add repulsive Gaussian potentials in the (`cv1`, `cv2`) space at regular intervals (`PACE`), discouraging the simulation from re-visiting conformations and pushing it over energy barriers
# - `PRINT` : This tells PLUMED to write the values of our CVs and the metadynamics bias energy to a file named `COLVAR` for later analysis.

if os.path.exists("HILLS"):
    os.unlink("HILLS")

setup = [
    f"UNITS LENGTH=A ENERGY={ase.units.mol / ase.units.kJ}",
    # define a collective variables using metatensor
    """
    cv: METATOMIC
        MODEL=custom-cv.pt
        EXTENSIONS_DIRECTORY=./extensions/
        SPECIES1=1-38
        SPECIES_TO_TYPES=18
    """,
    # extract the different components from the METATOMIC output into scalars
    # (METAD only accepts scalars, and METATOMIC output is a vector here)
    "cv1: SELECT_COMPONENTS ARG=cv COMPONENTS=1",
    "cv2: SELECT_COMPONENTS ARG=cv COMPONENTS=2",
    # run metadynamics with this collective variable
    """
    mtd: METAD
        ARG=cv1,cv2
    """,
    # Height of Gaussian hills in energy units
    """
        HEIGHT=0.05
    """,
    # Add a hill every 50 steps
    """
        PACE=50
    """,
    # Width of Gaussians for both CVs
    """
        SIGMA=1,2.5
    """,
    # Define the grid for free energy reconstruction
    """
        GRID_MIN=-20,-40
        GRID_MAX=20,40
        GRID_BIN=500,500
    """,
    # Well-Tempered Metadynamics factor
    """
        BIASFACTOR=5
    """,
    # File wor the history of the deposited hills
    """
        FILE=HILLS
    """,
    # prints out trajectory
    """    
    PRINT ARG=cv.*,mtd.* STRIDE=10 FILE=COLVAR
    """,
    # Flush often
    """
    FLUSH STRIDE=1
    """,
]

# %%
# Running dynamics I - `ase`
# ---------------------------
#
# The easiest way to generate a trajectory is to leverage `ase`. In subsequent
# sections we will use LAMMPS, as a more production worthy and correct molecular
# dynamics engine.
#

# Create the Plumed calculator, which wraps the LJ potential
atoms.calc = ase.calculators.plumed.Plumed(
    calc=lj_potential,
    input=setup,
    timestep=0.01,
    atoms=atoms,
    kT=0.1,
)
atoms.set_masses([1.0] * len(atoms))


md = ase.md.langevin.Langevin(
    atoms,
    timestep=0.01,
    temperature_K=0.1 / ase.units.kB,
    friction=1.0,
)

trajectory = []
for _ in range(100):
    md.run(steps=10)
    trajectory.append(atoms.copy())


# TODO: read the HILLS files & show the trajectory moving across the landscape
chemiscope.show(trajectory, mode="structure", settings=settings)
