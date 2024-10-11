r"""
Using machine learning collective variables in PLUMED with metatensor
=====================================================================

:Authors: Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_

TODO: introduction
"""

import os
os.environ["PLUMED_KERNEL"] = "PLEASE SET THIS!"


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
import numpy as np
import rascaline.torch
import torch


# %%
#
# TODO: introduction
#


# %%
#
# TODO: simulation starting point: random structure /w energy minimized
#

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
#
# TODO: where we want to go
#

minimal = ase.io.read("lj-oct-0k.xyz")
# FIXME: this does not look like the "other" stable structure
other = ase.io.read("lj38.xyz")

settings = {"structure": [{"playbackDelay": 50, "unitCell": True, "bonds": False}]}
chemiscope.show([minimal, other, atoms], mode="structure", settings=settings)


# %%
#
# Defining our custom collective variable
# ---------------------------------------
#


# TODO: build the code below step by step, explaining what's going on


# %%
#
# We can now put everything together in a single class. This class must follow the
# required interface, as defined in [TODO link].
#


class CollectiveVariable(torch.nn.Module):
    def __init__(self, cutoff, angular_list):
        super().__init__()

        self.max_angular = max(angular_list)
        # initialize and store the rascaline calculator inside the class
        self.soap = rascaline.torch.SphericalExpansion(
            cutoff=cutoff,
            max_radial=6,
            max_angular=self.max_angular,
            atomic_gaussian_width=0.3,
            radial_basis={"Gto": {}},
            center_atom_weight=1.0,
            cutoff_function={"ShiftedCosine": {"width": 0.5}},
        )
        self.selected_keys = mts.Labels(
            "o3_lambda", torch.tensor(angular_list).reshape(-1, 1)
        )

    def forward(
        self,
        systems: List[mts.atomistic.System],
        outputs: Dict[str, mts.atomistic.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:

        # execute the same code as above
        soap = self.soap(
            systems, selected_samples=selected_atoms, selected_keys=self.selected_keys
        )

        if len(soap) == 0:
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

        soap = mts.remove_dimension(soap, axis="keys", name="o3_sigma")
        soap = soap.keys_to_properties("neighbor_type")
        soap = soap.keys_to_samples("center_type")

        soap = mts.sum_over_samples(soap, sample_names=["atom", "center_type"])
        blocks: List[mts.TensorBlock] = []
        for block in soap.blocks():
            new_block = mts.TensorBlock(
                block.values.sum(dim=(1, 2)).reshape(-1, 1),
                samples=block.samples,
                components=[],
                properties=mts.Labels("n", torch.tensor([[0]])),
            )
            blocks.append(new_block)

        summed_q = mts.TensorMap(soap.keys, blocks)
        summed_q = summed_q.keys_to_properties("o3_lambda")

        # This model has a single output, named "features". This can be used by multiple
        # tools, including PLUMED where it defines a custom collective variable.
        return {"features": summed_q}


# %%
#
# Once we have defined our custom model, we can now annotate it with multiple metadata
# and export it to the disk. The resulting model file and extensions directory can then
# be loaded by PLUMED and other, without requiring a Python installation (for example on
# HPC systems).
#
# See [TODO link] for more information about exporting metatensor models.

# initialize the model
cutoff = 3.5
module = CollectiveVariable(cutoff, angular_list=[4, 6])

# metatdata about the model itself
metadata = mts.atomistic.ModelMetadata(name="TODO", description="TODO")

# metatdata about what the model can do
outputs = {"features": mts.atomistic.ModelOutput(per_atom=False)}
capabilities = mts.atomistic.ModelCapabilities(
    outputs=outputs,
    atomic_types=[18],
    interaction_range=cutoff,
    supported_devices=["cpu"],
    dtype="float64",
)

model = mts.atomistic.MetatensorAtomisticModel(
    module=module.eval(),
    metadata=metadata,
    capabilities=capabilities,
)

# finally, save the model to a standalone file
model.save("custom-cv.pt", collect_extensions="./extensions/")

# %%
# optional: show how one can check how the model is doing without leaving Python
featurizer = chemiscope.metatensor_featurizer(model)
# TODO: add settings once https://github.com/lab-cosmo/chemiscope/pull/378 is released
chemiscope.explore([minimal, other, atoms], featurize=featurizer)

# %%
#
# Using the model to run metadynamics with PLUMED
# -----------------------------------------------
#

if os.path.exists("HILLS"):
    os.unlink("HILLS")

setup = [
    f"UNITS LENGTH=A ENERGY={ase.units.mol / ase.units.kJ}",
    # define a collective variables using metatensor
    """
    cv: METATENSOR
        MODEL=custom-cv.pt
        EXTENSIONS_DIRECTORY=./extensions/
        SPECIES1=1-38
        SPECIES_TO_TYPES=18
    """,
    # extract the different components from METATENSOR output into scalar
    # (METAD only accepts scalars, and METATENSOR output is a vector here)
    "cv1: SELECT_COMPONENTS ARG=cv COMPONENTS=1",
    "cv2: SELECT_COMPONENTS ARG=cv COMPONENTS=2",
    # run metadynamics with this collective variable
    """
    METAD
        ARG=cv1,cv2
        HEIGHT=0.05
        PACE=50
        SIGMA=1,2.5
        GRID_MIN=-5,-40
        GRID_MAX=15,10
        GRID_BIN=500,500
        BIASFACTOR=5
        FILE=HILLS
    """,
]

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
