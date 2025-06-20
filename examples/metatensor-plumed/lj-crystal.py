# -*- coding: utf-8 -*-
"""
Exploring the Lennard-Jones 38 Cluster with Metadynamics
=========================================================

:Authors: Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_;
          Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_;
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

We shall demonstrate the usage of `metatomic models
<https://docs.metatensor.org/metatomic/latest/overview.html>`_ within enhanced
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
    cluster. We will use a custom CVs analogous to **Steinhardt order parameters**
    (:math:`Q_4` and :math:`Q_6`, a.k.a the bond-order parameters).
2.  Implement this custom CV using ``featomic``, ``metatensor``, and ``metatomic`` to
    create a portable ``metatomic`` model.
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
import featomic.torch
import matplotlib.pyplot as plt
import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
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
# structure and the two targets) using ``chemiscope``.

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
# face-centered cubic (FCC) packing of the global minimum, the **Steinhardt
# order parameters**, specifically :math:`Q_4` and :math:`Q_6`. These parameters
# are rotationally invariant and measure the local orientational symmetry around
# each atom. The standard caclulation works by summing over bond vectors within
# a cutoff radius which connect a central atom to the neighbors and does not use
# a weighing within the cutoff radius.
#
# - :math:`Q_6` is often high for both icosahedral and FCC-like structures,
#   making it a good measure of general "solidness".
# - :math:`Q_4` helps to distinguish between different crystal packing types. It
#   is close to zero for icosahedral structures but has a distinct non-zero value
#   for FCC structures.
#
# This works very well for the LJ38 and is also part of the standard PLUMED build.
#
# The key concept is that the geometry of the atomic neighborhood is described
# by projection onto a basis of spherical harmonics. With that in mind, we will
# demonstrate the usage of the SOAP power spectrum, which differs from the
# standard Steinhardt by operating on a smooth density field, and includes
# distance information through the aradial basis set. Additionally, unlike the
# sharp cutoff of the Steinhardt, we will use a cosine cutoff.


# %%
#
# Encapsulating the Logic in a ``torch.nn.Module``
# '''''''''''''''''''''''''''''''''''''''''''''''''
#
# To make this CV usable by PLUMED via the ``METATOMIC`` interface, we must wrap
# our calculation logic in a ``torch.nn.Module``. This class takes a list of
# atomic systems and returns a ``metatensor.TensorMap`` containing the
# calculated CV values. The interface is defined in
# `the PyTorch documentation <https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
# with more examples in the .
#
# Our descriptor is computed in a way that is closely related to
# the Smooth Overlap of Atomic Positions (SOAP) formalism, ensuring it is
# rotationally invariant and measures local orientational symmetry.
#
# We will use the power spectrum components for angular channels :math:`l=4` and
# :math:`l=6`.
#
# - The :math:`l=6` component is a good measure of general "solidness" or
#   ordering, as it is significant for both icosahedral and FCC-like structures.
# - The :math:`l=4` component helps to distinguish between different crystal
#   packing types. It has a distinct non-zero value for FCC structures but is
#   smaller for icosahedral symmetries.
#
# We will build a model that calculates global, system-averaged versions of
# these parameters, bearing in mind that as analogs to the Steinhardt we expect
# similar results.


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
            # These represent the degree of the spherical harmonics
            "o3_lambda",
            torch.tensor(angular_list).reshape(-1, 1),
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
# See the
# `upstream API documentation <https://docs.metatensor.org/metatomic/latest/torch/reference/models/export.html>`_
# and the
# `metatomic export example <https://docs.metatensor.org/metatomic/latest/examples/1-export-atomistic-model.html>`_
# for more information about exporting metatensor models.

# initialize the model
cutoff = 1.3
module = CollectiveVariable(cutoff, angular_list=[4, 6])

# metatdata about the model itself
metadata = mta.ModelMetadata(
    name="SOAP_order_params",
    description="Computes smoothed out versions of the Steinhardt order parameters",
)

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
chemiscope.explore([minimal, other, atoms], featurize=featurizer, settings=settings)

# %%
#
# Using the model to run metadynamics with PLUMED
# -----------------------------------------------
#
# With our model saved, we can now write the PLUMED input file. This file
# instructs PLUMED on what to do during the simulation.
# The input file consists of the following sections:
# - ``UNITS`` : Specifies the energy and length units
# - ``METATOMIC`` : Defines a collective variable which is essentially
#                 an exported metatomic model
# - ``SELECT_COMPONENTS`` : Splits the model output :math:`Q_4`
#                         and :math:`Q_6` parameters to scalars
# - ``METAD`` : sets up the metadynamics algorithm. It will add repulsive Gaussian
#             potentials in the (``cv1``, ``cv2``) space at regular intervals (``PACE``),
#             discouraging the simulation from re-visiting conformations and pushing it
#             over energy barriers
# - ``PRINT`` : This tells PLUMED to write the values of our CVs and the
#             metadynamics bias energy to a file named ``COLVAR`` for later analysis.

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
        HEIGHT=0.05
        PACE=50
        SIGMA=1,2.5
        GRID_MIN=-20,-40
        GRID_MAX=20,40
        GRID_BIN=500,500
        BIASFACTOR=5
        FILE=HILLS
    """,
    # prints out trajectory
    """
    PRINT ARG=cv.*,mtd.* STRIDE=10 FILE=COLVAR
    """,
    """
    FLUSH STRIDE=1
    """,
]

# %%
# Running dynamics I - ``ase``
# ---------------------------
#
# The easiest way to generate a trajectory is to leverage ``ase``. In subsequent
# sections we will use LAMMPS, as a more production worthy dynamics engine.
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

trajectory = [atoms.copy()]
for _ in range(100):
    md.run(steps=10)
    trajectory.append(atoms.copy())


# %%
# Running dynamics II - ``lammps``
# --------------------------------
#
# LAMMPS is easily amongst the most robust molecular dynamics engine.
#

plumed_input = f"""
UNITS LENGTH=A ENERGY={ase.units.mol / ase.units.kJ}
cv: METATOMIC ...
    MODEL=custom-cv.pt
    EXTENSIONS_DIRECTORY=./extensions/
    SPECIES1=1-38
    SPECIES_TO_TYPES=18
...
cv1: SELECT_COMPONENTS ARG=cv COMPONENTS=1
cv2: SELECT_COMPONENTS ARG=cv COMPONENTS=2
mtd: METAD ...
    ARG=cv1,cv2
    HEIGHT=0.05
    PACE=50
    SIGMA=1,2.5
    GRID_MIN=-20,-40
    GRID_MAX=20,40
    GRID_BIN=500,500
    BIASFACTOR=5
    FILE=HILLS
    TEMP=300
...
PRINT ARG=cv.*,mtd.* STRIDE=10 FILE=COLVAR
FLUSH STRIDE=1
"""

with open("plumed.dat", "w") as fname:
    fname.write(plumed_input)

atoms.set_atomic_numbers([1] * len(atoms))
atoms.set_masses([1.0] * len(atoms))
ase.io.write("structure.data", atoms, format="lammps-data")

# Get LJ parameters from the ASE calculator using the default for 'Ar'
lj_sigma = 3.405  # Angstrom
lj_epsilon = 0.010323  # eV
lj_cutoff = 2.5
# Simulation parameters
timestep = 0.01  # in ps
n_steps = 1000  # 100 iterations * 10 steps/iter
temperature = 0.1  # in eV (kT in ASE)
# The LAMMPS damp parameter is a damping time. In ASE, friction = 1.0
langevin_damp = 1.0  # in ps

lammps_in = f"""
# LAMMPS input script for Metadynamics of LJ38 cluster

# -- Initialization --
units           metal
atom_style      atomic
boundary        p p p
read_data       structure.data

# -- Potential --
# All atoms are type 1
pair_style      lj/cut {lj_cutoff}
pair_coeff      1 1 {lj_epsilon} {lj_sigma} {lj_cutoff}
mass            1 1.0

# -- Plumed integration --
fix             1 all plumed plumedfile plumed.dat outfile plumed.out

# -- NVT Dynamics (Langevin Thermostat) --
velocity        all create {temperature} 8675309 dist gaussian
fix             2 all nve
fix             3 all langevin {temperature} {temperature} {langevin_damp} 1995

# -- Simulation run --
timestep        {timestep}
thermo          100
dump            1 all xyz 10 traj.xyz
run             {n_steps}
"""

with open("lammps.in", "w") as f:
    f.write(lammps_in)

subprocess.run(["lmp", "-in", "lammps.in"], check=True, capture_output=True, text=True)
trajectory = [atoms.copy()]
trajectory.append(ase.io.read("traj.xyz", index=":"))

# %%
# Static visualization
# ---------------------------
#
# The dynamics on the free energy surface can be visualized using a static plot
# as follows.
#

# time, cv1, cv2, mtd.bias
colvar = np.loadtxt("COLVAR")
time = colvar[:, 0]
cv1_traj = colvar[:, 1]
cv2_traj = colvar[:, 2]

# HILLS has the free energy surface
# time, center_cv1, center_cv2, sigma_cv1, sigma_cv2, height
hills = np.loadtxt("HILLS")

# Visually pleasing grid for the FES based on the PLUMED input
grid_min = [-2.5, -5]
grid_max = [12, 20]
grid_bins = [500, 500]
grid_cv1 = np.linspace(grid_min[0], grid_max[0], grid_bins[0])
grid_cv2 = np.linspace(grid_min[1], grid_max[1], grid_bins[1])
X, Y = np.meshgrid(grid_cv1, grid_cv2)
FES = np.zeros_like(X)

# Sum over aussian hills to reconstruct the bias
for hill in hills:
    center_cv1, center_cv2 = hill[1], hill[2]
    sigma_cv1, sigma_cv2 = hill[3], hill[4]
    height = hill[5]

    term1 = (X - center_cv1) ** 2 / (2 * sigma_cv1**2)
    term2 = (Y - center_cv2) ** 2 / (2 * sigma_cv2**2)
    FES += height * np.exp(-(term1 + term2))

# The free energy surface is the -ve of the summed bias potential
# Shift for 0 minimum
FES = -FES
FES -= FES.min()

# Prepare the plot
plt.figure(figsize=(10, 7))
contour = plt.contourf(X, Y, FES, levels=np.linspace(0, FES.max(), 25), cmap="viridis")
plt.colorbar(contour, label="Free Energy (kJ/mol)")

# Overlay the trajectory
plt.plot(
    cv1_traj, cv2_traj, color="white", alpha=0.7, linewidth=1.5, label="MD Trajectory"
)

# Mark the start and end points
plt.scatter(
    cv1_traj[0], cv2_traj[0], c="red", marker="X", s=150, zorder=3, label="Start"
)
plt.scatter(
    cv1_traj[-1], cv2_traj[-1], c="cyan", marker="o", s=150, zorder=3, label="End"
)

plt.title("Free Energy Surface of LJ38 Cluster")
plt.xlabel("Collective Variable 1 ($q_4$)")
plt.ylabel("Collective Variable 2 ($q_6$)")
plt.xlim(grid_min[0], grid_max[0])
plt.ylim(grid_min[1], grid_max[1])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# %%
# Dynamic visualization
# ---------------------------
#
# The structures with the on the free energy surface can
# be visualized using a static plot as follows.
#

dyn_prop = {
    "cv1": {
        "target": "structure",
        "values": cv1_traj,
        "description": "Collective Variable 1 (q4)",
    },
    "cv2": {
        "target": "structure",
        "values": cv2_traj,
        "description": "Collective Variable 2 (q6)",
    },
    "time": {
        "target": "structure",
        "values": time,
        "description": "Simulation time",
        "units": "ps",
    },
}

# Configure the settings for the chemiscope visualization.
dyn_settings = chemiscope.quick_settings(
    x="cv1",
    y="cv2",
    color="time",
    trajectory=True,
    map_settings={
        "x": {"max": 12, "min": -2.5},
        "y": {"max": 20, "min": -5},
    },
)

# Show the trajectory in an interactive chemiscope widget.
chemiscope.show(
    frames=trajectory,
    properties=dyn_prop,
    settings=dyn_settings,
)
