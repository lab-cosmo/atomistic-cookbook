# -*- coding: utf-8 -*-
"""
Custom Collective Variables for Metadynamics with Pytorch and PLUMED
====================================================================

:Authors: Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_;
          Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_;
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example shows how to build a `metatomic model
<https://docs.metatensor.org/metatomic/latest/overview.html>`_ that computes
order parameters for a Lennard-Jones cluster, and how to use it with
the `PLUMED <https://www.plumed.org/>`_ package to run a metadynamics
simulation.

The LJ38 cluster is a classic benchmark system because its global minimum energy
structure is a truncated octahedron with :math:`O_h` symmetry, which is
difficult to find with simple optimization methods. The PES has a multi-funnel
landscape, meaning the system can easily get trapped in other local minima.
Our goal is to explore the PES, moving from a random initial configuration to
the low-energy structures. To do this, we will:

1.  Define a set of **collective variables (CVs)** that can distinguish between
    the disordered (liquid-like) and ordered (solid-like) states of the
    cluster. We will use a custom CVs analogous to **Steinhardt order parameters**
    (:math:`Q_4` and :math:`Q_6`, a.k.a the bond-order parameters).
2.  Implement this custom CV using ``featomic``, ``metatensor``, and ``metatomic``
    to create a portable ``metatomic`` model.
3.  Run metadynamics trajectories with LAMMPS, and visualize the system as it 
    explores different configurations.
4.  Show an example of integration with `i-PI <https://ipi-code.org/>`_, that 
    uses multiple time stepping to reduce the cost of computing complicated CVs.

As usual for these examples, the simulation is run on a small system and for
a short time, so that results will be fast but inaccurate. If you want to use
this exanmple as a template, you should set more appropriate parameters.
"""

# %%
import os
import subprocess
from typing import Dict, List, Optional

import ase.calculators.lj
import ase.io
import ase.optimize

#
import chemiscope
import featomic.torch
import matplotlib.pyplot as plt
import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
import torch
from matplotlib import colormaps

if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821

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

opt_atoms = atoms.copy()

# %%
# The Target Structures
# ---------------------
#
# The two most "famous" structures for LJ38 are the global minimum (a perfect
# truncated octahedron) and a lower-symmetry icosahedral structure which is a
# deep local minimum. Let's load them and visualize all three (our starting
# structure and the two targets) using ``chemiscope``.

minimal = ase.io.read("data/lj-oct.xyz")
icosaed = ase.io.read("data/lj-ico.xyz")

settings = {"structure": [{"playbackDelay": 50, "unitCell": True, "bonds": False}]}
chemiscope.show([minimal, icosaed, atoms], mode="structure", settings=settings)


# %%
# Defining our custom collective variable
# ---------------------------------------
#
# To distinguish between the liquid-like state and the highly ordered
# face-centered cubic (FCC) packing of the global minimum, the **Steinhardt
# order parameters**, specifically :math:`Q_4` and :math:`Q_6` are commmonly
# used. These parameters are rotationally invariant and measure the local
# orientational symmetry around each atom. The standard caclulation works by
# summing over bond vectors within a cutoff radius which connect a central atom
# to the neighbors and does not use a weighing within the cutoff radius.
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
# distance information through the radial basis set. Additionally, unlike the
# sharp cutoff of the Steinhardt, we will use a cosine cutoff.


# %%
#
# Encapsulating the Logic in a ``torch.nn.Module``
# '''''''''''''''''''''''''''''''''''''''''''''''''
#
# To make this CV usable by PLUMED via the ``METATOMIC`` interface, we must wrap
# our calculation logic in a ``torch.nn.Module``. This class takes a list of
# atomic systems and returns a ``metatensor.TensorMap`` containing the
# calculated CV values. The interface is defined in the `PyTorch`_
# documentation.
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


class SoapCV(torch.nn.Module):
    def __init__(self, cutoff, angular_list):
        super().__init__()

        self.max_angular = max(angular_list)
        # initialize and store the featomic calculator inside the class
        self.spex = featomic.torch.SphericalExpansion(
            **{
                "cutoff": {
                    "radius": 1.5,
                    "smoothing": {"type": "ShiftedCosine", "width": 0.5},
                },
                "density": {"type": "Gaussian", "width": 0.25},
                "basis": {
                    "type": "TensorProduct",
                    "max_angular": self.max_angular,
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
        
        blocks: List[mts.TensorBlock] = []
        for block in spex.blocks():
            # sums over both the m and the radial components
            new_block = mts.TensorBlock(
                (block.values**2).sum(dim=(1, 2)).reshape(-1, 1),
                samples=block.samples,
                components=[],
                properties=mts.Labels("n", torch.tensor([[0]])),
            )
            blocks.append(new_block)

        summed_q = mts.TensorMap(spex.keys, blocks)
        summed_q = summed_q.keys_to_properties("o3_lambda")
        summed_q = mts.mean_over_samples(summed_q, sample_names=["atom", "center_type"])

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
# `upstream API documentation
# <https://docs.metatensor.org/metatomic/latest/torch/reference/models/export.html>`_
# and the
# `metatomic export example
# <https://docs.metatensor.org/metatomic/latest/examples/1-export-atomistic-model.html>`_
# for more information about exporting metatensor models.

# initialize the model
cutoff = 1.5
module = SoapCV(cutoff, angular_list=[4, 6])

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
model.save("soap-cv.pt", collect_extensions="./extensions/")


# %%
# coordination histogram
def f_coord(y:torch.Tensor) -> torch.Tensor:
    """
    This function computes a switching function
    """
    cy = torch.zeros_like(y)
    cy.requires_grad_(True)
    
    # Apply conditions as per the Fortran code using torch.where
    cy = torch.where(y <= 0, torch.tensor(1.0, dtype=torch.float32), cy)
    cy = torch.where(y >= 1, torch.tensor(0.0, dtype=torch.float32), cy)
    mask = (y > 0) & (y < 1)
    cy = torch.where(mask, ((y - 1) ** 2) * (1 + 2 * y), cy)
    return cy

class CoordinationHistogram(torch.nn.Module):
    def __init__(self, cutoff, cn_list):
        super().__init__()

        self.cn_list = torch.tensor(cn_list, dtype=torch.int)
        self.cutoff = cutoff
        self.r0 = cutoff
        self.r1 = cutoff*4.0/5.0
        self.sigma2 = 0.5**2
        
    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:

        if len(systems[0].positions) == 0:
            # PLUMED will first call the model with 0 atoms to get the size of the
            # output, so we need to handle this case first
            keys = mts.Labels("_", torch.tensor([[0]]))
            block = mts.TensorBlock(
                torch.zeros((0, len(self.cn_list)), dtype=torch.float64),
                samples=mts.Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=mts.Labels("cn", self.cn_list.reshape(-1,1)),
            )
            return {"features": mts.TensorMap(keys, [block])}

        values = []
        isys = torch.arange(len(systems), dtype=torch.int32).reshape((-1,1))
        for s in systems:
            pos = s.positions

            # Calculate pairwise distances  
            dist = torch.cdist(pos, pos, p=2.0)
            coords = f_coord((dist-self.r1) / (self.r0-self.r1)).sum(dim=1) - 1.0

            cn_histo = torch.exp(-(coords - self.cn_list.reshape(-1, 1))**2*0.5/self.sigma2).sum(dim=1)
            values.append(cn_histo)

        keys = mts.Labels("_", torch.tensor([[0]]))
        values = torch.stack(values, dim=0)
        block = mts.TensorBlock(
            values=values,
            samples=mts.Labels("structure", isys),
            components=[],
            properties=mts.Labels("cn", self.cn_list.reshape(-1,1)),
        )
        mts_coords = mts.TensorMap(keys, [block])
        # This model has a single output, named "features". This can be used by multiple
        # tools, including PLUMED where it defines a custom collective variable.
        return {"features": mts_coords}

cutoff = 1.5
module = CoordinationHistogram(cutoff, cn_list=[6, 8])

# metatdata about the model itself
metadata = mta.ModelMetadata(
    name="Coordination histogram",
    description="Computes smooth histogram of coordination numbers",
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

model.save("histo-cv.pt", collect_extensions="./extensions/")


# %%

featurizer = chemiscope.metatomic_featurizer(model)
featurizer([minimal, icosaed, atoms], None)
# mta.systems_to_torch([ minimal, icosaed, atoms])
# model.forward(mta.systems_to_torch([ minimal, icosaed, atoms]), None, None)


# %%
# Optional: Test the Model in Python
# ''''''''''''''''''''''''''''''''''
#
# Before running the full simulation, we can use ``chemiscope``'s
# ``metatomic_featurizer`` to quickly check the output of our model on our
# initial structures. This is a great way to verify that the CVs produce
# different values for the different structures.
featurizer = chemiscope.metatomic_featurizer(model)
chemiscope.explore([minimal, icosaed, atoms], featurize=featurizer, settings=settings)

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
# - ``METAD`` : sets up the metadynamics algorithm. It will add repulsive
#             Gaussian potentials in the (``cv1``, ``cv2``) space at regular
#             intervals (``PACE``), discouraging the simulation from re-visiting
#             conformations and pushing it over energy barriers
# - ``PRINT`` : This tells PLUMED to write the values of our CVs and the
#             metadynamics bias energy to a file named ``COLVAR`` for later analysis.

if os.path.exists("HILLS"):
    os.unlink("HILLS")

if os.path.exists("COLVARS"):
    os.unlink("COLVARS")

with open("data/plumed.dat", "r") as fname:
    print(fname.read())

# %%
# Running dynamics with ``lammps``
# --------------------------------
#
# LAMMPS is easily amongst the most robust molecular dynamics engine.
#

lmp_atoms = opt_atoms.copy()
lmp_atoms.positions += lmp_atoms.cell[0,0]*0.5
lmp_atoms.set_masses([1.0] * len(atoms))
ase.io.write("data/firemin.data", lmp_atoms, format="lammps-data")

subprocess.run(["lmp", "-in", "data/lammps.plumed.in"], check=True, capture_output=True)
lmp_trajectory = [opt_atoms.copy()]
lmp_trajectory.append(ase.io.read("out/lj38.lammpstrj", index=":"))

# %%
# Static visualization - I
# ---------------------------
#
# The dynamics on the free energy surface can be visualized using a static plot
# with the trajectory overlaid as follows.
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
grid_min = [0,0]
grid_max = [1, 2]
grid_bins = [100, 200]
grid_cv1 = np.linspace(grid_min[0], grid_max[0], grid_bins[0])
grid_cv2 = np.linspace(grid_min[1], grid_max[1], grid_bins[1])
X, Y = np.meshgrid(grid_cv1, grid_cv2)
FES = np.zeros_like(X)

# Sum over Gaussian hills to reconstruct the bias
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
plt.colorbar(contour, label="Free Energy")

# Overlay the trajectory
plt.plot(
    cv1_traj,
    cv2_traj,
    color="white",
    alpha=0.7,
    linewidth=1.5,
    label="LAMMPS MD Trajectory",
)

# Mark significant points

feats = featurizer([minimal, icosaed], None)
plt.scatter(
    feats[0,0], feats[0,1], c="red", marker="X", s=150, zorder=3, label="octahedron"
)
plt.scatter(
    feats[1,0], feats[1,1], c="cyan", marker="o", s=150, zorder=3, label="icosahedral"
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
# Static visualization - II
# ---------------------------
#
# We can also just check the basins.
#

# Kanged from
# https://github.com/Sucerquia/ASE-PLUMED_tutorial/blob/master/files/plotterFES.py
p = subprocess.Popen(
    "plumed sum_hills --hills HILLS --outfile fes.dat"
    + " --bin 100,200 --min 0,0 --max 1,2",
    shell=True,
    stdout=subprocess.PIPE,
)
p.wait()

# %%
# Import free energy and reshape with the number of bins defined in the
# reconstruction process.
scm = np.loadtxt("fes.dat", usecols=0).reshape(101, 201)
tcm = np.loadtxt("fes.dat", usecols=1).reshape(101, 201)
fes = np.loadtxt("fes.dat", usecols=2).reshape(101, 201)

# Plot
fig, ax = plt.subplots(figsize=(10, 9))

# Plot free energy surface
im = ax.contourf(scm, tcm, fes, 10, cmap=colormaps["Blues_r"])  # cmo.tempo_r)
cp = ax.contour(scm, tcm, fes, 10, linestyles="-", colors="darkgray", linewidths=1.2)

# Plot parameters
ax.set_xlabel("MTA_Q4", fontsize=40)
ax.set_ylabel("MTA_Q6", fontsize=40)
ax.tick_params(axis="y", labelsize=25)
ax.tick_params(axis="x", labelsize=25)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(label=r"FES[$\epsilon$]", fontsize=40)
cbar.ax.tick_params(labelsize=32)

plt.tight_layout()
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
        "values": cv1_traj[::10],
        "description": "Collective Variable 1 (mta_q4)",
    },
    "cv2": {
        "target": "structure",
        "values": cv2_traj[::10],
        "description": "Collective Variable 2 (mta_q6)",
    },
    "time": {
        "target": "structure",
        "values": time[::10],
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

lmp_trajectory = ase.io.read("out/lj38.lammpstrj", index=":")
# Show the trajectory in an interactive chemiscope widget.
chemiscope.show(
     frames=lmp_trajectory,
     properties=dyn_prop,
     settings=dyn_settings,
)

# _PyTorch: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html

# %%
