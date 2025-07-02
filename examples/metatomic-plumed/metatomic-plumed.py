# -*- coding: utf-8 -*-
"""
ML collective variables in PLUMED with metatomic
================================================

:Authors: Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_;
          Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_;
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example shows how to build a `metatomic model
<https://docs.metatensor.org/metatomic/latest/overview.html>`_ that computes order
parameters for a Lennard-Jones cluster, and how to use it with the `PLUMED
<https://www.plumed.org/>`_ package to run a metadynamics simulation.

The LJ38 cluster is a classic benchmark system because its global minimum energy
structure is a truncated octahedron with :math:`O_h` symmetry, which is difficult to
find with simple optimization methods. The PES has a multi-funnel landscape, meaning the
system can easily get trapped in other local minima. Our goal is to explore the PES,
moving from a random initial configuration to the low-energy structures. To do this, we
will:

1.  Define a set of **collective variables (CVs)** that can distinguish between the
    disordered (liquid-like) and ordered (solid-like) states of the cluster. We will use
    two sets of CVs: histograms of the coordination number of atoms, and two CVs derived
    from SOAP descriptors that are analogous to the **Steinhardt order parameters**
    :math:`Q_4` and :math:`Q_6` (a.k.a the bond-order parameters).
2.  Implement these custom CV using ``featomic``, ``metatensor``, and ``metatomic`` to
    create a portable ``metatomic`` model.
3.  Run metadynamics trajectories with LAMMPS, and visualize the system as it explores
    different configurations.
4.  Show an example of integration with `i-PI <https://ipi-code.org/>`_, that uses
    multiple time stepping to reduce the cost of computing complicated CVs.

As usual for these examples, the simulation is run on a small system and for a short
time, so that results will be fast but inaccurate. If you want to use this example as a
template, you should set more appropriate parameters.
"""

import linecache
import os
import pathlib
import subprocess
from time import sleep
from typing import Dict, List, Optional

import ase.io
import chemiscope
import featomic.torch
import ipi
import matplotlib.pyplot as plt
import metatensor.torch as mts
import metatomic.torch as mta
import numpy as np
import torch


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821

# %%
#
# Target structures
# ^^^^^^^^^^^^^^^^^
#
# The two most "famous" structures for LJ38 are the global minimum (a perfect truncated
# octahedron) and a lower-symmetry icosahedral structure which is a deep local minimum.
# We can visualize them using `chemiscope <https://chemiscope.org/docs/>`_.

minimal = ase.io.read("data/lj-oct.xyz")
icosaed = ase.io.read("data/lj-ico.xyz")

settings = {"structure": [{"playbackDelay": 50, "unitCell": True, "bonds": True, "spaceFilling": True}]}
chemiscope.show([minimal, icosaed], mode="structure", settings=settings)


# %%
#
# Defining custom collective variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use two different approaches to define our custom CVs: in one case, we compute the
# CV manually starting from the atomic positions, in the other we build the descriptors
# based on a spherical-harmonics expansion of the neighbor density, computed using the
# `featomic <https://metatensor.github.io/featomic/>`_ package.
#
# Histogram of coordination numbers
# ---------------------------------
#
# As a first set of CVs, we use a histogram of coordination numbers, that was used in
# several examples studying this cluster (see e.g. `this paper
# <http://doi.org/10.1021/ct3010563>`_). The idea is to compute the coordination number
# of each atom in the cluster, and then to count how many atoms have a given
# coordination number. This makes it possible to differentiate clearly between the
# truncated octahedron and the icosahedral configurations, as well as distorted,
# disordered structures.
#
#
# To make this CV usable by PLUMED via the ``METATOMIC`` interface, we must wrap our
# calculation logic in a ``torch.nn.Module``. This class takes a list of atomic systems
# and returns a ``metatensor.TensorMap`` containing the calculated CV values. The
# interface is defined in the `metatomic
# <https://docs.metatensor.org/metatomic/latest/torch/reference/models/export.html>`_
# documentation.


def f_coord(y: torch.Tensor) -> torch.Tensor:
    """
    This function computes a switching function that we use
    to evaluate the coordination number.
    """
    cy = torch.zeros_like(y)
    # we use torch.where to be compatible with autodiff
    cy = torch.where(y <= 0, torch.tensor(1.0, dtype=torch.float32), cy)
    cy = torch.where(y >= 1, torch.tensor(0.0, dtype=torch.float32), cy)
    mask = (y > 0) & (y < 1)
    cy = torch.where(mask, ((y - 1) ** 2) * (1 + 2 * y), cy)
    return cy


class CoordinationHistogram(torch.nn.Module):
    def __init__(self, cutoff, cn_list):
        """
        ``cutoff`` provides the point at which the switching function levels off to
        zero. Note that for simplicity we still compute all distances.

        ``cn_list`` is the list of bins in the histogram. Strictly speaking, we don't
        compute a histogram, but a kernel density estimator centered on the values given
        in this list.
        """
        super().__init__()

        self.cn_list = torch.tensor(cn_list, dtype=torch.int32)
        self._nl_options = mta.NeighborListOptions(
            cutoff=cutoff, full_list=True, strict=True
        )
        self.cutoff = cutoff
        self.r0 = cutoff
        self.r1 = cutoff * 4.0 / 5.0
        self.sigma2 = 0.5**2

    def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
        # request a neighbor list to be computed and stored in the system passed to
        # `forward`.
        return [self._nl_options]

    def forward(
        self,
        systems: List[mta.System],
        outputs: Dict[str, mta.ModelOutput],
        selected_atoms: Optional[mts.Labels],
    ) -> Dict[str, mts.TensorMap]:
        if "features" not in outputs:
            return {}

        if outputs["features"].per_atom:
            raise ValueError("per-atoms features are not supported in this model")

        if selected_atoms is not None:
            raise ValueError("selected_atoms is not supported in this model")

        if len(systems[0].positions) == 0:
            # PLUMED will first call the model with 0 atoms to get the size of the
            # output, so we need to handle this case first
            keys = mts.Labels("_", torch.tensor([[0]]))
            block = mts.TensorBlock(
                torch.zeros((0, len(self.cn_list)), dtype=torch.float64),
                samples=mts.Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=mts.Labels("cn", self.cn_list.reshape(-1, 1)),
            )
            return {"features": mts.TensorMap(keys, [block])}

        values = []
        # loop over all systems
        system_index = torch.arange(len(systems), dtype=torch.int32).reshape((-1, 1))
        for system in systems:
            # the neighbor list has been computed by whoever is calling this model
            neighbors = system.get_neighbor_list(self._nl_options)

            # get all distances
            distances = torch.linalg.vector_norm(neighbors.values.reshape(-1, 3), dim=1)

            # switching function to evaluate the coordination number
            z = f_coord((distances - self.r1) / (self.r0 - self.r1))

            # Sum over neighbors
            coords = torch.zeros(len(system), dtype=z.dtype, device=z.device)
            coords.index_add_(
                dim=0, index=neighbors.samples.column("first_atom"), source=z
            )

            # Compute the KDE over the required bins
            cn_histo = torch.exp(
                -((coords - self.cn_list.reshape(-1, 1)) ** 2) * 0.5 / self.sigma2
            ).sum(dim=1)
            values.append(cn_histo)

        # Assembles a TensorMap
        keys = mts.Labels("_", torch.tensor([[0]]))
        block = mts.TensorBlock(
            values=torch.stack(values, dim=0),
            samples=mts.Labels("structure", system_index),
            components=[],
            properties=mts.Labels("cn", self.cn_list.reshape(-1, 1)),
        )
        mts_coords = mts.TensorMap(keys, [block])
        # This model has a single output, named "features". This can be used by multiple
        # tools, including PLUMED where it defines a custom collective variable.
        return {"features": mts_coords}


# %%
#
# SOAP-based Steinhardt parameters
# --------------------------------
#
# Rather than looking at the atomic coordination, one can also resort to order
# parameters that capture the angular order. The **Steinhardt order parameters**,
# specifically :math:`Q_4` and :math:`Q_6` are rotationally invariant and measure the
# local orientational symmetry around each atom. The standard caclulation works by
# summing over bond vectors within a cutoff radius which connect a central atom to the
# neighbors and does not use a weighing within the cutoff radius.
#
# - :math:`Q_6` is often high for both icosahedral and FCC-like structures, making it a
#   good measure of general "solidness".
# - :math:`Q_4` helps to distinguish between different crystal packing types. It is
#   close to zero for icosahedral structures but has a distinct non-zero value for FCC
#   structures.
#
# This works fairly well for the LJ38 and is also part of the standard PLUMED build.
#
# The key concept is that the geometry of the atomic neighborhood is described by
# projection onto a basis of spherical harmonics. With that in mind, we will use the
# SOAP power spectrum, which differs from the standard Steinhardt by operating on a
# smooth density field, and includes distance information through the radial basis set.
# Additionally, unlike the sharp cutoff of the Steinhardt, we will use a cosine cutoff.
#
# We will use the power spectrum components for angular channels :math:`l=4` and
# :math:`l=6`. SOAP features are computed from an expansion of the neighbor density in
# spherical harmonics and radial functions.  ``featomic`` is used to evaluate this
# spherical expansion, select the appropriate angular indices and then sum over the
# :math:`m` index, as well as over the radial dimension to recover order parameters
# analogous to :math:`Q_4` and :math:`Q_6`.


class SoapCV(torch.nn.Module):
    def __init__(self, cutoff, angular_list):
        super().__init__()

        self.max_angular = max(angular_list)
        # initialize and store the featomic calculator inside the class
        self.spex = featomic.torch.SphericalExpansion(
            **{
                "cutoff": {
                    "radius": cutoff,
                    "smoothing": {"type": "ShiftedCosine", "width": 1.5},
                },
                "density": {"type": "Gaussian", "width": 1.0},
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
        if "features" not in outputs:
            return {}

        # computes the spherical expansion
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

        # then manipulate the tensormap to remove some of the sparsity
        spex = mts.remove_dimension(spex, axis="keys", name="o3_sigma")
        spex = spex.keys_to_properties("neighbor_type")
        spex = spex.keys_to_samples("center_type")

        blocks: List[mts.TensorBlock] = []
        for block in spex.blocks():
            # squares, and sums over both the m and the radial components
            new_block = mts.TensorBlock(
                (block.values**2).sum(dim=(1, 2)).reshape(-1, 1),
                samples=block.samples,
                components=[],
                properties=mts.Labels("n", torch.tensor([[0]])),
            )
            blocks.append(new_block)

        # packs the resulting values in a tensormap
        summed_q = mts.TensorMap(spex.keys, blocks)
        summed_q = summed_q.keys_to_properties("o3_lambda")

        if not outputs["features"].per_atom:
            summed_q = mts.mean_over_samples(
                summed_q, sample_names=["atom", "center_type"]
            )

        # This model, like CoordinationHistogram has a single output, named "features".
        return {"features": summed_q}


# %%
#
# Exporting the Model
# -------------------
#
# Once we have defined our custom model, we can now annotate it with multiple metadata
# entries and export it to the disk. The resulting model file and extensions directory
# can then be loaded by PLUMED and other compatible engines, without requiring a Python
# installation (for example on HPC systems).
#
# See the `corresponding documentation
# <https://docs.metatensor.org/metatomic/latest/torch/reference/models/export.html>`_
# and `example
# <https://docs.metatensor.org/metatomic/latest/examples/1-export-atomistic-model.html>`_
# for more information about exporting metatomic models.

# generates a coordination histogram model
cutoff = 5.5
module = CoordinationHistogram(cutoff, cn_list=[6, 8])

# metatdata about the model itself
metadata = mta.ModelMetadata(
    name="Coordination histogram",
    description="Computes smooth histogram of coordination numbers",
)

# metatdata about what the model can do
capabilities = mta.ModelCapabilities(
    length_unit="Angstrom",
    outputs={"features": mta.ModelOutput(per_atom=False)},
    atomic_types=[18],
    interaction_range=cutoff,
    supported_devices=["cpu"],
    dtype="float64",
)

model_ch = mta.AtomisticModel(
    module=module.eval(),
    metadata=metadata,
    capabilities=capabilities,
)

model_ch.save("histo-cv.pt", collect_extensions="./extensions/")

# ... and a SOAP-based CV, with the same cutoff
module = SoapCV(cutoff, angular_list=[4, 6])

metadata = mta.ModelMetadata(
    name="Steinhardt-like SOAP CV",
    description="Computes smoothed out versions of the Steinhardt order parameters",
)

model_soap = mta.AtomisticModel(
    module=module.eval(),
    metadata=metadata,
    capabilities=capabilities,
)

# finally, save the model to a standalone file
model_soap.save("soap-cv.pt", collect_extensions="./extensions/")


# %%
#
# Optional: test the collective variables in Python
# -------------------------------------------------
#
# Before running the full simulation, we can use ``chemiscope``'s
# ``metatomic_featurizer`` to quickly check the output of our model on our initial
# structures. This is a great way to verify that the CVs produce different values for
# the different structures.

featurizer_ch = chemiscope.metatomic_featurizer(model_ch)
featurizer_soap = chemiscope.metatomic_featurizer(model_soap)

chemiscope.explore(
    [minimal, icosaed],
    featurize=featurizer_ch,
    # we can also add extra properties, here we use this
    # to include additional descriptors
    properties={"cv_soap": featurizer_soap([minimal, icosaed], None)},
    settings=settings,
)

# %%
#
# Using the model to run metadynamics with PLUMED
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# With our model saved, we can now write the PLUMED input file. This file instructs
# PLUMED on what to do during the simulation. The input file consists of the following
# sections:
#
# - ``UNITS``: Specifies the energy and length units;
# - ``METATOMIC``: Defines a collective variable using an exported metatomic model. We
#   load both the models we just created, and use the CV histogram that is faster to
#   compute (and more efficient with metadynamics);
# - ``SELECT_COMPONENTS``: Splits the model output to scalars;
# - ``METAD``: sets up the metadynamics algorithm. It will add repulsive Gaussian
#   potentials in the (``cv1``, ``cv2``) space at regular intervals (``PACE``),
#   discouraging the simulation from re-visiting conformations and pushing it over
#   energy barriers;
# - ``PRINT``: This tells PLUMED to write the values of our CVs and the metadynamics
#   bias energy to a file named ``COLVAR`` for later analysis.

with open("data/plumed.dat", "r") as fd:
    print(fd.read())

# %%
#
# Running metadynamics with LAMMPS
# --------------------------------
#
# We use a custom version of LAMMPS that is linked with ``metatomic`` and the
# ``metatomic``-enabled version of PLUMED. From the point of view of LAMMPS, all that is
# needed is to use ``fix_plumed`` to load the PLUMED input file, as the calculation of
# the custom collective variables is handled by PLUMED itself.

# write the LAMMPS structure file
lammps_initial = minimal.copy()
lammps_initial.cell = [20, 20, 20]
ase.io.write("data/minimal.data", lammps_initial, format="lammps-data")

print(linecache.getline("data/lammps.plumed.in", 25).strip())
subprocess.run(
    ["lmp", "-in", "data/lammps.plumed.in"], check=True, capture_output=False
)
lmp_trajectory = ase.io.read("lj38.lammpstrj", index=":")

# %%
#
# Static visualization
# --------------------
#
# The dynamics on the free energy surface can be visualized using a static plot with the
# trajectory overlaid as follows. NB: The accumulated bias is **not** the free energy
# when performing well-tempered metadynamics, and a re-scaling is required, cf. `the
# original paper <https://doi.org/10.1103/PhysRevLett.100.020603>`_
#
# NB: PLUMED provides dedicated CLI tools to perform these tasks
#

# time, cv1, cv2, mtd.bias, histo.1, histo.2, soap.1, soap.2
colvar = np.loadtxt("COLVAR")
time = colvar[:, 0]
cv1_traj = colvar[:, 1]
cv2_traj = colvar[:, 2]
soap1_traj = colvar[:, 6]
soap2_traj = colvar[:, 7]

# HILLS has the free energy surface
# time, center_cv1, center_cv2, sigma_cv1, sigma_cv2, height
hills = np.loadtxt("HILLS")

# Visually pleasing grid for the FES based on the PLUMED input
grid_min = [0, 0]
grid_max = [30, 20]
grid_bins = [200, 100]
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
FES *= 40 / (40 - 1)  # corrects for the well-tempered biasfactor

# Prepare the plot
plt.figure(figsize=(10, 7))
contour = plt.contourf(X, Y, FES, levels=np.linspace(0, FES.max(), 25), cmap="viridis")
plt.colorbar(contour, label="-bias (a.u.)")

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
feats = featurizer_ch([minimal, icosaed], None)
plt.scatter(
    feats[0, 0], feats[0, 1], c="red", marker="X", s=150, zorder=3, label="octahedron"
)
plt.scatter(
    feats[1, 0], feats[1, 1], c="cyan", marker="o", s=150, zorder=3, label="icosahedral"
)

plt.title("Free Energy Surface of LJ38 Cluster")
plt.xlabel("Collective Variable 1")
plt.ylabel("Collective Variable 2")
plt.xlim(grid_min[0], grid_max[0])
plt.ylim(grid_min[1], grid_max[1])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# %%
#
# Interactive visualization in ``chemiscope``
# -------------------------------------------
#
# The structures on the free energy surface can be visualized using a dynamic
# plot in ``chemiscope``.  We load both the histogram-based and the SOAP-based
# CVs, so you can see the difference in the two approaches by changing the x and
# y axes in the plot settings.

dyn_prop = {
    "cv1": {
        "target": "structure",
        "values": cv1_traj[::10],
        "description": "CV1 (histo c=6)",
    },
    "cv2": {
        "target": "structure",
        "values": cv2_traj[::10],
        "description": "CV2 (histo c=8)",
    },
    "soap1": {
        "target": "structure",
        "values": soap1_traj[::10],
        "description": "SOAP1 (~Q4)",
    },
    "soap2": {
        "target": "structure",
        "values": soap2_traj[::10],
        "description": "CV2 (~Q6)",
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
        "x": {"max": 30, "min": 0},
        "y": {"max": 15, "min": 0},
    },
    structure_settings={
        "playbackDelay": 50,  # ms between frames
        "unitCell": True,
        "bonds": True,
        "spaceFilling": True,
    },
)

lmp_trajectory = ase.io.read("lj38.lammpstrj", index=":")
# Show the trajectory in an interactive chemiscope widget.
chemiscope.show(
    frames=lmp_trajectory,
    properties=dyn_prop,
    settings=dyn_settings,
)

# %%
#
# Running metadynamics with i-PI
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``metatomic`` models can be used with any code that supports the PLUMED interface,
# including `i-PI <https://ipi-code.org/>`_. We take this opportunity to demonstrate the
# multiple time stepping feature of i-PI which reduces the cost of computing expensive
# CVs.
#
# We modify the PLUMED input file to use the SOAP CVs and to half the frequency of the
# metadynamics updates, since we will call PLUMED every two steps. We also change the
# grid and the Gaussian width, consistent with the different range of the SOAP CVs.

src = pathlib.Path("data/plumed.dat")
dst = pathlib.Path("data/plumed-mts.dat")

content = src.read_text()
dst.write_text(
    content.replace("ARG=histo", "ARG=soap")
    .replace("PACE=30", "PACE=15")
    .replace("GRID_MAX=30,20", "GRID_MAX=0.5,1.5")
    .replace("GRID_BIN=300,200", "GRID_BIN=100,300")
    .replace("SIGMA=0.5,0.25", "SIGMA=0.03,0.05")
)

# %%
#
# The i-PI input file defines PLUMED as a forcefield, but uses the ``<bias>``
# tag to specify that it should not be used  as a physical force. The input also
# demonstrates how to retrieve the CVs from PLUMED.
# See `this recipe
# <https://atomistic-cookbook.org/examples/pi-mts-rpc/mts-rpc.html#multiple-time-stepping>`_
# for more details on performing multiple time stepping with i-PI.

ipi_input = pathlib.Path("data/input-meta.xml")
print(ipi_input.read_text())

# %%
#
# We use LAMMPS to compute the LJ potential, and use i-PI in its client-server mode.
ipi_process = subprocess.Popen(["i-pi", "data/input-meta.xml"])
sleep(3)  # wait for i-PI to start
lmp_process = subprocess.Popen(["lmp", "-in", "data/lammps.ipi.in"])

# %%
#
# Wait for the processes to finish

ipi_process.wait()
lmp_process.wait()

# %%
#
# i-PI output trajectory
# ----------------------
#
# The SOAP CVs are much more expensive to compute than the histogram-based CVs,
# so we only run 2000 steps as a demonstration.

ipi_trajectory = ase.io.read("meta-md.pos_0.extxyz", ":")
ipi_properties, _ = ipi.utils.parsing.read_output("meta-md.out")
ipi_cv = ipi.utils.parsing.read_trajectory("meta-md.colvar_0", format="extras")[
    "cv1, cv2"
]

# %%
#
# The structure doesn't move much in this short simulation, but we can still see how
# trajectory moves around in the region surrounding the octahedral structure.

chemiscope.show(
    frames=ipi_trajectory,
    properties={
        "time": {
            "target": "structure",
            "values": ipi_properties["time"],
            "description": "Simulation time",
            "units": "a.u.",
        },
        "bias": {
            "target": "structure",
            "values": ipi_properties["ensemble_bias"],
            "description": "Bias energy",
            "units": "a.u.",
        },
        "soap1": {
            "target": "structure",
            "values": ipi_cv[:, 0],
            "description": "SOAP1 (~Q4)",
        },
        "soap2": {
            "target": "structure",
            "values": ipi_cv[:, 1],
            "description": "SOAP2 (~Q6)",
        },
    },
    settings=chemiscope.quick_settings(
        x="soap1", y="soap2", z="", color="bias", trajectory=True,
        structure_settings={
            "playbackDelay": 50,  # ms between frames
            "unitCell": True,
            "bonds": True,
            "spaceFilling": True,
        },
    ),
)

# %%
