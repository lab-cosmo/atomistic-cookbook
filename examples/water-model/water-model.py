"""
Atomistic Water Model for Molecular Dynamics
============================================

:Authors: Philip Loche `@PicoCentauri <https://github.com/picocentauri>`_,
          Marcel Langer `@sirmarcel <https://github.com/sirmarcel>`_ and
          Michele Ceriotti `@ceriottm <https://github.com/ceriottm>`_

In this example, we demonstrate how to construct a `metatensor atomistic model
<https://docs.metatensor.org/latest/atomistic>`_ for flexible three and four-point water
model, with parameters optimized for use together with quantum-nuclear-effects-aware
path integral simulations (cf. `Habershon et al., JCP (2009)
<http://dx.doi.org/10.1063/1.3167790>`_). The model also demonstrates the use of
``torch-pme``, a Torch library for long-range interactions, and uses the resulting model
to perform demonstrative molecular dynamics simulations.
"""

# sphinx_gallery_thumbnail_number = 3

# %%
import subprocess
from typing import Dict, List, Optional, Tuple

import ase.io

# Simulation and visualization tools
import chemiscope
import matplotlib.pyplot as plt

# Model wrapping and execution tools
import numpy as np
import torch

# Core atomistic libraries
import torchpme
from ase.optimize import LBFGS
from ipi.utils.parsing import read_output, read_trajectory
from ipi.utils.scripting import (
    InteractiveSimulation,
    forcefield_xml,
    motion_nvt_xml,
    simulation_xml,
)
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
)

# Integration with ASE calculator for metatensor atomistic models
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from vesin.torch.metatensor import NeighborList


# %%
#
# The q-TIP4P/F Model
# -------------------
#
# The q-TIP4P/F model uses simple (quasi)-harmonic terms to describe intra-molecular
# flexibility - with the use of a quartic term being a specific feature used to describe
# the covalent bond softening for a H-bonded molecule - a Lennard-Jones term describing
# dispersion and repulsion between O atoms, and an electrostatic potential between
# partial charges on the H atoms and the oxygen electron density. For a four-point
# model, the oxygen charge is slightly displaced from the oxygen's position, improving
# properties like the `dielectric constant <http://dx.doi.org/10.1021/jp410865y>`_. The
# fourth point, referred to as ``M``, is implicitly derived from the other atoms of each
# water molecule.
#
# Intra-molecular interactions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The molecular bond potential is usually defined as a harmonic potential
# of the form
#
# .. math::
#
#   V_\mathrm{bond}(r) = \frac{k_\mathrm{bond}}{2} (r - r_0)^2
#
# Here, :math:`k_\mathrm{bond}` is the force constant and :math:`r_0` is the equilibrium
# distance. Bonded terms like this require defining a *topology*, i.e. a list of
# pairs of atoms that are actually bonded to each other.
#
# q-TIP4P/F doesn't use a harmonic potential but a quartic approximation of a Morse
# potential, that allows describing the anharmonicity of the O-H covalent bond, and how
# the mean distance changes due to zero-point fluctuations and/or hydrogen bonding.
#
# .. math::
#
#   V_\mathrm{bond}(r) = \frac{k_r}{2} [(r-r_0)^2-\alpha_r (r-r_0)^3 +
#       \frac{7}{12}\alpha_r^2(r-r_0)^4]
#
# .. note::
#
#   The harmonic coefficient is related to the coefficients in the original paper
#   by :math:`k_r=2 D_r \alpha_r^2`.


def bond_energy(
    distances: torch.Tensor,
    coefficient_k: torch.Tensor,
    coefficient_alpha: torch.Tensor,
    coefficient_r0: torch.Tensor,
) -> torch.Tensor:
    """Harmonic potential for bond terms."""
    dr = distances - coefficient_r0
    alpha_dr = dr * coefficient_alpha

    return 0.5 * coefficient_k * dr**2 * (1 - alpha_dr + alpha_dr**2 * 7 / 12)


# %%
#
# The parameters reproduce the form of a Morse potential close to the minimum, but
# avoids the dissociation of the bond, due to the truncation to the quartic term. These
# are the parameters used for q-TIP4P/F

OH_kr = 116.09 * 2 * 2.287**2  # kcal/mol/Å**2
OH_r0 = 0.9419  # Å
OH_alpha = 2.287  # 1/Å

bond_distances = np.linspace(0.5, 1.65, 200)
bond_potential = bond_energy(
    distances=bond_distances,
    coefficient_k=OH_kr,
    coefficient_alpha=OH_alpha,
    coefficient_r0=OH_r0,
)

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.set_title("Bond Potential Between Oxygen and Hydrogen")

ax.plot(bond_distances, bond_potential)
ax.axvline(OH_r0, label="equilibrium distance", color="black", linestyle="--")

ax.set_xlabel("Distance / Å ")
ax.set_ylabel("Bond Potential / (kcal/mol)")

ax.legend()
fig.show()

# %%
#
# The harmonic angle potential describe the bending of the HOH angle, and is usually
# modeled as a (curvilinear) harmonic term, defined based on the angle
#
# .. math::
#
#   V_\mathrm{angle}(\theta) = \frac{k_\mathrm{angle}}{2} (\theta - \theta_0)^2
#
# where :math:`k_\mathrm{angle}` is the force constant and :math:`\theta_0` is the
# equilibrium angle between the three atoms.


def bend_energy(
    angles: torch.Tensor, coefficient: torch.Tensor, equilibrium_angle: torch.Tensor
):
    """Harmonic potential for angular terms."""
    return 0.5 * coefficient * (angles - equilibrium_angle) ** 2


# %%
#
#  We use the following parameters:

HOH_angle_coefficient = 87.85  # kcal/mol/rad^2
HOH_equilibrium_angle = 107.4 * torch.pi / 180  # radians

# %%
#
# We can plot the bend energy as a function of the angle that is, unsurprisingly,
# parabolic around the equilibrium angle

angle_distances = np.linspace(100, 115, 200)
angle_potential = bend_energy(
    angles=angle_distances * torch.pi / 180,
    coefficient=HOH_angle_coefficient,
    equilibrium_angle=HOH_equilibrium_angle,
)

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.set_title("Harmonic Angular Potential for a Water Molecule")

ax.plot(angle_distances, angle_potential)
ax.axvline(
    HOH_equilibrium_angle * 180 / torch.pi,
    label="equilibrium angle",
    color="black",
    linestyle=":",
)

ax.set_xlabel("Angle / degrees")
ax.set_ylabel("Harmonic Angular Potential / (kcal/mol)")

ax.legend()
fig.show()

# %%
# Lennard-Jones Potential
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The Lennard-Jones (LJ) potential describes the interaction between a pair of neutral
# atoms or molecules, balancing dispersion forces at longer ranges and repulsive forces
# at shorter ranges. The LJ potential is defined as:
#
# .. math::
#
#  V_\mathrm{LJ}(r) = 4 \epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left(
#  \frac{\sigma}{r} \right)^6 \right]
#
# where :math:`\epsilon` is the depth of the potential well and :math:`\sigma` the
# distance at which the potential is zero. For water there is usually only an
# oxygen-oxygen Lennard-Jones potential.
#
# We implement the Lennard-Jones potential as a function that takes distances, along
# with the parameters ``sigma``, ``epsilon``, and ``cutoff`` that indicates the distance
# at which the interaction is assumed to be zero. To ensure that there is no
# discontinuity an offset is included to shift the curve so it is zero at the cutoff
# distance.


def lennard_jones_pair(
    distances: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: torch.Tensor,
):
    """Shifted Lennard-Jones potential for pair terms."""
    c6 = (sigma / distances) ** 6
    c12 = c6**2
    lj = 4 * epsilon * (c12 - c6)

    sigma_cutoff_6 = (sigma / cutoff) ** 6
    offset = 4 * epsilon * sigma_cutoff_6 * (sigma_cutoff_6 - 1)

    return lj - offset


# %%
#
# We plot this potential to visualize its behavior, using q-TIP4P/F parameters. To
# highlight the offset, we use a cutoff of 5 Å instead of the usual 7 Å.

O_sigma = 3.1589  # Å
O_epsilon = 0.1852  # kcal/mol
cutoff = 5.0  # Å <- cut short, to highlight offset

lj_distances = np.linspace(3, cutoff, 200)

lj_potential = lennard_jones_pair(
    distances=lj_distances, sigma=O_sigma, epsilon=O_epsilon, cutoff=cutoff
)

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
ax.set_title("Lennard-Jones Potential Between Two Oxygen Atoms")
ax.axhline(0, color="black", linestyle="--")
ax.axhline(-O_epsilon, color="red", linestyle=":", label="Oxygen ε")
ax.axvline(O_sigma, color="black", linestyle=":", label="Oxygen σ")
ax.plot(lj_distances, lj_potential)
ax.set_xlabel("Distance / Å")
ax.set_ylabel("Lennard-Jones Potential / (kcal/mol)")
ax.legend()
fig.show()


# %%
#
# Due to our reduced cutoff the minimum of the blue line does not touch the red
# horizontal line.
#
# Electrostatic Potential
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The long-ranged nature of electrostatic interactions makes computing them
# in simulations non-trivial. For periodic systems the Coulomb energy is given by:
#
# .. math::
#
#  V_\mathrm{Coulomb} = \frac{1}{2} \sum_{i,j} \sideset{}{'}\sum_{\boldsymbol n \in
#   \mathcal{Z}} \frac{1}{4\pi\epsilon_0}
#   \frac{q_i q_j}{\left|\boldsymbol r_{ij} + \boldsymbol{n L}\right|}
#
# The sum over :math:`\boldsymbol n` takes into account the periodic images of the
# charges and the prime indicates that in the case :math:`i=j` the term :math:`n=0` must
# be omitted. Further :math:`\boldsymbol r_{ij} = \boldsymbol r_i - \boldsymbol r_j` and
# :math:`\boldsymbol L` is the length of the (cubic)simulation box.
#
# Since this sum is conditionally convergent it isn't computable using a direct sum.
# Instead the Ewald summation, published in 1921, remains a foundational method that
# effectively defines how to compute the energy and forces of such systems. To further
# speed the methods, mesh based algorithm suing fast Fourier transformation have been
# developed, such as the Particle-Particle Particle-Mesh (P3M) algorithm. For further
# details we refer to a paper by `Deserno and Holm
# <https://aip.scitation.org/doi/10.1063/1.477414>`_.
#
# We use a *Torch* implementation of the P3M method within the ``torch-pme`` package.
# The core class we use is the :class:`torchpme.P3MCalculator` class - you can read more
# and see specific examples in the `torchpme documentation
# <https://lab-cosmo.github.io/torch-pme>`_ As a demonstration we use two
# charges in a cubic cell, computing the electrostatic energy as a function of distance

O_charge = -0.84
coulomb_distances = torch.linspace(0.5, 9.0, 50)
cell = torch.eye(3) * 10.0


# %%
#
# We also use the parameter-tuning functionality of ``torchpme``, to provide efficient
# evaluation at the desired level of accuracy. This is achieved calling
# :func:`torchpme.tuning.tune_p3m` on a template of the target structure

charges = torch.tensor([O_charge, -O_charge]).unsqueeze(-1)
positions_coul = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
neighbor_indices = torch.tensor([[0, 1]])
neighbor_distances = torch.tensor([4.0])

p3m_smearing, p3m_parameters, _ = torchpme.tuning.tune_p3m(
    charges,
    cell,
    positions_coul,
    cutoff,
    neighbor_indices,
    neighbor_distances,
    accuracy=1e-4,
)
p3m_prefactor = torchpme.prefactors.kcalmol_A

# %%
#
# The hydrogen charge is derived from the oxygen charge as :math:`q_H = -q_O/2`. The
# ``smearing`` and ``mesh_spacing`` parameters are the central parameters for P3M and
# are crucial to ensure the correct energy calculation. We now compute the electrostatic
# energy between two point charges using the P3M algorithm.

p3m_calculator = torchpme.P3MCalculator(
    potential=torchpme.CoulombPotential(p3m_smearing),
    **p3m_parameters,
    prefactor=p3m_prefactor,
)


# %%
#
# For the inference, we need a neighbor list and distances which we compute
# "manually". Typically, the neighbors are provided by the simulation engine.

neighbor_indices = torch.tensor([[0, 1]])

potential = torch.zeros_like(coulomb_distances)

for i_dist, dist in enumerate(coulomb_distances):
    positions_coul = torch.tensor([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]])
    charges = torch.tensor([O_charge, -O_charge]).unsqueeze(-1)

    neighbor_distances = torch.tensor([dist])

    potential[i_dist] = p3m_calculator.forward(
        positions=positions_coul,
        cell=cell,
        charges=charges,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )[0]

# %%
#
# We plot the electrostatic potential between two point charges.

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

ax.set_title("Electrostatic Potential Between Two Point Charges")
ax.plot(coulomb_distances, potential)

ax.set_xlabel("Distance / Å")
ax.set_ylabel("Electrostatic Potential / (kcal/mol)")

fig.show()

# %%
#
# The potential shape may appear unusual due to computations within a periodic box. For
# small distances, the potential behaves like :math:`1/r`, but it increases again as
# charges approach across periodic boundaries.
#
# .. note::
#
#   In most water models, Coulomb interactions within each molecule are excluded, as
#   intramolecular energies are already parametrized by the bond and angle interactions.
#   Therefore, in our model defined below we first compute the electrostatic energy of
#   all atoms and then subtract interactions between bonded atoms.
#
# Implementation as a TIP4P/f ``torch`` module
# ----------------------------------------------
#
# In order to implement a Q-TIP4P/f potential in practice, we first build a class that
# follows the interface of a *metatomic* model. This requires defining the atomic
# structure in terms of a :class:`metatensor.torch.atomistic.System` object - a simple
# container for positions, cell, and atomic types, that can also be enriched with one or
# more :class:`metatensor.torch.atomistic.NeighborList` objects holding neighbor
# distance information. This is usually provided by the code used to perform a
# simulation, but can be also computed explicitly using ``ase`` or `vesin
# <https://luthaf.fr/vesin/latest/index.html>`_, as we do here.
#
# For running our simulation we use a small waterbox containing 32 water molecules.

atoms = ase.io.read("data/water_32.xyz")

chemiscope.show(
    [atoms],
    mode="structure",
    settings=chemiscope.quick_settings(structure_settings={"unitCell": True}),
)

# %%
#
# We transform the ase Atoms object into a metatensor atomistic system and define the
# options for the neighbor list.

system = System(
    types=torch.from_numpy(atoms.get_atomic_numbers()),
    positions=torch.from_numpy(atoms.positions),
    cell=torch.from_numpy(atoms.cell.array),
    pbc=torch.from_numpy(atoms.pbc),
)

nlo = NeighborListOptions(cutoff=7.0, full_list=False, strict=False)
calculator = NeighborList(nlo, length_unit="Angstrom")
neighbors = calculator.compute(system)
system.add_neighbor_list(nlo, neighbors)


# %%
#
# Neighbor lists are stored within ``metatensor`` as :class:`metatensor.TensorBlock`
# objects, if you're curious

neighbors

# %%
# Helper functions for molecular geometry
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In order to compute the different terms in the Q-TIP4P/f potential, we need to extract
# some information on the geometry of the water molecule. To keep the model class clean,
# we define a helper functions that do two things.
#
# First, it computes O-H covalent bond lengths and angle. We use heuristics to identify
# the covalent bonds as the shortest O-H distances in a simulation. This is necessary
# when using the model with an external code, that might re-order atoms internally (as
# is e.g. the case for LAMMPS). The heuristics here may fail in case molecules get too
# close together, or at very high temperature.
#
# Second, it computes the position of the virtual "M sites", the position of the O
# charge in a TIP4P-like model. We also need distances to compute range-separated
# electrostatics, which we obtain manipulating the neighbor list that is pre-attached to
# the system. Thanks to the fact we rely on ``torch`` autodifferentiation mechanism, the
# forces acting on the virtual sites will be automatically split between O and H atoms,
# in a way that is consistent with the definition.
#
# Forces acting on the M sites will be automatically split between O and H atoms, in a
# way that is consistent with the definition.


def get_molecular_geometry(
    system: System,
    nlo: NeighborListOptions,
    m_gamma: torch.Tensor,
    m_charge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, System, torch.Tensor, torch.Tensor]:
    """Compute bond distances, angles and charge site positions for each molecules."""
    neighbors = system.get_neighbor_list(nlo)
    species = system.types

    # get neighbor idx and vectors as torch tensors
    neigh_ij = neighbors.samples.view(["first_atom", "second_atom"]).values
    neigh_dij = neighbors.values.squeeze()

    # get all OH distances and their sorting order
    oh_mask = species[neigh_ij[:, 0]] != species[neigh_ij[:, 1]]
    oh_ij = neigh_ij[oh_mask]
    oh_dij = neigh_dij[oh_mask]
    oh_dist = torch.linalg.vector_norm(oh_dij, dim=1).squeeze()
    oh_sort = torch.argsort(oh_dist)

    # gets the oxygen indices in the bonds, sorted by distance
    oh_oidx = torch.where(species[oh_ij[:, 0]] == 8, oh_ij[:, 0], oh_ij[:, 1])[oh_sort]
    # makes sure we always consider bonds in the O->H direction
    oh_dij = oh_dij * torch.where(species[oh_ij[:, 0]] == 8, 1.0, -1.0).reshape(-1, 1)

    # we assume that the first n_oxygen*2 bonds cover all water molecules.
    # if not we throw an error
    o_idx = torch.nonzero(species == 8).squeeze()
    n_oh = len(o_idx) * 2
    oh_oidx_sort = torch.argsort(oh_oidx[:n_oh])

    oh_dij_oidx = oh_oidx[oh_oidx_sort]  # indices of the O atoms for each dOH
    # if each O index appears twice, this should be a vector of zeros
    twoo_chk = oh_dij_oidx[::2] - oh_dij_oidx[1::2]
    if (twoo_chk != 0).any():
        raise RuntimeError("Cannot assign OH bonds to water molecules univocally.")

    # finally, we compute O->H1 and O->H2 for each water molecule
    oh_dij_sort = oh_dij[oh_sort[:n_oh]][oh_oidx_sort]
    doh_1 = oh_dij_sort[::2]
    doh_2 = oh_dij_sort[1::2]

    oh_dist = torch.concatenate(
        [
            torch.linalg.vector_norm(doh_1, dim=1),
            torch.linalg.vector_norm(doh_2, dim=1),
        ]
    )

    if oh_dist.max() > 2.0:
        raise ValueError(
            "Unphysical O-H bond length. Check that the molecules are entire, and "
            "atoms are listed in the expected OHH order."
        )

    hoh_angles = torch.arccos(
        torch.sum(doh_1 * doh_2, dim=1)
        / (oh_dist[: len(doh_1)] * oh_dist[len(doh_2) :])
    )

    # now we use this information to build the M sites
    # we first put the O->H vectors in the same order as the
    # positions. This allows us to manipulate all atoms at once later
    oh1_vecs = torch.zeros_like(system.positions)
    oh1_vecs[oh_dij_oidx[::2]] = doh_1
    oh2_vecs = torch.zeros_like(system.positions)
    oh2_vecs[oh_dij_oidx[1::2]] = doh_2

    # we compute the vectors O->M displacing the O to the M sites
    om_displacements = (1 - m_gamma) * 0.5 * (oh1_vecs + oh2_vecs)

    # creates a new System with the m-sites
    m_system = System(
        types=system.types,
        positions=system.positions + om_displacements,
        cell=system.cell,
        pbc=system.pbc,
    )

    # adjust neighbor lists to point at the m sites rather than O atoms. this assumes
    # this won't have atoms cross the cutoff, which is of course only approximately
    # true, so one should use a slighlty larger-than-usual cutoff nb - this is reshaped
    # to match the values in a NL tensorblock
    nl = system.get_neighbor_list(nlo)
    i_idx = nl.samples.view(["first_atom"]).values.flatten()
    j_idx = nl.samples.view(["second_atom"]).values.flatten()
    m_nl = TensorBlock(
        nl.values
        + (om_displacements[j_idx] - om_displacements[i_idx]).reshape(-1, 3, 1),
        nl.samples,
        nl.components,
        nl.properties,
    )
    m_system.add_neighbor_list(nlo, m_nl)

    # set charges of all atoms
    charges = (
        torch.where(species == 8, -m_charge, 0.5 * m_charge)
        .reshape(-1, 1)
        .to(dtype=system.positions.dtype, device=system.positions.device)
    )

    # Create metadata for the charges TensorBlock
    samples = Labels(
        "atom", torch.arange(len(system), device=charges.device).reshape(-1, 1)
    )
    properties = Labels(
        "charge", torch.zeros(1, 1, device=charges.device, dtype=torch.int32)
    )
    data = TensorBlock(
        values=charges, samples=samples, components=[], properties=properties
    )

    tensor = TensorMap(
        keys=Labels("_", torch.zeros(1, 1, device=charges.device, dtype=torch.int32)),
        blocks=[data],
    )

    m_system.add_data(name="charges", tensor=tensor)

    # intra-molecular distances with M sites (for self-energy removal)
    hh_dist = torch.sqrt(((doh_1 - doh_2) ** 2).sum(dim=1))
    dmh_1 = doh_1 - om_displacements[oh_dij_oidx[::2]]
    dmh_2 = doh_2 - om_displacements[oh_dij_oidx[1::2]]
    mh_dist = torch.concatenate(
        [
            torch.linalg.vector_norm(dmh_1, dim=1),
            torch.linalg.vector_norm(dmh_2, dim=1),
        ]
    )

    return oh_dist, hoh_angles, m_system, mh_dist, hh_dist


# %%
# Defining the model
# ^^^^^^^^^^^^^^^^^^
#
# Armed with these functions, the definitions of bonded and LJ potentials, and the
# :class:`torchpme.P3MCalculator` class, we can define with relatively small effort the
# actual model. We do not hard-code the specific parameters of the Q-TIP4P/f potential
# (so that in principle this class can be used for classical TIP4P, and - by setting
# ``m_gamma`` to one - a three-point water model).
#
# A few points worth noting: (1) As discussed above we define a bare Coulomb potential
# (that pretty much computes :math:`1/r`) which we need to subtract the molecular "self
# electrostatic interaction"; (2) units are expected to be Å for distances, kcal/mol for
# energies, and radians for angles; (3) model parameters are registered as ``buffers``;
# (4) P3M parameters can also be given, in the format returned by
# :func:`torchpme.tuning.tune_p3m`.
#
# The ``forward`` call is a relatively straightforward combination of all the helper
# functions defined further up in this file, and should be relatively easy to follow.


class WaterModel(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        lj_sigma: float,
        lj_epsilon: float,
        m_gamma: float,
        m_charge: float,
        oh_bond_eq: float,
        oh_bond_k: float,
        oh_bond_alpha: float,
        hoh_angle_eq: float,
        hoh_angle_k: float,
        p3m_options: Optional[dict] = None,
    ):
        super().__init__()

        if p3m_options is None:
            # should give a ~1e-4 relative accuracy on forces
            p3m_options = (1.4, {"interpolation_nodes": 5, "mesh_spacing": 1.33}, 0)

        p3m_smearing, p3m_parameters, _ = p3m_options

        self.p3m_calculator = torchpme.metatensor.P3MCalculator(
            potential=torchpme.CoulombPotential(p3m_smearing),
            **p3m_parameters,
            prefactor=torchpme.prefactors.kcalmol_A,  # consistent units
        )

        self.coulomb = torchpme.CoulombPotential()

        # We use a half neighborlist and allow to have pairs farther than cutoff
        # (`strict=False`) since this is not problematic for PME and may speed up the
        # computation of the neigbors.
        self.nlo = NeighborListOptions(cutoff=cutoff, full_list=False, strict=False)

        self.register_buffer("cutoff", torch.tensor(cutoff))
        self.register_buffer("lj_sigma", torch.tensor(lj_sigma))
        self.register_buffer("lj_epsilon", torch.tensor(lj_epsilon))
        self.register_buffer("m_gamma", torch.tensor(m_gamma))
        self.register_buffer("m_charge", torch.tensor(m_charge))
        self.register_buffer("oh_bond_eq", torch.tensor(oh_bond_eq))
        self.register_buffer("oh_bond_k", torch.tensor(oh_bond_k))
        self.register_buffer("oh_bond_alpha", torch.tensor(oh_bond_alpha))
        self.register_buffer("hoh_angle_eq", torch.tensor(hoh_angle_eq))
        self.register_buffer("hoh_angle_k", torch.tensor(hoh_angle_k))

    def requested_neighbor_lists(self):
        """Returns the list of neighbor list options that are needed."""
        return [self.nlo]

    def _setup_systems(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels] = None,
    ) -> tuple[System, TensorBlock]:
        if len(systems) > 1:
            raise ValueError(f"only one system supported, got {len(systems)}")

        system_i = 0
        system = systems[system_i]

        # select only real atoms and discard ghosts
        # (this is to work in codes like LAMMPS that provide a neighbor
        # list that also includes "domain decomposition" neigbhbors)
        if selected_atoms is not None:
            current_system_mask = selected_atoms.column("system") == system_i
            current_atoms = selected_atoms.column("atom")
            current_atoms = current_atoms[current_system_mask].to(torch.long)

            types = system.types[current_atoms]
            positions = system.positions[current_atoms]
            system_clean = System(types, positions, system.cell, system.pbc)
            for nlo in system.known_neighbor_lists():
                system_clean.add_neighbor_list(nlo, system.get_neighbor_list(nlo))
        else:
            system_clean = system
        return system_clean, system.get_neighbor_list(self.nlo)

    def forward(
        self,
        systems: List[System],  # noqa
        outputs: Dict[str, ModelOutput],  # noqa
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:  # noqa

        if list(outputs.keys()) != ["energy"]:
            raise ValueError(
                f"`outputs` keys ({', '.join(outputs.keys())}) contain unsupported "
                "keys. Only 'energy' is supported."
            )

        if outputs["energy"].per_atom:
            raise NotImplementedError("per-atom energies are not supported.")

        system, neighbors = self._setup_systems(systems, selected_atoms)

        # compute non-bonded LJ energy
        neighbor_indices = neighbors.samples.view(["first_atom", "second_atom"]).values
        species = system.types
        oo_mask = (species[neighbor_indices[:, 0]] == 8) & (
            species[neighbor_indices[:, 1]] == 8
        )
        oo_distances = torch.linalg.vector_norm(neighbors.values[oo_mask], dim=1)
        energy_lj = lennard_jones_pair(
            oo_distances, self.lj_sigma, self.lj_epsilon, self.cutoff
        ).sum()

        d_oh, a_hoh, m_system, mh_dist, hh_dist = get_molecular_geometry(
            system, self.nlo, self.m_gamma, self.m_charge
        )

        # intra-molecular energetics
        e_bond = bond_energy(
            d_oh, self.oh_bond_k, self.oh_bond_alpha, self.oh_bond_eq
        ).sum()
        e_bend = bend_energy(a_hoh, self.hoh_angle_k, self.hoh_angle_eq).sum()

        # now this is the long-range part - computed over the M-site system
        # m_system, mh_dist, hh_dist = get_msites(system, self.m_gamma, self.m_charge)
        m_neighbors = m_system.get_neighbor_list(self.nlo)

        potentials = self.p3m_calculator(m_system, m_neighbors).block(0).values
        charges = m_system.get_data("charges").block().values
        energy_coulomb = (potentials * charges).sum()

        # this is the intra-molecular Coulomb interactions, that must be removed
        # to avoid double-counting
        energy_self = (
            (
                self.coulomb.from_dist(mh_dist).sum() * (-self.m_charge)
                + self.coulomb.from_dist(hh_dist).sum() * 0.5 * self.m_charge
            )
            * 0.5
            * self.m_charge
            * torchpme.prefactors.kcalmol_A
        )

        # combines all energy terms
        energy_tot = e_bond + e_bend + energy_lj + energy_coulomb - energy_self

        # Rename property label to follow metatensor's convention for an atomistic model
        samples = Labels(
            ["system"],
            torch.arange(len(systems), device=energy_tot.device).reshape(-1, 1),
        )
        properties = Labels(["energy"], torch.tensor([[0]], device=energy_tot.device))

        block = TensorBlock(
            values=torch.sum(energy_tot).reshape(-1, 1),
            samples=samples,
            components=torch.jit.annotate(List[Labels], []),
            properties=properties,
        )
        return {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=energy_tot.device)), [block]
            ),
        }


# %%
#
# All this class does is take a ``System`` and return its energy (as a
# :class:`metatensor.TensorMap``).

qtip4pf_parameters = dict(
    cutoff=7.0,
    lj_sigma=3.1589,
    lj_epsilon=0.1852,
    m_gamma=0.73612,
    m_charge=1.1128,
    oh_bond_eq=0.9419,
    oh_bond_k=2 * 116.09 * 2.287**2,
    oh_bond_alpha=2.287,
    hoh_angle_eq=107.4 * torch.pi / 180,
    hoh_angle_k=87.85,
)
qtip4pf_model = WaterModel(
    **qtip4pf_parameters
    #   uncomment to override default options
    #    p3m_options = (1.4, {"interpolation_nodes": 5, "mesh_spacing": 1.33}, 0)
)

# %%
#
# We re-initilize the ``system`` to ask for gradients

system = System(
    types=torch.from_numpy(atoms.get_atomic_numbers()),
    positions=torch.from_numpy(atoms.positions).requires_grad_(),
    cell=torch.from_numpy(atoms.cell.array).requires_grad_(),
    pbc=torch.from_numpy(atoms.pbc),
)
system.add_neighbor_list(nlo, calculator.compute(system))

energy_unit = "kcal/mol"
length_unit = "angstrom"

outputs = {"energy": ModelOutput(quantity="energy", unit=energy_unit, per_atom=False)}

nrg = qtip4pf_model.forward([system], outputs)
nrg["energy"].block(0).values.backward()

print(
    f"""
Energy is {nrg["energy"].block(0).values[0].item()} kcal/mol

The forces on the first molecule (in kcal/mol/Å) are
{system.positions.grad[:3]}

The stress is
{system.cell.grad}
"""
)

# %%
#
# Build and save a ``MetatensorAtomisticModel``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This model can be wrapped into a
# :class:`metatensor.torch.atomistic.MetatensorAtomisticModel` class, that provides
# useful helpers to specify the capabilities of the model, and to save it as a
# ``torchscript`` module.
#
# Model options include a definition of its units, and a description of the quantities
# it can compute.
#
# .. note::
#
#   We neeed to specify that the model has infinite interaction range because of the
#   presence of a long-range term that means one cannot assume that forces decay to zero
#   beyond the cutoff.

options = ModelEvaluationOptions(
    length_unit=length_unit, outputs=outputs, selected_atoms=None
)

model_capabilities = ModelCapabilities(
    outputs=outputs,
    atomic_types=[1, 8],
    interaction_range=torch.inf,
    length_unit=length_unit,
    supported_devices=["cuda", "cpu"],
    dtype="float32",
)

atomistic_model = MetatensorAtomisticModel(
    qtip4pf_model.eval(), ModelMetadata(), model_capabilities
)

atomistic_model.save("qtip4pf-mta.pt")


# %%
#
# Other water models
# ^^^^^^^^^^^^^^^^^^
#
# The `WaterModel` class is flexible enough that one can also implement (and export)
# other 4-point models, or even 3-point models if one sets the `m_gamma` parameter to
# one. For instance, we can implement the (classical) SPC/Fw model (`Wu et al., JCP
# (2006) <http://doi.org/10.1063/1.2136877>`_)

spcf_parameters = dict(
    cutoff=7.0,
    lj_sigma=3.16549,
    lj_epsilon=0.155425,
    m_gamma=1.0,
    m_charge=0.82,
    oh_bond_eq=1.012,
    oh_bond_k=1059.162,
    oh_bond_alpha=0.0,
    hoh_angle_eq=113.24 * torch.pi / 180,
    hoh_angle_k=75.90,
)
spcf_model = WaterModel(**spcf_parameters)

atomistic_model = MetatensorAtomisticModel(
    spcf_model.eval(), ModelMetadata(), model_capabilities
)

atomistic_model.save("spcfw-mta.pt")

# %%
#
# Using the Q-TIP4P/f model
# -------------------------
#
# The ``torchscript`` model can be reused with any simulation software compatible with
# the ``metatomic`` API. Here we give a couple of examples, designed to demonstrate the
# usage more than to provide realistic use cases.
#
# Geometry optimization with ``ase``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We begin with an example based on an ``ase``-compatible calculator. To this end,
# ``metatensor`` provides a compatible
# :class:`metatensor.torch.atomistic.MetatensorCalculator` wrapper to a model. Note how
# the metadata associated with the model are used to convert energy into the units
# expected by ``ase`` (eV and Å).

atomistic_model = load_atomistic_model("qtip4pf-mta.pt")
mta_calculator = MetatensorCalculator(atomistic_model)

atoms.calc = mta_calculator
nrg = atoms.get_potential_energy()

print(
    f"""
Energy is {nrg} eV, corresponding to {nrg*23.060548} kcal/mol
"""
)


# %%
#
# We then use one of the built-in ``ase`` functions to run the structural relaxation.
# The relaxation is split into short segments just to be able to visualize the
# trajectory.
#
# ``fmax`` is the threshold on the maximum force component and the optimization will
# stop when the threshold is reached

opt_trj = []
opt_nrg = []

for _ in range(10):
    opt_trj.append(atoms.copy())
    opt_nrg.append(atoms.get_potential_energy())
    opt = LBFGS(atoms, restart="lbfgs_restart.json")
    opt.run(fmax=0.001, steps=5)

opt.run(fmax=0.001, steps=10)
nrg_final = atoms.get_potential_energy()


# %%
#
# Use `chemiscope <http://chemiscope.org>`_ to visualize the geometry optimization
# together with the convergence of the energy to a local minimum.

chemiscope.show(
    frames=opt_trj,
    properties={
        "step": 1 + np.arange(0, len(opt_trj)),
        "energy": opt_nrg - nrg_final,
    },
    mode="default",
    settings=chemiscope.quick_settings(
        map_settings={
            "x": {"property": "step", "scale": "log"},
            "y": {"property": "energy", "scale": "log"},
        },
        structure_settings={
            "unitCell": True,
        },
        trajectory=True,
    ),
)


# %%
# Path integral molecular dynamics with ``i-PI``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use `i-PI <http://ipi-code.org>`_ to perform path integral molecular-dynamics
# simulations of bulk water to include nuclear quntum effects. See `this recipe
# <https://atomistic-cookbook.org/examples/thermostats/thermostats.html>`_ for an
# introduction to constant-temperatur MD using i-PI.
#
# First, the ``XML`` input of the i-PI simulation is created using a few utility
# functions. This input could also be written to file and used with the command-line
# version of i-PI. We use the structure twice to generate a path integral with two
# replicas: this is far from converged, see also `this recipe
# <https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html>`_ if you
# have never run path integral simulations before.


data = ase.io.read("data/water_32.xyz")
input_xml = simulation_xml(
    structures=[data, data],
    forcefield=forcefield_xml(
        name="qtip4pf",
        mode="direct",
        pes="metatensor",
        parameters={"model": "qtip4pf-mta.pt", "template": "data/water_32.xyz"},
    ),
    motion=motion_nvt_xml(timestep=0.5 * ase.units.fs),
    temperature=300,
    prefix="qtip4pf-md",
)

print(input_xml)

# %%
#
# Then, we create an ``InteractiveSimulation`` object and run a short simulation (purely
# for demonstrative purposes)

sim = InteractiveSimulation(input_xml)
sim.run(400)

# %%
#
# The simulation generates output files that can be parsed and visualized from Python

data, info = read_output("qtip4pf-md.out")
trj = read_trajectory("qtip4pf-md.pos_0.xyz")

# %%

fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)

ax.plot(data["time"], data["potential"], "b-", label="potential")
ax.plot(data["time"], data["conserved"] - 4, "k-", label="conserved")
ax.set_xlabel("t / ps")
ax.set_ylabel("energy / ev")
ax.legend()

# %%

chemiscope.show(
    frames=trj,
    properties={
        "time": data["time"][::10],
        "potential": data["potential"][::10],
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
# Molecular dynamics with ``LAMMPS``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``metatomic`` model can also be run with `LAMMPS <https://lammps.org>`_
# and used to perform all kinds of atomistic simulations with it.
# This only requires defining a ``pair_metatensor`` potential, and specifying
# the mapping between LAMMPS atom types and those used in the model.
#
# Note also that the ``metatomic`` interface takes care of converting the
# model units to those used in the LAMMPS file, so it is possible to use
# energies in eV even if the model outputs kcal/mol.

with open("data/spcfw.in", "r") as file:
    lines = file.readlines()

for line in lines[:7] + lines[16:]:
    print(line, end="")


# %%
#
# This specific example runs a short MD trajectory, using a Langevin thermostat. Given
# that this is a classical MD trajectory, we use the SPC/Fw model that is fitted to
# reproduce (some) water properties even with a classical description of the nuclei.
#
# We save to geometry to a LAMMPS data file and run the simulation

ase.io.write("water_32.data", atoms, format="lammps-data", masses=True)

subprocess.check_call(["lmp_serial", "-in", "data/spcfw.in"])
