"""
Atomistic Water Model for Molecular Dynamics
============================================

:Authors: Philip Loche `@PicoCentauri <https://github.com/picocentauri>`_
         Marcel Langer `@sirmarcel <https://github.com/sirmarcel>`_

In this example, we demonstrate how to construct a `metatensor atomistic model
<https://docs.metatensor.org/latest/atomistic>`_ for q-TIP4P/F, a flexible four-point water
model, with parameters optimized for use together with quantum-nuclear-effects-aware
path integral simulations (cf. `Habershon et al., JCP (2009)
<http://dx.doi.org/10.1063/1.3167790>`_)
The model also demonstrates the use of ``torch-pme``, a Torch library for
long-range interactions, and uses the resulting model to perform demonstrative
molecular dynamics simulations.
"""

# %%

# sphinx_gallery_thumbnail_number = 5
from typing import Dict, List, Optional

import importlib.machinery
import importlib.util

# Simulation and visualization tools
import ase.md
import ase.md.velocitydistribution
import ase.visualize.plot
import chemiscope
import matplotlib.pyplot as plt

# Model wrapping and execution tools
import numpy as np
import torch

# Core libraries
import torchpme
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
)

# Integration with ASE calculator for metatensor atomistic models
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator


get_ipython().run_line_magic('matplotlib', 'inline')

# %%
#
# The q-TIP4P/F Model
# -------------------
#
# As described above, this example implements a flexible four-point water
# models. This model uses simple (quasi)-harmonic terms to describe intra-molecular
# flexibility - with the use of a quartic term being a specific feature used
# to describe the covalent bond softening for a H-bonded molecule - a Lennard-Jones
# term describing dispersion and repulsion between O atoms, and an electrostatic
# potential between partial charges on the H atoms and the oxygen electron density.
# For a four-point model, the oxygen charge is slightly displaced from the
# oxygen's position, improving properties like the `dielectric constant
# <http://dx.doi.org/10.1021/jp410865y>`_. The fourth point, referred to as ``M``, is
# implicitly derived from the other atoms of each water molecule.
#
#

# %%
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
# distance. Bonded terms like this require defining a `topology`, i.e. a list of
# pairs of atoms that are actually bonded to each other.
#
# q-TIP4P/F uses a quartic approximation of a Morse potential, that allows
# describing the anharmonicity of the O-H covalent bond, and how the mean distance
# changes due to zero-point fluctuations and/or hydrogen bonding.
#
# .. math::
#
#   V_\mathrm{bond}(r) = D_r[\alpha_r^2(r-r_0)^2-\alpha_r^3(r-r_0)^3+\frac{7}{12}\alpha_r^4(r-r_0)^4]


def bond_energy(
    distances: torch.Tensor,
    coefficient_d: torch.Tensor,
    coefficient_alpha: torch.Tensor,
    coefficient_r0: torch.Tensor,
):
    """Harmonic potential for bond terms."""
    alpha_dr = (distances - coefficient_r0) * coefficient_alpha

    return coefficient_d * alpha_dr**2 * (1 - alpha_dr + alpha_dr**2 * 7.0 / 12.0)


# %%
#
# The parameters reproduce the form of a Morse potential close to the minimum, but avoids the
# dissociation of the bond, due to the truncation to the quartic term
# These are the parameters used for q-TIP4P/F

OH_Dr = 116.09  # kcal/mol
OH_r0 = 0.9419  # Å
OH_alpha = 2.287  # 1/Å

bond_distances = np.linspace(0.5, 1.5, 200)
bond_potential = bond_energy(
    distances=bond_distances,
    coefficient_d=OH_Dr,
    coefficient_alpha=OH_alpha,
    coefficient_r0=OH_r0,
)

plt.title("Bond Potential Between Oxygen and Hydrogen")
plt.plot(bond_distances, bond_potential)
plt.xlabel("Distance [Å]")
plt.ylabel("Bond Potential [kcal/mol]")

plt.show()

# %%
# The harmonic angle potential describe the bending of the HOH angle, and is usually
# modeled as a (curvilinear) harmonic term, defined based on the angle
#
# .. math::
#
#   V_\mathrm{angle}(\theta) = \frac{k_\mathrm{angle}}{2} (\theta - \theta_0)^2
#
# where :math:`k_\mathrm{angle}` is the force constant and :math:`\theta_0` is the
# equilibrium angle between the three atoms. We use the following parameters:


def bend_energy(
    angles: torch.Tensor, coefficient: torch.Tensor, equilibrium_angle: torch.Tensor
):
    """Harmonic potential for angular terms."""
    return 0.5 * coefficient * (angles - equilibrium_angle) ** 2


# %%
#
# We can plot the bend energy as a function of the angle that is,
# unsurprisingly, parabolic around the equilibrium angle

HOH_angle_coefficient = 87.85  # kcal/mol/rad^2
HOH_equilibrium_angle = 107.4 * np.pi / 180  # radians

angle_distances = np.linspace(100, 115, 200)
angle_potential = bend_energy(
    angles=angle_distances * np.pi / 180,
    coefficient=HOH_angle_coefficient,
    equilibrium_angle=HOH_equilibrium_angle,
)

plt.title("Harmonic Angle Potential in Water Molecule")
plt.plot(angle_distances, angle_potential)
plt.xlabel("Angle [degrees]")
plt.ylabel("Harmonic Angle Potential [kcal/mol]")
plt.show()

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
# where :math:`\epsilon` is the depth of the potential well and :math:`\sigma` is the
# distance at which the potential is zero. For water there is only an oxygen-oxygen
# Lennard-Jones potential. In a typical 4-points water model are no LJ interactions
# between hydrogen atoms and between hydrogen and oxygen atoms.


# %%
#
# We implement the Lennard-Jones potential as a function that takes distances, along
# with the parameters ``sigma``, ``epsilon``, and ``cutoff`` that indicates the distance
# at which the interaction is assumed to be zero. To ensure that there is no discontinuity
# an offset is included to shift the curve so it is zero at the cutoff distance.


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
# We plot this potential to visualize its behavior, using
# q-TIP4P/F parameters

O_sigma = 3.1589  # Å
O_epsilon = 0.1852  # kcal/mol
cutoff = 5.0  #  Å <- cut short, to highlight offset

lj_distances = np.linspace(3, cutoff, 200)

lj_potential = lennard_jones_pair(
    distances=lj_distances, sigma=O_sigma, epsilon=O_epsilon, cutoff=cutoff
)


plt.title("Lennard-Jones Potential Between Two Oxygen Atoms")
plt.axhline(0, color="black", linestyle="--")
plt.axhline(-O_epsilon, color="red", linestyle=":", label="Oxygen ε")
plt.axvline(O_sigma, color="black", linestyle=":", label="Oxygen σ")
plt.plot(lj_distances, lj_potential)
plt.xlabel("Distance [Å]")
plt.ylabel("Lennard-Jones Potential [kcal/mol]")
plt.legend()

plt.show()


# %%
#
# Electrostatic Potential
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Since electrostatic interactions are long-ranged it is not fully trivial to compute
# these in simulations. For periodic systems the Coulomb energy is given by:
#
# .. math::
#
#  V_\mathrm{Coulomb} = \frac{1}{2} \sum_{i,j} \sideset{}{'}\sum_{\boldsymbol n \in
#   \mathcal{Z}} \frac{q_i q_j}{\left|\boldsymbol r_{ij} + \boldsymbol{n L}\right|}
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

# %%
# We use a `Torch` implementation of the P3M method within the ``torch-pme`` package.
# The core class we use is the :class:`torchpme.P3MCalculator` class - you can read more
# and see specific examples in the `torchpme documentation
# <https://lab-cosmo.github.io/torch-pme/latest/>`_
# As a demonstration we use two charges in a cubic cell, computing the electrostatic energy
# as a function of distance

O_charge = -0.84
coulomb_distances = torch.linspace(0.5, 9.0, 50)
cell = torch.eye(3) * 10.0


# %%
# We also use the parameter-tuning functionality of ``torchpme``, 
# to provide efficient evaluation at the desired level of accuracy. 
# This is achieved calling :func:`torchpme.tuning.tune_p3m` on a 
# template of the target structure

charges = torch.tensor([O_charge, -O_charge]).unsqueeze(-1)
positions_coul = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
neighbor_indices = torch.tensor([[0, 1]])
neighbor_distances = torch.tensor([4.0])

p3m_smearing, p3m_parameters, _ = torchpme.tuning.tune_p3m(charges, cell, positions_coul, 
                                      cutoff, neighbor_indices, 
                                      neighbor_distances, accuracy=1e-4)
p3m_prefactor = torchpme.prefactors.kcalmol_A

# %%
#
# The hydrogen charge is derived from the oxygen charge as :math:`q_H = -q_O/2`. The
# ``smearing`` and ``mesh_spacing`` parameters are the central parameters for P3M and
# are crucial to ensure the correct energy calculation. Here, we base these values on
# the ``cutoff`` distance with will ensures good convergence but not necessarly the
# fasted evaluation. For a faster evaluation parameters, refer to the ``torch-pme``
# package and its tuning functions like :func:`torchpme.tuning.tune_p3m`. We now compute
# the electrostatic energy between two point charges using the P3M algorithm.

p3m_calculator = torchpme.P3MCalculator(
    potential=torchpme.CoulombPotential(p3m_smearing),
    **p3m_parameters
    prefactor=p3m_prefactor
)



# %%
#
# For the P3M algorithm, we need a neighbor list and distances which we compute
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

plt.title("Electrostatic Potential Between Two Point Charges")
plt.plot(coulomb_distances, potential)
plt.xlabel("Distance [Å]")
plt.ylabel("Electrostatic Potential [kcal/mol]")
plt.show()

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
#   Therefore, we first compute the electrostatic energy of all atoms and then subtract
#   interactions between bonded atoms.


# %%
# Implementation as a ``torch`` module
# ------------------------------------

from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

atoms = ase.io.read("data/water_32.pdb")
system = System(
    types=torch.from_numpy(atoms.get_atomic_numbers()),
    positions=torch.from_numpy(atoms.positions),
    cell=torch.from_numpy(atoms.cell.array),
    pbc=torch.from_numpy(atoms.pbc),
)
nlo = NeighborListOptions(cutoff=6.0, full_list=False, strict=False)


from vesin.torch.metatensor import NeighborList
from metatensor.torch import Labels, TensorBlock, TensorMap
calculator = NeighborList(nlo, length_unit="Angstrom")
neighbors = calculator.compute(system)
system.add_neighbor_list(nlo, neighbors)

# %%
# 
neighbors

# %% 
#

def get_bonds_angles(positions: torch.Tensor):
    """ Return the list of bond distances, angles and charge site 
    positions for the water molecules. These are assumed not to 
    be "folded", and to be listed in the input as OHHOHHOHH."""
    
    o_pos = positions[::3]
    h1_pos = positions[1::3]
    h2_pos = positions[2::3]

    oh_dist = torch.concatenate([
            torch.sqrt(((h1_pos-o_pos)**2).sum(dim=1)),
            torch.sqrt(((h2_pos-o_pos)**2).sum(dim=1)),
    ])
    if oh_dist.max() > 2.0: 
        raise ValueError("Unphysical O-H bond length. Check that the molecules are entire, and atoms are listed in the expected OHH order.")

    hoh_angles = torch.arccos(torch.sum( (h1_pos-o_pos)*(h2_pos-o_pos), dim=1 )/(
        oh_dist[:len(o_pos)]*oh_dist[len(o_pos):]
    ))
    
    return oh_dist, hoh_angles

def get_msites(system: System, m_gamma: torch.Tensor, m_charge: torch.Tensor):
    
    positions = system.positions
    o_pos = positions[::3]
    h1_pos = positions[1::3]
    h2_pos = positions[2::3]
    
    # NB: this is enough to get the correct forces on the `system` atoms. thanks autodiff!
    m_pos = m_gamma * o_pos + (1-m_gamma)*0.5*(h1_pos+h2_pos)
    
    # creates a new System with the m-sites
    m_system = System(
        types=torch.tensor([8, 1, 1]*len(o_pos)),
        positions = torch.stack([m_pos, h1_pos, h2_pos], dim=1).reshape(-1, 3),
        cell=system.cell,
        pbc=system.pbc
    )

    # adjust neighbor lists to point at the m sites rather than O atoms.
    # this assumes this won't have atoms cross the cutoff, which is 
    # of course only approximately true, so one should use a slighlty 
    # larger-than-usual cutoff
    # nb - this is reshaped to match the values in a NL tensorblock
    om_displacements = torch.stack([m_pos-o_pos, torch.zeros_like(h1_pos), torch.zeros_like(h2_pos)], dim=1).reshape(-1, 3, 1)

    for nlo in system.known_neighbor_lists():
        nl = system.get_neighbor_list(nlo)
        i_idx = nl.samples.view(["first_atom"]).values.flatten()
        j_idx = nl.samples.view(["second_atom"]).values.flatten()
        print(nl.values.shape,
              om_displacements[j_idx].shape)
        m_nl = TensorBlock(
            nl.values + om_displacements[j_idx] - om_displacements[i_idx],
            nl.samples,
            nl.components,
            nl.properties,
        )
        m_system.add_neighbor_list(nlo, m_nl)

    # set charges of all atoms (now we bundle O first and H last)
    charges = torch.ones_like(o_pos).reshape(-1,1)
    charges[::3] = -m_charge
    charges[1::3] = 0.5*m_charge
    charges[2::3] = 0.5*m_charge
    
    # Create metadata for the charges TensorBlock
    samples = Labels("atom", torch.arange(len(system), device=charges.device).reshape(-1, 1))
    properties = Labels("charge", torch.zeros(1, 1, device=charges.device, dtype=torch.int32))
    data = TensorBlock(
        values=charges, samples=samples, components=[], properties=properties
    )
    m_system.add_data(name="charges", data=data)    

    # intra-molecular distances (for self-energy removal)
    hh_dist = torch.sqrt(((h1_pos-h2_pos)**2).sum(dim=1))
    mh_dist = torch.concatenate([
            torch.sqrt(((h1_pos-m_pos)**2).sum(dim=1)),
            torch.sqrt(((h2_pos-m_pos)**2).sum(dim=1)),
    ])
    return m_system, mh_dist, hh_dist    



# %%
mol_data = get_bonds_angles(system.positions)    
mol_data[-1]

m_sys = get_msites(system, 0.8, 0.85)
m_nl = m_sys[0].get_neighbor_list(nlo)
p3m_calculator = torchpme.metatensor.P3MCalculator(
                potential=torchpme.CoulombPotential(p3m_smearing),
                **p3m_parameters
            )

# %%
pots = p3m_calculator.forward(m_sys[0], m_nl).block(0).values
charges = m_sys[0].get_data("charges").values

# %%
(pots*charges).sum()

# %%


# %%
class QTIP4PfModel(torch.nn.Module):
    def __init__(self, 
                cutoff: float,
                lj_sigma: float,
                lj_epsilon: float,
                m_gamma: float,
                m_charge: float,
                oh_bond_eq: float,
                oh_bond_d: float,
                oh_bond_alpha: float,
                hoh_angle_eq: float,
                hoh_angle_k: float,
                p3m_options: Optional[dict]=None,
                dtype: Optional[torch.dtype]=None,
                device: Optional[torch.device]=None,
    ):
        super().__init__()

        if p3m_options is None:
            # sane defaults, should give a ~1e-4 relative accuracy on forces 
            p3m_options = (1.4, {'interpolation_nodes': 5, 'mesh_spacing': 1.33}, 0)
        p3m_smearing, p3m_parameters, _ = p3m_options
        
        self.p3m_calculator = torchpme.metatensor.P3MCalculator(
                potential=torchpme.CoulombPotential(p3m_smearing),
                **p3m_parameters,
                prefactor=torchpme.prefactors.kcalmol_A
            )
        
        self.coulomb = torchpme.CoulombPotential()

        # We use a half neighborlist and allow to have pairs farther than cutoff
        # (`strict=False`) since this is not problematic for PME and may speed up the
        # computation of the neigbors.
        self.nlo = NeighborListOptions(cutoff=cutoff, full_list=False, strict=False)

        # registers model parameters as buffers
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.device = device if device is not None else torch.get_default_device()
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=self.dtype))
        self.register_buffer("lj_sigma", torch.tensor(lj_sigma, dtype=self.dtype))
        self.register_buffer("lj_epsilon", torch.tensor(lj_epsilon, dtype=self.dtype))
        self.register_buffer("m_gamma", torch.tensor(m_gamma, dtype=self.dtype))
        self.register_buffer("m_charge", torch.tensor(m_charge, dtype=self.dtype))
        self.register_buffer("oh_bond_eq", torch.tensor(oh_bond_eq, dtype=self.dtype))
        self.register_buffer("oh_bond_d", torch.tensor(oh_bond_d, dtype=self.dtype))
        self.register_buffer("oh_bond_alpha", torch.tensor(oh_bond_alpha, dtype=self.dtype))
        self.register_buffer("hoh_angle_eq", torch.tensor(hoh_angle_eq, dtype=self.dtype))
        self.register_buffer("hoh_angle_k", torch.tensor(hoh_angle_k, dtype=self.dtype))
        

    def requested_neighbor_lists(self):
        return [self.nlo]

    def _setup_systems(
        self,
        systems: list[System],
        selected_atoms: Optional[Labels] = None,
    ) -> tuple[System, TensorBlock]:
        
        """Remove possible ghost atoms."""
        if len(systems) > 1:
            raise ValueError(f"only one system supported, got {len(systems)}")

        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")
        return systems[0], systems[0].get_neighbor_list(self.nlo)

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
        
        system, neighbors = self._setup_systems(systems, selected_atoms)

        # gets information about water molecules, to compute intra-molecular and electrostatic terms
        d_oh, a_hoh = get_bonds_angles(system.positions)

        # intra-molecular energetics
        e_bond = bond_energy(d_oh, self.oh_bond_d, self.oh_bond_alpha, self.oh_bond_eq).sum()
        e_bend = bend_energy(a_hoh, self.hoh_angle_k, self.hoh_angle_eq).sum()
        
        # compute non-bonded LJ energy
        neighbor_indices = neighbors.samples.view(["first_atom", "second_atom"]).values
        species = system.types
        oo_mask = (species[neighbor_indices[:,0]]==8) & (species[neighbor_indices[:,1]]==8)
        oo_distances = torch.linalg.vector_norm(neighbors.values[oo_mask], dim=1)
        energy_lj = lennard_jones_pair(oo_distances, self.lj_sigma, self.lj_epsilon, self.cutoff).sum()
        
        # now this is the long-range part - computed over the M-site system
        m_system, mh_dist, hh_dist = get_msites(system, self.m_gamma, self.m_charge)
        potentials = self.p3m_calculator(m_system, neighbors).block(0).values
        charges = m_system.get_data("charges").values
        print(charges, potentials, charges.sum())
        energy_coulomb = (potentials*charges).sum()        
        energy_self = (
            self.coulomb.from_dist(mh_dist).sum()*charges[0] + 
            self.coulomb.from_dist(hh_dist).sum()*charges[1]
        )*charges[1]*torchpme.prefactors.kcalmol_A

        print("energies", e_bond, e_bend, energy_lj, energy_coulomb, energy_self)

        energy_tot = e_bond + e_bend + energy_lj + energy_coulomb - energy_self
        # Rename property label to follow metatensor's covention for an atomistic model
        samples = Labels(
            ["system"], torch.arange(len(systems), device=self.device).reshape(-1, 1)
        )
        block = TensorBlock(
            values=torch.sum(energy_tot).reshape(-1, 1),
            samples=samples,
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]], device=self.device)),
        )
        return {
            "energy": TensorMap(
                Labels("_", torch.tensor([[0]], device=self.device)), [block]
            ),
        }



# %%

model=QTIP4PfModel(6.0, 3.1589, 0.1852, 0.73612, 1.128, 0.9419, 116.09, 2.287, 107.4*np.pi /180, 87.85)
# %%
nrg = model.forward([system], {"energy":""})


# %%

torch.jit.script(model)

# %%
# Wraps the torch model into a metatomic calculator


energy_unit = "kcal/mol"
length_unit = "angstrom"
outputs = {"energy": ModelOutput(quantity="energy", unit=energy_unit, per_atom=False)}
options = ModelEvaluationOptions(
    length_unit=length_unit, outputs=outputs, selected_atoms=None
)

model_capabilities = ModelCapabilities(
    outputs=outputs,
    atomic_types=[1, 8],
    interaction_range=torch.inf,
    length_unit=length_unit,
    supported_devices=["cpu", "cuda"],
    dtype="float32",
)

atomistic_model = MetatensorAtomisticModel(
    model.eval(), ModelMetadata(), model_capabilities
)
atomistic_model.save("qtip4pf-mta.pt")
mta_calculator = MetatensorCalculator(atomistic_model)

# %%

atoms = ase.io.read("data/water_32.pdb")
atoms.calc=mta_calculator
nrg = atoms.get_potential_energy()
atoms.calc.todict=(lambda : None)
opt_trj = []
opt_nrg = []

# %%
from ase.optimize import LBFGS
# fmax is the threshold on the maximum force component. 
# optimization will stop when the threshold is reached
for i in range(100):
    opt_trj.append(atoms.copy())
    opt_nrg.append(atoms.get_potential_energy())
    opt = LBFGS(atoms, restart="lbfgs_restart.json")
    opt.run(fmax=0.001, steps=3)


# the optimized geometry is stored in the `structure` object
# %%

# %%
import chemiscope

chemiscope.show(frames=opt_trj, mode="structure", settings=
                chemiscope.quick_settings(trajectory=True))
# %%

plt.plot(opt_nrg)
# %%

opt_nrg
# %%
