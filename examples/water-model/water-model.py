"""
Atomistic Water Model for Molecular Dynamics
============================================

:Authors: Philip Loche `@PicoCentauri <https://github.com/picocentauri>`_
         Marcel Langer `@sirmarcel <https://github.com/sirmarcel>`_

In this example, we demonstrate how to construct a `metatensor atomistic model
<https://docs.metatensor.org/latest/atomistic>`_ for three- and four-body flexible water
models. The model will be used to run a brief molecular dynamics (MD) simulation. The
bonds between the oxygens and hydrogens are harmonic, as are the angles within the water
molecule. Long range electrostatic energies are handled using the P3M algorithm.
"""

# %%

# sphinx_gallery_thumbnail_number = 5

import importlib.machinery
import importlib.util

# Analysis tools
import ase.geometry.rdf

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


# %%
#
# The Simulation System
# ---------------------
#
# We will simulate a system containing 213 water molecules in a cubic box with periodic
# boundary conditions, stored in an GROMACS structure file. The side length of the box
# is 18.8 Å.

atoms = ase.io.read("conf.gro")

# %%
#
# The system was pre-equilibrated in the Npt ensemble at 300 K and 1 bar pressure for 1
# ns. We can visualize the system with `chemiscope <https://chemiscope.org>`_.

chemiscope.show(
    [atoms],
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True, structure_settings={"unitCell": True}
    ),
)

# %%
#
# The Model
# ---------
#
# As described above, our model will handle any flexible three- or four-point water
# models. For a four-point model, the oxygen charge is slightly displaced from the
# oxygen's position, improving properties like the `dielectric constant
# <http://dx.doi.org/10.1021/jp410865y>`_. The fourth point, referred to as ``M``, is
# implicitly derived from the other atoms of each water molecule. Refer to
# `10.1063/1.3167790`_ for details on its derivation.
#
# Initially, we will model a three-point model called the flexible q-SPC/Fw model as
# introduced in `10.1063/1.3167790`_. The model implementation is lengthy and thus not
# shown here by default. You can find it in :download:`model.py.example` or in the
# foldable code block below.
#
# .. _`10.1063/1.3167790`: https://doi.org/10.1063/1.3167790
#
# .. details:: Show the code used to create the model file
#
#     .. literalinclude:: model.py.example
#         :language: python
#
# Instead, we import the model code using ``importlib`` and discuss the central
# properties of the model and its usage in a simulation.

spec = importlib.util.spec_from_loader(
    "model", importlib.machinery.SourceFileLoader("model", "./model.py.example")
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# %%
# .. warning::
#
#  The model assumes that molecules are not fragmented across the periodic box when
#  computing bond and angle interactions. If molecules are incomplete, expect severe
#  artifacts in the simulation results!
#
# Lennard-Jones Potential
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The Lennard-Jones (LJ) potential describes the interaction between a pair of neutral
# atoms or molecules, balancing attractive forces at longer ranges and repulsive forces
# at shorter ranges. The LJ potential is defined as:
#
# .. math::
#
#  V_\mathrm{LJ}(r) = 4 \epsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left(
#  \frac{\sigma}{r} \right)^6 \right]
#
# where :math:`\epsilon` is the depth of the potential well and :math:`\sigma` is the
# distance at which the potential is zero. For water there is only an oxygen-oxygen
# Lennard-Jones potential. There is no LJ potential between hydrogen atoms and between
# hydrogen and oxygen atoms. The parameters for the oxygen-oxygen interaction are:

O_sigma = 3.1655  # Å
O_epsilon = 0.1554  # kcal/mol
cutoff = 9.0  # Å

# %%
#
# We implement the Lennard-Jones potential as a function that takes distances, along
# with the parameters ``sigma``, ``epsilon``, and ``cutoff``. The ``cutoff`` shifts the
# potential to zero at the cutoff distance.

lj_distances = np.linspace(3, cutoff, 1000)

lj_potential = model.lennard_jones_pair(
    distances=lj_distances, sigma=O_sigma, epsilon=O_epsilon, cutoff=cutoff
)

# %%
#
# We plot this potential to visualize its behavior.

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
# The plot shows the typical Lennard-Jones potential crossing zero at σ, with the
# minimum energy of ε.
#
# Harmonic Bond Potential
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we define the harmonic bond potential, modeled as:
#
# .. math::
#
#   V_\mathrm{bond}(r) = \frac{k_\mathrm{bond}}{2} (r - r_0)^2
#
# Here, :math:`k_\mathrm{bond}` is the force constant and :math:`r_0` is the equilibrium
# distance. For the water model there is a harmonic bond between the oxygen and each of
# the two hydrogens. We use the following parameters:

OH_bond_coefficient = 1059.162  # kcal/mol/Å^2
OH_equilibrium_distance = 1.0  # Å

# %%
#
# Similarly, we compute the harmonic bond potential.

bond_distances = np.linspace(0.5, 1.5, 1000)
bond_potential = model.harmonic_distance_pair(
    distances=bond_distances,
    coefficient=OH_bond_coefficient,
    equilibrium_distance=OH_equilibrium_distance,
)

plt.title("Harmonic Bond Potential Between Oxygen and Hydrogen")
plt.plot(bond_distances, bond_potential)
plt.xlabel("Distance [Å]")
plt.ylabel("Harmonic Bond Potential [kcal/mol]")

plt.show()

# %%
#
# The plot displays the harmonic bond potential with a minimum at the equilibrium
# distance of 1.0 Å and a quadratic increase.
#
# Harmonic Angle Potential
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The harmonic angle potential is an interaction potential between three atoms that
# create a plan and is defined as:
#
# .. math::
#
#   V_\mathrm{angle}(\theta) = \frac{k_\mathrm{angle}}{2} (\theta - \theta_0)^2
#
# where :math:`k_\mathrm{angle}` is the force constant and :math:`\theta_0` is the
# equilibrium angle between the three atoms. We use the following parameters:

HOH_angle_coefficient = 75.90  # kcal/mol/rad^2
HOH_equilibrium_angle = 112.0  # degrees

# %%
#
# and compute the harmonic angle potential similarly as before.

angle_distances = np.linspace(104, 115, 1000)
angle_potential = model.harmonic_angular(
    angles=angle_distances,
    coefficient=HOH_angle_coefficient,
    equilibrium_angle=HOH_equilibrium_angle,
)

plt.title("Harmonic Angle Potential in Water Molecule")
plt.plot(angle_distances, angle_potential)
plt.xlabel("Angle [degrees]")
plt.ylabel("Harmonic Angle Potential [kcal/mol]")
plt.show()

# %%
#
# Electrostatic Potential
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# Since electrostatic interactions are long-ranged it is not fullt trivial to compute
# these in simulations. For periodic systems the Coulomb energy is given by:
#
# .. math::
#
#  V_\mathrm{Coulomb} = \frac{1}{2} \sum_{i,j} \sideset{}{'}\sum_{\boldsymbol n \in
#   \mathcal{Z}} \frac{q_i q_j}{\boldsymbol r_{ij} + \boldsymbol n L}
#
# The sum over :math:`\boldsymbol n` takes into account the periodic images of the
# charges and the prime indicates that in the case :math:`i=j` the term :math:`n=0` must
# be omitted. Further :math:`boldsymbol r_{ij} = \boldsymbol r_i - \boldsymbol r_j` and
# :math:`\boldsymbol L` is the length of the simulation box.
#
# Since this sum is conditionally convergent it isn't computable using a direct sum.
# Instead the Ewald summation, published in 1921, remains a foundational method that
# effectively defines how to compute the energy and forces of such systems. To further
# speed the methods, mesh based algorithm suing fast Fourier transformation have been
# developed, such as the Particle-Particle Particle-Mesh (P3M) algorithm. For further
# details we refer to a paper by `Deserno and Holm
# <https://aip.scitation.org/doi/10.1063/1.477414>`_ The parameters for the P3M
# algorithm are:

O_charge = -0.84
pme_smearing = cutoff / 5.0
pme_mesh_spacing = pme_smearing / 8.0
pme_interpolation_nodes = 4
pme_prefactor = torchpme.prefactors.kcalmol_A

# %%
#
# The hydrogen charge is derived from the oxygen charge as :math:`q_H = -q_O/2`. The
# ``smearing`` and ``mesh_spacing`` parameters are the central parameters for P3M and
# are crucial to ensure the correct energy calculation. Here, we base these values on
# the ``cutoff`` distance with will ensures good convergence but not necessarly the
# fasted evaluation. For a faster evaluation parameters, refer to the ``torch-pme``
# package and its tuning functions like :func:`torchpme.tuning.tune_p3m`. We now compute
# the electrostatic energy between two point charges using the P3M algorithm.

pme_calculator = torchpme.P3MCalculator(
    potential=torchpme.CoulombPotential(pme_smearing),
    mesh_spacing=pme_mesh_spacing,
    interpolation_nodes=pme_interpolation_nodes,
    prefactor=pme_prefactor,
)

coulomb_distances = torch.linspace(0.5, 9.0, 50)
cell = torch.eye(3) * 10.0

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

    potential[i_dist] = pme_calculator.forward(
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
#   Coulomb interactions within a single water molecule are excluded, as atomic
#   interaction are already parametrized by the bond and angle interactions. Therefore,
#   we first compute the electrostatic energy of all atoms and then subtract
#   interactions between bonded atoms.
#
# Define a Calculator
# -------------------
#
# We now are familiar with the paramterization of water. We initialize the model object
# and wrap it in a metatensor atomistic model, defining all necessary metadata,
# including energy and length units. This is necessary since differenrt simulation
# engines have different energy units. For example in ASE, all energies are in supposed
# to be in eV. Metatensor will convert our energy units from kcal/mol to eV
# automatically.

energy_unit = "kcal/mol"
length_unit = "angstrom"

# %%
#
# We specify the model's capabilities using
# :class:`metatensor.torch.atomistic.ModelOutput`. The model computes energy in
# kcal/mol. It's crucial to set ``interaction_range`` to **infinite** to ensure the
# entire system is considered which is a requirement for the P3M algorithm.

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

# %%
#
# Initialize and Wrap the Model
# -----------------------------
#
# We use the previously defined values to initialize the model.

spc_f_model = model.WaterModel(
    cutoff=cutoff,
    O_sigma=O_sigma,
    O_epsilon=O_epsilon,
    O_charge=O_charge,
    OH_bond_coefficient=OH_bond_coefficient,
    OH_equilibrium_distance=OH_equilibrium_distance,
    HOH_angle_coefficient=HOH_angle_coefficient,
    HOH_equilibrium_angle=HOH_equilibrium_angle,
    pme_smearing=pme_smearing,
    pme_mesh_spacing=pme_mesh_spacing,
    pme_interpolation_nodes=pme_interpolation_nodes,
    pme_prefactor=pme_prefactor,
    four_point_model=False,
)
spc_f_model.eval()

spc_f_model_atomistic_model = MetatensorAtomisticModel(
    spc_f_model, ModelMetadata(), model_capabilities
)

# %%
#
# We run the MD simulation in the constant volume/temperature (NVT) ensemble at 300
# Kelvin, using a Langevin thermostat for integration. First we set the
# ``atomistic_model`` as the calculator for our system.

water_mta_calculator = MetatensorCalculator(spc_f_model_atomistic_model)
atoms.calc = water_mta_calculator

# %%
#
# To test we can compute the potential energy of the system.

print(f"Potential energy of the system: {atoms.get_potential_energy():.2f} eV")

# %%
#
# Set initial velocities according to the Maxwell-Boltzmann distribution at 300 Kelvin.

ase.md.velocitydistribution.MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# %%
#
# Set up the Langevin thermostat for the NVT ensemble.

integrator = ase.md.Langevin(
    atoms,
    timestep=0.5 * ase.units.fs,
    temperature_K=300,
    friction=1 / ase.units.fs,
)

# %%
#
# Run the Simulation
# ------------------
#
# We run the simulation for 50 steps (:math:`0.5\,\mathrm{fs}`) and collect potential,
# kinetic, and total energy, as well as temperature and pressure.

n_steps = 50

potential_energy = np.zeros(n_steps)
kinetic_energy = np.zeros(n_steps)
total_energy = np.zeros(n_steps)
temperature = np.zeros(n_steps)
pressure = np.zeros(n_steps)
trajectory = []

for i_step in range(n_steps):
    integrator.run(1)

    # Collect simulation data
    trajectory.append(atoms.copy())
    potential_energy[i_step] = atoms.get_potential_energy()
    kinetic_energy[i_step] = atoms.get_kinetic_energy()
    total_energy[i_step] = atoms.get_total_energy()
    temperature[i_step] = atoms.get_temperature()
    pressure[i_step] = -np.diagonal(atoms.get_stress(voigt=False)).mean()

# %%
#
# Visualize the trajectory using `chemiscope <https://chemiscope.org>`_. For better
# visualization, we include the unit cell.

chemiscope.show(
    trajectory,
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True, structure_settings={"unitCell": True}
    ),
)

# %%
# Analyze the Results
# -------------------
#
# Time Evolution of Physical Properties
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We examine the time evolution of physical properties. For better comparison, an
# arbitrary constant is subtracted from the potential energy.

fig, ax = plt.subplots(3, figsize=(8, 5), sharex=True)

time = 0.5 * np.arange(n_steps)

ax[0].plot(time, potential_energy - potential_energy[-1], label="Potential Energy")
ax[0].plot(time, kinetic_energy, label="Kinetic Energy")
ax[0].plot(time, total_energy - potential_energy[-1], label="Total Energy")
ax[0].legend(ncol=3)
ax[0].set_ylabel("Energy [eV]")

ax[1].plot(time, temperature, label="Temperature")
ax[1].axhline(300, color="black", linestyle="--", label="Target Temperature")
ax[1].legend(ncol=2)
ax[1].set_ylabel("Temperature [K]")

ax[2].plot(time, pressure)
ax[2].axhline(0, color="black", linestyle="--")
ax[2].set_ylabel("Pressure [eV Å$^{-3}$]")

ax[-1].set_xlabel("Time [fs]")

fig.align_labels()
plt.show()

# %%
#
# We find that temperature remains around the target of 300 K, and the pressure
# stabilizes around a positive value.
#
# Radial Distribution Function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We compute and plot the partial radial distribution functions (RDF) between all pairs
# in the system, using the last 20 steps to ensure at least some equilibration from the
# initial configuration.

OO_rdf_l = []
OH_rdf_l = []
HH_rdf_l = []

for atoms in trajectory[-20:]:
    OO_rdf_step, rdf_dists = ase.geometry.rdf.get_rdf(
        atoms, rmax=6.0, nbins=200, elements=(8, 8)
    )
    OH_rdf_step, _ = ase.geometry.rdf.get_rdf(
        atoms, rmax=6.0, nbins=200, elements=(8, 1)
    )
    HH_rdf_step, _ = ase.geometry.rdf.get_rdf(
        atoms, rmax=6.0, nbins=200, elements=(1, 1)
    )

    OO_rdf_l.append(OO_rdf_step)
    OH_rdf_l.append(OH_rdf_step)
    HH_rdf_l.append(HH_rdf_step)

# %%
#
# We average the RDFs and renormalize them. ASE normalizes partial RDFs based on the
# total number of atoms. To ensure RDFs converge to 1 at large distances, we normalize
# by the corresponding number fractions.

OO_rdf = np.mean(OO_rdf_l, axis=0) / (1 / 3)
OH_rdf = np.mean(OH_rdf_l, axis=0) / (2 / 3)
HH_rdf = np.mean(HH_rdf_l, axis=0) / (2 / 3)

fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(6, 5))

ax[0].axhline(1, c="k", ls="dotted")
ax[1].axhline(1, c="k", ls="dotted")
ax[2].axhline(1, c="k", ls="dotted")

ax[0].plot(rdf_dists, OO_rdf)
ax[1].plot(rdf_dists, OH_rdf)
ax[2].plot(rdf_dists, HH_rdf)

ax[0].set_ylabel(r"$g_\mathrm{OO}(r)$")
ax[1].set_ylabel(r"$g_\mathrm{OH}(r)$")
ax[2].set_ylabel(r"$g_\mathrm{HH}(r)$")
ax[2].set_xlabel("Distance [Å]")
ax[2].set_xlim(0.8, 6)
ax[2].set_ylim(-0.6, 3)

fig.tight_layout()
plt.show()

# %%
#
# The RDFs display typical peaks for oxygen-oxygen, oxygen-hydrogen, and
# hydrogen-hydrogen pairs, decaying to 1 at larger distances.
#
# For convenience, we summarize the flexible SPC model parameters in a dictionary. Note
# that all energy units are in kcal/mol, distances in Angstroms, angles in degrees, and
# charges in electron charges.

spc_fw_parameters = {
    "cutoff": 9.0,
    "O_sigma": 3.1655,
    "O_epsilon": 0.1554,
    "O_charge": -0.84,
    "OH_bond_coefficient": 1059.162,
    "OH_equilibrium_distance": 1.0,
    "HOH_angle_coefficient": 75.90,
    "HOH_equilibrium_angle": 112.0,
    "pme_smearing": 1.8,
    "pme_mesh_spacing": 0.225,
    "pme_interpolation_nodes": 4,
    "pme_prefactor": 332.06371,
    "four_point_model": False,
}

# %%
#
# Export for General Simulation Engines
# -------------------------------------
#
# An atomistic model can be used in other engines. Refer to the metatensor atomistic
# documentation on `supported simulation engines
# <https://docs.metatensor.org/latest/atomistic/engines>`_.
#
# .. caution::
#
#   The model requires a strict atom order: ``OHHOHHOHH...``. Some simulation engines
#   (e.g., LAMMPS) may not maintain atom order even if initially sorted.
#
# To use the model in general simulation engines, save it to disk by

spc_f_model_atomistic_model.save("spc_fw_model.pt")

# %%
#
# Four-Point Model
# ----------------
#
# We can create the q-TIP4P/F four-point model using the following parameters

tip4p_f_parameters = {
    "cutoff": 9.0,
    "O_sigma": 3.1589,
    "O_epsilon": 0.1852,
    "O_charge": -1.1128,
    "OH_bond_coefficient": 116.09,
    "OH_equilibrium_distance": 0.9419,
    "HOH_angle_coefficient": 87.85,
    "HOH_equilibrium_angle": 107.4,
    "pme_smearing": 1.8,
    "pme_mesh_spacing": 0.225,
    "pme_interpolation_nodes": 4,
    "pme_prefactor": 332.06371,
    "four_point_model": True,
}

tip4p_model = model.WaterModel(**tip4p_f_parameters)

# %%
#
# and wrap the four-point model similarly to compute the potential energy of the system.

tip4p_model.eval()

tip4p_atomistic_model = MetatensorAtomisticModel(
    tip4p_model, ModelMetadata(), model_capabilities
)

tip4p_water_mta_calculator = MetatensorCalculator(tip4p_atomistic_model)
atoms.calc = tip4p_water_mta_calculator

print(f"Potential energy of the system: {atoms.get_potential_energy():.2f} eV")
