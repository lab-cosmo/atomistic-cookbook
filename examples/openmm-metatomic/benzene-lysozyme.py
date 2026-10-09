"""
Protein-ligand interaction energies with OpenMM-ML
==================================================

:Authors: Eric Boittier `@EricBoittier <https://github.com/EricBoittier>`_

QM/MM is a method that combines
fast, empircal force fields (MM) with more detailed models (QM), for a trade-off between accuracy and computational cost.
MLIPs fit to QM data can be faithful oracles that provide an even better trade-off to speed and quality.
This notebook evaluates the interaction energy of benzene and o-xylene (ML region) with the protein lysozyme (MM region).

The ligand is modelled with PET-SPICE-S in metatomic. Lysozyme is modelled by the Amber ff14SB force field.
The total energy is the sum of the MM energy of the protein and ligand, the ML energy of the ligand, and the MM energy of the ligand.

.. math::

    E = E_\\text{MM}(\\text{protein + ligand})
      + E_\\text{ML}(\\text{ligand})
      - E_\\text{MM}(\\text{ligand}).

This approach is called a mechanical embedding, since the ligand is not polarised by the protein charges and only interacts
using the molecular mechanics force field [7]_.
"""


# %%
# Setuping Metatomic and OpenMM-ML
# -----
#
# OpenMM-ML loads each potential
# from an entry point in the group ``openmmml.potentials``. The wrapper in ``openmm_metatomic`` 
# next to this file, so ``MLPotential("metatomic")`` uses the OpenMM-ML already
# installed. 

import sys
import warnings
from importlib.metadata import Distribution, DistributionFinder
from pathlib import Path

import ase
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from atomistic_cookbook_utils import download_with_retry, run_command
from openff.toolkit import Molecule
from openff.toolkit import Topology as OffTopology
from openff.units.openmm import to_openmm
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

sys.path.insert(0, str(Path.cwd()))


class _MetatomicDistribution(Distribution):
    def read_text(self, filename):
        if filename == "METADATA":
            return "Metadata-Version: 2.1\nName: openmmml-metatomic\nVersion: 0\n"
        if filename == "entry_points.txt":
            return (
                "[openmmml.potentials]\n"
                "metatomic = openmm_metatomic.metatomicpotential:"
                "MetatomicPotentialImplFactory\n"
            )
        return None

    def locate_file(self, path):
        return path


class _MetatomicFinder(DistributionFinder):
    def find_distributions(self, context=DistributionFinder.Context()):
        if context.name in (None, "openmmml-metatomic"):
            yield _MetatomicDistribution()


sys.meta_path.append(_MetatomicFinder())
from openmmml import MLPotential  # noqa: E402


WORKSHOP = "https://raw.githubusercontent.com/openmm/openmm_workshops/main/section_1"
for filename in ("lysozyme.pdb", "benzene.sdf", "o-xylene.sdf"):
    download_with_retry(f"{WORKSHOP}/{filename}", filename)

model_path = Path("pet-spice-s.pt")
if not model_path.is_file():
    checkpoint = (
        "https://huggingface.co/lab-cosmo/upet/resolve/main/"
        "models/pet-spice-s-v0.2.0.ckpt"
    )
    run_command(f"mtt export {checkpoint} -o {model_path}", print_output=True)


# Download the workshop structures, export , and build each
# complex. An OpenFF molecule is appended to the protein with ``Modeller``. 
# OpenFF 2.2.1 expects AM1-BCC charges. The NAGL model ``openff-gnn-am1bcc-1.0.0``
# assigns them, so AmberTools is not required.

potential = MLPotential("metatomic", model=str(model_path), device="cpu")
print("Embeddings:", potential.getSupportedEmbeddings())

CPU = mm.Platform.getPlatformByName("CPU")


def to_atoms(topology, positions):
    symbols = [atom.element.symbol for atom in topology.atoms()]
    if isinstance(positions, unit.Quantity):
        xyz = positions.value_in_unit(unit.angstrom)
    else:
        xyz = unit.Quantity(positions, unit.nanometer).value_in_unit(unit.angstrom)
    return ase.Atoms(symbols=symbols, positions=np.asarray(xyz, dtype=np.float64))


def potential_energy(system, positions):
    """Potential energy of one configuration, in kJ/mol."""
    context = mm.Context(system, mm.VerletIntegrator(1.0 * unit.femtoseconds), CPU)
    context.setPositions(positions)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return energy.value_in_unit(unit.kilojoule_per_mole)

# %%
# Building the MM and ML systems
# --------------------
#
# The protein is the Amber region. It is the same structure for both
# ligands, and it is the receptor that remains when the ligand is removed
# from the supermolecule difference.
protein_pdb = app.PDBFile("lysozyme.pdb")
n_protein = protein_pdb.topology.getNumAtoms()
protein_atoms = to_atoms(protein_pdb.topology, protein_pdb.positions)
protein_xyz = protein_atoms.positions
print(f"Total number of atoms in the complex: {protein_pdb.topology.getNumAtoms()}")
print(f"Number of bonds in the MM system: {protein_pdb.topology.getNumBonds()}")
print(f"Number of bonds in the ML system: {protein_pdb.topology.getNumBonds()}")
chemiscope.show(
    [protein_atoms], mode="structure", settings={"structure": [{"bonds": True}]}
)


molecules = [Molecule.from_file(name) for name in ("benzene.sdf", "o-xylene.sdf")]
for molecule in molecules:
    molecule.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
warnings.filterwarnings(  # no virtual sites on these hydrocarbons
    "ignore", message="Preset charges were provided"
)
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
forcefield.registerTemplateGenerator(
    SMIRNOFFTemplateGenerator(molecules=molecules, forcefield="openff-2.2.1").generator
)

# We will print the number of atoms in the MM and ML systems for each ligand
# and the total number of atoms in the complex, and the number of bonds
systems = {}
for name, molecule in zip(("benzene", "o-xylene"), molecules):
    ligand_topology = OffTopology.from_molecules(molecules=[molecule]).to_openmm()
    ligand_positions = to_openmm(molecule.conformers[0])
    modeller = app.Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_topology, ligand_positions)
    systems[name] = {
        "topology": modeller.topology,
        "positions": modeller.positions,
        "ligand_topology": ligand_topology,
        "ligand_positions": ligand_positions,
        "ligand_atoms": list(range(n_protein, modeller.topology.getNumAtoms())),
    }
    print(
        f"{name}: {len(systems[name]['ligand_atoms'])} ligand atoms, "
        f"{modeller.topology.getNumAtoms()} in the complex"
        f"{modeller.topology.getNumBonds()} bonds"
    )



# %%
# Ligands
# -------
#
# Benzene and o-xylene are the metatomic region. PET-SPICE sees only these
# atoms. Use the structure list to switch from one ligand to the other.

ligand_frames = []
for name in ("benzene", "o-xylene"):
    entry = systems[name]
    ligand_frames.append(to_atoms(entry["ligand_topology"], entry["ligand_positions"]))

chemiscope.show(
    ligand_frames,
    properties={"ligand": ["benzene", "o-xylene"]},
    mode="structure",
    settings={"structure": [{"bonds": True, "keepOrientation": True, "spaceFilling": True}]},
)

# %%
# Complexes
# ---------
#
# Each complex is the protein plus one ligand, still in vacuum. The
# closest heavy-atom contact is printed so a steric clash can be told
# from a packed pose before any energy is interpreted.

complex_frames = []
for name in ("benzene", "o-xylene"):
    entry = systems[name]
    frame = to_atoms(entry["topology"], entry["positions"])
    ligand_xyz = np.asarray(
        entry["ligand_positions"].value_in_unit(unit.angstrom), dtype=np.float64
    )
    contact = np.linalg.norm(
        ligand_xyz[:, None, :] - protein_xyz[None, :, :], axis=-1
    ).min()
    print(f"{name}: closest contact {contact:.2f} A")
    complex_frames.append(frame)

chemiscope.show(
    complex_frames,
    properties={"ligand": ["benzene", "o-xylene"]},
    mode="structure",
    settings={"structure": [{"bonds": True, "keepOrientation": True}]},
)

# %%
# OpenMM energy contributions
# ---------------------------
#
# A force field is a list of ``Force`` objects. OpenMM has no query for
# one ``Force``, but each ``Force`` can be placed in its own force group
# and `Context.getState <https://docs.openmm.org/latest/userguide/theory/05_other_features.html>`_
# then returns that group's energy. Groups do not change the integrator:
# unless ``setIntegrationForceGroups`` says otherwise, every group is
# still part of the dynamics.
#
# The mixed system is benzene only. Bonds, angles, and torsions that lie
# entirely inside the ligand have been removed, so those classical terms
# are protein terms.  The nonbonded term is
# the protein plus the classical protein–ligand coupling.

benzene = systems["benzene"]
vacuum = dict(
    nonbondedMethod=app.NoCutoff,
    constraints=app.HBonds,
    rigidWater=True,
    removeCMMotion=False,
)
mm_vacuum = forcefield.createSystem(benzene["topology"], **vacuum)
mixed = potential.createMixedSystem(
    benzene["topology"],
    mm_vacuum,
    benzene["ligand_atoms"],
    embedding="mechanical",
    removeConstraints=True,
)
for index, force in enumerate(mixed.getForces()):
    force.setForceGroup(index)

context = mm.Context(mixed, mm.VerletIntegrator(1.0 * unit.femtoseconds), CPU)
context.setPositions(benzene["positions"])
names = []
energies = []
for index, force in enumerate(mixed.getForces()):
    group = context.getState(getEnergy=True, groups={index})
    energy = group.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    name = "metatomic ligand" if isinstance(force, mm.PythonForce) else force.getName()
    names.append(name)
    energies.append(energy)
    print(f"{name:24s} {energy:14.2f} kJ/mol")
total = context.getState(getEnergy=True).getPotentialEnergy()
print(f"{'sum of groups':24s} {sum(energies):14.2f} kJ/mol")
print(f"{'total':24s} {total.value_in_unit(unit.kilojoule_per_mole):14.2f} kJ/mol")

fig, axis = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
axis.barh(names, energies)
axis.set_xlabel("energy / kJ/mol")

# %%
# Constraints and NVE
# -------------------
#
# ``constraints=app.HBonds`` replaces every bond to hydrogen with a
# holonomic constraint (SHAKE), held to the force field's equilibrium
# length. ``app.AllBonds`` also constrains heavy-atom bonds, and
# ``app.HAngles`` constrains the H–X–H angle. Water added later is held
# rigid by SETTLE when ``rigidWater=True``.
#
# ``createMixedSystem(..., removeConstraints=True)`` drops constraints
# whose two atoms are both in the ligand. Benzene has six hydrogens, so
# six constraints leave with it. Those distances belong to the model;
# leaving the constraint in as well would fight it. Protein X–H
# constraints stay. The ligand hydrogens are then the fastest motion, so
# the Verlet step is 0.5 fs rather than the 2 fs an all-constrained
# hydrogen timestep would allow. ``setConstraintTolerance`` is the
# allowed error in a constrained distance.
#
# Velocities are drawn once from a 300 K distribution. The Verlet
# integrator has no thermostat, so the trajectory is NVE and the total
# energy, kinetic plus potential, is the conserved quantity.
# Center-of-mass motion is left in the Hamiltonian
# (``removeCMMotion=False``). A short minimization removes the worst
# contacts in the deposited coordinates before the first step. The run
# is benzene only, and it stays in vacuum.

print(
    f"constraints: {mm_vacuum.getNumConstraints()} on the MM system, "
    f"{mixed.getNumConstraints()} after the ligand constraints are removed"
)
constrained = None
for index in range(mixed.getNumConstraints()):
    i, j, distance = mixed.getConstraintParameters(index)
    if i < n_protein and j < n_protein:
        constrained = (i, j, distance.value_in_unit(unit.angstrom))
        break
ligand_set = set(benzene["ligand_atoms"])
free = None
for bond in benzene["topology"].bonds():
    i = bond.atom1.index
    j = bond.atom2.index
    if i in ligand_set and j in ligand_set:
        symbols = {bond.atom1.element.symbol, bond.atom2.element.symbol}
        if symbols == {"C", "H"}:
            free = (i, j)
            break
print(
    f"protein constraint atoms {constrained[0]}, {constrained[1]} "
    f"at {constrained[2]:.4f} A"
)
print(f"unconstrained ligand C-H atoms {free[0]}, {free[1]}")

dt = 0.5 * unit.femtoseconds
n_steps = 100
integrator = mm.VerletIntegrator(dt)
integrator.setConstraintTolerance(1e-5)
simulation = app.Simulation(benzene["topology"], mixed, integrator, CPU)
simulation.context.setPositions(benzene["positions"])
mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=25)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)

times = []
totals = []
protein_d = []
ligand_d = []
md_frames = []
for step in range(n_steps + 1):
    if step:
        simulation.step(1)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    xyz = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    total_energy = state.getPotentialEnergy() + state.getKineticEnergy()
    times.append(step * 0.5)
    totals.append(total_energy.value_in_unit(unit.kilojoule_per_mole))
    protein_d.append(float(np.linalg.norm(xyz[constrained[0]] - xyz[constrained[1]])))
    ligand_d.append(float(np.linalg.norm(xyz[free[0]] - xyz[free[1]])))
    if step % 25 == 0:
        md_frames.append(to_atoms(benzene["topology"], state.getPositions()))

nve_positions = simulation.context.getState(getPositions=True).getPositions()
print(
    f"total energy drift over {n_steps * 0.5:.0f} fs: {totals[-1] - totals[0]:.3f} kJ/mol"
)
print(f"constrained distance std: {np.std(protein_d):.3e} A")
print(f"ligand C-H std:           {np.std(ligand_d):.3e} A")

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
axes[0].plot(times, totals)
axes[0].set_xlabel("time / fs")
axes[0].set_ylabel("total energy / kJ/mol")
axes[1].plot(times, protein_d, label="protein X–H")
axes[1].plot(times, ligand_d, label="ligand C–H")
axes[1].axhline(constrained[2], color="0.5", linewidth=0.8)
axes[1].set_xlabel("time / fs")
axes[1].set_ylabel("distance / Å")
axes[1].legend()

chemiscope.show(
    md_frames,
    mode="structure",
    settings={
        "structure": [{"bonds": True, "keepOrientation": True, "playbackDelay": 200}]
    },
)

# %%
# Pulling the ligand out
# ----------------------
#
# CustomForces available in OpenMM-ML can be used for enhanced sampling methods. Here, we will simply
# pull the ligand away from the protein, using an harmonic
# spring whose center moves at a fixed speed [9]_.
#
# .. math::
#
#     r = \mathbf{u} \cdot (\mathbf{R}_\text{ligand}
#     - \mathbf{R}_\text{protein}),
#     \qquad
#     V = \tfrac{1}{2} k (r - r_0)^2.
#
# :math:`r_0` starts at the bound value and then moves at constant speed.



xyz_nm = to_atoms(benzene["topology"], nve_positions).positions * 0.1
masses = np.array(
    [
        mixed.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(mixed.getNumParticles())
    ]
)
protein_ids = np.arange(n_protein)
ligand_ids = np.asarray(benzene["ligand_atoms"])
protein_com = np.average(xyz_nm[protein_ids], axis=0, weights=masses[protein_ids])
ligand_com = np.average(xyz_nm[ligand_ids], axis=0, weights=masses[ligand_ids])
origin = xyz_nm[ligand_ids].mean(axis=0)
delta = xyz_nm[protein_ids] - origin
n_directions = 200
index = np.arange(n_directions)
golden = np.pi * (3.0 - np.sqrt(5.0))
height = 1.0 - 2.0 * (index + 0.5) / n_directions
radius = np.sqrt(1.0 - height * height)
directions = np.stack(
    [
        radius * np.cos(golden * index),
        radius * np.sin(golden * index),
        height,
    ],
    axis=1,
)
clearance = np.empty(n_directions)
for i, direction in enumerate(directions):
    along = delta @ direction
    perpendicular = np.linalg.norm(delta - along[:, None] * direction, axis=1)
    # a protein atom within 3 Å of the ray, and not on top of the ligand
    blocked = (perpendicular < 0.3) & (along > 0.1)
    clearance[i] = along[blocked].min() if blocked.any() else np.inf
u = directions[int(np.argmax(clearance))]
r_init = float(np.dot(ligand_com - protein_com, u))
projection = (
    f"({u[0]:.8f}) * (x1 - x2) + ({u[1]:.8f}) * (y1 - y2) + ({u[2]:.8f}) * (z1 - z2)"
)
cv = mm.CustomCentroidBondForce(2, projection)
cv.addBond(
    [
        cv.addGroup(ligand_ids.tolist()),
        cv.addGroup(protein_ids.tolist()),
    ]
)
k_pull = 8000.0  # kJ/mol/nm^2
pulling = mm.CustomCVForce("0.5 * k * (r - r0)^2")
pulling.addGlobalParameter("k", k_pull)
pulling.addGlobalParameter("r0", r_init)
pulling.addCollectiveVariable("r", cv)
pulling.setForceGroup(mixed.getNumForces())
mixed.addForce(pulling)
print(f"pull axis = [{u[0]: .3f}, {u[1]: .3f}, {u[2]: .3f}]")
print(f"clearance along that axis: {np.max(clearance) * 10:.2f} A")
print(f"bound projection r0 = {r_init:.3f} nm, k = {k_pull:.0f} kJ/mol/nm^2")

dt = 0.5 * unit.femtoseconds
n_pull = 1600
increment = 5
v_pull = 3.0  # nm/ps
dt_ps = dt.value_in_unit(unit.picosecond)
pull_integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, dt
)
pull_integrator.setConstraintTolerance(1e-5)
pulled = app.Simulation(benzene["topology"], mixed, pull_integrator, CPU)
pulled.context.setPositions(nve_positions)
pulled.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)

r0 = r_init
pull_times = []
pull_r = []
pull_r0 = []
contact_times = []
contacts = []
pull_frames = []
for step in range(0, n_pull + 1, increment):
    if step:
        pulled.step(increment)
        r0 += v_pull * dt_ps * increment
        pulled.context.setParameter("r0", r0)
    pull_times.append(step * dt_ps)
    pull_r.append(pulling.getCollectiveVariableValues(pulled.context)[0])
    pull_r0.append(r0)
    if step % 50 == 0:
        state = pulled.context.getState(getPositions=True)
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        contacts.append(
            np.linalg.norm(
                xyz[ligand_ids][:, None, :] - xyz[protein_ids][None, :, :],
                axis=-1,
            ).min()
        )
        contact_times.append(step * dt_ps)
        if step % 200 == 0:
            pull_frames.append(to_atoms(benzene["topology"], state.getPositions()))

print(
    f"r0 moved from {r_init:.3f} nm to {r0:.3f} nm; "
    f"projection followed to {pull_r[-1]:.3f} nm"
)
print(f"closest contact at the end of the pull: {contacts[-1]:.2f} A")
spring = pulled.context.getState(getEnergy=True, groups={pulling.getForceGroup()})
print(
    "pulling restraint "
    f"{spring.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.2f} kJ/mol"
)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
axes[0].plot(np.asarray(pull_times) * 1000.0, pull_r0, label=r"$r_0$")
axes[0].plot(np.asarray(pull_times) * 1000.0, pull_r, label=r"$r$")
axes[0].set_xlabel("time / fs")
axes[0].set_ylabel("projection / nm")
axes[0].legend()
axes[1].plot(np.asarray(contact_times) * 1000.0, contacts)
axes[1].set_xlabel("time / fs")
axes[1].set_ylabel("closest contact / Å")

chemiscope.show(
    pull_frames,
    mode="structure",
    settings={
        "structure": [{"bonds": True, "keepOrientation": True, "playbackDelay": 200}]
    },
)

# %%
# Solvated interaction energy
# ---------------------------
#
# Solvent is added to the last vacuum frame of the benzene trajectory, and
# to the deposited o-xylene coordinates. ``addSolvent`` neutralizes the
# box. Nothing is integrated after this: each number is one configuration.
# Benzene's closest contact in the workshop file is about 2.1 Å. The
# o-xylene file is nearer 1 Å, a steric clash, so that interaction energy
# is the clash rather than a binding enthalpy.
#
# Both ligands are neutral, so deleting the ligand does not change the
# charge of the box. The Amber and mechanical evaluations share that box.
# The metatomic ligand energy is the isolated ligand, which is also how
# the mixed system evaluates the ligand region. Because that energy
# cancels in :math:`E_\\mathrm{int}`, the mechanical and Amber interaction
# energies are two ways of computing the same classical coupling.

solvated_kwargs = dict(
    nonbondedMethod=app.PME,
    constraints=app.HBonds,
    rigidWater=True,
    removeCMMotion=False,
)
rows = []
for name in ("benzene", "o-xylene"):
    entry = systems[name]
    positions = nve_positions if name == "benzene" else entry["positions"]
    solvated = app.Modeller(entry["topology"], positions)
    solvated.addSolvent(forcefield, model="tip3p", padding=1.0 * unit.nanometer)
    print(
        f"{name}: {entry['topology'].getNumAtoms()} atoms before solvent, "
        f"{solvated.topology.getNumAtoms()} after"
    )

    mm_complex = forcefield.createSystem(solvated.topology, **solvated_kwargs)
    mixed_complex = potential.createMixedSystem(
        solvated.topology,
        mm_complex,
        entry["ligand_atoms"],
        embedding="mechanical",
        removeConstraints=True,
    )
    ligand_positions = [positions[i] for i in entry["ligand_atoms"]]
    ligand_ml = potential.createSystem(entry["ligand_topology"])
    ligand_box = app.Modeller(entry["ligand_topology"], ligand_positions)
    ligand_box.topology.setPeriodicBoxVectors(solvated.topology.getPeriodicBoxVectors())
    mm_ligand = forcefield.createSystem(ligand_box.topology, **solvated_kwargs)

    receptor = app.Modeller(solvated.topology, solvated.positions)
    ligand_index = set(entry["ligand_atoms"])
    receptor.delete(
        [atom for atom in receptor.topology.atoms() if atom.index in ligand_index]
    )
    mm_receptor = forcefield.createSystem(receptor.topology, **solvated_kwargs)

    e_mixed = potential_energy(mixed_complex, solvated.positions)
    e_mm = potential_energy(mm_complex, solvated.positions)
    e_receptor = potential_energy(mm_receptor, receptor.positions)
    e_ml = potential_energy(ligand_ml, ligand_positions)
    e_mm_ligand = potential_energy(mm_ligand, ligand_box.positions)
    mechanical = e_mixed - e_receptor - e_ml
    amber = e_mm - e_receptor - e_mm_ligand
    print(f"  mechanical {mechanical:10.2f} kJ/mol    Amber {amber:10.2f} kJ/mol")
    rows.append({"ligand": name, "mechanical": mechanical, "amber": amber})

fig, axes = plt.subplots(1, len(rows), figsize=(7.2, 3.2), constrained_layout=True)
for axis, row in zip(axes, rows):
    axis.bar(["Amber", "mechanical"], [row["amber"], row["mechanical"]])
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.set_title(row["ligand"])
    axis.set_ylabel("interaction energy / kJ/mol")

# %%
# References
# ----------
#
# .. [1] Eriksson, Baase, Wozniak, and Matthews, Nature 355, 371 (1992).
#    `DOI:10.1038/355371a0 <https://doi.org/10.1038/355371a0>`_
# .. [2] Morton, Baase, and Matthews, Biochemistry 34, 8564 (1995).
#    `DOI:10.1021/bi00027a006 <https://doi.org/10.1021/bi00027a006>`_
# .. [3] Deng and Roux, J. Chem. Theory Comput. 2, 1255 (2006).
#    `DOI:10.1021/ct060037v <https://doi.org/10.1021/ct060037v>`_
# .. [4] Mobley et al., J. Mol. Biol. 371, 1118 (2007).
#    `DOI:10.1016/j.jmb.2007.06.002 <https://doi.org/10.1016/j.jmb.2007.06.002>`_
# .. [5] Maseras and Morokuma, J. Comput. Chem. 16, 1170 (1995).
#    `DOI:10.1002/jcc.540160911 <https://doi.org/10.1002/jcc.540160911>`_
# .. [6] Svensson et al., J. Phys. Chem. 100, 19357 (1996).
#    `DOI:10.1021/jp962071j <https://doi.org/10.1021/jp962071j>`_
# .. [7] Bakowies and Thiel, J. Phys. Chem. 100, 10580 (1996).
#    `DOI:10.1021/jp9536514 <https://doi.org/10.1021/jp9536514>`_
# .. [8] Eastman et al., Sci. Data 10, 11 (2023).
#    `DOI:10.1038/s41597-022-01882-6 <https://doi.org/10.1038/s41597-022-01882-6>`_
# .. [9] Park, Khalili-Araghi, Tajkhorshid, and Schulten, J. Chem. Phys.
#    119, 3559 (2003).
#    `DOI:10.1063/1.1590311 <https://doi.org/10.1063/1.1590311>`_
# .. [10] Grubmüller, Heymann, and Tavan, Science 271, 997 (1996).
#    `DOI:10.1126/science.271.5251.997 <https://doi.org/10.1126/science.271.5251.997>`_
