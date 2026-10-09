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
from ase.data import vdw_radii
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


def ligand_spheres(frames, ligand_atoms):
    """van der Waals spheres on the ligand, drawn over a ball-and-stick protein."""
    colors = {1: 0xFFFFFF, 6: 0x909090, 7: 0x3050F8, 8: 0xFF0D0D}
    structure = []
    for frame, atoms in zip(frames, ligand_atoms):
        numbers = frame.numbers[list(atoms)]
        structure.append(
            {
                "centers": frame.positions[list(atoms)].tolist(),
                "radii": [float(vdw_radii[z]) for z in numbers],
                "colors": [colors.get(int(z), 0xFF1493) for z in numbers],
            }
        )
    return {"ligand": {"kind": "spheres", "parameters": {"structure": structure}}}


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
    settings={
        "structure": [{"bonds": True, "keepOrientation": True, "spaceFilling": True}]
    },
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
    shapes=ligand_spheres(
        complex_frames,
        [systems[name]["ligand_atoms"] for name in ("benzene", "o-xylene")],
    ),
    mode="structure",
    settings={
        "structure": [{"bonds": True, "keepOrientation": True, "shape": "ligand"}]
    },
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
# ``app.HAngles`` constrains the H–X–H angle.
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
    shapes=ligand_spheres(md_frames, [benzene["ligand_atoms"]] * len(md_frames)),
    mode="structure",
    settings={
        "structure": [
            {
                "bonds": True,
                "keepOrientation": True,
                "playbackDelay": 200,
                "shape": "ligand",
            }
        ]
    },
)

# %%
# A custom force
# --------------
#
# ``CustomCVForce`` is a spring written on top of another force. The
# collective variable is the distance between the protein and benzene
# centers,
#
# .. math::
#
#     V = \tfrac{1}{2} k (r - r_0)^2.


cv = mm.CustomCentroidBondForce(2, "distance(g1,g2)")
cv.addBond(
    [
        cv.addGroup(list(range(n_protein))),
        cv.addGroup(benzene["ligand_atoms"]),
    ]
)
pulling = mm.CustomCVForce("0.5 * k * (r - r0)^2")
pulling.addGlobalParameter("k", 5000.0)
pulling.addGlobalParameter("r0", 0.0)
pulling.addCollectiveVariable("r", cv)
mixed.addForce(pulling)

pulled = mm.Context(mixed, mm.VerletIntegrator(1.0 * unit.femtoseconds), CPU)
pulled.setPositions(simulation.context.getState(getPositions=True).getPositions())
r = pulling.getCollectiveVariableValues(pulled)[0]
pulled.setParameter("r0", r)
print(f"centroid distance r = r0 = {r:.3f} nm")

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
