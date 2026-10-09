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


# %%
# Ligands
# -------
#
# Benzene and o-xylene are the metatomic region. PET-SPICE sees only these
# atoms. Use the structure list to switch from one ligand to the other.

# We will print the number of atoms in the MM and ML systems for each ligand
# and the total number of atoms in the complex, and the number of bonds
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

# %%
# Switching the ligand potential
# ------------------------------
#
# ``interpolate=True`` adds ``lambda_interpolate``. At 0 the ligand is
# Amber. At 1 the Amber energy of the ligand is off and the metatomic
# model is on. The mix is linear, so the Amber piece left in the energy
# is :math:`E(\lambda) - E(1)`, and that piece is zero at
# :math:`\lambda = 1`. The coupling to the protein is not in the mix. It
# stays Amber at every lambda.

switched = potential.createMixedSystem(
    benzene["topology"],
    mm_vacuum,
    benzene["ligand_atoms"],
    embedding="mechanical",
    removeConstraints=True,
    interpolate=True,
)
switch = mm.Context(switched, mm.VerletIntegrator(1.0 * unit.femtoseconds), CPU)
switch.setPositions(benzene["positions"])
lambdas = np.linspace(0.0, 1.0, 5)
switched_energy = []
for lam in lambdas:
    switch.setParameter("lambda_interpolate", float(lam))
    switched_energy.append(
        switch.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
amber_left = np.asarray(switched_energy) - switched_energy[-1]
print(f"Amber ligand energy at lambda 0: {amber_left[0]:.2f} kJ/mol")
print(f"Amber ligand energy at lambda 1: {amber_left[-1]:.2f} kJ/mol")

fig, axis = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
axis.plot(lambdas, amber_left, marker="o")
axis.set_xlabel(r"$\lambda$")
axis.set_ylabel(r"$E(\lambda) - E(1)$ / kJ/mol")

# %%
# Constraints
# -----------
#
# ``constraints=app.HBonds`` replaces each bond to hydrogen with a SHAKE
# constraint. ``removeConstraints=True`` drops the ones inside the
# ligand, because those distances belong to the model. What remains is
# a short NVE segment, benzene only, in vacuum.

print(
    f"constraints: {mm_vacuum.getNumConstraints()} on the MM system, "
    f"{mixed.getNumConstraints()} after the ligand constraints are removed"
)
integrator = mm.VerletIntegrator(0.5 * unit.femtoseconds)
integrator.setConstraintTolerance(1e-5)
simulation = app.Simulation(benzene["topology"], mixed, integrator, CPU)
simulation.context.setPositions(benzene["positions"])
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
start = simulation.context.getState(getEnergy=True)
simulation.step(20)
end = simulation.context.getState(getEnergy=True)
drift = (end.getPotentialEnergy() + end.getKineticEnergy()) - (
    start.getPotentialEnergy() + start.getKineticEnergy()
)
print(
    f"total energy drift over 10 fs: {drift.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol"
)

# %%
# A custom force
# --------------
#
# We can use OpenMM's custom force module to introduce a ``CustomCVForce``. The
# collective variable (CV) is the distance between the protein and benzene
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
pulled.setPositions(benzene["positions"])
r = pulling.getCollectiveVariableValues(pulled)[0]
pulled.setParameter("r0", r)
print(f"centroid distance r = r0 = {r:.3f} nm")

# %%
# References
# ----------
#
# .. [7] Bakowies and Thiel, J. Phys. Chem. 100, 10580 (1996).
#    `DOI:10.1021/jp9536514 <https://doi.org/10.1021/jp9536514>`_
