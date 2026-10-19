"""
Mechanical embedding
====================

:Authors: Eric Boittier `@EricBoittier <https://github.com/EricBoittier>`_

Mechanical embedding is a type of 'subtractive' scheme (literally, see the
equation below) used in QM/MM. A small region is described by a higher-level
model and the rest by a force field. The higher-level region is not polarized
by the surroundings. That speeds the calculation up at the cost of accuracy.
The interaction between the two regions stays in the force field, and the
force-field energy of the small region is subtracted so as not to
double-count it [1]_.

.. math::

    E = E_\\text{MM}(\\text{all})
      + E_\\text{ML}(\\text{region})
      - E_\\text{MM}(\\text{region}).

We use a familiar test system in small-molecule design. Here the small region
is a ligand in the T4 lysozyme L99A cavity, benzene or o-xylene, and the
higher-level model is PET-SPICE-S, a metatomic potential trained on SPICE.
Lysozyme is Amber ff14SB. This example includes ``openmm_metatomic`` and
registers it on the ``openmmml.potentials`` entry point before OpenMM-ML
is imported.

.. [1] Bakowies and Thiel, J. Phys. Chem. 100, 10580 (1996).
   `DOI:10.1021/jp9536514 <https://doi.org/10.1021/jp9536514>`_
"""

# sphinx_gallery_thumbnail_path = '../../examples/openmm-metatomic/mechanical-embedding.png'

# %%
# Structures
# ----------
#
# The coordinates are T4 lysozyme L99A from the OpenMM workshop, with
# benzene or o-xylene in the cavity. ``prepare.py`` registers the
# metatomic potential and exports PET-SPICE-S. The ligand charges and the
# valence terms Amber does not know come from OpenFF 2.2.1.

import sys
import warnings
from pathlib import Path

import ase
import chemiscope
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from ase.data import vdw_radii
from openff.toolkit import Molecule
from openff.toolkit import Topology as OffTopology
from openff.units.openmm import to_openmm
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

sys.path.insert(0, str(Path.cwd()))
from prepare import potential  # noqa: E402

CPU = mm.Platform.getPlatformByName("CPU")

# CPK colors for the ligand overlay. Anything else is gray.
_CPK = {1: 0xFFFFFF, 6: 0x909090, 7: 0x3050F8, 8: 0xFF0D0D}


def topology_to_atoms(topology, positions):
    """ASE atoms from an OpenMM topology, with positions in angstrom."""
    symbols = [atom.element.symbol for atom in topology.atoms()]
    if isinstance(positions, unit.Quantity):
        xyz = positions.value_in_unit(unit.angstrom)
    else:
        xyz = unit.Quantity(positions, unit.nanometer).value_in_unit(unit.angstrom)
    return ase.Atoms(symbols=symbols, positions=np.asarray(xyz))


def ligand_vdw_spheres(frames, ligand_atoms):
    """Chemiscope shape: van der Waals spheres on the ligand atoms."""
    structure = []
    for frame, atoms in zip(frames, ligand_atoms):
        numbers = frame.numbers[list(atoms)]
        structure.append(
            {
                "centers": frame.positions[list(atoms)].tolist(),
                "radii": [float(vdw_radii[z]) for z in numbers],
                "colors": [_CPK.get(int(z), 0x808080) for z in numbers],
            }
        )
    return {"ligand": {"kind": "spheres", "parameters": {"structure": structure}}}


protein_pdb = app.PDBFile("lysozyme.pdb")
n_protein = protein_pdb.topology.getNumAtoms()

molecules = [Molecule.from_file(name) for name in ("benzene.sdf", "o-xylene.sdf")]
# OpenFF 2.2.1 expects AM1-BCC charges. The NAGL model assigns them.
for molecule in molecules:
    molecule.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
warnings.filterwarnings("ignore", message="Preset charges were provided")
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
forcefield.registerTemplateGenerator(
    SMIRNOFFTemplateGenerator(molecules=molecules, forcefield="openff-2.2.1").generator
)

names = ("benzene", "o-xylene")
systems = {}
complex_frames = []
for name, molecule in zip(names, molecules):
    ligand_topology = OffTopology.from_molecules(molecules=[molecule]).to_openmm()
    ligand_positions = to_openmm(molecule.conformers[0])
    modeller = app.Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_topology, ligand_positions)
    ligand_atoms = list(range(n_protein, modeller.topology.getNumAtoms()))
    systems[name] = {
        "topology": modeller.topology,
        "positions": modeller.positions,
        "ligand_atoms": ligand_atoms,
    }
    frame = topology_to_atoms(modeller.topology, modeller.positions)
    complex_frames.append(frame)

chemiscope.show(
    complex_frames,
    properties={"ligand": list(names)},
    shapes=ligand_vdw_spheres(
        complex_frames, [systems[name]["ligand_atoms"] for name in names]
    ),
    mode="structure",
    settings={
        "structure": [{"bonds": True, "keepOrientation": True, "shape": "ligand"}]
    },
)

# %%
# Energies
# --------
#
# ``Context.getState`` can return the energy of a single force if that
# force is placed in its own force group. The metatomic contribution is
# the ligand. Bonded terms that lie entirely inside the ligand are
# removed, so the Amber terms that remain are the protein and the
# protein-ligand interaction.

vacuum = {
    "nonbondedMethod": app.NoCutoff,
    "constraints": app.HBonds,
    "rigidWater": True,
    "removeCMMotion": False,
}
for name in names:
    entry = systems[name]
    mm_system = forcefield.createSystem(entry["topology"], **vacuum)
    mixed = potential.createMixedSystem(
        entry["topology"],
        mm_system,
        entry["ligand_atoms"],
        embedding="mechanical",
        removeConstraints=True,
    )
    for index, force in enumerate(mixed.getForces()):
        force.setForceGroup(index)
    context = mm.Context(mixed, mm.VerletIntegrator(1.0 * unit.femtoseconds), CPU)
    context.setPositions(entry["positions"])
    print(name)
    for index, force in enumerate(mixed.getForces()):
        group = context.getState(getEnergy=True, groups={index})
        energy = group.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if isinstance(force, mm.PythonForce):
            label = "metatomic ligand"
        else:
            label = force.getName()
        print(f"  {label:24s} {energy:14.2f} kJ/mol")
    total = context.getState(getEnergy=True).getPotentialEnergy()
    total = total.value_in_unit(unit.kilojoule_per_mole)
    print(f"  {'total':24s} {total:14.2f} kJ/mol")
