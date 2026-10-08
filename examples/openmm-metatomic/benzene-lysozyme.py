"""
Ligand-protein interaction energies with OpenMM-ML
==================================================

:Authors: Eric Boittier `@EricBoittier <https://github.com/EricBoittier>`_

Benzene and o-xylene bound in the L99A cavity of T4 lysozyme are a standard
OpenMM / Amber protein-ligand example. The structures used here are the
prepared files from the `OpenMM workshop notebook
<https://github.com/openmm/openmm_workshops/blob/main/section_1/protein_ligand_complex.ipynb>`_.
The protein is described with Amber ff14SB and each ligand with OpenFF 2.2.1.

This recipe evaluates the interaction energy of each ligand with the protein
at that one geometry, and repeats the evaluation while changing how the
ligand is embedded:

- ``amber``: the whole complex is classical.
- each embedding reported by :meth:`MLPotential.getSupportedEmbeddings`
  (for metatomic models this is mechanical embedding): the ligand's internal
  energy comes from a metatomic model, and Amber computes the
  ligand-protein coupling.
- ``metatomic``: protein, ligand, and complex are all evaluated with the
  metatomic model. There is no classical coupling.

The interaction energy at a single configuration is

.. math::

    E_\\text{int} = E(\\text{complex}) - E(\\text{protein}) - E(\\text{ligand}).

That difference is the configurational contribution to the interaction
enthalpy. A thermodynamic enthalpy would average it over an ensemble; the
workshop's solvated trajectory is omitted here so the example stays short.
The same ``createMixedSystem`` call works after ``Modeller.addSolvent``.

.. warning::

    PET-SPICE is trained on small organic molecules, not on proteins. The
    energies below show how the interface is used. They are not a prediction
    of the experimental binding enthalpy. The complex is also in vacuum, so
    the absolute energies are large: the protein is charged and there is no
    solvent screening. Compare interaction energies, not raw totals.
"""

# %%
# Setup
# -----
#
# Download the workshop structures, export PET-SPICE-S to a metatomic
# ``.pt`` file, and load it as an OpenMM-ML potential.

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
from openmmml import MLPotential


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

potential = MLPotential(
    "metatomic",
    model=str(model_path),
    device="cpu",
    # A protein lies far outside the training set, so the per-atom
    # uncertainty check would warn on every evaluation.
    uncertaintyThreshold=None,
)
embeddings = potential.getSupportedEmbeddings()
print("Embeddings supported by this potential:", embeddings)

CPU = mm.Platform.getPlatformByName("CPU")


def potential_energy(system, positions):
    """Potential energy of one configuration, in kJ/mol."""
    context = mm.Context(system, mm.VerletIntegrator(0.001 * unit.picoseconds), CPU)
    context.setPositions(positions)
    energy = context.getState(getEnergy=True).getPotentialEnergy()
    return energy.value_in_unit(unit.kilojoule_per_mole)


# %%
# Classical protein
# -----------------
#
# The protein is the same for both ligands, so its Amber and metatomic
# energies are computed once. ``NoCutoff`` keeps every pair, which makes
# the later subtraction exact: there is no PME self-energy and no shifted
# cutoff to account for.

protein_pdb = app.PDBFile("lysozyme.pdb")
n_protein = protein_pdb.topology.getNumAtoms()
amber = app.ForceField("amber14-all.xml")
protein_mm_system = amber.createSystem(
    protein_pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=None,
)
protein_mm = potential_energy(protein_mm_system, protein_pdb.positions)
protein_ml = potential_energy(
    potential.createSystem(protein_pdb.topology),
    protein_pdb.positions,
)
print(f"Protein atoms: {n_protein}")
print(f"Protein Amber energy:     {protein_mm:12.1f} kJ/mol")
print(f"Protein metatomic energy: {protein_ml:12.1f} kJ/mol")

# %%
# One complex, several embeddings
# -------------------------------
#
# For each ligand we build the complex the way the workshop does: an OpenFF
# molecule is appended to the protein with ``Modeller``. The interaction
# energy uses the same coordinates for the complex, the protein, and the
# ligand, so bonded terms inside each molecule cancel.

ligands = (("benzene", "benzene.sdf"), ("o-xylene", "o-xylene.sdf"))
rows = []
frames = []

for ligand_name, ligand_file in ligands:
    ligand = Molecule.from_file(ligand_file)
    ligand_topology = OffTopology.from_molecules(molecules=[ligand]).to_openmm()
    ligand_positions = to_openmm(ligand.conformers[0])

    forcefield = app.ForceField("amber14-all.xml")
    forcefield.registerTemplateGenerator(
        SMIRNOFFTemplateGenerator(
            molecules=[ligand], forcefield="openff-2.2.1"
        ).generator
    )
    modeller = app.Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_topology, ligand_positions)
    assert modeller.topology.getNumAtoms() == (
        n_protein + ligand_topology.getNumAtoms()
    )
    ligand_atoms = list(range(n_protein, modeller.topology.getNumAtoms()))

    mm_kwargs = {"nonbondedMethod": app.NoCutoff, "constraints": None}
    complex_mm_system = forcefield.createSystem(modeller.topology, **mm_kwargs)
    ligand_mm_system = forcefield.createSystem(ligand_topology, **mm_kwargs)
    ligand_mm = potential_energy(ligand_mm_system, ligand_positions)
    ligand_ml = potential_energy(
        potential.createSystem(ligand_topology),
        ligand_positions,
    )

    descriptions = [("amber", complex_mm_system, ligand_mm, protein_mm)]
    for embedding in embeddings:
        mixed = potential.createMixedSystem(
            modeller.topology,
            forcefield.createSystem(modeller.topology, **mm_kwargs),
            ligand_atoms,
            embedding=embedding,
        )
        descriptions.append((embedding, mixed, ligand_ml, protein_mm))
    descriptions.append(
        (
            "metatomic",
            potential.createSystem(modeller.topology),
            ligand_ml,
            protein_ml,
        )
    )

    symbols = [atom.element.symbol for atom in modeller.topology.atoms()]
    if isinstance(modeller.positions, unit.Quantity):
        positions = modeller.positions.value_in_unit(unit.angstrom)
    else:
        positions = unit.Quantity(modeller.positions, unit.nanometer).value_in_unit(
            unit.angstrom
        )
    positions = np.asarray(positions, dtype=np.float64)
    protein_xyz = np.asarray(protein_pdb.positions.value_in_unit(unit.angstrom))
    ligand_xyz = np.asarray(ligand_positions.value_in_unit(unit.angstrom))
    contact = np.linalg.norm(
        ligand_xyz[:, None, :] - protein_xyz[None, :, :], axis=-1
    ).min()
    print(
        f"\n{ligand_name} ({len(ligand_atoms)} atoms, closest contact {contact:.2f} A)"
    )
    for name, system, ligand_energy, protein_energy in descriptions:
        total = potential_energy(system, modeller.positions)
        interaction = total - protein_energy - ligand_energy
        print(
            f"  {name:12s}  interaction {interaction:10.2f} kJ/mol"
            f"   ligand {ligand_energy:10.2f} kJ/mol"
        )
        rows.append(
            {
                "ligand": ligand_name,
                "embedding": name,
                "interaction": interaction,
                "ligand_internal": ligand_energy,
                "total": total,
            }
        )
        frame = ase.Atoms(symbols=symbols, positions=positions)
        frame.info["ligand"] = ligand_name
        frame.info["embedding"] = name
        frames.append(frame)

# %%
# Interaction energies
# --------------------
#
# Mechanical embedding replaces the ligand's internal Amber energy with the
# metatomic energy and leaves the coupling classical, so the interaction
# energy matches Amber. The fully metatomic number is a supermolecule
# difference on the model's own energy zero.
#
# Benzene sits in the cavity (closest contact about 2.1 Å) and the Amber
# interaction is favorable. The o-xylene file from the workshop is a second
# crystal pose; a contact near 1 Å makes the Amber interaction a steric
# clash, and mechanical embedding reports that same clash. PET-SPICE is
# single precision and the total energies are ~10^8 kJ/mol, so the
# metatomic difference is only reliable to a few tens of kJ/mol. Benzene's
# metatomic interaction is therefore lost in the cancellation, while the
# o-xylene clash is large enough to remain.

ligand_names = list(dict.fromkeys(row["ligand"] for row in rows))
fig, axes = plt.subplots(
    1, len(ligand_names), figsize=(8.2, 3.4), constrained_layout=True
)
for axis, ligand in zip(axes, ligand_names):
    subset = [row for row in rows if row["ligand"] == ligand]
    axis.bar(
        [row["embedding"] for row in subset],
        [row["interaction"] for row in subset],
    )
    axis.axhline(0.0, color="0.5", linewidth=0.8)
    axis.set_title(ligand)
    axis.set_ylabel("interaction energy / kJ/mol")

# %%
# The same numbers on the structures
# ----------------------------------
#
# Each point is one ligand and one embedding. The horizontal axis is the
# interaction energy and the vertical axis is the ligand internal energy, so
# changing the embedding moves the point even when the coordinates do not.
# Switching structures keeps the view fixed on the binding pose.

properties = {
    "ligand": [row["ligand"] for row in rows],
    "embedding": [row["embedding"] for row in rows],
    "interaction": {
        "target": "structure",
        "values": np.array([row["interaction"] for row in rows]),
        "units": "kJ/mol",
    },
    "ligand_internal": {
        "target": "structure",
        "values": np.array([row["ligand_internal"] for row in rows]),
        "units": "kJ/mol",
    },
    "total": {
        "target": "structure",
        "values": np.array([row["total"] for row in rows]),
        "units": "kJ/mol",
    },
}

chemiscope.show(
    frames,
    properties=properties,
    settings=chemiscope.quick_settings(
        x="interaction",
        y="ligand_internal",
        map_color="interaction",
        symbol="embedding",
        structure_settings={"bonds": True, "keepOrientation": True},
    ),
)
