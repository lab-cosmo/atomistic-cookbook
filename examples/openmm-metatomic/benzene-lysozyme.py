"""
ML-MM simulations with OpenMM & metatomic
=========================================

:Authors: Eric Boittier `@EricBoittier <https://github.com/EricBoittier>`_

In an ML-MM simulation, a small part of the system uses a machine-learning
(ML) potential and the rest uses a classical force field (molecular mechanics,
MM). The small part is where the chemistry of interest happens, for example a
ligand in a binding pocket. The ML potential is trained on quantum-chemistry
data, so it describes this part better than the force field. The force field
keeps the cost of the large environment low.

This recipe runs an ML-MM calculation with `OpenMM <https://openmm.org>`_ and
`OpenMM-ML <https://github.com/openmm/openmm-ml>`_. The system is T4 lysozyme
L99A, a protein with a small hydrophobic cavity that is often used to test
ligand binding. The cavity holds benzene or o-xylene. The protein uses the
Amber ff14SB force field. The ligand uses PET-SPICE-S, a metatomic model
trained on the SPICE dataset of small molecules.

The two parts are coupled with *mechanical embedding*, the simplest ML-MM
scheme. The force field describes the whole system. The ML energy of the
ligand is added, and the force-field energy of the ligand is removed so that
it is not counted twice [1]_:

.. math::

    E = E_\\text{MM}(\\text{all})
      + E_\\text{ML}(\\text{ligand})
      - E_\\text{MM}(\\text{ligand}).

The interaction between the ligand and the protein stays in the force field,
so the ML model does not see the charges of the protein. This is fast, but
less accurate than electrostatic embedding, where it does.

.. [1] Bakowies and Thiel, J. Phys. Chem. 100, 10580 (1996).
   `DOI:10.1021/jp9536514 <https://doi.org/10.1021/jp9536514>`_
"""

# sphinx_gallery_thumbnail_path = '../../examples/openmm-metatomic/mechanical-embedding.png'

# %%
# Structures and force field
# --------------------------
#
# The coordinates of T4 lysozyme L99A and of the two ligands come from the
# OpenMM workshop. Amber ff14SB has parameters for proteins, but not for
# arbitrary small molecules. The ligand parameters come from the `Open Force
# Field <https://openforcefield.org>`_ (OpenFF) 2.2.1 force field, which assigns
# parameters from chemical patterns instead of atom types, so it covers most
# drug-like molecules. ``openmmforcefields`` adds them to the OpenMM force field.
# Mechanical embedding still needs them: the protein-ligand interaction uses the
# ligand charges and Lennard-Jones parameters.
#
# OpenFF 2.2.1 expects AM1-BCC partial charges. We assign them with NAGL, a graph
# neural network trained to reproduce AM1-BCC charges, which is faster than the
# semi-empirical calculation.

import math
import warnings

import chemiscope
import MDAnalysis as mda
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch
import upet
import vesin.metatomic
from atomistic_cookbook_utils import download_with_retry
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
    pick_device,
    pick_output,
)
from openff.toolkit import Molecule
from openff.toolkit import Topology as OffTopology
from openff.units.openmm import to_openmm
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmml import MLPotential
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

CPU = mm.Platform.getPlatformByName("CPU")

WORKSHOP = "https://raw.githubusercontent.com/openmm/openmm_workshops/main/section_1"
for filename in ("lysozyme.pdb", "benzene.sdf", "o-xylene.sdf"):
    download_with_retry(f"{WORKSHOP}/{filename}", filename)

protein_pdb = app.PDBFile("lysozyme.pdb")
n_protein = protein_pdb.topology.getNumAtoms()

molecules = [Molecule.from_file(name) for name in ("benzene.sdf", "o-xylene.sdf")]
for molecule in molecules:
    molecule.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
warnings.filterwarnings("ignore", message="Preset charges were provided")
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
forcefield.registerTemplateGenerator(
    SMIRNOFFTemplateGenerator(molecules=molecules, forcefield="openff-2.2.1").generator
)

names = ("benzene", "o-xylene")
systems = {}
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

# %%
# MDAnalysis reads the OpenMM ``Modeller`` directly, including residues and
# chains, so chemiscope can draw the protein as a cartoon. Both ligands sit in
# the same cavity; ``modeller`` holds the last complex, o-xylene.

chemiscope.show(
    mda.Universe(modeller).atoms,
    mode="structure",
    settings=chemiscope.quick_settings(structure_settings={"cartoon": True}),
)

# %%
# Metatomic potential
# -------------------
#
# OpenMM-ML does not include a metatomic backend, so we define one here. Any
# package can add a backend to OpenMM-ML under a name of its choice.
# ``MetatomicPotentialImpl`` loads a metatomic model and adds it
# to the OpenMM system as a ``PythonForce``. OpenMM calls this force at every
# step, and it returns the ML energy and forces of the selected atoms.
#
# .. raw:: html
#
#    <details>
#    <summary>Click to see the metatomic backend for OpenMM-ML</summary>

# System-level inputs PET models request, with the values used here.
SYSTEM_INPUTS = {"charge": (0.0, "e"), "spin_multiplicity": (1.0, "")}
# Neighbor-list skin in nm (2 A).
NL_SKIN = 0.2


class MetatomicPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, model, device=None):
        return MetatomicPotentialImpl(model, device)


class MetatomicPotentialImpl(MLPotentialImpl):
    """A metatomic model, evaluated as one ``PythonForce``."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def getMLLongRange(self):
        return False

    def addForces(self, topology, system, atoms, forceGroup, **args):
        if args:
            raise TypeError(f"unsupported arguments: {', '.join(sorted(args))}")

        indices = None if atoms is None else list(atoms)
        topology_atoms = list(topology.atoms())
        if indices is not None:
            topology_atoms = [topology_atoms[i] for i in indices]

        model = load_atomistic_model(self.model)
        capabilities = model.capabilities()
        if indices is not None and math.isinf(capabilities.interaction_range):
            raise ValueError(
                "this model has an infinite interaction range and cannot be "
                "applied to a subset of atoms"
            )
        energy_key = pick_output("energy", capabilities.outputs)

        numbers = [atom.element.atomic_number for atom in topology_atoms]
        missing = sorted(set(numbers) - set(capabilities.atomic_types))
        if missing:
            raise ValueError(f"the model does not support atomic types {missing}")

        desired = None if self.device is None else str(self.device)
        device = torch.device(pick_device(capabilities.supported_devices, desired))
        dtype = getattr(torch, capabilities.dtype)
        model = model.to(device=device)
        types = torch.tensor(numbers, dtype=torch.int32, device=device)

        # A subset of atoms (the ML region) is evaluated as an isolated molecule.
        periodic = indices is None and (
            topology.getPeriodicBoxVectors() is not None
            or system.usesPeriodicBoundaryConditions()
        )
        pbc = torch.tensor([periodic] * 3, dtype=torch.bool, device=device)

        neighbor_lists = [
            vesin.metatomic.NeighborList(
                options=options, length_unit="nm", check_consistency=False, skin=NL_SKIN
            )
            for options in model.requested_neighbor_lists()
        ]
        extras = system_inputs(
            model.requested_inputs(use_new_names=True), dtype, device
        )
        options = ModelEvaluationOptions(
            length_unit="nm",
            outputs={energy_key: ModelOutput(unit="kJ/mol", sample_kind="system")},
        )
        force = mm.PythonForce(
            MetatomicForce(
                model, types, pbc, extras, neighbor_lists, options, dtype, energy_key
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        if indices is not None:
            force.setParticles(indices)
        system.addForce(force)


def system_inputs(requested, dtype, device):
    """Charge and spin, the only extra inputs this example fills in."""
    extras = {}
    for name, option in requested.items():
        if name not in SYSTEM_INPUTS or option.sample_kind != "system":
            raise ValueError(f"the model requests '{name}', which is not provided")
        value, input_unit = SYSTEM_INPUTS[name]
        block = TensorBlock(
            values=torch.tensor([[value]], dtype=dtype),
            samples=Labels(["system"], torch.zeros((1, 1), dtype=torch.int32)),
            components=[],
            properties=Labels([name], torch.tensor([[0]])),
        )
        tensor = TensorMap(Labels(["_"], torch.tensor([[0]])), [block])
        tensor.set_info("unit", input_unit)
        extras[name] = tensor.to(dtype=dtype, device=device)
    return extras


class MetatomicForce:
    """``PythonForce`` callback: energy in kJ/mol and forces in kJ/mol/nm."""

    def __init__(
        self, model, types, pbc, extras, neighbor_lists, options, dtype, energy_key
    ):
        self.model = model
        self.types = types
        self.pbc = pbc
        self.extras = extras
        self.neighbor_lists = neighbor_lists
        self.options = options
        self.dtype = dtype
        self.energy_key = energy_key

    def __call__(self, state):
        def tensor(quantity):
            value = np.asarray(quantity.value_in_unit(unit.nanometer), dtype=np.float64)
            return torch.tensor(value, dtype=self.dtype, device=self.types.device)

        positions = tensor(state.getPositions(asNumpy=True)).requires_grad_(True)
        if bool(self.pbc.any()):
            cell = tensor(state.getPeriodicBoxVectors(asNumpy=True))
        else:
            cell = torch.zeros((3, 3), dtype=self.dtype, device=self.types.device)

        structure = System(self.types, positions, cell, self.pbc)
        for name, value in self.extras.items():
            structure.add_data(name, value)
        for neighbors in self.neighbor_lists:
            neighbors.add_neighbor_list(systems=[structure], copy=False)

        outputs = self.model([structure], self.options, False)
        energy = outputs[self.energy_key].block().values.sum()
        energy.backward()
        forces = -structure.positions.grad.detach().cpu().numpy()
        return float(energy.detach()), forces


# %%
#
# .. raw:: html
#
#    </details>
#
# We register the backend under the name ``"metatomic"`` and download
# PET-SPICE-S with ``upet``. If an installed package already provides a
# ``"metatomic"`` backend (one is proposed in `openmm-ml#174
# <https://github.com/openmm/openmm-ml/pull/174>`_), the registration and the code
# above are not needed.

MLPotential.registerImplFactory("metatomic", MetatomicPotentialImplFactory())
upet.save_upet(model="pet-spice", size="s", version="0.2.0", output="pet-spice-s.pt")
potential = MLPotential("metatomic", model="pet-spice-s.pt", device="cpu")

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
