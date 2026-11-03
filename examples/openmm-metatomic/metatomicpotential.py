"""
Evaluate a metatomic model as an OpenMM-ML potential.

The model is a TorchScript file. Energy and forces are added as one
``PythonForce``. A subset of atoms, as in mechanical embedding, is
evaluated as an isolated molecule.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Eric D. Boittier, Guillaume Fraux

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
from collections.abc import Iterable

import numpy as np
import openmm
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

# System-level inputs PET models request, with the values used here.
_SYSTEM_INPUTS = {"charge": (0.0, "e"), "spin_multiplicity": (1.0, "")}
# Neighbor-list skin in nm (2 A). Set to -1 once vesin supports auto skin.
_NL_SKIN = 0.2


class MetatomicPotentialImplFactory(MLPotentialImplFactory):
    """Build :class:`MetatomicPotentialImpl` objects."""

    def createImpl(self, name, model, device=None):
        return MetatomicPotentialImpl(model, device)


class MetatomicPotentialImpl(MLPotentialImpl):
    """A metatomic TorchScript model, evaluated as one ``PythonForce``."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def getMLLongRange(self) -> bool:
        return False

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        **args,
    ):
        try:
            import torch
            from metatomic.torch import (
                ModelEvaluationOptions,
                ModelOutput,
                load_atomistic_model,
                pick_device,
                pick_output,
            )
        except ImportError as error:
            raise ImportError(
                "Failed to import metatomic. Install it with "
                "'pip install metatomic-torch'."
            ) from error

        if args:
            unexpected = ", ".join(sorted(args))
            raise TypeError(f"unsupported argument: {unexpected}")

        topology_atoms = list(topology.atoms())
        indices = None if atoms is None else list(atoms)
        if indices is not None:
            topology_atoms = [topology_atoms[i] for i in indices]
        if any(atom.element is None for atom in topology_atoms):
            raise ValueError("Every atom needs an element.")

        model = load_atomistic_model(self.model)
        capabilities = model.capabilities()
        if indices is not None and math.isinf(capabilities.interaction_range):
            raise ValueError(
                "this model has an infinite interaction range and cannot be "
                "applied to a subset of atoms"
            )
        energy_key = pick_output("energy", capabilities.outputs)

        numbers = [atom.element.atomic_number for atom in topology_atoms]
        allowed = set(capabilities.atomic_types)
        missing = sorted({z for z in numbers if z not in allowed})
        if missing:
            raise ValueError(f"the model does not support atomic types {missing}")

        desired = None if self.device is None else str(self.device)
        device = torch.device(pick_device(capabilities.supported_devices, desired))
        dtype = getattr(torch, capabilities.dtype)
        model = model.to(device=device)
        types = torch.tensor(numbers, dtype=torch.int32, device=device)

        if indices is None:
            periodic = (
                topology.getPeriodicBoxVectors() is not None
                or system.usesPeriodicBoundaryConditions()
            )
            pbc = torch.tensor(
                [periodic, periodic, periodic], dtype=torch.bool, device=device
            )
        else:
            # Mechanical embedding evaluates the subset as an isolated molecule.
            pbc = torch.zeros(3, dtype=torch.bool, device=device)

        neighbor_lists = _neighbor_lists(model)
        extras = _system_inputs(
            model.requested_inputs(use_new_names=True), dtype, device
        )
        options = ModelEvaluationOptions(
            length_unit="nm",
            outputs={energy_key: ModelOutput(unit="kJ/mol", sample_kind="system")},
        )
        force = openmm.PythonForce(
            _MetatomicForce(
                model, types, pbc, extras, neighbor_lists, options, dtype, energy_key
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(bool(pbc.any()))
        if indices is not None:
            force.setParticles(indices)
        system.addForce(force)


def _neighbor_lists(model):
    requested = list(model.requested_neighbor_lists())
    if not requested:
        return []
    try:
        import vesin.metatomic
    except ImportError as error:
        raise ImportError(
            "Failed to import vesin. Install it with 'pip install vesin'."
        ) from error
    return [
        vesin.metatomic.NeighborList(
            options=options,
            length_unit="nm",
            check_consistency=False,
            skin=_NL_SKIN,
        )
        for options in requested
    ]


def _system_inputs(requested, dtype, device):
    """Charge and spin, the only extra inputs this example fills in."""
    import torch
    from metatensor.torch import Labels, TensorBlock, TensorMap

    extras = {}
    for name, option in requested.items():
        if name not in _SYSTEM_INPUTS or option.sample_kind != "system":
            raise ValueError(
                f"the model requests '{name}' "
                f"(sample_kind={option.sample_kind!r}), which this example "
                "does not provide"
            )
        value, input_unit = _SYSTEM_INPUTS[name]
        block = TensorBlock(
            values=torch.tensor([[float(value)]], dtype=dtype),
            samples=Labels(["system"], torch.zeros((1, 1), dtype=torch.int32)),
            components=[],
            properties=Labels([name], torch.tensor([[0]])),
        )
        tensor = TensorMap(Labels(["_"], torch.tensor([[0]])), [block])
        tensor.set_info("unit", input_unit)
        extras[name] = tensor.to(dtype=dtype, device=device)
    return extras


class _MetatomicForce:
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
        import torch
        from metatomic.torch import System

        device = self.types.device
        positions = torch.tensor(
            np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer),
                dtype=np.float64,
            ),
            dtype=self.dtype,
            device=device,
        )
        if bool(self.pbc.any()):
            box = state.getPeriodicBoxVectors(asNumpy=True)
            cell = torch.tensor(
                np.asarray(box.value_in_unit(openmm.unit.nanometer), dtype=np.float64),
                dtype=self.dtype,
                device=device,
            )
        else:
            cell = torch.zeros((3, 3), dtype=self.dtype, device=device)

        positions.requires_grad_(True)
        structure = System(self.types, positions, cell, self.pbc)
        for name, tensor in self.extras.items():
            structure.add_data(name, tensor)
        if self.neighbor_lists:
            if structure.device.type not in ("cpu", "cuda"):
                structure = structure.to(device="cpu")
            for neighbors in self.neighbor_lists:
                neighbors.add_neighbor_list(systems=[structure], copy=False)
            if structure.device != device:
                structure = structure.to(device=device)

        outputs = self.model([structure], self.options, False)
        energy = outputs[self.energy_key].block().values.sum()
        energy.backward()
        gradient = structure.positions.grad
        if gradient is None:
            raise RuntimeError("the energy does not depend on the positions")
        forces = (-gradient).detach().cpu().numpy()
        return float(energy.detach()), forces
