"""
Running a legacy potential with TorchModel
==============================================

A saved ``metatomic.torch.AtomisticModel`` can run through the new
``metatomic.execute_model`` interface using ``LegacyModel``. Here we export a
small Lennard-Jones potential, reload it, and compare its energy, forces, and
stress with the legacy ASE engine. No trained-model download is needed.

.. warning::

    This recipe uses unreleased APIs. Its environment pins the wrapper branch;
    see the `wrapper PR
    <https://github.com/metatensor/metatomic/pull/360>`_ and the `TorchModel proposal
    <https://github.com/metatensor/metatomic/pull/354>`_. Until core's gradient
    unit conversion is fixed, use matching input and model length units.

The original model keeps its TorchScript implementation. The wrapper converts
containers while preserving the tensor autograd graph, allowing TorchModel to
compute energy derivatives inside the model interface.
"""

# %%
# Export and reload a model
# -----------------------------
# For an existing file, replace this section with
# ``model = LegacyModel.load("model.pt", extensions_directory="extensions")``.
# The extension directory is only needed when the model uses custom operators.

import tempfile
from pathlib import Path

import ase.build
import matplotlib.pyplot as plt
import metatomic
import metatomic_lj_test
import numpy as np
from metatomic_ase import MetatomicCalculator
from metatomic_torch.legacy import LegacyModel, system_with_pairs

legacy = metatomic_lj_test.lennard_jones_model(
    atomic_type=28,
    cutoff=5.0,
    sigma=1.5808,
    epsilon=0.1729,
    length_unit="angstrom",
    energy_unit="eV",
    with_extension=False,
)
with tempfile.TemporaryDirectory() as directory:
    path = str(Path(directory) / "lj.pt")
    legacy.save(path)
    model = LegacyModel.load(path)

# %%
# Build the system and request derivatives
# --------------------------------------------
# Pair-list cutoffs use the model's length unit. This example uses angstrom for
# both the model and the ASE coordinates. Rebuild pairs when coordinates change;
# an old neighbor list is not valid for arbitrary new geometries.
#
# The result list follows request order. A positions gradient is ``dE/dr``;
# forces have the opposite sign. A strain gradient is ``dE/d(strain)``;
# dividing by cell volume gives stress in ASE's sign convention.

atoms = ase.build.bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
atoms.positions += np.random.default_rng(42).normal(0, 0.03, atoms.positions.shape)
energy_request = metatomic.Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)


def evaluate(structure, gradients=True):
    request = (
        energy_request
        if gradients
        else metatomic.Quantity(name="energy", unit="eV", sample_kind="system")
    )
    system = system_with_pairs(
        structure.numbers,
        structure.positions,
        structure.cell[:],
        structure.pbc,
        model.requested_pair_lists([request]),
        length_unit="angstrom",
        model_length_unit=model.capabilities().length_unit,
    )
    (result,) = metatomic.execute_model(
        model, [system], [request], check_consistency=True
    )
    return result.block()


block = evaluate(atoms)
energy = np.asarray(block.values).item()
positions_gradient = block.gradient("positions")
# Use sample labels instead of assuming rows are ordered by atom index.
forces = np.empty((len(atoms), 3))
forces[positions_gradient.samples.column("atom")] = -np.asarray(
    positions_gradient.values
)[:, :, 0]
stress = np.asarray(block.gradient("strain").values)[0, :, :, 0] / atoms.get_volume()
print(f"Energy: {energy:.8f} eV")
print(f"Maximum force: {np.linalg.norm(forces, axis=1).max():.8f} eV/angstrom")

# %%
# Compare with the old engine
# -------------------------------
# In the legacy path, the ASE calculator runs the backward pass. In the new
# path, TorchModel has already added the requested gradients to the TensorMap.
# Both should describe the same potential energy surface.

atoms.calc = MetatomicCalculator(
    legacy, check_consistency=True, uncertainty_threshold=None
)
reference_forces = atoms.get_forces()
np.testing.assert_allclose(energy, atoms.get_potential_energy(), rtol=1e-10)
np.testing.assert_allclose(forces, reference_forces, rtol=1e-10, atol=1e-12)
np.testing.assert_allclose(
    stress, atoms.get_stress(voigt=False), rtol=1e-10, atol=1e-12
)
print("Energy, forces, and stress agree with the legacy ASE engine.")

# %%
# Independently check a force with finite differences
# -------------------------------------------------------
# Displace one coordinate and rebuild pairs on either side. Central differences
# should approach the autograd value before floating-point cancellation dominates.

steps = np.logspace(-2, -5, 7)
errors = []
for step in steps:
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[0, 0] += step
    minus.positions[0, 0] -= step
    derivative = (
        np.asarray(evaluate(plus, gradients=False).values).item()
        - np.asarray(evaluate(minus, gradients=False).values).item()
    ) / (2 * step)
    errors.append(abs(-derivative - forces[0, 0]))
np.testing.assert_allclose(-derivative, forces[0, 0], rtol=1e-5, atol=1e-8)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].scatter(reference_forces.ravel(), forces.ravel(), s=12)
limits = [reference_forces.min(), reference_forces.max()]
axes[0].plot(limits, limits, "k--", linewidth=1)
axes[0].set(xlabel="Legacy force (eV/angstrom)", ylabel="Wrapped force (eV/angstrom)")
axes[1].loglog(steps, errors, "o-")
axes[1].set(xlabel="Displacement (angstrom)", ylabel="Force error (eV/angstrom)")
fig.tight_layout()
plt.show()

# %%
# Explicit gradients and direct forces
# ----------------------------------------
# A legacy model's ``explicit_gradients`` become ``Quantity.gradients`` in the
# wrapper's capabilities. TorchModel preserves those derivatives and computes
# only the missing ones. For example, a model can supply analytical position
# gradients while autograd supplies strain gradients.
#
# ``non_conservative_force`` is a separate output, not an energy gradient.
# Requesting it does not replace ``-dE/dr``. Models which predict direct forces
# need a separate request, and such forces need not conserve energy in dynamics.
# This LJ recipe deliberately compares conservative energy derivatives.
#
# The bridge currently supports Python engines. Direct loading of legacy models
# by C++ engines requires separate plugin integration.
