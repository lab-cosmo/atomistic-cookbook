from io import StringIO
from pathlib import Path

import ase.build
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.emt import EMT
from atomistic_cookbook_utils import run_command

calculator = EMT()
structure = ase.build.bulk("Al", "fcc", cubic=True)

structures = []
for i in range(1000):
    atoms = structure.copy()
    atoms.rattle(0.3, seed=i)
    atoms.calc = calculator
    atoms.info["energy"] = atoms.get_potential_energy()
    atoms.arrays["forces"] = atoms.get_forces()
    atoms.info["stress"] = atoms.get_stress(voigt=False)
    atoms.calc = None
    structures.append(atoms)

ase.io.write("dataset.xyz", structures[:50])
ase.io.write("evaluation.xyz", structures[50:])

run_command("mtt train options.yaml -o model.pt")
run_command("mtt train options-llpr.yaml -o model-llpr.pt")
run_command("mtt eval model-llpr.pt eval.yaml -b 20 -e extensions")

# %%
# We can now inspect LLPR and ensemble uncertainties for the held-out structures. The
# plot compares each uncertainty estimate with the absolute energy error.

reference_structures = ase.io.read("evaluation.xyz", ":")

# ASE's extended-XYZ reader cannot parse ``::`` in per-atom property names,
# so temporarily encode the affected part of the Properties schema.
# Only the in-memory text is modified; output.xyz remains unchanged.

property_name = "mtt::aux::cutoff_stats"
encoded_property_name = "mttAuxCutoffStatsEncoded"

property_schema = f"{property_name}:R:2"
encoded_property_schema = f"{encoded_property_name}:R:2"

output_text = Path("output.xyz").read_text()

# Check for a collision before introducing the temporary name.
if encoded_property_name in output_text:
    raise ValueError(
        f"Temporary property name already exists: {encoded_property_name}"
    )

output_text = output_text.replace(
    property_schema,
    encoded_property_schema,
)

evaluated_structures = ase.io.read(
    StringIO(output_text),
    index=":",
    format="extxyz",
)

# Restore the original namespaced property name after parsing.
for atoms in evaluated_structures:
    if encoded_property_name in atoms.arrays:
        atoms.arrays[property_name] = atoms.arrays.pop(encoded_property_name)


predicted_energies = np.array(
    [atoms.get_potential_energy() for atoms in evaluated_structures]
)
true_energies = np.array(
    [atoms.get_potential_energy() for atoms in reference_structures]
)
errors = np.abs(predicted_energies - true_energies)
llpr_uncertainties = np.array(
    [atoms.info["energy_uncertainty"] for atoms in evaluated_structures]
)
ensemble_uncertainties = np.array(
    [atoms.info["energy_ensemble"].std() for atoms in evaluated_structures]
)


def positive_log_limits(array: np.ndarray) -> tuple[float, float]:
    values = np.ravel(array)
    values = values[np.isfinite(values) & (values > 0.0)]
    return values.min(), values.max()


quantile_lines = [0.00916, 0.10256, 0.4309805, 1.71796, 2.5348, 3.44388]
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

for ax, uncertainty, title, xlabel in [
    (
        axes[0],
        llpr_uncertainties,
        "LLPR",
        "Predicted energy uncertainty / eV",
    ),
    (
        axes[1],
        ensemble_uncertainties,
        "LLPR-derived ensemble",
        "Ensemble standard deviation / eV",
    ),
]:
    lower, upper = positive_log_limits(uncertainty)
    ax.plot([lower, upper], [lower, upper], "k--", lw=0.75)
    for factor in quantile_lines:
        ax.plot([lower, upper], [factor * lower, factor * upper], "k:", lw=0.75)
    ax.scatter(uncertainty, errors, s=10)
    ax.set(xscale="log", yscale="log", xlabel=xlabel, title=title)
    ax.grid()

axes[0].set_ylabel("Absolute energy error / eV")
fig.tight_layout()

# %%
# The exported LLPR model can also be used directly through the metatomic ASE
# calculator. Requesting ``energy_uncertainty`` returns the calibrated LLPR
# uncertainty, while requesting ``energy_ensemble`` returns all shallow-ensemble
# energies.

for i in range(5):
    print(
        f"structure {i:2d}: "
        f"energy = {predicted_energies[i]: .6f} eV, "
        f"LLPR uncertainty = {llpr_uncertainties[i]: .6f} eV, "
        f"ensemble std = {ensemble_uncertainties[i]: .6f} eV"
    )

# %%
# The exported model can be used directly as a drop-in calculator for any
# inference workflow. For an end-to-end demonstration — single-point uncertainty
# estimates on a validation dataset, vacancy formation energies with ensemble
# error bars, and uncertainty propagation through an MD trajectory — see the
# :doc:`Uncertainty Quantification with PET-MAD </examples/pet-mad-uq/pet-mad-uq>`
# example.
#
#
