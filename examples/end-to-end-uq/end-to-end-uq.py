"""
End-to-end uncertainty quantification
=====================================

:Authors: Filippo Bigi `@frostedoyster <https://github.com/frostedoyster>`_,
          Johannes Spies `@johannes-spies <https://github.com/johannes-spies>`_

This recipe joins three uncertainty quantification (UQ) workflows into a single
example:

1. Train a small baseline potential and wrap it with LLPR, including a shallow
   last-layer ensemble.
2. Use PET-MAD 1.5 ensemble predictions for static potential-energy
   uncertainties, including a vacancy-formation energy.
3. Propagate PET-MAD 1.5 ensemble uncertainty to an O-H radial distribution
   observable by trajectory reweighting.

The LLPR training section follows the workflow from the
`uq4ml tutorial <https://github.com/frostedoyster/uq4ml_tutorial>`_. The PET-MAD
sections use the current `UPET <https://github.com/lab-cosmo/upet>`_ interface,
which exposes PET-MAD 1.5 and its energy UQ methods.

Getting Started
---------------

At the bottom of the page, you'll find a ZIP file containing the whole example. It
includes the ``environment.yml`` file and the small metatrain option files needed to
execute the script.
"""

# %%
# Imports
# -------

import os

import ase.build
import ase.io
import ase.units
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.emt import EMT
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import LBFGS
from atomistic_cookbook_utils import run_command
from upet.calculator import UPETCalculator


# %%
# LLPR integration
# ----------------
#
# LLPR can be used to turn a baseline model into a model that returns analytical
# uncertainties and, optionally, a shallow last-layer ensemble. We first build the
# same kind of small aluminum dataset used in the LLPR tutorial: randomly distorted
# fcc cells evaluated with the EMT potential.

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

# %%
# The option files bundled with this recipe define a small PET model and the LLPR
# wrapper. The second command trains LLPR and samples a last-layer ensemble. These
# options are copied from the LLPR tutorial.
#
# .. code-block:: bash
#
#     mtt train options.yaml -o model.pt
#     mtt train options-llpr.yaml -o model-llpr.pt
#     mtt eval model-llpr.pt eval.yaml -b 20
#
# The example runs these commands directly. We then read ``output.xyz`` using the same
# workflow as the LLPR tutorial notebook.

run_command("mtt train options.yaml -o model.pt")
run_command("mtt train options-llpr.yaml -o model-llpr.pt")
run_command("mtt eval model-llpr.pt eval.yaml -b 20")

# %%
# We can now inspect LLPR and ensemble uncertainties for the held-out structures. The
# plot compares each uncertainty estimate with the absolute energy error.

reference_structures = ase.io.read("evaluation.xyz", ":")
evaluated_structures = ase.io.read("output.xyz", ":")

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
# Static PET-MAD 1.5 uncertainties
# --------------------------------
#
# For the rest of the recipe we use PET-MAD 1.5 through ``UPETCalculator``. The
# calculator downloads the selected checkpoint on first use and then caches it. The
# UQ-enabled PET-MAD 1.5 models expose both a scalar energy uncertainty and the full
# shallow ensemble.

pet_mad = UPETCalculator(model="pet-mad-xs", version="1.5.0", device="cpu")


def print_pet_mad_energy_summary(atoms, label: str) -> None:
    atoms = atoms.copy()
    atoms.calc = pet_mad
    energy = atoms.get_potential_energy()
    uncertainty = np.ravel(pet_mad.get_energy_uncertainty(atoms, per_atom=False))[0]
    ensemble = np.ravel(pet_mad.get_energy_ensemble(atoms, per_atom=False))
    print(
        f"{label}: {len(atoms)} atoms, "
        f"energy = {energy:.6f} eV, "
        f"uncertainty = {uncertainty:.6f} eV, "
        f"ensemble std = {ensemble.std(ddof=1):.6f} eV, "
        f"uncertainty/atom = {uncertainty / len(atoms):.6f} eV"
    )


water = ase.build.molecule("H2O")
water.center(vacuum=6.0)

al_bulk = ase.build.bulk("Al", "fcc", a=4.05, cubic=True)

distorted_al = al_bulk.copy()
distorted_al.rattle(0.15, seed=10)

print_pet_mad_energy_summary(water, "isolated water")
print_pet_mad_energy_summary(al_bulk, "Al fcc cell")
print_pet_mad_energy_summary(distorted_al, "distorted Al fcc cell")

# %%
# UQ on a vacancy-formation energy
# --------------------------------
#
# Derived quantities should be computed for each ensemble member before taking
# statistics. For a neutral vacancy in a one-component bulk material, the formation
# energy estimate is
#
# .. math::
#
#    E_f = E_{\\mathrm{vac}} - \\frac{N - 1}{N} E_{\\mathrm{bulk}}.
#
# We compute this for every PET-MAD ensemble member.

bulk = ase.build.bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2)
bulk.calc = pet_mad
num_atoms = len(bulk)
bulk_ensemble = np.ravel(pet_mad.get_energy_ensemble(bulk, per_atom=False))

vacancy = bulk.copy()
del vacancy[-1]
vacancy.calc = pet_mad
unrelaxed_vacancy_ensemble = np.ravel(
    pet_mad.get_energy_ensemble(vacancy, per_atom=False)
)

optimizer = LBFGS(vacancy, logfile=os.devnull)
optimizer.run(fmax=0.05, steps=20)
relaxed_vacancy_ensemble = np.ravel(
    pet_mad.get_energy_ensemble(vacancy, per_atom=False)
)

unrelaxed_formation = unrelaxed_vacancy_ensemble - (num_atoms - 1) / num_atoms * (
    bulk_ensemble
)
relaxed_formation = relaxed_vacancy_ensemble - (num_atoms - 1) / num_atoms * (
    bulk_ensemble
)

defect_energy_samples = [
    ("bulk energy", bulk_ensemble),
    ("unrelaxed vacancy energy", unrelaxed_vacancy_ensemble),
    ("relaxed vacancy energy", relaxed_vacancy_ensemble),
    ("unrelaxed vacancy formation energy", unrelaxed_formation),
    ("relaxed vacancy formation energy", relaxed_formation),
]

for name, values in defect_energy_samples:
    print(
        f"{name}: mean = {values.mean():.6f} eV, " f"std = {values.std(ddof=1):.6f} eV"
    )

# %%
# A compact plot helps compare the uncertainty scales of the directly predicted
# energies and the derived formation energies. We plot them in separate panels because
# the raw total energies and the formation energies live on very different scales.

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
defect_means = np.array([values.mean() for _, values in defect_energy_samples])
defect_stds = np.array([values.std(ddof=1) for _, values in defect_energy_samples])

for ax, selection, title in [
    (axes[0], slice(0, 3), "Total energies"),
    (axes[1], slice(3, 5), "Formation energies"),
]:
    x = np.arange(len(defect_energy_samples[selection]))
    ax.errorbar(
        x,
        defect_means[selection],
        yerr=defect_stds[selection],
        fmt="o",
        capsize=4,
    )
    ax.set(
        xticks=x,
        xticklabels=[
            label.replace(" ", "\n") for label, _ in defect_energy_samples[selection]
        ],
        title=title,
    )
    ax.grid(axis="y")

fig.supylabel("Energy / eV")
fig.tight_layout()

# %%
# Trajectory reweighting for an O-H RDF
# -------------------------------------
#
# We now propagate ensemble uncertainty to a thermodynamic observable. The input
# ``water.xyz`` contains a liquid-water structure. The O-H RDF below is computed from
# all O-H distances using the minimum-image convention.
#
# We run the reference PET-MAD trajectory, evaluate the PET-MAD ensemble energy for
# each sampled configuration, and then reweight each configuration for every ensemble
# member.

temperature_K = 300.0
water_md = ase.io.read("water.xyz")
water_md.calc = pet_mad

MaxwellBoltzmannDistribution(water_md, temperature_K=temperature_K, rng=np.random)
Stationary(water_md)
ZeroRotation(water_md)

integrator = Bussi(
    water_md,
    timestep=0.5 * ase.units.fs,
    temperature_K=temperature_K,
    taut=10.0 * ase.units.fs,
)

rdf_edges = np.linspace(0.5, 4.5, 250)
rdf_centers = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
shell_volumes = 4.0 * np.pi / 3.0 * (rdf_edges[1:] ** 3 - rdf_edges[:-1] ** 3)
oxygen_indices = [i for i, atom in enumerate(water_md) if atom.symbol == "O"]
hydrogen_indices = [i for i, atom in enumerate(water_md) if atom.symbol == "H"]
hydrogen_density = len(hydrogen_indices) / water_md.get_volume()


def oh_distance_distribution(atoms) -> np.ndarray:
    distances = atoms.get_all_distances(mic=True)
    distances = distances[np.ix_(oxygen_indices, hydrogen_indices)].ravel()
    histogram, _ = np.histogram(distances, bins=rdf_edges)
    return histogram / (len(oxygen_indices) * hydrogen_density * shell_volumes)


energies = []
sampled_structures = []


def collect_sample() -> None:
    sampled_structures.append(water_md.copy())
    energies.append(water_md.get_potential_energy())


integrator.attach(collect_sample, interval=2)
integrator.run(1000)

energies = np.asarray(energies)
ensemble_energies = np.array(
    [np.ravel(pet_mad.get_energy_ensemble(atoms)) for atoms in sampled_structures]
)
oh_rdfs = np.array([oh_distance_distribution(atoms) for atoms in sampled_structures])

# %%
# The ensemble member :math:`m` assigns a Boltzmann reweighting factor to frame
# :math:`t` based on the energy difference between the member energy
# :math:`E_m(x_t)` and the sampled reference energy :math:`E_0(x_t)`.

beta = 1.0 / (ase.units.kB * temperature_K)
log_weights = -beta * (ensemble_energies - energies[:, None])
log_weights -= log_weights.max(axis=0, keepdims=True)
weights = np.exp(log_weights)
weights /= weights.sum(axis=0, keepdims=True)

reweighted_rdf_by_member = weights.T @ oh_rdfs

reference_rdf_mean = oh_rdfs.mean(axis=0)
reweighted_rdf_mean = reweighted_rdf_by_member.mean(axis=0)
reweighted_rdf_std = reweighted_rdf_by_member.std(axis=0, ddof=1)

# %%
# Finally, we plot the O-H RDF from all PET-MAD 1.5 ensemble members, and then the
# corresponding mean and standard-deviation band. The reweighted interval includes the
# effect that each ensemble member would sample the trajectory with slightly different
# Boltzmann probabilities.

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
member_ax, mean_ax = axes

for i, member_rdf in enumerate(reweighted_rdf_by_member):
    label = "Ensemble members" if i == 0 else None
    member_ax.plot(
        rdf_centers, member_rdf, color="tab:orange", alpha=0.08, lw=0.5, label=label
    )
member_ax.set(
    xlabel="O-H distance / angstrom",
    ylabel="Probability density",
    title="Reweighted ensemble members",
    ylim=(0.0, 3.0),
)
member_ax.grid()
member_ax.legend()

mean_ax.plot(rdf_centers, reweighted_rdf_mean, label="Reweighted ensemble", lw=1.5)
mean_ax.fill_between(
    rdf_centers,
    reweighted_rdf_mean - 3.0 * reweighted_rdf_std,
    reweighted_rdf_mean + 3.0 * reweighted_rdf_std,
    alpha=0.25,
    label="3 ensemble standard deviations",
)
mean_ax.set(
    xlabel="O-H distance / angstrom",
    title="Ensemble mean and spread",
)
mean_ax.grid()
mean_ax.legend()
fig.tight_layout()

# %%
