"""
Uncertainty Quantification with PET-MAD
=======================================

:Authors: Johannes Spies `@johannes-spies <https://github.com/johannes-spies>`_,
          Filippo Bigi `@frostedoyster <https://github.com/frostedoyster>`_

This recipe demonstrates three ways of computing errors on the outputs of
ML potential-driven simulations, using as an example the PET-MAD model and its
built-in uncertainty quantification (UQ) capabilities.

In particular, it demonstrates:

1. Estimating uncertainties for single-point calculations on a
   full validation dataset.
2. Computing energies in simple functions of energy predictions,
   namely the value of vacancy formation energies
3. Propagating errors from energy predictions to thermodynamic averages
   computed over a constant-temperature MD simulation.


For more information on PET-MAD, have a look at
`Mazitov et al., 2025. <https://arxiv.org/abs/2503.14118>`_
The LLPR uncertainties are introduced in `Bigi et al., 2024.
<https://arxiv.org/abs/2403.02251>`_ For more
information on dataset calibration and error propagation, see
`Imabalzano et al., 2021. <https://arxiv.org/abs/2011.08828>`_

The PET-MAD model used here already includes LLPR and ensemble UQ. To train a
custom model with these capabilities from scratch using
`metatrain <https://metatensor.github.io/metatrain/>`_, see the
:doc:`Training a Model with UQ from Scratch </examples/end-to-end-uq/end-to-end-uq>`
example.

Getting Started
---------------

At the bottom of the page, you'll find a ZIP file containing the whole example. Note
that it comes with an `environment.yml` file specifying all dependencies required
to execute the script.
"""

# %%
import os
import subprocess

import upet
from atomistic_cookbook_utils import download_with_retry

import ase.geometry.rdf
import ase.units
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io.cif import read_cif
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize.bfgs import BFGS
from ipi.utils.scripting import InteractiveSimulation
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator
from metatrain.utils.data import Dataset, read_systems, read_targets
from metatrain.utils.data.system_to_ase import system_to_ase


# %%
# Model Loading
# -------------
# All examples require a PET-MAD model with ensemble and LLPR prediction.
# PET-MAD v1.5.0 includes built-in LLPR uncertainty quantification, so we
# can use it directly. Here we use the extra-small (xs) model for speed.
# For production calculations, the small (s) model (``size="s"``) is far
# more accurate and still very fast.
# We use the ``upet`` package to download the model, and load it using the
# ASE-compatible MetatomicCalculator wrapper, which conveniently hides computing
# neighbor lists in the calculator.

model_path = "models/pet-mad-xs-v1.5.0.pt"
os.makedirs("models", exist_ok=True)
upet.save_upet(model="pet-mad", size="xs", version="1.5.0", output=model_path)

calculator = MetatomicCalculator(model_path, device="cpu")

# %%
# Uncertainties on a Dataset
# ----------------------------------------------
# This first example shows how to use PET-MAD to estimate uncertainties on a reference
# dataset. We use a reduced subset (100 structures, because of limited compute power
# in the CI runner) of the MAD 1.5 validation set, which contains r2SCAN references
# matching the level of theory used to train PET-MAD v1.5.0.
#
# We download the full validation set from
# `Materials Cloud <https://archive.materialscloud.org/records/18tke-tt476>`_
# and extract 100 structures with a constant stride.
# Then, we prepare the dataset and pass it through the model. In the
# final step, we visualize the predicted uncertainties and compare them to a
# ground truth method.

if not os.path.exists("data/mad-val-100.xyz"):
    mad_val_full = "data/mad-1.5-r2scan-val.xyz"
    download_with_retry(
        "https://archive.materialscloud.org/records/18tke-tt476/"
        "files/mad-1.5-r2scan-val.xyz",
        mad_val_full,
    )
    # Extract 100 structures with constant stride from the full validation set
    full_dataset = ase.io.read(mad_val_full, index=":")
    stride = max(1, len(full_dataset) // 100)
    subset = full_dataset[::stride][:100]
    ase.io.write("data/mad-val-100.xyz", subset, format="extxyz")
    os.remove(mad_val_full)

# Read the dataset's structures.
systems = read_systems("data/mad-val-100.xyz")

# Read the dataset's targets.
target_config = {
    "energy": {
        "quantity": "energy",
        "read_from": "data/mad-val-100.xyz",
        "reader": "ase",
        "key": "atomization_energy",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_subtargets": 1,
        "forces": False,
        "stress": False,
        "virial": False,
    },
}
targets, infos = read_targets(target_config)  # type: ignore

# Wrap in a `metatrain` compatible way.
dataset = Dataset.from_dict({"system": systems, **targets})

# %%
# After preparation, the dataset can be passed through the model using the calculator
# to obtain energy predictions and LLPR scores.

# Convert the systems to an ASE-native `Atoms` object
systems = [system_to_ase(sample["system"]) for sample in dataset]
outputs = {
    # Request the uncertainty in the atomic energy predictions
    "energy": ModelOutput(),  # (Needed to request the uncertainties)
    "energy_uncertainty": ModelOutput(),
    "energy_ensemble": ModelOutput(),
}
results = calculator.run_model(systems, outputs)

# Extract the requested results
predicted_energies = results["energy"][0].values.squeeze()
predicted_uncertainties = results["energy_uncertainty"][0].values.squeeze()
ensemble_raw = results["energy_ensemble"][0].values
print(
    "energy_ensemble shape:", ensemble_raw.shape
)  # (n_structures, n_ensemble_members)
predicted_ensemble_std = ensemble_raw.std(dim=-1).squeeze()

# %%
# Compute the true prediction error by comparing the predicted energy to the reference
# value from dataset.

# Reference values from dataset.
ground_truth_energies = torch.stack(
    [sample["energy"][0].values.squeeze() for sample in dataset]
)

# Compute squared distance between predicted energy and reference value.
empirical_errors = torch.abs(predicted_energies - ground_truth_energies)

# %%
# After gathering predicted uncertainties and computing ground truth error metrics, we
# can compare them to each other. Similar to figure S4 of the PET-MAD paper, we present
# the data using a parity plot. The side-by-side panels compare the calibrated LLPR
# uncertainty (analytical) to the ensemble standard deviation. For more information
# about interpreting this type of plot, see Appendix F.7 of
# `Bigi et al., 2024 <https://arxiv.org/abs/2403.02251>`_.
# Because we are using a heavily reduced dataset (only 100 structures) from the MAD
# validation set, the parity plots look very sparse.

quantile_lines = [0.00916, 0.10256, 0.4309805, 1.71796, 2.5348, 3.44388]
min_val, max_val = 2.5e-2, 2.5

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
for ax, uncertainty, title, xlabel in [
    (axes[0], predicted_uncertainties, "LLPR", "Predicted energy uncertainty / eV"),
    (axes[1], predicted_ensemble_std, "Ensemble", "Ensemble standard deviation / eV"),
]:
    ax.set(xscale="log", yscale="log", title=title, xlabel=xlabel)
    ax.grid()
    ax.plot([min_val, max_val], [min_val, max_val], "k--")
    for factor in quantile_lines:
        ax.plot([min_val, max_val], [factor * min_val, factor * max_val], "k:", lw=0.75)
    ax.scatter(uncertainty, empirical_errors, s=10)

axes[0].set_ylabel("Absolute energy error / eV")
fig.tight_layout()

# %%
# Uncertainties in Vacancy Formation Energies
# -------------------------------------------
# One can use ensemble uncertainty quantification to estimate the error in predicting
# `vacancy formation <https://en.wikipedia.org/wiki/Vacancy_defect>`_
# energies, which we show in this example.
#
# In this part, we use an aluminum crystal as an example system. The structure file can
# be downloaded from
# `Material Project <https://legacy.materialsproject.org/materials/mp-134/>`_
# as a `.cif` file. We've included such a file with the recipe.
#
# The following code loads the structure, computes the energy before creating a defect,
# creates a defect, runs a structural optimization, and computes the energy after the
# optimization. The energy difference can be used to estimate the vacancy formation
# energy.

# Load the crystal from the Materials Project and create a supercell (not strictly
# necessary).
crystal_structure = "data/Al_mp-134_conventional_standard.cif"
atoms: Atoms = read_cif(crystal_structure)  # type: ignore
supercell = atoms * 2
supercell.calc = calculator
N = len(supercell)  # store the number of atoms

# %%
# We now compute the vacancy formation energy by keeping track of the ensemble energies
# at different stages. Note that calling `.get_potential_energy()` on an `Atoms` object
# triggers computing the ensemble values.

# Get ensemble energy before creating the vacancy
outputs = ["energy", "energy_ensemble"]
outputs = {o: ModelOutput() for o in outputs}
results = calculator.run_model(supercell, outputs)
bulk_ens = results["energy_ensemble"][0].values

# Remove an atom (last atom in this case) to create a vacancy
i = -1
supercell.pop(i)

# Get ensemble energy right after creating the vacancy
results = calculator.run_model(supercell, outputs)
right_after_vacancy_ens = results["energy_ensemble"][0].values

# Run structural optimization optimizing both positions and cell layout.
ecf = FrechetCellFilter(supercell)
bfgs = BFGS(ecf)  # type: ignore
bfgs.run()

# get ensembele energy after optimization
results = calculator.run_model(supercell, outputs)
vacancy_ens = results["energy_ensemble"][0].values

# %%
# Compute vacancy formation energy for each ensemble member.

vacancy_formation_ens = vacancy_ens - (N - 1) / N * bulk_ens
unrelaxed_formation_ens = right_after_vacancy_ens - (N - 1) / N * bulk_ens

# %%
# A compact plot helps compare the uncertainty scales of the directly predicted
# energies and the derived formation energies. We plot them in separate panels because
# the raw total energies and the formation energies live on very different scales.

defect_energy_samples = [
    ("Bulk energy", bulk_ens.detach().numpy().squeeze()),
    ("Unrelaxed vacancy energy", right_after_vacancy_ens.detach().numpy().squeeze()),
    ("Relaxed vacancy energy", vacancy_ens.detach().numpy().squeeze()),
    ("Unrelaxed VFE", unrelaxed_formation_ens.detach().numpy().squeeze()),
    ("Relaxed VFE", vacancy_formation_ens.detach().numpy().squeeze()),
]

for name, values in defect_energy_samples:
    print(f"{name}: mean = {values.mean():.6f} eV, std = {values.std(ddof=1):.6f} eV")

# %%

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
#
# Uncertainty Propagation with Python-based MD
# ---------------------------------------------
#
# We now propagate ensemble uncertainty to a thermodynamic observable. We run a short
# NVT simulation on a box of 32 water molecules, evaluate the ensemble energy for each
# sampled configuration, and reweight each frame for every ensemble member.
# As an observable we use the O-H `Radial Distribution Function
# (RDF) <https://en.wikipedia.org/wiki/Radial_distribution_function>`_.

temperature_K = 300.0
water_md = ase.io.read("data/h2o-32.xyz")
water_md.set_cell([9.865916, 9.865916, 9.865916])
water_md.set_pbc(True)
water_md.calc = calculator

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

# %%
# After the simulation, we evaluate the ensemble energy for each frame and compute
# the O-H RDF.

energies = np.asarray(energies)
ensemble_outputs = {"energy_ensemble": ModelOutput()}
ensemble_energies = np.array(
    [
        calculator.run_model(atoms, ensemble_outputs)["energy_ensemble"][0]
        .values.detach()
        .numpy()
        .squeeze()
        for atoms in sampled_structures
    ]
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

reweighted_rdf_mean = reweighted_rdf_by_member.mean(axis=0)
reweighted_rdf_std = reweighted_rdf_by_member.std(axis=0, ddof=1)

# %%

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
member_ax, mean_ax = axes

for i, member_rdf in enumerate(reweighted_rdf_by_member):
    label = "Ensemble members" if i == 0 else None
    member_ax.plot(
        rdf_centers, member_rdf, color="tab:orange", alpha=0.08, lw=0.5, label=label
    )
member_ax.set(
    xlabel="O-H distance / Å",
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
mean_ax.set(xlabel="O-H distance / Å", title="Ensemble mean and spread")
mean_ax.grid()
mean_ax.legend()
fig.tight_layout()

# %%
#
# Uncertainty Propagation with i-PI
# ----------------------------------
#
# The same propagation can be performed with `i-PI
# <https://ipi-code.org>`_, which supports more advanced MD
# algorithms and handles the reweighting as a post-processing step. In this
# example, we use a box with period boundary conditions housing 32 water molecules.
# As an observable, we inspect the `Radial Distribution
# Function (RDF) <https://en.wikipedia.org/wiki/Radial_distribution_function>`_ between
# hydrogen-hydrogen and oxygen-oxygen bonds.
#
# First, we run a simulation with i-PI generating a trajectory and logging other
# metrics. The trajectory and committee energies can be used in a subsequent
# postprocessing step to obtain RDFs using ASE. These can be re-weighted to propagate
# errors from the committee uncertainties to the observed RDFs.
#
# Note also that we set a `uncertainty_threshold` option in the driver. When running
# from the command line, this will output a warning every time one of the atomic energy
# is estimated to have an uncertainty above that threshold (in eV/atom).

# Load configuration and run simulation.
with open("data/h2o-32.xml") as f:
    xml_input = f.read()

# prints the relevant sections of the input file
print(xml_input[:883][-334:])

sim = InteractiveSimulation(xml_input)

# %%
# Run the simulation.

# NB: To get better estimates, set this to a higher number (perhaps 10000) to
# run the simulation for a longer time.
sim.run(400)

# %%
# Load the trajectories and compute the per-frame RDFs
# Note that ASE applies a weird normalization to the partial RDFs,
# which require a correction to recover the usual asymptotic
# behavior at large distances.

frames: list[Atoms] = ase.io.read("h2o-32.pos_0.extxyz", ":")  # type: ignore

# Our simulation should only include water molecules. (types: hydrogen=1, oxygen=8)
assert set(frames[0].numbers.tolist()) == set([1, 8])

# Compute the RDF of each frame (for H-H and for O-O)
num_bins = 250
rdfs_hh = []
rdfs_oo = []
xs = None
for atoms in frames:
    # Compute H-H distances
    bins, xs = ase.geometry.rdf.get_rdf(  # type: ignore
        atoms, 4.5, num_bins, elements=[1, 1]
    )

    # smoothen the RDF a bit (not enough data...)
    bins[2:-2] = (
        bins[:-4] * 0.1
        + bins[1:-3] * 0.2
        + bins[2:-2] * 0.4
        + bins[3:-1] * 0.2
        + bins[4:] * 0.1
    )
    rdfs_hh.append(bins)

    # Compute O-O distances
    bins, xs = ase.geometry.rdf.get_rdf(  # type: ignore
        atoms, 4.5, num_bins, elements=[8, 8]
    )
    bins[2:-2] = (
        bins[:-4] * 0.1
        + bins[1:-3] * 0.2
        + bins[2:-2] * 0.4
        + bins[3:-1] * 0.2
        + bins[4:] * 0.1
    )
    rdfs_oo.append(bins)

rdfs_hh = np.stack(rdfs_hh, axis=0)
rdfs_oo = np.stack(rdfs_oo, axis=0)

# %%
# Run the i-PI re-weighting utility as a post-processing step.

# Save RDFs such that they can be read from i-PI.
np.savetxt("h2o-32_rdfs_h-h.txt", rdfs_hh)
np.savetxt("h2o-32_rdfs_o-o.txt", rdfs_oo)

# Run the re-weighting tool from i-PI for H-H and O-O
for ty in ["h-h", "o-o"]:
    infile = f"h2o-32_rdfs_{ty}.txt"
    outfile = f"h2o-32_rdfs_{ty}_reweighted.txt"
    cmd = (
        f"i-pi-committee-reweight h2o-32.committee_pot_0 {infile} --input"
        " data/h2o-32.xml"
    )
    print("Executing command:", "\t" + cmd, sep="\n")
    cmd = cmd.split()
    with open(outfile, "w") as out:
        process = subprocess.run(cmd, stdout=out)

# %%
# Load and display the RDFs after re-weighting. Note that the results might not noisy
# due to the small number of MD steps.

# Load the reweighted RDFs.
rdfs_hh_reweighted = np.loadtxt("h2o-32_rdfs_h-h_reweighted.txt")
rdfs_oo_reweighted = np.loadtxt("h2o-32_rdfs_o-o_reweighted.txt")

# Extract columns.
rdfs_hh_reweighted_mu = rdfs_hh_reweighted[:, 0]
rdfs_hh_reweighted_err = rdfs_hh_reweighted[:, 1]
rdfs_hh_reweighted_committees = rdfs_hh_reweighted[:, 2:]

rdfs_oo_reweighted_mu = rdfs_oo_reweighted[:, 0]
rdfs_oo_reweighted_err = rdfs_oo_reweighted[:, 1]
rdfs_oo_reweighted_committees = rdfs_oo_reweighted[:, 2:]

# Display results.
fig, axs = plt.subplots(figsize=(6, 3), sharey=True, ncols=2, constrained_layout=True)
for title, ax, mus, std, xlim in [
    ("H-H", axs[0], rdfs_hh_reweighted_mu, rdfs_hh_reweighted_err, (1.0, 4.5)),
    ("O-O", axs[1], rdfs_oo_reweighted_mu, rdfs_oo_reweighted_err, (2.0, 4.5)),
]:
    ylabel = "RDF" if title == "H-H" else None
    ax.set(title=title, xlabel="Distance", ylabel=ylabel, xlim=xlim, ylim=(-0.1, 3.7))
    ax.grid()
    ax.plot(xs, mus, label="Mean", lw=0.5)
    z95 = 1.96
    rdfs_ci95 = (mus - z95 * std, mus + z95 * std)
    ax.fill_between(xs, *rdfs_ci95, alpha=0.5, label="CI95")
    ax.legend()

# %%
