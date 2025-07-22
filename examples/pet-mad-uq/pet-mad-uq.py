"""
Uncertainty Quantification with PET-MAD
=======================================

:Authors: Johannes Spies `@johannes-spies <https://github.com/johannes-spies>`_

This reciles demonstrates three ways of computing errors on the outputs of
ML potential-driven simulations, using as an example the PET-MAD model and its
built-in uncertainty quantification capabilities.

In particular, we show

1. Estimating uncertainties for single-point calculations on a
   full validation dataset.
2. Computing energies in simple functions of energy predictions,
   namely the value of vacancy formation energies
3. Propagate errors from energy predictions to thermodynamic averages
   computed over a constant-temperature MD simulation.


For more information on PET-MAD, have a look at
`Mazitov et al., 2025. <https://arxiv.org/abs/2503.14118>`_
The LLPR uncertainties are introduced in `Bigi et al., 2024.
<https://arxiv.org/abs/2403.02251>`_ For more
information on dataset calibration and error propagation, see
`Imabalzano et al., 2021. <https://arxiv.org/abs/2011.08828>`_

Models compatible with `metatomic <https://metatensor.github.io/metatomic/>`_ can be
equipped with uncertainty quantification capabilities through the
`LLPRUncertaintyModel` wrapper included with
`metatrain <https://metatensor.github.io/metatrain/>`_. For running this recipe, you can
use a prebuilt model (the example itself downloads a model from Hugging Face).
For adding support for uncertainty quantification to an existing model, have a look at
the following scaffold. For more information on loading a dataset with the
infrastructure, have a look at
`this section <https://metatensor.github.io/metatrain/latest/dev-docs/utils/data>`_
of the documentation. The pseudocode below also shows how to create an ensemble model
from the last-layer parameters of a model.

.. code-block:: python

    from metatrain.utils.llpr import LLPRUncertaintyModel
    from metatomic.torch import AtomisticModel, ModelMetadata

    # You need to provide a model and datasets (wrapped in PyTorch dataloaders).
    model = ...
    dataloaders = {"train": ..., "val": ...}

    # Wrap the model in a module capable of estimating uncertainties, estimate the
    # inverse covariance on the training set, and calibrate the model on the validation
    # set.
    llpr_model = LLPRUncertaintyModel(model)
    llpr_model.compute_covariance(dataloaders["train"])
    llpr_model.compute_inverse_covariance(regularizer=1e-4)
    llpr_model.calibrate(dataloaders["val"])

    # In the next step, we show how to enable ensembles in PET-MAD. For that, it is
    # necessary to extract the last-layer parameters of the model. The ensemble
    # generation expects the parameters in a flat-vector format.
    last_layer_parameters = ...

    # Generate an ensemble with 128 members to compare ensemble uncertainties to LLPR
    # scores.
    llpr_model.generate_ensemble({"energy": last_layer_parameters}, 128)

    # Save the model to disk using metatomic.
    exported_model = AtomisticModel(
        llpr_model.eval(),
        ModelMetadata(),
        llpr_model.capabilities,
    )
    exported_model.save("models/model-with-llpr.pt")

At the bottom of the page, you'll find a ZIP file containing the whole example. Note
that it comes with an `environment.yml` file specifying all dependencies required
to execute the script.
"""

# %%
import os
import subprocess
from urllib.request import urlretrieve

import ase.cell
import ase.ga.utilities
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io.cif import read_cif
from ase.optimize.bfgs import BFGS
from ipi.utils.scripting import InteractiveSimulation
from metatomic.torch import ModelEvaluationOptions, ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from metatrain.utils.data import Dataset, read_systems, read_targets
from metatrain.utils.io import load_model
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


# %%
# Model Preparation
# -----------------
# All examples require a PET-MAD model with ensemble and LLPR prediction. The
# following
# code loads a pre-built model if available or builds it on-demand. Note that, by
# default, PET-MAD does not (yet) come equipped with such extended capabilities.
if not os.path.exists("models/pet-mad-latest-llpr.pt"):
    os.makedirs("models", exist_ok=True)
    urlretrieve(
        "https://huggingface.co/jospies/pet-mad-llpr/resolve/main/"
        "pet-mad-latest-llpr.pt?download=true",
        "models/pet-mad-latest-llpr.pt",
    )
model = load_model("models/pet-mad-latest-llpr.pt")

# %%
# Uncertainties on a Dataset
# ----------------------------------------------
# This first example shows how to use PET-MAD to estimate uncertainties on a reference
# dataset. We use a reduced version (because of limited compute power in the CI runner)
# of the MAD validation set.
#
# For this, we first download the correspond MAD validation dataset record from
# Materials Cloud. Then, we prepare the dataset and pass it through the model. In the
# final step, we visualize the predicted uncertainties and compare them to a
# ground truth method.

if not os.path.exists("data/mad-val-100.xyz"):
    os.makedirs("data", exist_ok=True)
    urlretrieve(
        "https://huggingface.co/jospies/pet-mad-llpr/resolve/main/mad-val-100.xyz"
        "?download=true",
        "data/mad-val-100.xyz",
    )

# Read the dataset's structures.
systems = read_systems("data/mad-val-100.xyz")
systems = [system.to(dtype=torch.float32) for system in systems]
requested_neighbor_lists = get_requested_neighbor_lists(model)
systems_with_nb_lists = [
    get_system_with_neighbor_lists(system, requested_neighbor_lists)
    for system in systems
]

# Read the dataset's targets.
target_config = {
    "energy": {
        "quantity": "energy",
        "read_from": "data/mad-val-100.xyz",
        "reader": "ase",
        "key": "energy",
        "unit": "kcal/mol",
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
dataset = Dataset.from_dict({"system": systems_with_nb_lists, **targets})

# %%
# After preparation, the dataset can be passed through the model to obtain energy
# predictions and LLPR scores.
evaluation_options = ModelEvaluationOptions(
    length_unit="angstrom",
    outputs={
        # request the uncertainty in the atomic energy predictions
        "energy": ModelOutput(),  # needed to request the uncertainties
        "energy_uncertainty": ModelOutput(),
    },
    selected_atoms=None,
)
outputs = model(
    [sample["system"] for sample in dataset],
    evaluation_options,
    check_consistency=True,
)
predicted_energies = outputs["energy"][0].values.squeeze()
predicted_uncertainties = outputs["energy_uncertainty"][0].values.squeeze()

# %%
# Compute the true prediction error by comparing the predicted energy to the reference
# value from dataset.

# Reference values from dataset.
ground_truth_energies = torch.stack(
    [sample["energy"][0].values.squeeze() for sample in dataset]
)

# Compute squared distance between predicted energy and reference value.
ground_truth_uncertainties = torch.square(predicted_energies - ground_truth_energies)

# %%
# After gathering predicted uncertainties and computing ground truth error metrics, we
# can compare them to each other. Similar to figure S4 of the PET-MAD paper, we present
# the data in using a parity plot. For more information about interpreting this type of
# plot, see Appendix F.7 of `Bigi et al., 2024 <https://arxiv.org/abs/2403.02251>`_.
# Note that both the x- and the y-axis use a logarithmic scale, which is more suitable
# for inspecting uncertainty values. Because we are using a heavily reduced dataset
# (only 100 structures) from the MAD validation set, the parity plot looks very sparse.

# Hard-code the zoomed in region of the plot and iso-lines.
min_val, max_val = 2.5e-2, 2.5
lower_bounds = [
    [0.010775, 0.034072, 0.107746, 0.340723, 1.077459],
    [0.002564, 0.008109, 0.025643, 0.08109, 0.25643],
    [0.000229, 0.000724, 0.002288, 0.007237, 0.022885],
]
upper_bounds = [
    [0.042949, 0.135817, 0.429491, 1.358168, 4.294906],
    [0.06337, 0.200392, 0.633695, 2.00392, 6.336952],
    [0.086097, 0.272264, 0.860975, 2.722641, 8.609747],
]
sigmas = np.geomspace(min_val, max_val, 5)

# Create the parity plot.
plt.figure(figsize=(4, 4))
plt.grid()
plt.gca().set(
    title="Parity of Uncertainties",
    ylabel="Errors",
    xlabel="Uncertainties",
)
plt.loglog()

# Plot iso lines.
plt.plot([min_val, max_val], [min_val, max_val], ls="--", c="k")
for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
    plt.plot(sigmas, lower_bound, color="black", lw=0.75)
    plt.plot(sigmas, upper_bound, color="black", lw=0.75)

# Add actual samples.
plt.scatter(predicted_uncertainties, ground_truth_uncertainties)

plt.tight_layout()

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
N = len(supercell)  # store the number of atoms

# Attach the loaded model and enable returning ensemble energies when computing the
# potential energy
output_options = {
    "energy": ModelOutput(),
    "energy_ensemble": ModelOutput(),
    "energy_uncertainty": ModelOutput(),
}
calculator = MetatomicCalculator(model, additional_outputs=output_options, device="cpu")
supercell.calc = calculator

# %%
# We now compute the vacancy formation energy by keeping track of the ensemble energies
# at different stages. Note that calling `.get_potential_energy()` on an `Atoms` object
# triggers computing the ensemble values.

supercell.get_potential_energy()
bulk = calculator.additional_outputs["energy_ensemble"][0].values

# Remove an atom (last atom in this case) to create a vacancy
i = -1
supercell.pop(i)

supercell.get_potential_energy()
right_after_vacancy = calculator.additional_outputs["energy_ensemble"][0].values

# Run structural optimization optimizing both positions and cell layout.
ecf = FrechetCellFilter(supercell)
bfgs = BFGS(ecf)  # type: ignore
bfgs.run()

supercell.get_potential_energy()
vacancy = calculator.additional_outputs["energy_ensemble"][0].values

# %%
# Compute vacancy formation energy for each ensemble member.

vacancy_formation = vacancy - (N - 1) / N * bulk

# %%
# Put all ensemble energies in a dataframe and compute the desired statistics.

# This dataframe contains each stage's energies in a single column.
named_stages = [
    ("Before creating vacancy", bulk),
    ("Right after creating vacancy", right_after_vacancy),
    ("Energy of optimized vacancy", vacancy),
    ("Vacancy formation energy", vacancy_formation),
]
df = pd.DataFrame.from_dict(
    {
        # Convert the PyTorch tensors to flat NumPy vectors
        k: v.detach().numpy().squeeze()
        for k, v in named_stages
    }
)

# Compute statistics (mean, variance, and standard deviation) on the ensemble energies.
df = pd.DataFrame(dict(mean=df.mean(), var=df.var(), std=df.std()))
df

# %%
# Uncertainty Propagation with MD
# -------------------------------
# This example shows how to use i-PI to propagate error estimates from an ensemble to
# output observables. In this example, we use a box with period boundary conditions
# housing 32 water molecules. As an observable, we inspect the `Radial Distribution
# Function (RDF) <https://en.wikipedia.org/wiki/Radial_distribution_function>`_ between
# hydrogen-hydrogen and oxygen-oxygen bonds.
#
# First, we run a simulation with i-PI generating a trajectory and logging other
# metrics. The trajectory and committee energies can be used in a subsequent
# postprocessing step to obtain RDFs using ASE. These can be re-weighted to propagate
# errors from the committee uncertainties to the observed RDFs.

# Load configuration and run simulation.
with open("data/h2o-32.xml") as f:
    sim = InteractiveSimulation(f.read())

# %%
# Right now, the model does not produce the `energy_ensemble` field on its own. The
# ensemble energy is only produced if the `energy_uncertainty` field is populated in
# the model evaluation option's output field. This does not happen by default in the
# `MetatomicDriver` of i-PI, but the following hack ensures that all outputs are
# properly requested for all force fields.

for _, ff in sim.__dict__["fflist"].items():
    outputs = ff.driver.evaluation_options.outputs
    outputs["energy_uncertainty"] = ModelOutput()
    ff.driver.evaluation_options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs=outputs,
    )

# %%
# Run the simulation.

# NB: To get better estimates, set this to a higher number (perhaps 10000) to
# run the simulation for a longer time.
sim.run(200)

# %%
# Load the trajectories and compute the per-frame RDFs
frames: list[Atoms] = ase.io.read("h2o-32.pos_0.xyz", ":")  # type: ignore

# Our simulation should only include water molecules. (types: hydrogen=1, oxygen=8)
assert set(frames[0].numbers.tolist()) == set([1, 8])

# Compute the RDF of each frame (for H-H and for O-O)
num_bins = 100
rdfs_hh = []
rdfs_oo = []
xs = None
for atoms in frames:
    atoms.pbc = True
    atoms.cell = ase.cell.Cell(9.86592 * np.eye(3))

    # Compute H-H distances
    bins, xs = ase.ga.utilities.get_rdf(  # type: ignore
        atoms, 4.5, num_bins, elements=[1, 1]
    )
    rdfs_hh.append(bins)

    # Compute O-O distances
    bins, xs = ase.ga.utilities.get_rdf(  # type: ignore
        atoms, 4.5, num_bins, elements=[8, 8]
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
fig, axs = plt.subplots(figsize=(6, 3), sharey=True, ncols=2)
for title, ax, mus, errs, xlim in [
    ("H-H", axs[0], rdfs_hh_reweighted_mu, rdfs_hh_reweighted_err, (0.0, 4.5)),
    ("O-O", axs[1], rdfs_oo_reweighted_mu, rdfs_oo_reweighted_err, (2.0, 4.5)),
]:
    ylabel = "RDF" if title == "H-H" else None
    ax.set(title=title, xlabel="Distance", ylabel=ylabel, xlim=xlim, ylim=(-1, 3.0))
    ax.grid()
    ax.plot(xs, mus, label="Mean", lw=2)
    z95 = 1.96
    rdfs_ci95 = (mus - z95 * errs, mus + z95 * errs)
    ax.fill_between(xs, *rdfs_ci95, alpha=0.4, label="CI95")
    ax.legend()
