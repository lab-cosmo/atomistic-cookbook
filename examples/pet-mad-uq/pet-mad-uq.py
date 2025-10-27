"""
Uncertainty Quantification with PET-MAD
=======================================

:Authors: Johannes Spies `@johannes-spies <https://github.com/johannes-spies>`_

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

Optional: Adding UQ to a Model
------------------------------

Models compatible with `metatomic <https://metatensor.github.io/metatomic/>`_ can be
equipped with UQ capabilities through the
`LLPRUncertaintyModel` wrapper included with
`metatrain <https://metatensor.github.io/metatrain/>`_. For running this recipe, you can
use a prebuilt model (the example itself downloads a model from Hugging Face).
For adding UQ support to an existing model, have a look at
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

Getting Started
---------------

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
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from metatrain.utils.data import Dataset, read_systems, read_targets
from metatrain.utils.data.system_to_ase import system_to_ase


# %%
# Model Loading
# -------------
# All examples require a PET-MAD model with ensemble and LLPR prediction. The
# following
# code loads a pre-trained model using the ASE-compatible calculator wrapper. Using the
# calculator instead of calling the model directly conveniently hides computing
# neighbor lists in the calculator.
if not os.path.exists("models/pet-mad-latest-llpr.pt"):
    os.makedirs("models", exist_ok=True)
    urlretrieve(
        "https://huggingface.co/jospies/pet-mad-llpr/resolve/main/"
        "pet-mad-latest-llpr.pt?download=true",
        "models/pet-mad-latest-llpr.pt",
    )

calculator = MetatomicCalculator("models/pet-mad-latest-llpr.pt", device="cpu")

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
}
results = calculator.run_model(systems, outputs)

# Extract the requested results
predicted_energies = results["energy"][0].values.squeeze()
predicted_uncertainties = results["energy_uncertainty"][0].values.squeeze()

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
# the data in using a parity plot. For more information about interpreting this type of
# plot, see Appendix F.7 of `Bigi et al., 2024 <https://arxiv.org/abs/2403.02251>`_.
# Note that both the x- and the y-axis use a logarithmic scale, which is more suitable
# for inspecting uncertainty values. Because we are using a heavily reduced dataset
# (only 100 structures) from the MAD validation set, the parity plot looks very sparse.

# Hard-code the zoomed in region of the plot and iso-lines.
quantile_lines = [0.00916, 0.10256, 0.4309805, 1.71796, 2.5348, 3.44388]
min_val, max_val = 2.5e-2, 2.5

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
for factor in quantile_lines:
    plt.plot([min_val, max_val], [factor * min_val, factor * max_val], "k:", lw=0.75)

# Add actual samples.
plt.scatter(predicted_uncertainties, empirical_errors)

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
supercell.calc = calculator
N = len(supercell)  # store the number of atoms

# %%
# We now compute the vacancy formation energy by keeping track of the ensemble energies
# at different stages. Note that calling `.get_potential_energy()` on an `Atoms` object
# triggers computing the ensemble values.

# Get ensemble energy before creating the vacancy
outputs = ["energy", "energy_uncertainty", "energy_ensemble"]
outputs = {o: ModelOutput() for o in outputs}
results = calculator.run_model(supercell, outputs)
bulk = results["energy_ensemble"][0].values

# Remove an atom (last atom in this case) to create a vacancy
i = -1
supercell.pop(i)

# Get ensemble energy right after creating the vacancy
results = calculator.run_model(supercell, outputs)
right_after_vacancy = results["energy_ensemble"][0].values

# Run structural optimization optimizing both positions and cell layout.
ecf = FrechetCellFilter(supercell)
bfgs = BFGS(ecf)  # type: ignore
bfgs.run()

# get ensembele energy after optimization
results = calculator.run_model(supercell, outputs)
vacancy = results["energy_ensemble"][0].values

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
#
# Uncertainty Propagation with MD
# -------------------------------
#
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
sim.run(100)

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
