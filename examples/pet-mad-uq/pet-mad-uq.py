"""
Uncertainty Quantification with PET-MAD
=======================================

:Authors: Johannes Spies `@johannes-spies <https://github.com/johannes-spies>`_

This example includes three ways of using PET-MAD's uncertainty quantification.

1. Estimating uncertainties on a full validation dataset.
2. Computing vacancy formation energies and prediction uncertainties.
3. Propagate errors from energy predictions to output observables.

For more information on PET-MAD, have a look at
`Mazitov et al., 2025. <https://arxiv.org/abs/2503.14118>`_ The LLPR uncertainties are
introduced in `Bigi et al., 2024. <https://arxiv.org/abs/2403.02251>`_ For more
information on dataset calibration and error propagation, see the
`Imabalzano et al., 2021. <https://arxiv.org/abs/2011.08828>`_ The re-weighting scheme
used for error propagation can be found in
`Kellner & Ceriotti, 2024. <https://arxiv.org/abs/2402.16621>`_


At the bottom of the page, you'll find a ZIP file containing the whole example. Note
that it comes with an `environment.yml` file specifying all dependencies required
to run this example.
"""

# %%
import os
import shlex
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
from scipy.optimize import root_scalar
from scipy.stats import norm
from torch.jit import ScriptModule


# %%
# Model Preparation
# -----------------
# All examples require a PET-MAD model with ensemble and LLPR prediction. The
# following
# code loads a pre-built model if available or builds it on-demand. Note that, by
# default, PET-MAD does not (yet) come equipped with such extended capabilities.
if not os.path.exists("models/pet-mad-latest-llpr.pt"):
    # TODO build the LLPR scores, but -- for now -- just download a prebuilt model
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
# flake
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
# Define helper functions to create iso lines for the parity plot.


def pdf(x, sigma):
    return x * np.exp(-(x**2) / (2 * sigma**2)) * 1.0 / (sigma * np.sqrt(2 * np.pi))


def find_where_pdf_is_c(c, sigma):
    # Finds the two values of x where the pdf is equal to c
    mode_value = pdf(sigma, sigma)
    if c > mode_value:
        raise ValueError("c must be less than mode_value")
    where_below_mode = root_scalar(lambda x: pdf(x, sigma) - c, bracket=[0, sigma]).root
    where_above_mode = root_scalar(
        lambda x: pdf(x, sigma) - c, bracket=[sigma, 100]
    ).root
    return where_below_mode, where_above_mode


def pdf_integral(sigma, c):
    # Calculates the integral (analytical) of the pdf from x1 to x2,
    # where x1 and x2 are the two values of x where the pdf is equal to c
    x1, x2 = find_where_pdf_is_c(c, sigma)
    return np.exp(-(x1**2) / (2 * sigma**2)) - np.exp(-(x2**2) / (2 * sigma**2))


def find_fraction(sigma, fraction):
    # Finds the value of c where the integral of the pdf from x1 to x2 is equal to
    # fraction, where x1 and x2 are the two values of x where the pdf is equal to c
    mode_value = pdf(sigma, sigma)
    return root_scalar(
        lambda x: pdf_integral(sigma, x) - fraction,
        x0=mode_value - 0.01,
        x1=mode_value - 0.02,
    ).root


desired_fractions = [
    norm.cdf(1, 0.0, 1.0) - norm.cdf(-1, 0.0, 1.0),  # 1 sigma
    norm.cdf(2, 0.0, 1.0) - norm.cdf(-2, 0.0, 1.0),  # 2 sigma
    norm.cdf(3, 0.0, 1.0) - norm.cdf(-3, 0.0, 1.0),  # 3 sigma
]

# Hard-code the zoomed in region of the plot.
min_val = 2.5e-2
max_val = 2.5

# Create iso lines.
sigmas = np.geomspace(1e-2, 10, 5)
sigmas = np.geomspace(min_val, max_val, 5)
lower_bounds = []
upper_bounds = []
for desired_fraction in desired_fractions:
    lower_bound = []
    upper_bound = []
    for sigma in sigmas:
        isoline_value = find_fraction(sigma, desired_fraction)
        x1, x2 = find_where_pdf_is_c(isoline_value, sigma)
        lower_bound.append(x1)
        upper_bound.append(x2)
    lower_bounds.append(lower_bound)
    upper_bounds.append(upper_bound)


# %%
# After gathering predicted uncertainties and computing ground truth error metrics, we
# can compare them to each other. Similar to figure S4 of the PET-MAD paper, we present
# the data in using a parity plot. For more information about interpreting this type of
# plot, see Appendix F.7 of `Bigi et al., 2024 <https://arxiv.org/abs/2403.02251>`_.
# Note that both the x- and the y-axis use a logarithmic scale, which is more suitable
# for inspecting uncertainty values. Because we are using a heavily reduced dataset
# (only 100 structures) from the MAD validation set, the parity plot looks very sparse.

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
for i, desired_fraction in enumerate(desired_fractions):
    plt.plot(sigmas, lower_bounds[i], color="black", lw=0.75)
    plt.plot(sigmas, upper_bounds[i], color="black", lw=0.75)

# Add actual samples.
plt.scatter(predicted_uncertainties, ground_truth_uncertainties)

plt.tight_layout()

# %%
# Uncertainties in Vacancy Formation Energies
# -------------------------------------------
# One interesting use of the LLPR scores are the error propagation capabilities from
# the predicted uncertainties for predicting `vacancy formation <https://en.wikipedia.org/wiki/Vacancy_defect>`_ # noqa
# energies. Albeit not completely intuitive, his can be done using the last-layer
# features of PET-MAD. Furthermore, this example also computes energies and variances
# using an ensemble method.
#
# In this part, we use an aluminum crystal as an example system. The structure file can
# be downloaded from `Material Project <https://legacy.materialsproject.org/materials/mp-134/>`_ # noqa
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

# Attach the loaded model
calculator = MetatomicCalculator(model)
supercell.calc = calculator

# %%
# The pretrained model comes with ensemble prediction capabilities that allow for
# uncertainty quantification by measuring the agreement between members of the
# ensemble. Similar energies correspond to a high certainty; a high spread in the
# energies corresponds to a high uncertainty. The pretrained version of PET-MAD comes
# with an ensemble of 128 members.
#
# Along with the `energy_uncertainty` and `energy_ensemble`, we also explicitly request
# the last layer features, which are needed in the analytical computation of the
# uncertainties of the vacancy formation energies that we attempt further down.

output_options = {
    # request the uncertainty in the atomic energy predictions
    "energy": ModelOutput(),  # needed to request the uncertainties
    "energy_uncertainty": ModelOutput(),
    "mtt::aux::energy_last_layer_features": ModelOutput(),
    "energy_ensemble": ModelOutput(),  # needed to request energy ensemble predictions
}

# %%
# Before running the computation, we define the following function that collects
# information at a single stage (before vacancy, before optimization, after
# optimization) in a dict to remove code duplication.


def collect_state(name: str, atoms: Atoms) -> tuple[str, dict[str, torch.Tensor]]:
    """
    Obtain and return results from the configuration in `atoms`.

    Each stage of this script emits a set of results that are all similar in shape and
    layout. This function computes shared quantities and collects them in a dict that
    can be read into a dataframe.
    """
    outputs = calculator.run_model(supercell, output_options)
    energy_ensemble = outputs["energy_ensemble"][0].values
    results = dict(
        energy=atoms.get_potential_energy(),
        energy_uncertainty=outputs["energy_uncertainty"][0].values.squeeze(),
        last_layer_features=outputs["mtt::aux::energy_last_layer_features"][0].values,
        energy_ensemble_mean=torch.mean(energy_ensemble),
        energy_ensemble_var=torch.var(energy_ensemble),
        energy_ensemble=energy_ensemble,
    )
    return name, results


# %%
# We can now run the different stages while keeping track of the information generated
# in each step.

bulk = collect_state("Before creating vacancy", supercell)

# Remove an atom (last atom in this case) to create a vacancy
i = -1
supercell.pop(i)

right_after_vacancy = collect_state("Right after creating vacancy", supercell)

# Run structural optimization optimizing both positions and cell layout.
ecf = FrechetCellFilter(supercell)
bfgs = BFGS(ecf)  # type: ignore
bfgs.run()

vacancy = collect_state("Energy of optimized vacancy", supercell)

# %%
# After running all stages, we can now compute the vacancy formation (VF) energy using
# the following formula.
#
# .. math ::
#
#    E_\mathrm{VF}=E_\mathrm{vacancy}-\frac{N-1}{N}E_\mathrm{bulk}
#
# In this equation, :math:`N` denotes the number of atoms, and the factor of
# :math:`\frac{N-1}{N}` on :math:`E_\mathrm{bulk}` accounts for the missing atom.

vacancy_formation_energy = vacancy[1]["energy"] - (N - 1) / N * bulk[1]["energy"]

# %%
# From the ensemble, we can estimate both the vacancy formation energy and the
# variance from the different predictions.

vacancy_formation_energy_ensemble = torch.mean(
    vacancy[1]["energy_ensemble"] - (N - 1) / N * bulk[1]["energy_ensemble"]
)
vacancy_formation_energy_ensemble_var = torch.var(
    vacancy[1]["energy_ensemble"] - (N - 1) / N * bulk[1]["energy_ensemble"]
)

# %%
# The uncertainty in the vacancy formation energy cannot be computed in a similar,
# straightforward way similar to the vacancy formation energy. As mentioned previously,
# one has to use the last-layer features of the model to estimate the desired variance.
#
# The covariance on the vacancy formation energy can be computed using this formula.
# Note that :math:`\alpha` denotes the dataset calibration constant and :math:`\Sigma`
# is the approximation to the covariance matrix stored in the `LLPRUncertaintyModel`
# wrapping PET-MAD in this example. :math:`\mathbf{f}_i` denote the respective
# last-layer features, :math:`i` can either be `bulk` or `vacancy`. :math:`N` is the
# number of atoms.
#
# .. math ::
#
#    \sigma_\mathrm{VF}^2=\alpha^2\left(\mathbf{f}_\mathrm{vacancy}-\frac{N-1}{N}
#    \mathbf{f}_\mathrm{bulk}\right)^\top\Sigma^{-1}\left(\mathbf{f}_\mathrm{vacancy}-
#    \frac{N-1}{N}\mathbf{f}_\mathrm{bulk}\right)
#
# This formula is implemented in the last line of `estimate_llpr_for_vacancy_formation`.


def find_llpr_module(model) -> ScriptModule | None:
    "Return a `LLPRUncertaintyModel` module if found."
    for child in model.children():
        if (
            isinstance(child, ScriptModule)
            and child.original_name == "LLPRUncertaintyModel"
        ):
            return child
        else:
            find_llpr_module(child)


def estimate_llpr_for_vacancy_formation(
    calculator: MetatomicCalculator,
    f_bulk: torch.Tensor,
    f_vacancy: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    """
    Return the LLPR scores for the vacancy formation energy, i.e. the uncertainty/
    variance, given the last-layer features for the bulk and the vacancy structure.
    """
    # Find the LLPR uncertainty quantification module (it holds the approximation to
    # the inverse covariance matrix)
    llpr_model = find_llpr_module(calculator._model)
    if llpr_model is None:
        raise RuntimeError(
            "Unable to find LLPRUncertaintyModel in MetatomicCalculator. Ensure that"
            " the model supports computing LLPR scores with PET-MAD."
        )

    # Manually extract the inverse covariance and calibration constant from the LLPR
    # module
    inv_covariance = llpr_model.inv_covariance_energy_uncertainty
    alpha_sq = llpr_model.multiplier_energy_uncertainty

    # Create the feature vector and estimate the variance
    f = f_vacancy - (num_atoms - 1) / num_atoms * f_bulk
    return alpha_sq * torch.einsum("...i,ij,...j->...", f, inv_covariance, f)


# %%
# Estimate the uncertainty in the vacancy formation energy using the helper function
# that deals with the last-layer features.
vacancy_formation_uncertainty = estimate_llpr_for_vacancy_formation(
    calculator,
    bulk[1]["last_layer_features"],
    vacancy[1]["last_layer_features"],
    N,
)
vacancy_formation_uncertainty = vacancy_formation_uncertainty.item()

# %%
# At this point, all energies and uncertainties have been computed. They can now be
# gathered in a format suitable to print in a table.

# Wrap information about vacancy formation in a dict to include it as a stage.
vacancy_formation = dict(
    energy=vacancy_formation_energy,
    energy_uncertainty=vacancy_formation_uncertainty,
    energy_ensemble_mean=vacancy_formation_energy_ensemble,
    energy_ensemble_var=vacancy_formation_energy_ensemble_var,
)
vacancy_formation = "Vacancy Formation", vacancy_formation

# Extract index and (printable) values from the stages.
stages = [bulk, right_after_vacancy, vacancy, vacancy_formation]
index, values = zip(*stages)  # split names off (name, value) tuples

# Ensure only printable fields are retained (as floats).
for vs in values:
    for field in ["last_layer_features", "energy_ensemble"]:
        if field in vs:
            del vs[field]
    for field in vs:
        if isinstance(vs[field], torch.Tensor):
            vs[field] = vs[field].item()

# %%
# The following table summarizes the values computed using different methods.
pd.DataFrame(values, index=index)

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
    cmd = shlex.split(cmd)
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
