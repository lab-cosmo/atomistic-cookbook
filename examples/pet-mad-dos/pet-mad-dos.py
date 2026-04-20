"""
A Guide to PET-MAD-DOS: Model Inference and Fine-Tuning
=========================================================
:Authors: How Wei Bin `@HowWeiBin <https://github.com/HowWeiBin/>`_

This tutorial would describe two ways that one can use PET-MAD-DOS to
predict the electronic density of states (DOS) of a structure, 1) Treating
PET-MAD-DOS as a universal model and deploying it out of the box, or 2)
Treating PET-MAD-DOS as a foundation model and fine-tuning it for specific
applications.

In this tutorial, we would demonstrate the first approach using the upet
package to showcase how one can easily obtain high quality DOS predictions
for a structure of interest. Then, we would illustrate how one can use the
`metatrain` package to fine-tune the PET-MAD-DOS model for a specific
application. In the process, we would also go through the necessary
data processing pipeline, starting from basic DFT outputs.

First, lets begin by importing the necessary packages and helper functions
"""
# sphinx_gallery_thumbnail_number = 2
# %%

from upet.calculator import PETMADDOSCalculator
import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import urllib.request
import os
import subprocess

# %%
# Using PET-MAD-DOS out of the box
# -----------------------------------
#
# In this section, we are going to treat PET-MAD-DOS as a universal
# model and use it out-of-the-box to obtain predictions of the DOS for a sample
# of structures.
#


# %%
# Step 1: Loading Sample Structures
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here, we will demonstrate PET-MAD-DOS on a small sample of 5 structures in
# the training dataset used in the `original PET-MAD-DOS publication
# <https://arxiv.org/abs/2508.17418>`_.
#


# %%

# Load the structures
sample_MAD_structures = ase.io.read("data/MAD_sample_structures.xyz", ":")

# Extract the DOS and mask
true_DOS = []
true_mask = []
for i in sample_MAD_structures:
    true_DOS.append(i.info["DOS"])
    true_mask.append(i.info["mask"])
true_DOS = np.stack(true_DOS)
true_mask = np.stack(true_mask)
print(f"The shape of true_DOS is: {true_DOS.shape}")
print(f"The shape of true_mask is: {true_mask.shape}")

# Define the energy grid for the DOS
lower_bound = -148.1456 - 1.5
upper_bound = 79.1528 + 1.5
interval = 0.05
n_points = np.ceil(np.array(upper_bound - lower_bound) / interval)
true_energy_grid = np.arange(n_points) * interval + lower_bound
print(f"The shape of true_energy_grid is: {true_energy_grid.shape}")

# %%
# The true DOS is computed based on eigenvalaues obtained from DFT calculations
# and projected on an energy grid (``true_energy_grid``) of size 4606. However,
# due to eigenvalue truncation in the dataset, the DOS is not necessarily
# well-defined at every point on the energy grid. For instance, if the highest
# computed eigenvalue of a structure A is at 3eV, the computed DOS would
# indicate that there are no states past 3eV. However, that is false and is
# merely an artefact of eigenvalue truncation. Hence, an additional mask is
# included for each structure to show the regions where the DOS is
# well-defined, 1 indicates that the DOS is well defined and 0 indicates that
# it is not. The DOS is considered well-defined up to 0.9eV below the minimum
# energy of the highest band in the DFT calculation. In the following, we plot
# these quantities, where the mask is multiplied by 10 to enhance visibility
# in the plot. DOS values where the mask is 0 should not be considered reliable
# and should be ignored when comparing both spectra.

# %%
plt.plot(true_energy_grid, true_DOS[0], label="DFT DOS", color="red")
plt.plot(
    true_energy_grid,
    true_mask[0] * 10,
    label="Mask x 10",
    linestyle="--",
    color="black",
)
plt.xlim(-60, 20)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy - $\mathrm{E_F}$ [eV]", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
# %%
# Here, we see a visualization of the DOS and the mask for the first sample
# structure. In this case, the mask is multiplied by 10 so that it is more
# visible on the plot. The DOS is aligned such that the Fermi level is at 0eV.

# %%
# Step 2: Loading PET-MAD-DOS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we will load the PET-MAD-DOS model using the upet package and use it to
# predict the DOS. Although the target has size 4606 as seen in the cells above,
# in order to accomodate the energy reference agnostic loss function,
# PET-MAD-DOS predicts on a larger energy grid (`energies`) of size 4806.


# Load the calculator
pet_mad_dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")

# Generate predictions for structures
energies, pred_DOS = pet_mad_dos_calculator.calculate_dos(sample_MAD_structures)

print(f"The shape of pred_DOS is: {pred_DOS.shape}")
print(f"The shape of energies is: {energies.shape}")

# %%
# Step 3: Visualize the predictions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since the absolute energy reference is not well-defined for bulk systems, we
# need to compare the predicted DOS and the true DOS in a way that is agnostic
# to the energy reference used. To do this, we will compare these spectra using
# the energy reference that minimizes the error between them.


# Define a function that minimizes the RMSE between the predicted and true DOS
# by shifting the energy axis
def get_dynamic_shift_agnostic_mse(predictions, targets, cutoff_mask):
    if predictions.shape[1] < targets.shape[1]:
        smaller = predictions
        bigger = targets
    else:
        smaller = targets
        bigger = predictions

    bigger_unfolded = bigger.unfold(1, smaller.shape[1], 1)
    smaller_expanded = smaller[:, None, :]
    delta = smaller_expanded - bigger_unfolded
    dynamic_delta = delta * cutoff_mask.unsqueeze(dim=1)
    device = predictions.device
    losses = torch.sum(dynamic_delta * dynamic_delta, dim=2)
    front_tail = torch.cumsum(predictions**2, dim=1)
    shape_difference = predictions.shape[1] - targets.shape[1]
    additional_error = torch.hstack(
        [
            torch.zeros(len(predictions), device=device).reshape(-1, 1),
            front_tail[:, :shape_difference],
        ]
    )
    total_losses = losses + additional_error
    final_loss, shift = torch.min(total_losses, dim=1)
    result = torch.mean(final_loss)

    return result, shift


# Compute the optimal shift and the associated MSE
MSEs, shifts = get_dynamic_shift_agnostic_mse(
    pred_DOS, torch.tensor(true_DOS), torch.tensor(true_mask)
)


# Function that adjusts the true DOS and mask so that it is aligned to the
# predictions based on the optimal shift obtained above. This is useful for
# visualizing the predictions and the true DOS on the same plot.
def adjust_DOS_and_mask(target, mask, shift):
    adjusted_DOSes = []
    adjusted_mask = []
    for index, s in enumerate(shift):
        front_pad = torch.zeros(s)
        end_pad = torch.zeros(200 - s)  # 200 is because the dimensionality of
        # the prediction is 200 larger than
        # that of the target
        adjusted_DOSes.append(torch.hstack([front_pad, target[index], end_pad]))
        adjusted_mask.append(torch.hstack([front_pad, mask[index], end_pad]))
    return torch.vstack(adjusted_DOSes), torch.vstack(adjusted_mask)


# Adjust the DOS and mask based on the optimal shifts obtained above
adjusted_true_DOS, adjusted_true_mask = adjust_DOS_and_mask(
    torch.tensor(true_DOS), torch.tensor(true_mask), shifts
)

# Visualize the predictions and the true DOS on the same plot
plt.plot(energies, pred_DOS[0], label="Predicted DOS", color="blue")

plt.plot(energies, adjusted_true_DOS[0], label="DFT DOS", linestyle="--", color="red")

plt.plot(
    energies,
    adjusted_true_mask[0] * 10,
    label="Mask x 10",
    linestyle="-.",
    color="black",
)

plt.xlim(-60, 20)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy - $\mathrm{E_F}$ [eV]", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
# %%
# Step 4: Predicting the Bandgap
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additionally, PET-MAD-DOS comes with a CNN bandgap model that predicts the gap
# of a system based on the predicted DOS. Due to the inherent model noise and
# the high sensitivity of the bandgap to small errors in the DOS, obtaining the
# bandgap via a CNN model is more robust than deriving it from the predicted DOS.


pred_bandgap = pet_mad_dos_calculator.calculate_bandgap(
    sample_MAD_structures, dos=pred_DOS
)

true_bandgap = []
for i in sample_MAD_structures:
    true_bandgap.append(i.info["gap"])
true_bandgap = torch.tensor(true_bandgap, dtype=torch.float32)

print(f"The predicted bandgaps are : {pred_bandgap}")
print(f"The DFT bandgaps are : {(true_bandgap)}")

# %%
# Visualze the predicted and true bandgaps on a parity plot

plt.scatter(pred_bandgap, true_bandgap, color="blue")
plt.plot([0, 1.4], [0, 1.4], label="y=x", color="red")
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Predicted Gap [eV]", size=16)
plt.ylabel(r"DFT Gap [eV]", size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
# %%
# Finetuning PET-MAD-DOS on specific applications
# --------------------------------------------------
# In this section, we are going to treat PET-MAD-DOS as a foundation
# model and finetune it on a Gallium Arsenide (GaAs) system. We will first
# demonstrate the data processing pipeline starting from DFT outputs, and then
# we will show how to use the `metatrain` package to finetune the model.

# %%
# Step 1: Data Processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The data processing pipeline uses the computed eigenvalues at each k-point,
# along with its associated k-point weight (where relevant) to generate the DOS
# and mask for each structure. The DOS is generated by applying gaussian
# broadening on each eigenenergy, and projected on an energy grid. For the
# purposes of fine-tuning, the energy grid will be identical to the one used for
# the training of PET-MAD-DOS.


n_extra_targets = 200  # PET-MAD-DOS predicts 200 more DOS
# values per structure (4806 - 4606 = 200)
# Load the GaAs sample structures with eigenvalues and k-point weights. These
# files are hosted on zenodo due to their large size

url = "https://zenodo.org/records/19655792/files/GaAs_sample_structures.xyz?download=1"
filename = "GaAs_sample_structures.xyz"
urllib.request.urlretrieve(url, filename)
GaAs_sample_structures = ase.io.read("GaAs_sample_structures.xyz", ":")

eigenvalues = []
kweights = []

for structure_i in GaAs_sample_structures:
    eigenvalues.append(structure_i.info["eigvals"])  # shape (n_kpoints, n_bands)
    kweights.append(
        structure_i.info["kweight"]
    )  # shape (n_kpoints,), sums to 2 in this example

smearing = 0.3  # Same smearing value as the one used in the MAD training data
energy_grid = true_energy_grid

normalization = 1 / np.sqrt(2 * np.pi * smearing**2)  # Gaussian normalization factor

# After defining the energy grid, we can now compute the DOS and the
# mask for each structure.
for index in range(len(GaAs_sample_structures)):
    confident_energy_upper_bound = (
        np.min(eigenvalues[index][:, -1]) - 0.9
    )  # 0.9eV below the minimum of the highest band

    # Flatten eigenvalues to shape (n_kpoints * n_bands,)
    eigenvalues_i = eigenvalues[index].flatten()

    # Ensure that the eigenvalues are mapped to the correct weight,
    # shape (n_kpoints * n_bands,)
    kweights_i = kweights[index].repeat(eigenvalues[index].shape[1])

    # Compute the DOS on the energy grid by applying Gaussian broadening
    # on each eigenvalue
    delta_E = (energy_grid - eigenvalues_i[:, None]) / smearing
    gaussian_weights = np.exp(-0.5 * delta_E**2)
    dos_i = (
        np.sum(kweights_i[:, None] * gaussian_weights, axis=0) * normalization
    )  # Apply Gaussian smearing and sum contributions

    # Define the mask
    mask_i = (energy_grid <= confident_energy_upper_bound).astype(int)

    # To prepare the data for fine-tuning, we will pad the DOS and mask with
    # zeros in front based on the number of extra targets that PET-MAD-DOS
    # predicts. These values will be ignored during loss computation.
    dos_i_padded = np.concatenate([np.zeros(n_extra_targets), dos_i])
    mask_i_padded = np.concatenate([np.zeros(n_extra_targets), mask_i])

    # Store the dos and mask in the structure info for training
    GaAs_sample_structures[index].info["DOS"] = dos_i_padded.astype(np.float32)
    GaAs_sample_structures[index].info["mask"] = mask_i_padded.astype(np.float32)

# %%
# Alternatively, if one uses precomputed DOS data, like the ones on the
# `MaterialsCloud archive
# <https://archive.materialscloud.org/records/8ee9k-b7865>`_, one can skip to
# the end of the data processing pipeline as follows


GaAs_sample_train_structures = ase.io.read("data/GaAs_sample_train_structures.xyz", ":")
GaAs_sample_val_structures = ase.io.read("data/GaAs_sample_val_structures.xyz", ":")
GaAs_sample_test_structures = ase.io.read("data/GaAs_sample_test_structures.xyz", ":")

# To prepare the structures for training
extra_targets = 200
for i in GaAs_sample_train_structures:
    dos_i_padded = np.concatenate([np.zeros(n_extra_targets), i.info["DOS"]])
    mask_i_padded = np.concatenate([np.zeros(n_extra_targets), i.info["mask"]])

    i.info["trainingDOS"] = dos_i_padded
    i.info["trainingmask"] = mask_i_padded

for i in GaAs_sample_val_structures:
    dos_i_padded = np.concatenate([np.zeros(n_extra_targets), i.info["DOS"]])
    mask_i_padded = np.concatenate([np.zeros(n_extra_targets), i.info["mask"]])

    i.info["trainingDOS"] = dos_i_padded
    i.info["trainingmask"] = mask_i_padded

for i in GaAs_sample_test_structures:
    dos_i_padded = np.concatenate([np.zeros(n_extra_targets), i.info["DOS"]])
    mask_i_padded = np.concatenate([np.zeros(n_extra_targets), i.info["mask"]])

    i.info["trainingDOS"] = dos_i_padded
    i.info["trainingmask"] = mask_i_padded

ase.io.write("GaAs_processed_train.xyz", GaAs_sample_train_structures)
ase.io.write("GaAs_processed_val.xyz", GaAs_sample_val_structures)
ase.io.write("GaAs_processed_test.xyz", GaAs_sample_test_structures)

# %%
# Step 2: Model Loading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we need to download the PET-MAD-DOS checkpoint and place it in the
# local directory so that it can be referenced in the metatrain YAML
# configuration files for fine-tuning.


# Download the checkpoint from Zenodo and copy it to the local directory
url = "https://zenodo.org/records/19655792/files/pet-mad-dos-v1.0.ckpt?download=1"

checkpoint_path = "pet-mad-dos-v1.0.ckpt"
if not os.path.exists(checkpoint_path):
    print("Downloading PET-MAD-DOS checkpoint...")
    urllib.request.urlretrieve(url, checkpoint_path)

# %%
# Step 3: Fine-tuning the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we are ready to fine-tune the model using the `metatrain` package. We
# will use the `mtt train` command to train the model, alongside with a
# supporting YAML configuration file that specifies the training hyperparameters.

# %%
# .. literalinclude:: finetune.yaml
#   :language: yaml


# Begin finetuning
subprocess.run(
    ["mtt", "train", "finetune.yaml", "-o", "fine_tune-model.pt"], check=True
)

# %%
# Step 4: Evaluating the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After training, we can evaluate the model on the test set using the `mtt eval`
# command, alongside with a supporting YAML configuration file that specifies the
# evaluation hyperparameters.

# %%
# .. literalinclude:: eval.yaml
#   :language: yaml
#

subprocess.run(
    ["mtt", "eval", "fine_tune-model.pt", "eval.yaml", "-o", "pred.xyz"], check=True
)

output = ase.io.read("pred.xyz", ":")

# %%
# As we have only fine-tuned for 10 epochs on a tiny dataset, the model
# performance is not expected to be good. However, this simply serves as a
# demonstration of how to use the fine-tuned model.

predicted_DOS = []
for i in output:
    predicted_DOS.append(i.info["mtt::dos"])
predicted_DOS = torch.tensor(predicted_DOS)

true_DOS = []
true_mask = []
for i in GaAs_sample_test_structures:
    true_DOS.append(i.info["DOS"])
    true_mask.append(i.info["mask"])
true_DOS = np.stack(true_DOS)
true_mask = np.stack(true_mask)

# Compute the optimal shift and the associated MSE
MSEs, shifts = get_dynamic_shift_agnostic_mse(
    predicted_DOS, torch.tensor(true_DOS), torch.tensor(true_mask)
)

# Adjust the DOS and mask based on the optimal shifts obtained above
adjusted_true_DOS, adjusted_true_mask = adjust_DOS_and_mask(
    torch.tensor(true_DOS), torch.tensor(true_mask), shifts
)

# Visualize the predictions and the true DOS on the same plot

plt.plot(energies, predicted_DOS[0], label="Predicted DOS", color="blue")

plt.plot(energies, adjusted_true_DOS[0], label="DFT DOS", linestyle="--", color="red")

plt.plot(
    energies,
    adjusted_true_mask[0] * 100,
    label="Mask x 100",
    linestyle="-.",
    color="black",
)

plt.xlim(-30, 20)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy - $\mathrm{E_F}$ [eV]", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
