"""
A Guide to PET-MAD-DOS: Model Inference and Fine-Tuning
=========================================================
:Authors: How Wei Bin `@HowWeiBin <https://github.com/HowWeiBin/>`_

This tutorial describes two ways in which one can use PET-MAD-DOS to
predict the electronic density of states (DOS) of a structure:

1. **Treating PET-MAD-DOS as a universal model**: using the upet package,
we will obtain high quality DOS predictions out of the box for a structure
of interest.

2. **Treating PET-MAD-DOS as a foundation model**: we will illustrate
how one can use the ``metatrain`` package to fine-tune the PET-MAD-DOS
model for a specific application.

In the process, we will also go through the necessary
data processing pipeline, starting from basic DFT outputs.

First, let's begin by importing the necessary packages and helper functions
"""
# sphinx_gallery_thumbnail_number = 2
# %%

import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch
import urllib.request
import os
import subprocess
from upet.calculator import PETMADDOSCalculator

# %%
# Using PET-MAD-DOS out of the box
# -----------------------------------
#
# In this section, we are going to treat PET-MAD-DOS as a universal
# model and use it out-of-the-box to obtain predictions of the DOS for a sample
# of structures.
#


# %%
# Step 1: Loading Sample Structures and Calculator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will start by loading a sample of structures, along with their associated
# DOS and mask. These are 5 structures from the training dataset used in the
# original `PET-MAD-DOS publication <https://arxiv.org/abs/2508.17418>`_. We also load
# the ``PETMADDOSCalculator`` from the upet package, which will be used to predict the
# DOS for these structures.
#


# %%

# Load the structures
url = "https://zenodo.org/records/19655792/files/MAD_sample_structures.xyz?download=1"
filename = "MAD_sample_structures.xyz"
urllib.request.urlretrieve(url, filename)
structs = ase.io.read("MAD_sample_structures.xyz", ":")

# Extract the DOS and mask
true_DOS = torch.tensor(np.stack([s.info["DOS"] for s in structs])).float()
true_mask = torch.tensor(np.stack([s.info["mask"] for s in structs])).float()
print(f"The shape of true_DOS is: {true_DOS.shape}")
print(f"The shape of true_mask is: {true_mask.shape}")

# Load the calculator
pet_mad_dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")

# %%
# The true DOS is computed based on eigenvalues obtained from DFT calculations
# and projected on an energy grid of size 4606. However, due to eigenvalue truncation
# in the dataset, the DOS is not necessarily well-defined at every point on the energy
# grid. For instance, if the highest computed eigenvalue of a structure A is at 3eV,
# the computed DOS would indicate that there are no states past 3 eV. However, that is
# false and is merely an artifact of eigenvalue truncation. Hence, an additional mask
# is included for each structure to show the regions where the DOS is well-defined.
# The DOS is considered well-defined up to 0.9eV below the minimum energy of the
# highest band in the DFT calculation. Following, we plot an example DOS together
# with its mask.
#

i_struct = 1  # index of the structure to visualize

# Plot DOS of the structure
plt.plot(
    pet_mad_dos_calculator._target_energy_grid,
    true_DOS[i_struct],
    label="DFT DOS",
    color="red",
)
# Plot mask for the structure (multiplied by 10 for better visualization)
plt.plot(
    pet_mad_dos_calculator._target_energy_grid,
    true_mask[i_struct] * 10,
    label="Mask x 10",
    linestyle="--",
    color="black",
)
plt.xlim(-80, 10)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy - $\mathrm{E_F}$ [eV]", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
# %%
# In this plot, the DOS is aligned such that the Fermi level is at 0eV.
# DOS values where the mask is 0 should not be considered reliable
# and should be ignored when comparing against the predicted DOS spectra.


# %%
# Step 2: Predicting the DOS with PET-MAD-DOS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Although the target has size 4606 as seen in the cells above,
# in order to accomodate the energy reference agnostic loss function,
# PET-MAD-DOS predicts on a larger energy grid (``energies``)of size 4806.
#

energies, pred_DOS = pet_mad_dos_calculator.calculate_dos(structs)

print(f"The shape of pred_DOS is: {pred_DOS.shape}")
print(f"The shape of energies is: {energies.shape}")

# %%
# Additionally, the calculate_dos function also includes a ``denoise`` parameter that
# implements a post-processing denoising step to remove the high-frequency noise in the
# predicted DOS. The denoising procedure is detailed in the `PET-MAD-DOS publication
# <https://arxiv.org/abs/2508.17418>`_. By default, the denoising is turned off,
# but one can set ``denoise=True`` to obtain the denoised predictions.
#

energies, denoised_pred_DOS = pet_mad_dos_calculator.calculate_dos(
    structs,
    denoise=True,
)
print(f"The shape of denoised_pred_DOS is: {denoised_pred_DOS.shape}")


# %%
# Step 3: Visualize the predictions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can plot the predicted DOS and the denoised DOS on the same plot
# to see how they compare.
#


# Visualize the raw predictions and the denoised predictions on the same plot
plt.plot(
    energies,
    denoised_pred_DOS[i_struct],
    label="Denoised DOS",
    color="blue",
    linestyle="-",
)
plt.plot(energies, pred_DOS[i_struct], label="Raw DOS", color="green", linestyle="--")

plt.xlim(-80, 10)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# %%
# As we can see, the denoising step removes the high-frequency noise in the raw
# predictions, while preserving the overall DOS profile. In the next part, we will
# compare the denoised predictions with the true DOS. However, since the prediction and
# the true DOS are on different energy grids and are not necessarily aligned in energy
# as the energy reference is not well-defined, we will need to do some processing to
# align them before we can compare them directly.
#

denoised_DOS, aligned_true_DOS, aligned_true_masks = pet_mad_dos_calculator.align_dos(
    denoised_pred_DOS, true_DOS.float(), true_mask.float()
)

# Visualize the predictions and the true DOS on the same plot
plt.plot(
    energies, denoised_DOS[i_struct], label="Denoised DOS", color="blue", linestyle="-"
)
plt.plot(
    energies, aligned_true_DOS[i_struct], label="DFT DOS", linestyle="--", color="red"
)
plt.plot(
    energies,
    aligned_true_masks[i_struct] * 10,
    label="Mask x 10",
    linestyle="-.",
    color="black",
)
plt.xlim(-80, 10)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy [eV]", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()


# %%
# The plot shows that the predicted DOS is actually very close to the
# true DOS for the region where the DOS values are physical. The DOS profiles are
# nearly identical, with the errors lying primarily in the height of the peaks.
#

# %%
# Step 4: Predicting the Bandgap
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Additionally, PET-MAD-DOS comes with a CNN bandgap model that predicts the gap
# of a system based on the predicted DOS. Due to inherent model noise and
# the high sensitivity of the bandgap to small errors in the DOS, obtaining the
# bandgap via a CNN model is more robust than deriving it from the predicted DOS.
#

pred_bandgap = pet_mad_dos_calculator.calculate_bandgap(structs, dos=pred_DOS)

true_bandgap = torch.tensor(np.stack([s.info["gap"] for s in structs])).float()

print(f"The predicted bandgaps are : {pred_bandgap}")
print(f"The DFT bandgaps are : {(true_bandgap)}")

# %%
# The band gaps show good agreement, following we generate a parity plot:

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
# we will show how to use the ``metatrain`` package to finetune the model.
#

# %%
# Step 1: Data Processing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following cells will show how to process a dataset containing eigenvalues
# to obtain the corresponding DOS and mask that can be used for fine-tuning
# PET-MAD-DOS.
# We will start by loading some GaAs sample structures containing eigenvalues and
# k-point weights from zenodo.
#

url = "https://zenodo.org/records/19655792/files/GaAs_sample_structures.xyz?download=1"
filename = "GaAs_sample_structures.xyz"
urllib.request.urlretrieve(url, filename)
GaAs_sample_structures = ase.io.read("GaAs_sample_structures.xyz", ":")

# %%
# Then we compute the DOS and mask for each structure using the
# ``compute_DOS_and_mask_from_eigenvalues`` function in the
# ``PETMADDOSCalculator`` calculator, which applies Gaussian broadening on the eigenvalues
# and projects them onto the energy grid of PET-MAD-DOS to obtain the DOS, and defines the
# mask based on the energy range where the DOS is well-defined.
#

for struct in GaAs_sample_structures:
    dos_i, mask_i = pet_mad_dos_calculator.compute_DOS_and_mask_from_eigenvalues(
        torch.tensor(struct.info["eigvals"]), torch.tensor(struct.info["kweight"])
    )
    struct.info["DOS"] = dos_i.numpy().astype(np.float32)
    struct.info["mask"] = mask_i.numpy().astype(np.float32)

# Store the processed structures as a new XYZ file for fine-tuning
# ase.io.write("GaAs_processed_structures.xyz", GaAs_sample_structures)

# %%
# One would save the processed structures as a new XYZ file to to the
# fine-tuning, as demonstrated in the commented line in the cell above. However,
# for the purposes of this tutorial, we will demonstrate on datasets that have already
# been processed, as can be obtained from the `MaterialsCloud archive
# <https://archive.materialscloud.org/records/8ee9k-b7865>`_.
#

for i in ["train", "val", "test"]:
    url = (
        f"https://zenodo.org/records/19655792/files/"
        f"GaAs_sample_{i}_structures.xyz?download=1"
    )
    filename = f"GaAs_sample_{i}_structures.xyz"
    urllib.request.urlretrieve(url, filename)

    GaAs_sample_structures = ase.io.read(filename, ":")

    for struct in GaAs_sample_structures:
        DOS = torch.tensor(struct.info["DOS"])
        mask = torch.tensor(struct.info["mask"])
        padded_dos, padded_mask = pet_mad_dos_calculator.pad_dos_and_mask_for_training(
            DOS, mask
        )
        struct.info["trainingDOS"] = padded_dos.numpy().astype(np.float32)
        struct.info["trainingmask"] = padded_mask.numpy().astype(np.int32)
    ase.io.write(f"GaAs_processed_{i}.xyz", GaAs_sample_structures)


# %%
# Step 2: Model Loading
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we need to download the PET-MAD-DOS checkpoint and place it in the
# local directory so that it can be referenced in the metatrain YAML
# configuration files for fine-tuning.


# Download the checkpoint from Zenodo and copy it to the local directory
url = "https://zenodo.org/records/19655792/files/pet-mad-dos-v1.0.ckpt?download=1"

checkpoint_path = "pet-mad-dos-v1.0.ckpt"
download_with_retry(url, checkpoint_path)

# %%
# Step 3: Fine-tuning the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we are ready to fine-tune the model using the ``metatrain`` package. We
# will use the ``mtt train`` command to train the model, alongside with a
# supporting YAML configuration file that specifies the training hyperparameters.

# %%
# .. literalinclude:: finetune.yaml
#   :language: yaml


# Begin finetuning
run_command("mtt train finetune.yaml -o fine_tune-model.pt")

# %%
# Step 4: Evaluating the model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# After training, we can evaluate the model on the test set using the ``mtt eval``
# command, alongside with a supporting YAML configuration file that specifies the
# evaluation hyperparameters.

# %%
# .. literalinclude:: eval.yaml
#   :language: yaml
#

run_command("mtt eval fine_tune-model.pt eval.yaml -o pred.xyz")

output = ase.io.read("pred.xyz", ":")

# %%
# As we have only fine-tuned for 10 epochs on a tiny dataset, the model
# performance is not expected to be good. However, this simply serves as a
# demonstration of how to use the fine-tuned model.

predicted_DOS = torch.tensor(np.stack([s.info["mtt::dos"] for s in output])).float()
true_DOS = torch.tensor(
    np.stack([s.info["DOS"] for s in GaAs_sample_structures])
).float()
true_mask = torch.tensor(
    np.stack([s.info["mask"] for s in GaAs_sample_structures])
).float()

predicted_DOS, aligned_true_DOS, aligned_true_masks = pet_mad_dos_calculator.align_dos(
    predicted_DOS, true_DOS, true_mask
)

# Visualize the predictions and the true DOS on the same plot
i_struct = 0  # index of the structure to visualize

plt.plot(energies, predicted_DOS[i_struct], label="Raw Predicted DOS", color="green")

plt.plot(
    energies, aligned_true_DOS[i_struct], label="DFT DOS", linestyle="--", color="red"
)

plt.plot(
    energies,
    aligned_true_masks[i_struct] * 100,
    label="Mask x 100",
    linestyle="-.",
    color="black",
)

plt.xlim(-80, 10)
plt.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
plt.xlabel(r"Energy", size=16)
plt.ylabel(r"DOS [$\mathrm{states}/eV$]", size=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
