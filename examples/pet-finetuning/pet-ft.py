"""
Fine-tuning the PET-MAD universal potential
===========================================

:Authors: Davide Tisi `@DavideTisi <https://github.com/DavideTisi>`_,
          Zhiyi Wang `@0WellyWang0 <https://github.com/0WellyWang0>`_,
          Cesare Malosso `@cesaremalosso <https://github.com/cesaremalosso>`_

This example demonstrates how to finetune the PET-MAD model with `metatrain
<https://github.com/metatensor/metatrain>`_. PET-MAD is a "universal" machine-learning
forcefield trained on the MAD dataset that aims to incorporate a very high degree of
structural diversity.

The Point-Edge Transformer (PET) is an unconstrained architecture that achieves a high
degree of symmetry compliance through data augmentation during training. The
unconstrained nature of the model simplifies its implementation and structure, making it
computationally efficient and very expressive. See the original `PET paper
<https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_
for more details.

The Massive Atomic Diversity (MAD) dataset was constructed to emphasize structural
diversity by distorting the composition and structure of stable configurations. It
combines stable inorganic structures from the `MC3D
<https://mc3d.materialscloud.org/>`_, 2D structures from the `MC2D
<https://mc2d.materialscloud.org/>`_, and molecular crystals from the `ShiftML
<https://archive.materialscloud.org/record/2022.147>`_. Despite containing fewer than
100,000 structures, PET-MAD trained on this dataset has already achieved
state-of-the-art performance. Electronic structure calculations use the PBEsol
functional, optimized for consistency rather than per-structure accuracy. Further
details are available in the MAD dataset `preprint <https://arxiv.org/abs/2506.19674>`_.

While PET-MAD is trained as a universal model capable of handling a broad range of
atomic environments, fine-tuning of it allows to adapt this general-purpose model to a
more specialized task by retraining it on a smaller domain-specific dataset.
"""

# %%
#
# In this example, we will finetune PET-MAD on a very simple dataset composed of 100
# structres of ethanol. To get the PET-MAD checkpoint and obtain all the necessary
# dependencies, you can simply use pip to install the `PET-MAD package
# <https://github.com/lab-cosmo/pet-mad>`_:
#
# .. code-block:: bash
#
#     pip install pet-mad

import subprocess
from collections import Counter
from urllib.request import urlretrieve

import ase.io
import numpy as np
import torch
from metatrain.pet import PET
from sklearn.linear_model import LinearRegression


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821


# %%
# Download the checkpoint
# -----------------------
#
# To get the checkpoint, you can use wget or curl to download it from the `HuggingFace
# repository <https://huggingface.co/lab-cosmo/pet-mad>`_:
#
# .. code-block:: bash
#
#  wget https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt
#
# We'll download it directly:

url = "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"
checkpoint_path = "pet-mad-latest.ckpt"

urlretrieve(url, checkpoint_path)

# %%
# Prepare the dataset
# -------------------
#
# DFT-calculated energies often contain systematic shifts due to the choice of
# functional, basis set, or pseudopotentials. If left uncorrected, such shifts can
# mislead the fine-tuning process. To align our finetuning dataset with PET-MAD energy
# reference, we apply a linear correction based on atomic compositions. First, we define
# a helper functions:


def load_reference_energies(checkpoint_path):
    """
    Extract atomic reference energies from PET-MAD checkpoint.

    It returns a mapping element number to their reference energies (eV), e.g.: {'1':
    -1.23, '2': -5.67, ...}
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    pet_model = PET.load_checkpoint(checkpoint, "finetune")

    energy_values = pet_model.additive_models[0].weights["energy"].block().values
    atomic_numbers = checkpoint["model_data"]["dataset_info"].atomic_types

    return dict(zip(atomic_numbers, energy_values))


def get_compositional_energy(atoms, energy_per_atom):
    """Calculates total energy from atomic composition and per-atom energies"""
    counts = Counter(atoms.get_atomic_numbers())
    return sum(energy_per_atom[Z] * count for Z, count in counts.items())


# %%
# Energy correction
# ^^^^^^^^^^^^^^^^^
#
# The dataset is composed of 100 structures of ethanol. We fit a linear model based on
# atomic compositions to apply energy correction.

dataset = ase.io.read("data/ethanol_reduced_100.xyz", index=":", format="extxyz")

# Extract DFT energies and atomic compositions
dft_energies = [atoms.get_potential_energy() for atoms in dataset]
compositions = [Counter(atoms.get_atomic_numbers()) for atoms in dataset]
elements = sorted({element for composition in compositions for element in composition})

X = np.array(
    [[composition.get(elem, 0) for elem in elements] for composition in compositions]
)
y = np.array(dft_energies)

# Fit linear model to estimate per-element DFT contributions
correction_model = LinearRegression()
correction_model.fit(X, y)

coeffs = dict(zip(elements, correction_model.coef_))

# Get reference energies from PET-MAD
ref_energies = load_reference_energies(checkpoint_path)

# Apply correction to each structure
for atoms, E_dft in zip(dataset, dft_energies):
    E_comp_dft = get_compositional_energy(atoms, coeffs)
    E_comp_ref = get_compositional_energy(atoms, ref_energies)

    corrected_energy = E_dft - E_comp_dft + E_comp_ref - correction_model.intercept_

    atoms.info["energy-corrected"] = corrected_energy.item()

ase.io.write("data/ethanol_corrected.xyz", dataset, format="extxyz")


# %%
# Model finetuning
# ----------------
#
# We will use the `metatrain` package to finetune the model. There are multiple
# strategies to apply finetuning, each described in `metatrain documentation
# <https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html>`_
# The process is configured by yaml file (``options.yaml``) and executed with the simple
# command:
#
# .. code-block:: bash
#
# mtt train data/options.yaml

# TODO: explain quickly what is in options to finetune

subprocess.run(["mtt", "train", "data/options.yaml"], check=True)

# %%
#
# Small example of LoRA
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO

# %%
#
# Finetuning on the forces
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# TODO
