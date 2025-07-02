"""
Fine-tuning the PET-MAD universal potential
===========================================

:Authors: Davide Tisi `@DavideTisi <https://github.com/DavideTisi>`_,
          Zhiyi Wang `@0WellyWang0 <https://github.com/0WellyWang0>`_,
          Cesare Malosso `@cesaremalosso <https://github.com/cesaremalosso>`_

This example demonstrates fine-tuning the PET-MAD model with `metatrain
<https://github.com/metatensor/metatrain>`_. PET-MAD is a universal machine-learning
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
atomic environments, fine-tuning it allows to adapt this general-purpose model to a more
specialized task by retraining it on a smaller domain-specific dataset.

In this example, we fine-tune PET-MAD on a minimal dataset of 100 ethanol structures
using three strategies: full fine-tuning, LoRA, and learning of forces directly. The
goal is to demonstrate the process, not to reach high accuracy. Adjust the dataset size
and hyperparameters accordingly if adapting this for a real case.
"""

# %%

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
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# First, we need to get a checkpoint of the pre-trained PET-MAD model to start training
# from it. The checkpoint is stored in `Hugging Face repository
# <https://huggingface.co/lab-cosmo/pet-mad>`_ and can be fetched using wget or curl:
#
# .. code-block:: bash
#
#    wget https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt  # noqa: E501
#
# We'll download it directly:

url = "https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt"
checkpoint_path = "pet-mad-latest.ckpt"

urlretrieve(url, checkpoint_path)

# %%
# Prepare the fine-tuning dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# DFT-calculated energies often contain systematic shifts due to the choice of
# functional, basis set, or pseudopotentials. If left uncorrected, such shifts can
# mislead the fine-tuning process. We apply a linear correction based on atomic
# compositions to align our fine-tuning dataset with PET-MAD energy reference. First, we
# define a helper function to load reference energies from PET-MAD.


def load_reference_energies(checkpoint_path):
    """
    Extract atomic reference energies from the PET-MAD checkpoint.

    It returns a mapping of elements to their reference energies (eV), e.g.: {'1':
    -1.23, '2': -5.67, ...}
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    pet_model = PET.load_checkpoint(checkpoint, "finetune")

    energy_values = pet_model.additive_models[0].weights["energy"].block().values
    atomic_numbers = checkpoint["model_data"]["dataset_info"].atomic_types

    return dict(zip(atomic_numbers, energy_values))


# %%
# Energy correction
# +++++++++++++++++
#
# The dataset is composed of 100 structures of ethanol. We fit a linear model based on
# atomic compositions to apply energy correction.

dataset = ase.io.read("data/ethanol_reduced_100.xyz", index=":", format="extxyz")

# Extract DFT energies and compositions
dft_energies = [atoms.get_potential_energy() for atoms in dataset]
compositions = [Counter(atoms.get_atomic_numbers()) for atoms in dataset]
elements = sorted({element for composition in compositions for element in composition})

X = np.array([[comp.get(elem, 0) for elem in elements] for comp in compositions])
y = np.array(dft_energies)

# Fit linear model to estimate DFT per-element energy
correction_model = LinearRegression()
correction_model.fit(X, y)

coeffs = dict(zip(elements, correction_model.coef_))

# %%
#
# Apply correction to each structure


def get_compositional_energy(atoms, energy_per_atom):
    """Calculates total energy from atomic composition and per-atom energies"""
    counts = Counter(atoms.get_atomic_numbers())
    return sum(energy_per_atom[Z] * count for Z, count in counts.items())


# Get reference energies from PET-MAD
ref_energies = load_reference_energies(checkpoint_path)

# Apply correction
for atoms, E_dft in zip(dataset, dft_energies):
    E_comp_dft = get_compositional_energy(atoms, coeffs)
    E_comp_ref = get_compositional_energy(atoms, ref_energies)

    corrected_energy = E_dft - E_comp_dft + E_comp_ref - correction_model.intercept_

    atoms.info["energy-corrected"] = corrected_energy.item()


# Split corrected dataset and save it
np.random.seed(42)
indices = np.random.permutation(len(dataset))
n = len(dataset)
n_val = n_test = int(0.1 * n)
n_train = n - n_val - n_test

train = [dataset[i] for i in indices[:n_train]]
val = [dataset[i] for i in indices[n_train : n_train + n_val]]
test = [dataset[i] for i in indices[n_train + n_val :]]

ase.io.write("data/ethanol_train.xyz", train, format="extxyz")
ase.io.write("data/ethanol_val.xyz", val, format="extxyz")
ase.io.write("data/ethanol_test.xyz", test, format="extxyz")


# %%
# Model fine-tuning
# ^^^^^^^^^^^^^^^^^
#
# As the fine-tuning dataset is prepared, we will proceed with the training. We use the
# ``metatrain`` package to finetune the model. There are multiple strategies to apply
# fine-tuning, each described in `metatrain documentation
# <https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html>`_.
# The process is configured by ``options.yaml``.
#
# Basic fine-tuning
# +++++++++++++++++
#
# Here is an example of a configuration for a basic full-finetuning strategy, which
# adapts all model weights to the new dataset:
#
# .. literalinclude:: basic_options.yaml
#   :language: yaml
#
# To launch training, run:
#
# .. code-block:: bash
#
#    mtt train options.yaml
#
# Or from Python:

subprocess.run(["mtt", "train", "basic_options.yaml"], check=True)

# %%
#
# After the training, ``mtt train`` outputs the ``model.ckpt`` (fine-tuned checkpoint)
# and ``model.pt`` (exported fine-tuned model) files in both the current directory and
# ``output/YYYY-MM-DD/HH-MM-SS/``.
#
# We evaluate the model on the test set using the ``metatrain`` command-line interface:
#
# .. code-block:: bash
#
#    mtt eval eval.yaml
#
# The evaluation YAML file contains lists the structures and corresponding reference
# quantities for the evaluation:
#
# .. literalinclude:: eval.yaml
#   :language: yaml
#
# We run evaluation from Python:

subprocess.run(["mtt", "eval", "model.pt", "eval.yaml"], check=True)


# %%
#
# The result of running the fine-tuning for 1000 epochs in visualised with
# ``chemiscope`` below:
import chemiscope  # noqa: E402


chemiscope.show_input("full_finetune_example.chemiscope.json")

# %%
#
# Fine-tuning with LoRA
# +++++++++++++++++++++
#
# LoRA updates a low-rank subset of parameters. To use this strategy, we will reuse the
# same configuration file with update with LoRA hyperparameters, ``alpha`` and ``rank``.
# Parameter details are available in `metatrain documentation
# <https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html#lora-fine-tuning>`_.
#
# .. code-block:: yaml
#
#  architecture:
#    training:
#      finetune:
#        method: "lora"
#        read_from: pet-mad-latest.ckpt
#        config:
#          alpha: 0.1  # scaling factor
#          rank: 4  # rank of the low-rank adaptation matrices
#
subprocess.run(["mtt", "train", "lora_options.yaml"], check=True)


# %%
#
# Fine-tuning on the forces
# +++++++++++++++++++++++++
#
# To include non-conservative forces in training, set the ``non_conservative_forces``
# target as a per-atom vector in the ``options.yaml``:
#
# .. code-block:: yaml
#
#  training_set:
#    systems:
#      read_from: "data/ethanol_corrected.xyz"  # path to the finetuning dataset
#      length_unit: angstrom
#    targets:
#      energy:
#        key: "energy-corrected"  # name of the target value
#        unit: "eV"
#      non_conservative_forces:
#        quantity: force
#        key: "forces"
#        unit: "eV/A"
#        per_atom: true
#        type:
#          cartesian:
#            rank: 1

subprocess.run(["mtt", "train", "learn_forces_options.yaml"], check=True)
