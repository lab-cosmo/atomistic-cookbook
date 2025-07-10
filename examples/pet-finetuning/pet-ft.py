"""
Fine-tuning the PET-MAD universal potential
===========================================

:Authors: Davide Tisi `@DavideTisi <https://github.com/DavideTisi>`_,
          Zhiyi Wang `@0WellyWang0 <https://github.com/0WellyWang0>`_,
          Cesare Malosso `@cesaremalosso <https://github.com/cesaremalosso>`_,
          Sofiia Chorna `@sofiia-chorna <https://github.com/sofiia-chorna>`_

This example demonstrates fine-tuning the PET-MAD model with `metatrain
<https://github.com/metatensor/metatrain>`_ on a small dataset of ethanol structures.

We cover two examples: 1) simple fine-tuning the pretrained PET-MAD model to adapt it to
specialized task by retraining it on a domain-specific dataset, and 2) two-stage
training strategy, first on non-conservative forces for efficiency, followed by
fine-tuning on conservative forces to ensure physical consistency.

PET-MAD is a universal machine-learning forcefield trained on `the MAD dataset
<https://arxiv.org/abs/2506.19674>`_ that aims to incorporate a very high degree of
structural diversity. It uses `the Point-Edge Transformer (PET)
<https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_,
an unconstrained architecture that achieves symmetry compliance through data
augmentation during training.

The goal of this recipe is to demonstrate the process, not to reach high accuracy.
Adjust the dataset size and hyperparameters accordingly if adapting this for a real
case.
"""

# %%

import subprocess
from collections import Counter
from glob import glob
from urllib.request import urlretrieve

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch
from metatrain.pet import PET
from sklearn.linear_model import LinearRegression


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821


# %%
# Fine-tuning of pretrained PET-MAD
# ---------------------------------
# While PET-MAD is trained as a universal model capable of handling a broad range of
# atomic environments, fine-tuning it allows to adapt this general-purpose model to a
# more specialized task. First, we need to get a checkpoint of the pre-trained PET-MAD
# model to start training from it. The checkpoint is stored in `Hugging Face repository
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
#
# The dataset is composed of 100 structures of ethanol. We fit a linear model based on
# atomic compositions to apply energy correction.

dataset = ase.io.read("data/ethanol.xyz", index=":", format="extxyz")

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
# Simple model fine-tuning
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# As the fine-tuning dataset is prepared, we will proceed with the training. We use the
# ``metatrain`` package to finetune the model. There are multiple strategies to apply
# fine-tuning, each described in `metatrain documentation
# <https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html>`_.
# In this example we demostrate a basic full fine-tuning strategy, which adapts all
# model weights to the new dataset starting from the pre-trained PET-MAD checkpoint. The
# process is configured by ``options.yaml``.
#
# .. literalinclude:: basic_ft_options.yaml
#   :language: yaml
#
# To launch training, run:
#
# .. code-block:: bash
#
#    mtt train options.yaml
#
# Or from Python:

subprocess.run(["mtt", "train", "basic_ft_options.yaml"], check=True)

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
#    mtt eval eval_ex1.yaml
#
# The evaluation YAML file contains lists the structures and corresponding reference
# quantities for the evaluation:
#
# .. literalinclude:: eval_ex1.yaml
#   :language: yaml
#
# We run it from Python:

subprocess.run(["mtt", "eval", "model.pt", "eval_ex1.yaml"], check=True)

# %%
#
# We visualize learning curves over epoch. The training log is stored in CSV format in
# the outputs directory.


def display_loss(csv_file):
    with open(csv_file, encoding="utf-8") as f:
        headers = f.readline().strip().split(",")

    cleaned_names = [h.strip().replace(" ", "_") for h in headers]

    train_log = np.genfromtxt(
        csv_file,
        delimiter=",",
        skip_header=2,
        names=cleaned_names,
    )

    plt.figure(figsize=(6, 4))
    plt.loglog(train_log["Epoch"], train_log["training_loss"], label="Training")
    plt.loglog(train_log["Epoch"], train_log["validation_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


csv_file = glob("outputs/*/*/train.csv")[-1]
display_loss(csv_file)


# %%
#
# The result of running the fine-tuning for 1000 epoches in visualised with
# ``chemiscope`` below:
import chemiscope  # noqa: E402


chemiscope.show_input("full_finetune_example.chemiscope.json")

# %%
#
# The learning curves for 1000 epoches displayed on the figure below.
#
# .. image:: basic_ft_loss.png
#    :align: center
#
#
# Two stage training strategy
# ---------------------------
#
# This approach accelerates training by first using non-conservative forces, which
# avoids costly backpropagation, then fine-tuning on conservative forces to ensure
# physical consistency. Non-conservative forces can lead to pathological behavior (see
# `preprint <https://arxiv.org/abs/2412.11569>`_), but this strategy mitigates such
# issues while maintaining efficiency.
#
# Non-conservative force training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Configure ``options.yaml`` to include non-conservative forces as a per-atom vector
# target:
#
# .. literalinclude:: nc_train_options.yaml
#   :language: yaml
#
# We run training from Python:

subprocess.run(["mtt", "train", "nc_train_options.yaml"], check=True)

# %%
#
# Let's evaluate the model:
#
# .. code-block:: bash
#
#    mtt eval model.pt eval_ex2.yaml
#
# Or from Python:

subprocess.run(["mtt", "eval", "model.pt", "eval_ex2.yaml"], check=True)


# %%
#
# The result of running non-conservative force learning for 1000 structures for 100
# epochs is present on the parity plot below. The left plot shows that the model's force
# predictions deviate due to the non-conservative training. On the right plot with the
# non-conservative forces align closely with targets but lack physical constraints,
# potentially leading to unphysical behavior.
#
# .. image:: nc_learning_res.png
#    :align: center
#    :width: 700px
#
# Finetuning on conservative forces
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The next step is to fine-tune the non-conservative checkpoint to conservative forces.
# Enable ``forces: on`` to compute them via backward propagation of gradients.
# Expectedly, the training will be slower.
#
# .. code-block:: yaml
#
#     training_set:
#       systems:
#         read_from: data/ethanol_train.xyz
#         length_unit: angstrom
#       targets:
#         energy:
#           unit: eV
#           forces: on
#         non_conservative_forces:
#           key: forces
#           type:
#             cartesian:
#               rank: 1
#           per_atom: true
#
# Run training, restarting from the previous checkpoint:
#
# .. code-block:: bash
#
#    mtt train c_ft_options.yaml --restart model.ckpt
#
# Or in Python:

subprocess.run(
    ["mtt", "train", "c_ft_options.yaml", "--restart", "model.ckpt"], check=True
)


# %%
#
# Let's evaluate the forces again:
#
# .. code-block:: bash
#
#    mtt eval model.pt eval_ex2.yaml
#
# Or from Python:

subprocess.run(["mtt", "eval", "model.pt", "eval_ex2.yaml"], check=True)

# %%
#
# After fine-tuning for 50 epochs, the updated parity plots show improved force
# predictions (left) with conservative forces.
#
# .. image:: c_ft_res.png
#    :align: center
#    :width: 700px
