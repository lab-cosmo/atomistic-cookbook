"""
Equivariant model for tensorial properties based on scalar features
===================================================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`_

In this example, we demonstrate how to train a `metatensor atomistic model
<https://docs.metatensor.org/latest/atomistic>`_ on dipole moments and polarizabilities
of small molecular systems. The model is trained with
`metatrain<https://metatensor.github.io/metatrain/latest/index.html>`_
and can then be used in an ASE calculator.
"""

# %%
# Core packages
import subprocess
from glob import glob

import ase.io

# Simulation and visualization tools
import chemiscope
import matplotlib.pyplot as plt
import metatensor as mts
import numpy as np

# Model wrapping and execution tools
from featomic.clebsch_gordan import cartesian_to_spherical
from metatensor import Labels, TensorBlock, TensorMap


# %%
# Load the training data
# ^^^^^^^^^^^^^^^^^^^^^^
# We load a simple dataset of :math:`\mathrm{C}_5\mathrm{NH}_7` molecules and
# their polarizability tensors stored in extended XYZ format.
# We also visualize the polarizability as ellipsoids to demonstrate the
# anisotropy of this molecular property.

molecules = ase.io.read("data/qm7x_reduced_100_CHNO.xyz", ":")

arrows = chemiscope.ase_vectors_to_arrows(molecules, "mu", scale=5)
arrows["parameters"]["global"]["color"] = "#008000"

ellipsoids = chemiscope.ase_tensors_to_ellipsoids(molecules, "alpha", scale=0.15)
ellipsoids["parameters"]["global"]["color"] = "#FF8800"
cs = chemiscope.show(
    molecules,
    shapes={"mu": arrows, "alpha": ellipsoids},
    mode="structure",
    settings=chemiscope.quick_settings(
        structure_settings={"shape": ["mu", "alpha"]},
        trajectory=True,
    ),
)

cs

# %%
# Prepare the targets for training
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We extract the dipole moments and polarizability tensors from the extended XYZ file
# and store them in a :class:`metatensor.torch.TensorMap` as Cartesian tensors.

np.random.seed(0)

indices = np.arange(len(molecules))
n_train = int(0.80 * len(molecules))
n_val = int(0.10 * len(molecules))
n_test = int(0.10 * len(molecules))

np.random.shuffle(indices)

train_indices = indices[:n_train]
val_indices = indices[n_train : n_train + n_val]
test_indices = indices[n_train + n_val : n_train + n_val + n_test]

for idx, filename in zip(
    [train_indices, val_indices, test_indices], ["training", "validation", "test"]
):
    subset = [molecules[i] for i in idx]
    ase.io.write(
        f"{filename}_set.xyz",
        subset,
        write_info=False,
    )

    # Create Cartesian tensormaps
    mu = np.array([molecule.info["mu"] for molecule in subset])
    cartesian_mu = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                samples=Labels("system", np.arange(len(subset)).reshape(-1, 1)),
                components=[Labels.range("xyz", 3)],
                properties=Labels.single(),
                values=mu.reshape(len(subset), 3, 1),
            )
        ],
    )

    alpha = np.array([molecule.info["alpha"].reshape(3, 3) for molecule in subset])
    cartesian_alpha = TensorMap(
        Labels.single(),
        [
            TensorBlock(
                samples=Labels("system", np.arange(len(subset)).reshape(-1, 1)),
                components=[Labels.range(f"xyz_{i}", 3) for i in range(1, 3)],
                properties=Labels.single(),
                values=alpha.reshape(len(subset), 3, 3, 1),
            )
        ],
    )

    # Convert to spherical tensormaps
    spherical_mu = mts.remove_dimension(
        cartesian_to_spherical(cartesian_mu, ["xyz"]), "keys", "_"
    )
    spherical_alpha = mts.remove_dimension(
        cartesian_to_spherical(cartesian_alpha, ["xyz_1", "xyz_2"]), "keys", "_"
    )

    # Save the spherical tensormaps to disk, ensuring contiguous memory layout
    mts.save(f"{filename}_dipoles.mts", mts.make_contiguous(spherical_mu))
    mts.save(f"{filename}_polarizabilities.mts", mts.make_contiguous(spherical_alpha))

# %%
# Model architecture
# ^^^^^^^^^^^^^^^^^^
# TODO: explain the model architecture

# %%
# Training and evaluation of the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# We use the `metatrain` package to train the model. The training is performed
# using the `metatrain` command line interface. The command to start the training
# is:
# .. code-block:: shell
#    mtt train data/options.yaml
#
# The options file contains information about the model architecture and the
# training parameters:
#
# .. literalinclude:: data/options.yaml
#    :language: yaml
#

# %%
subprocess.run(
    [
        "mtt",
        "train",
        "data/options.yaml",
    ],
    check=True,
)
# %%
# We can visualize the training and validation errors as functions of the epoch.
# The training log is stored in csv format in the `outputs` folder.

train_log = np.genfromtxt(
    glob("outputs/*/*/train.csv")[-1],
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)[1:]

plt.loglog(train_log["Epoch"], train_log["training_loss"], label="Training")
plt.loglog(train_log["Epoch"], train_log["validation_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and validation loss")

# %%
# We can evaluate the model on the test set using again the `metatrain` command line
# interface. The command to evaluate the model is
# is:
# .. code-block:: shell
#    mtt eval model.pt eval.yaml -e extensions -o test_results.mts
#
# The evaluation file contains information about the structures to evaluate and the
# target quantities:
#
# .. literalinclude:: eval.yaml
#    :language: yaml
#

# %%
subprocess.run(
    [
        "mtt",
        "eval",
        "model.pt",
        "data/eval.yaml",
        "-e",
        "extensions",
        "-o",
        "test_results.mts",
    ],
    check=True,
)

# %%
# We can now load the predictions and the targets from the test set and compare them.
# The predictions are stored in the `test_results_mtt::dipole.mts` and
# `test_results_mtt::polarizability.mts` files and the targets are in `test_dipoles.mts`
# and `test_polarizabilities.mts` files. We can load them using the `metatensor.load`
# function.

prediction_test = {
    "dipole": mts.load("test_results_mtt::dipole.mts"),
    "polarizability": mts.load("test_results_mtt::polarizability.mts"),
}

target_test = {
    "dipole": mts.load("test_dipoles.mts"),
    "polarizability": mts.load("test_polarizabilities.mts"),
}

test_set_molecules = ase.io.read("test_set.xyz", ":")
natm = np.array([len(mol) for mol in test_set_molecules])

# %%
# We can now compare the predictions and the targets by visualzing a parity plot

color_per_lambda = {0: "C0", 1: "C1", 2: "C2"}

fig, axes = plt.subplots(1, 2)
for ax, key in zip(axes, prediction_test):

    ax.set_aspect("equal")

    pred = prediction_test[key]
    target = target_test[key]

    for k in target.keys:
        assert k in pred.keys
        o3_lambda = int(k["o3_lambda"])
        label = rf"$\lambda={o3_lambda}$"
        x = target[k].values[..., 0] / natm[:, np.newaxis]
        y = pred[k].values[..., 0] / natm[:, np.newaxis]
        ax.plot(
            x.flatten(),
            y.flatten(),
            ".",
            color=color_per_lambda[o3_lambda],
            label=label,
        )

    xmin, xmax = ax.get_xlim()
    ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    ax.set_xlabel("Target (a.u./atom)")
    ax.set_ylabel("Prediction (a.u./atom)")

    ax.set_title(key.capitalize())

    ax.legend()
fig.tight_layout()

# %%
