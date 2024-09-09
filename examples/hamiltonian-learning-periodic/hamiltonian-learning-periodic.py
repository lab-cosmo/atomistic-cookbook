"""
Periodic Hamiltonian learning Tutorial
======================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`_
"""

# %%
# This tutorial explains how to train a machine learning model for the
# electronic Hamiltonian of a periodic system.
#


# %%
# First, import the necessary packages
#

import os
import warnings
import zipfile

import lightning.pytorch as pl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from lightning.pytorch.callbacks import EarlyStopping
from matplotlib.animation import FuncAnimation
from mlelec.callbacks.logging import LoggingCallback
from mlelec.data.derived_properties import compute_eigenvalues
from mlelec.data.mldataset import MLDataset
from mlelec.data.qmdataset import QMDataset
from mlelec.models.equivariant_nonlinear_lightning import (
    LitEquivariantNonlinearModel,
    MLDatasetDataModule,
    MSELoss,
)
from mlelec.models.equivariant_nonlinear_model import EquivariantNonLinearity
from mlelec.utils.pbc_utils import blocks_to_matrix
from mlelec.utils.plot_utils import plot_bands_frame
from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys_integrated
from torchviz import make_dot


warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


# %%
# Step 0: Get Data and Prepare Data Set
# -------------------------------------
#


# %%
# Download Data
# ~~~~~~~~~~~~~
#

filename = "data.zip"
if not os.path.exists(filename):
    url = (
        "https://github.com/curiosity54/mlelec/raw/"
        "tutorial_periodic/examples/periodic_tutorial/data.zip"
    )
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)

with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("./")


# %%
# Load structures and DFT data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# %%
# The DFT calculations are done on a minimal STO-3G basis. The $n$, $l$, $m$
# quantum numbers for species C are given below.
#

basis = "sto-3g"
orbitals = {
    "sto-3g": {6: [[1, 0, 0], [2, 0, 0], [2, 1, -1], [2, 1, 0], [2, 1, 1]]},
}

# %%
# DFT data is stored in a `QMDataset` instance.
#

qmdata = QMDataset.from_file(
    frames_path="data/C2.xyz",
    fock_realspace_path="data/graphene_fock.npy",
    overlap_realspace_path="data/graphene_ovlp.npy",
    kmesh_path="data/kmesh.dat",
    dimension=2,
    device="cpu",
    orbs_name=basis,
    orbs=orbitals[basis],
)


# %%
# Visualize the equivariant structure of the Hamiltonians
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# Under a global rotation of the atomic system, the Hamiltonian matrix
# elements rotate according to the equivariant character of the involved
# orbitals.
#
# Here is an animation of a trajectory along a Lissajous curve in 3D
# space, alongside a colormap representing the Hamiltonian matrix elements
# of the graphene unit cell in a minimal STO-3G basis.
#


image_files = sorted(
    [
        f"data/frames/{f}"
        for f in os.listdir("./data/frames")
        if f.startswith("rot_") and f.endswith(".png")
    ]
)
images = [mpimg.imread(img) for img in image_files]
fig, ax = plt.subplots()
img_display = ax.imshow(images[0])  # Initialize with the first image
ax.axis("off")  # Turn off axis


def update(frame):
    img_display.set_data(images[frame])
    return [img_display]


# Create the animation using FuncAnimation
ani = FuncAnimation(fig, update, frames=len(images), interval=20, blit=True)

# %%
# Instantiate machine learning data set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# Define the hyperparameters for the ACDC descriptors. There are hypers
# for the single-center (SC) :math:`\lambda`-SOAP and two-center (TC)
# descriptors.
#


# %%
# The single and two-center descriptors have very similar hyperparameters,
# except for the cutoff radius, which is larger for the two-center one, in
# order to explicitly include far away pairs of atoms.
#

SC_HYPERS = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.5,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
}

TC_HYPERS = {
    "cutoff": 6.0,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
}


# %%
# We then use the above defined hyperparameters to compute the descriptor
# and initialize a ``MLDataset`` instance.
#
# In addition to computing the descriptors, ``MLDataset`` takes the data
# stored in the ``QMDataset`` instance and puts it in a form required to
# train a ML model.
#


# %%
# ``item_names`` contains the names of the quantities we want to compute
# to target in the ML model training or to be able to access later.
#
# ``fock_blocks`` is a ``metatensor.Tensormap`` containing the coupled
# blocks the Hamiltonian matrices have been divided into.
#

mldata = MLDataset(
    qmdata,
    item_names=["fock_blocks", "overlap_kspace"],
    hypers_atom=SC_HYPERS,
    hypers_pair=TC_HYPERS,
    lcut=4,
    train_frac=0.7,
    val_frac=0.2,
    test_frac=0.1,
    shuffle=True,
    model_basis=orbitals["sto-3g"],
)


# %%
# ``mldata.features`` is ``metatensor.TensorMap`` containing the
# stuctures’ descriptors
#

mldata.features


# %%
# ``mldata.items`` can then be accessed as elements of a ``namedtuple``,
# e.g.:
#

mldata.items.fock_blocks


# %%
# Step 1: Build a machine learning model for the electronic Hamiltonian of
# graphene in a minimal basis
# --------------------------------------------------------------------------------
#


# %%
# Instantiate a ``pytorch_lightning`` data module from the ``MLDataset``
# instance
#

data_module = MLDatasetDataModule(mldata, batch_size=16, num_workers=0)

model = LitEquivariantNonlinearModel(
    mldata=mldata,
    # The number of hidden layers
    nlayers=1,
    # The number of neurons in each hidden layer
    nhidden=64,
    # What nonlinear activation function to apply to the invariant hidden
    # features
    activation="SiLU",
    # Type of optimizer
    optimizer="adam",
    # Initial learning rate for the optimizer
    learning_rate=1e-3,
    # learning rate scheduler settings
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.7,
    lr_scheduler_min_lr=1e-6,
    # Use the mean square error as loss function
    loss_fn=MSELoss(),
)


# %%
# This is what the architecture of one of the submodels we use for each
# Hamiltonian block looks like.
#
# Here we visualize the models’ graphs using ``torchviz``
#


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dots = {}
    for k in model.model.model.in_keys:
        submodel = model.model.model.get_module(k)
        descriptor = map_targetkeys_to_featkeys_integrated(mldata.features, k).values
        output = submodel.forward(descriptor)
        dots[tuple(k.values.tolist())] = make_dot(
            output, dict(submodel.named_parameters())
        )
        dots[tuple(k.values.tolist())].graph_attr.update(size="150,150")


# %%
# The first submodel
#

list(dots.values())[0].render("graph_output", format="png")
img = mpimg.imread("graph_output.png")
plt.figure(figsize=(10, 20))
plt.imshow(img)
plt.axis("off")


# %%
# We apply nonlinear activation to invariants obtained from the
# equivariant blocks
#

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    m = EquivariantNonLinearity(torch.nn.SiLU(), layersize=10)
    y = m.forward(torch.randn(3, 3, 10))
    dot = make_dot(y, dict(m.named_parameters()))
    dot.graph_attr.update(size="150,150")
    dot.render("graph_output", format="png")
    img = mpimg.imread("graph_output.png")
    plt.figure(figsize=(6, 10))
    plt.imshow(img)
    plt.axis("off")


# %%
# Set up the training loop
# ~~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# We set up a callback for logging training information such as the
# training and validation losses.
#

logger_callback = LoggingCallback(log_every_n_epochs=5)


# %%
# We set up an early stopping criterion to stop the training when the
# validation loss function stops decreasing.
#

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=1e-3, patience=10, verbose=False, mode="min"
)


# %%
# We define a ``lighting.pytorch.Trainer`` instance to handle the training
# loop. We train for 200 epochs
#

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="cpu",
    check_val_every_n_epoch=10,
    callbacks=[early_stopping, logger_callback],
    logger=False,
    enable_checkpointing=False,
)

trainer.fit(model, data_module)


# %%
# Evaluate model accuracy
# -----------------------
#


# %%
# We compute the test set loss to assess the model accuracy on an unseen
# set of structures
#

trainer.test(model, data_module)


# %%
# We can compute the predicted Hamiltonians from the trained model
#

predicted_blocks = model.forward(mldata.features)


# %%
# And convert the coupled blocks to Hamiltonian matrices
#

frames_dict = {A: qmdata.structures[A] for A in range(len(qmdata))}
HT = blocks_to_matrix(predicted_blocks, orbitals["sto-3g"], frames_dict, detach=True)
Hk = qmdata.bloch_sum(HT, is_tensor=True)


# %%
# We can then compute the eigenvalues to assess the model accuracy in
# predicting the band structure
#

target_eigenvalues = compute_eigenvalues(qmdata.fock_kspace, qmdata.overlap_kspace)
predicted_eigenvalues = compute_eigenvalues(Hk, qmdata.overlap_kspace)

Hartree = 27.211386024367243  # eV

plt.rcParams["font.size"] = 14
fig, ax = plt.subplots()
ax.set_aspect("equal")

x_text = 0.38
y_text = 0.2
d = 0.06

for i, (idx, label) in enumerate(
    zip(
        [mldata.train_idx, mldata.val_idx, mldata.test_idx],
        ["train", "validation", "test"],
    )
):

    target = (
        torch.stack([target_eigenvalues[i] for i in idx]).flatten().detach() * Hartree
    )
    prediction = (
        torch.stack([predicted_eigenvalues[i] for i in idx]).flatten().detach()
        * Hartree
    )

    non_core_states = target > -100
    rmse = np.sqrt(
        np.mean(
            (target.numpy()[non_core_states] - prediction.numpy()[non_core_states]) ** 2
        )
    )
    ax.scatter(target, prediction, marker=".", label=label, alpha=0.5)
    ax.text(
        x=x_text,
        y=y_text - d * i,
        s=rf"$\mathrm{{RMSE_{{{label}}}={rmse:.2f}\,eV}}$",
        transform=ax.transAxes,
    )

xmin, xmax = ax.get_xlim()
ax.plot([xmin, xmax], [xmin, xmax], "--k")
ax.set_xlim(xmin, xmax)
ax.set_ylim(xmin, xmax)
ax.legend()
ax.set_xlabel("Target eigenvalues (eV)")
ax.set_ylabel("Predicted eigenvalues (eV)")

# %%
# The core-state are more difficult to align compared to higher-energy states.
# Longer trainings are needed to accurately capture their energies.

# %%
# Graphene band structure
# ~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# Apart from the eigenvalues on a mesh in the Brillouin zone, we can use
# the real-space Hamiltonians predicted by the model to compute the band
# structure along high-symmetry paths.
#

fig, ax = plt.subplots(figsize=(8, 4.8))

idx = 0

handles = []

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Plot reference
    ax, h_ref = plot_bands_frame(
        qmdata.fock_realspace[idx], idx, qmdata, ax=ax, color="blue", lw=2
    )

    # Plot prediction
    ax, h_pred = plot_bands_frame(
        HT[idx], idx, qmdata, ax=ax, color="lime", ls="--", lw=2
    )

ax.set_ylim(-30, 30)
ax.legend(
    [h_ref, h_pred],
    ["Reference", "Prediction"],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
fig.tight_layout()

# %%
# Even with a poorly converged model, the band structure is quite accurate,
# especially for occupied states.
