"""
Hamiltonian learning
=============================

This tutorial explains how to train a machine learning model for the a molecular system.
"""

# %%
# First, import the necessary packages
#

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from ase.units import Hartree
from IPython.utils import io
from mlelec.features.acdc import compute_features_for_target
from mlelec.targets import drop_zero_blocks
from mlelec.utils.plot_utils import plot_losses


os.environ["PYSCFAD_BACKEND"] = "torch"
import mlelec.metrics as mlmetrics  # noqa: E402
from mlelec.data.dataset import MLDataset, MoleculeDataset, get_dataloader  # noqa: E402
from mlelec.models.linear import LinearTargetModel  # noqa: E402
from mlelec.train import Trainer  # noqa: E402
from mlelec.utils.property_utils import instantiate_mf  # noqa: E402


torch.set_default_dtype(torch.float64)

# sphinx_gallery_thumbnail_number = 3


# %%
# Get Data and Prepare Data Set
# -----------------------------
#
# We use here as an example a set of 100 ethane structures.
# For all structures, we ran pyscf calculations with the minimal
# basis set STO-3G and def2-TZVP.
# We stored the output hamiltonian, dipole moments, polarisability
# and overlap matrices in hickle format.
# The first three will be used as target in the following learning exercise.

# %%
# Set parameters for training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Set the parameters for the training, including the dataset set size 
# and split, the batch size, learning rate and weights for the individual
# components of eigenvalues, dipole and polarisability.
# We additionally define a folder name, in which the results are saved.
# Optionally, noise can be added to the ridge regression fit.


NUM_FRAMES = 100  #
BATCH_SIZE = 4  # 50
NUM_EPOCHS = 100
SHUFFLE_SEED = 42
TRAIN_FRAC = 0.7
TEST_FRAC = 0.1
VALIDATION_FRAC = 0.2
EARLY_STOP_CRITERION = 20
VERBOSE = 10
DUMP_HIST = 50
LR = 1e-1  # 5e-4
VAL_INTERVAL = 1
W_EVA = 1  # 1e4
W_DIP = 0  # 1e3
W_POL = 0  # 1e2
DEVICE = "cpu"

ORTHOGONAL = True  # set to 'FALSE' if working in the non-orthogonal basis
FOLDER_NAME = "FPS/ML_orthogonal_eva"
NOISE = False

# %%
# Create folders and save parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs(f"{FOLDER_NAME}/model_output", exist_ok=True)


def save_parameters(file_path, **params):
    with open(file_path, "w") as file:
        for key, value in params.items():
            file.write(f"{key}: {value}\n")


# Call the function with your parameters
save_parameters(
    f"{FOLDER_NAME}/parameters.txt",
    NUM_FRAMES=NUM_FRAMES,
    BATCH_SIZE=BATCH_SIZE,
    NUM_EPOCHS=NUM_EPOCHS,
    SHUFFLE_SEED=SHUFFLE_SEED,
    TRAIN_FRAC=TRAIN_FRAC,
    TEST_FRAC=TEST_FRAC,
    VALIDATION_FRAC=VALIDATION_FRAC,
    LR=LR,
    VAL_INTERVAL=VAL_INTERVAL,
    W_EVA=W_EVA,
    W_DIP=W_DIP,
    W_POL=W_POL,
    DEVICE=DEVICE,
    ORTHOGONAL=ORTHOGONAL,
    FOLDER_NAME=FOLDER_NAME,
)


# %%
# Create Datasets
# ---------------
#
# We use the dataloader of the mlelec package, and load the ethane
# dataset for the defined number of slices. 
# First, we load all relavant data (geometric structures, 
# auxiliary matrices (overlap and orbitals), and
# targets (fock, dipole moment and polarisablity)) into a molecule dataset.
# We do this for the minimal (STO-3G), as well as a larger basis (lb, def2-TZVP). 
# The larger basis has additional basis functions on the valence electrons.
# The dataset, we can then load into our dataloader ml_data, together with some
# settings on how to get data from the dataloader.
# Finally, we define the desired dataset split for training, validation,
# and testing.

molecule_data = MoleculeDataset(
    mol_name="ethane",
    use_precomputed=False,
    path="data/ethane/",
    aux_path="data/ethane/sto-3g",
    frame_slice=slice(0, NUM_FRAMES),
    device=DEVICE,
    aux=["overlap", "orbitals"],
    lb_aux=["overlap", "orbitals"],
    target=["fock"],
    lb_target=["fock"],
)

ml_data = MLDataset(
    molecule_data=molecule_data,
    device=DEVICE,
    model_strategy="coupled",
    shuffle=True,
    shuffle_seed=SHUFFLE_SEED,
    orthogonal=ORTHOGONAL,
)

ml_data._split_indices(
    train_frac=TRAIN_FRAC, val_frac=VALIDATION_FRAC, test_frac=TEST_FRAC
)

# %%
# Compute Features
# ----------------
#
# We use `featomic <https://github.com/metatensor/featomic/>`_
# to compute the atomic and pair-wise features for each structure
# in the trainingset. The output is a tensormap, containing all atomic
# and pair-wise features ordered by the atoms, and descriptor parameters
# (inversion_sigma, spherical_harmonics_l, species_center, species_neighbor,
# and block type).
# We then assign the computed features to the machine learning dataloader (ml_data).

hypers = {
    "cutoff": 2.5,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}
hypers_pair = {
    "cutoff": 2.5,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    # "radial_scaling": {"Willatt2018": {"scale": 2.0, "rate": 1.0, "exponent": 4}},
}

features = compute_features_for_target(
    ml_data, device=DEVICE, hypers=hypers, hypers_pair=hypers_pair
)
ml_data._set_features(features)


train_dl, val_dl, test_dl = get_dataloader(
    ml_data, model_return="blocks", batch_size=BATCH_SIZE
)

# Depending on the diversity of the structures in the datasets, it can occure that 
# some blocks are empty, because certain structural features are only present in 
# certain structures (e.g. if we would have some organic molecules
# with oxygen and some without).
# We drop these blocks, so that the dataloader does not 
# try to load them during training.

ml_data.target_train, ml_data.target_val, ml_data.target_test = drop_zero_blocks(
    ml_data.target_train, ml_data.target_val, ml_data.target_test
)

ml_data.feat_train, ml_data.feat_val, ml_data.feat_test = drop_zero_blocks(
    ml_data.feat_train, ml_data.feat_val, ml_data.feat_test
)


# %%
# Prepare training
# ----------------
#
# Next, we set up our linear neural network model.
# As a good initial starting guess, we fit a ridge regression model
# on the trainingset. This gives us quickly and reliably
# parameters for the neural network, that are better than random
# initialization. We will later compare the pred_fock from
# the thus initialized model to the one after batch-wise training.
# 

model = LinearTargetModel(
    dataset=ml_data, nlayers=1, nhidden=16, bias=False, device=DEVICE
)

pred_ridges, ridges = model.fit_ridge_analytical(
    alpha=np.logspace(-8, 3, 12),
    cv=3,
    set_bias=False,
)

pred_fock = model.forward(
    ml_data.feat_train,
    return_type="tensor",
    batch_indices=ml_data.train_idx,
    ridge_fit=True,
    add_noise=NOISE,
)

# We 
ref_eva_lb = []
for i in range(len(molecule_data.lb_target["fock"])):
    f = molecule_data.lb_target["fock"][i]
    s = molecule_data.lb_aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)
    ref_eva_lb.append(torch.from_numpy(eig))


ref_eva = []
for i in range(len(molecule_data.target["fock"])):
    f = molecule_data.target["fock"][i]
    s = molecule_data.aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)
    ref_eva.append(torch.from_numpy(eig))

# var_eva = torch.cat([ref_eva_lb[i].flatten() for i in range(len(ref_eva_lb))]).var()

with io.capture_output() as captured:
    all_mfs, fockvars = instantiate_mf(
        ml_data,
        fock_predictions=None,
        batch_indices=list(range(len(ml_data.structures))),
    )

# %%
# Training parameters and training
# --------------------------------
#

loss_fn = mlmetrics.mse_per_atom
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3,  # 20,
)

# Initialize trainer
trainer = Trainer(model, optimizer, scheduler, DEVICE)

# Define necessary arguments for the training and validation process
fit_args = {
    "ml_data": ml_data,
    "all_mfs": all_mfs,
    "loss_fn": loss_fn,
    "ref_eva": ref_eva_lb,
    "ref_dipole": None,
    "ref_polar": None,
    "var_eva": None,
    "var_dipole": None,
    "var_polar": None,
    "weight_eva": W_EVA,
    "weight_dipole": W_DIP,
    "weight_polar": W_POL,
    "ORTHOGONAL": ORTHOGONAL,
}


# Train and validate
history = trainer.fit(
    train_dl,
    val_dl,
    NUM_EPOCHS,
    EARLY_STOP_CRITERION,
    FOLDER_NAME,
    VERBOSE,
    DUMP_HIST,
    **fit_args,
)

# %%
# Plot loss
# ---------
#


np.save(f"{FOLDER_NAME}/model_output/loss_stats.npy", history)

plot_losses(history, save=True, savename=f"{FOLDER_NAME}/loss_vs_epoch.pdf")

# %%
# Parity plot

eva_test = []
eva_test_pred = []
f_pred = model.forward(
    ml_data.feat_test,
    return_type="tensor",
    batch_indices=ml_data.test_idx,
)

for j, i in enumerate(test_dl.dataset.test_idx):
    f = molecule_data.lb_target["fock"][i]
    s = molecule_data.lb_aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)[: molecule_data.target["fock"][i].shape[0]]
    eva_test.append(torch.from_numpy(eig))

    eig_pred = torch.linalg.eigvalsh(f_pred[j])
    eva_test_pred.append(eig_pred)


plt.figure()
plt.plot([-25, 20], [-25, 20], "k--")
plt.plot(
    torch.cat(eva_test).detach().numpy() * Hartree,
    torch.cat([ref_eva[i] for i in ml_data.test_idx]).detach().numpy() * Hartree,
    "o",
    alpha=0.7,
    color="gray",
    markeredgecolor="black",
    markeredgewidth=0.2,
    label="STO-3G",
)
plt.plot(
    torch.cat(eva_test).detach().numpy() * Hartree,
    torch.cat(eva_test_pred).detach().numpy() * Hartree,
    "o",
    alpha=0.7,
    color="royalblue",
    markeredgecolor="black",
    markeredgewidth=0.2,
    label="ML",
)
plt.ylim(-25, 20)
plt.xlim(-25, 20)
plt.xlabel("Reference eigenvalues")
plt.ylabel("Predicted eigenvalues")
plt.savefig(f"{FOLDER_NAME}/parity_eva.pdf")
plt.legend()
plt.show()
