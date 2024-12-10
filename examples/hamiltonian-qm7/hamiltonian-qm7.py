"""
Hamiltonian learning
=============================

This tutorial explains how to train a machine learning model for the a molecular system. 
"""

# %%
# First, import the necessary packages
#

import os
import time

import ase
import matplotlib.pyplot as plt
import metatensor
import numpy as np
import scipy
import torch
from ase.units import Hartree
from IPython.utils import io
from metatensor import Labels
from tqdm import tqdm

import mlelec.metrics as mlmetrics
from mlelec.data.dataset import MoleculeDataset, get_dataloader
from mlelec.features.acdc import compute_features_for_target
from src.mlelec.data.dataset import MLDataset
from src.mlelec.models.linear import LinearTargetModel
from src.mlelec.utils.dipole_utils import compute_batch_polarisability, instantiate_mf

torch.set_default_dtype(torch.float64)

# sphinx_gallery_thumbnail_number = 3


# %%
# Get Data and Prepare Data Set
# -----------------------------
#


# %%
# Set parameters for training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Set the parameters for the dataset


NUM_FRAMES = 10
BATCH_SIZE = 5
NUM_EPOCHS = 600
SHUFFLE_SEED = 0
TRAIN_FRAC = 0.7
TEST_FRAC = 0.1
VALIDATION_FRAC = 0.2

LR = 5e-4
VAL_INTERVAL = 1
W_EVA = 1e4
W_DIP = 0
W_POL = 0
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
    NOISE=NOISE,
)



# %%
# define functions
# -----------------------------
#

def drop_zero_blocks(train_tensor, val_tensor, test_tensor):
    for i1, b1 in train_tensor.items():
        if b1.values.shape[0] == 0:
            train_tensor = metatensor.drop_blocks(
                train_tensor, Labels(i1.names, i1.values.reshape(1, -1))
            )

    for i2, b2 in val_tensor.items():
        if b2.values.shape[0] == 0:
            val_tensor = metatensor.drop_blocks(
                val_tensor, Labels(i2.names, i2.values.reshape(1, -1))
            )

    for i3, b3 in test_tensor.items():
        if b3.values.shape[0] == 0:
            test_tensor = metatensor.drop_blocks(
                test_tensor, Labels(i3.names, i3.values.reshape(1, -1))
            )

    return train_tensor, val_tensor, test_tensor


# loss function to have different combinations of losses
def loss_fn_combined(
    ml_data,
    pred_focks,
    orthogonal,
    mfs,
    indices,
    loss_fn,
    frames,
    eigval,
    polar,
    dipole,
    var_polar,
    var_dipole,
    var_eigval,
    weight_eigval=1.0,
    weight_polar=1.0,
    weight_dipole=1.0,
):

    pred_dipole, pred_polar, pred_eigval = compute_batch_polarisability(
        ml_data, pred_focks, indices, mfs, orthogonal
    )

    loss_polar = loss_fn(frames, pred_polar, polar) / var_polar
    loss_dipole = loss_fn(frames, pred_dipole, dipole) / var_dipole
    loss_eigval = loss_fn(frames, pred_eigval, eigval) / var_eigval

    # weighted sum of the various loss contributions
    return (
        weight_eigval * loss_eigval
        + weight_dipole * loss_dipole
        + weight_polar * loss_polar,
        loss_eigval,
        loss_dipole,
        loss_polar,
    )



# %%
# Load dataset
# ~~~~~~~~~~~~
#
start_time_pred_lbfgs = time.time()

molecule_data = MoleculeDataset(
    mol_name="qm7",
    use_precomputed=True,
    path="data",
    aux_path="data/qm7/sto-3g",
    frame_slice=slice(0, NUM_FRAMES),
    device=DEVICE,
    aux=["overlap", "orbitals"],
    lb_aux=["overlap", "orbitals"],
    target=["fock", "dipole_moment", "polarisability"],
    lb_target=["fock", "dipole_moment", "polarisability"],
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
# Set hyperparameters for features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
hypers = {
    "cutoff": 3.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}
hypers_pair = {
    "cutoff": 5.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    #"radial_scaling": {"Willatt2018": {"scale": 2.0, "rate": 1.0, "exponent": 4}},
}

features = metatensor.load("qm7_1000_feats_cutoff.npz")
features = features.to(arrays="torch")
ml_data._set_features(features)

train_dl, val_dl, test_dl = get_dataloader(
    ml_data, model_return="blocks", batch_size=BATCH_SIZE
)

ml_data.target_train, ml_data.target_val, ml_data.target_test = drop_zero_blocks(
    ml_data.target_train, ml_data.target_val, ml_data.target_test
)

ml_data.feat_train, ml_data.feat_val, ml_data.feat_test = drop_zero_blocks(
    ml_data.feat_train, ml_data.feat_val, ml_data.feat_test
)

# %%
# Setup model
# ------------
#
model = LinearTargetModel(
    dataset=ml_data, nlayers=1, nhidden=16, bias=False, device=DEVICE
)
#model.load_state_dict(torch.load(f'{FOLDER_NAME}/model_output/model_epoch_350.pt'))

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

ref_polar_lb = molecule_data.lb_target["polarisability"]
ref_dip_lb = molecule_data.lb_target["dipole_moment"]

ref_eva_lb = []
for i in range(len(molecule_data.lb_target["fock"])):
    f = molecule_data.lb_target["fock"][i]
    s = molecule_data.lb_aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)
    ref_eva_lb.append(torch.from_numpy(eig))

ref_polar = molecule_data.target["polarisability"]
ref_dip = molecule_data.target["dipole_moment"]
ref_eva = []
for i in range(len(molecule_data.target["fock"])):
    f = molecule_data.target["fock"][i]
    s = molecule_data.aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)
    ref_eva.append(torch.from_numpy(eig))

var_eigval = torch.cat([ref_eva_lb[i].flatten() for i in range(len(ref_eva_lb))]).var()
var_dipole = torch.cat([ref_dip_lb[i].flatten() for i in range(len(ref_dip_lb))]).var()
var_polar = torch.cat(
    [ref_polar_lb[i].flatten() for i in range(len(ref_polar_lb))]
).var()

with io.capture_output() as captured:
    all_mfs, fockvars = instantiate_mf(
        ml_data,
        fock_predictions=None,
        batch_indices=list(range(len(ml_data.structures))),
    )

best = float("inf")
early_stop_criteria = 10
loss_fn = getattr(mlmetrics, "mse_per_atom")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=20,
)

# %%
# Training
# --------
#
val_interval = VAL_INTERVAL
early_stop_count = 0

# Initialize lists to store different losses
losses = []
val_losses = []
losses_eva = []
val_losses_eva = []
losses_polar = []
val_losses_polar = []
losses_dipole = []
val_losses_dipole = []

iterator = tqdm(range(NUM_EPOCHS))
for epoch in range(NUM_EPOCHS):
    model.train(True)
    train_loss = 0
    train_loss_eva = 0
    train_loss_polar = 0
    train_loss_dipole = 0

    for data in train_dl:
        optimizer.zero_grad()
        idx = data["idx"]

        # Forward pass
        pred = model(
            data["input"], return_type="tensor", batch_indices=[i.item() for i in idx]
        )
        train_polar_ref = ref_polar_lb[[i.item() for i in idx]]
        train_dip_ref = ref_dip_lb[[i.item() for i in idx]]
        train_eva_ref = [
            ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in idx
        ]

        loss, loss_eva, loss_dipole, loss_polar = loss_fn_combined(
            ml_data,
            pred,
            ORTHOGONAL,
            all_mfs,
            idx,
            loss_fn,
            data["frames"],
            train_eva_ref,
            train_polar_ref,
            train_dip_ref,
            var_polar,
            var_dipole,
            var_eigval,
            W_EVA,
            W_POL,
            W_DIP,
        )
        train_loss += loss.item()
        train_loss_eva += loss_eva.item()
        train_loss_polar += loss_polar.item()
        train_loss_dipole += loss_dipole.item()

        # Backward pass
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    avg_train_loss = train_loss / len(train_dl)
    avg_train_loss_eva = train_loss_eva / len(train_dl)
    avg_train_loss_polar = train_loss_polar / len(train_dl)
    avg_train_loss_dipole = train_loss_dipole / len(train_dl)

    losses.append(avg_train_loss)
    losses_eva.append(avg_train_loss_eva)
    losses_polar.append(avg_train_loss_polar)
    losses_dipole.append(avg_train_loss_dipole)

    lr = optimizer.param_groups[0]["lr"]

    model.eval()
    if epoch % val_interval == 0:
        val_loss = 0
        val_loss_eva = 0
        val_loss_polar = 0
        val_loss_dipole = 0

        for data in val_dl:
            idx = data["idx"]
            val_pred = model(
                data["input"],
                return_type="tensor",
                batch_indices=[i.item() for i in idx],
            )

            val_polar_ref = ref_polar_lb[[i.item() for i in idx]]
            val_dip_ref = ref_dip_lb[[i.item() for i in idx]]
            val_eva_ref = [
                ref_eva_lb[i][: ml_data.target.tensor[i].shape[0]] for i in idx
            ]

            vloss, vloss_eva, vloss_dipole, vloss_polar = loss_fn_combined(
                ml_data,
                val_pred,
                ORTHOGONAL,
                all_mfs,
                idx,
                loss_fn,
                data["frames"],
                val_eva_ref,
                val_polar_ref,
                val_dip_ref,
                var_polar,
                var_dipole,
                var_eigval,
                W_EVA,
                W_POL,
                W_DIP,
            )

            val_loss += vloss.item()
            val_loss_eva += vloss_eva.item()
            val_loss_polar += vloss_polar.item()
            val_loss_dipole += vloss_dipole.item()

        avg_val_loss = val_loss / len(val_dl)
        avg_val_loss_eva = val_loss_eva / len(val_dl)
        avg_val_loss_polar = val_loss_polar / len(val_dl)
        avg_val_loss_dipole = val_loss_dipole / len(val_dl)

        val_losses.append(avg_val_loss)
        val_losses_eva.append(avg_val_loss_eva)
        val_losses_polar.append(avg_val_loss_polar)
        val_losses_dipole.append(avg_val_loss_dipole)

        new_best = avg_val_loss < best
        if new_best:
            best = val_loss
            # torch.save(model.state_dict(), 'model_output_combined/best_model_dipole.pt')
            early_stop_count = 0
        else:
            early_stop_count += 1
        if early_stop_count > early_stop_criteria:
            print(f"Early stopping at epoch {epoch}")
            print(
                f"Epoch {epoch}, train loss {avg_train_loss}, val loss {avg_val_loss}"
            )
            # Save last best model
            break

        scheduler.step(avg_val_loss)

    if epoch % 50 == 0:
        torch.save(
            model.state_dict(), f"{FOLDER_NAME}/model_output/model_epoch_{epoch}.pt"
        )

    if epoch % 1 == 0:
        print(
            "Epoch:",
            epoch,
            "train loss:",
            f"{avg_train_loss:.4g}",
            "val loss:",
            f"{avg_val_loss:.4g}",
            "learning rate:",
            f"{lr:.4g}",
        )
        print(
            "Train Loss Polar:",
            f"{avg_train_loss_polar:.4g}",
            "Train Loss eva:",
            f"{avg_train_loss_eva:.4g}",
            "Train Loss dipole:",
            f"{avg_train_loss_dipole:.4g}",
        )
        print(
            "Val Loss Polar:",
            f"{avg_val_loss_polar:.4g}",
            "Val Loss eva:",
            f"{avg_val_loss_eva:.4g}",
            "Val Loss dipole:",
            f"{avg_val_loss_dipole:.4g}",
        )
    if epoch % 1 == 0:
        iterator.set_postfix(train_loss=avg_train_loss, Val_loss=avg_val_loss, lr=lr)

end_time_pred_lbfgs = time.time()
lbfgs_prediction_time = end_time_pred_lbfgs - start_time_pred_lbfgs
print(f"L-BFGS prediction time: {lbfgs_prediction_time:.4f} seconds")



