"""
Hamiltonian learning
=============================

This tutorial explains how to train a machine learning model for the a molecular system.
"""

# %%
# First, import the necessary packages
#

import os
from ase.units import Hartree

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from IPython.utils import io
from mlelec.features.acdc import compute_features_for_target
from mlelec.targets import drop_zero_blocks
from mlelec.utils.plot_utils import plot_losses


os.environ["PYSCFAD_BACKEND"] = "torch"
import mlelec.metrics as mlmetrics  # noqa: E402
from mlelec.data.dataset import MLDataset, MoleculeDataset, get_dataloader  # noqa: E402
from mlelec.models.linear import LinearTargetModel  # noqa: E402
from mlelec.train import Trainer  # noqa: E402
from mlelec.utils.property_utils import compute_batch_polarisability  # noqa: E402
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

# Set the parameters for the dataset
torch.manual_seed(42)

NUM_FRAMES = 20  #
BATCH_SIZE = 10  # 50
NUM_EPOCHS = 100
SHUFFLE_SEED = 42
TRAIN_FRAC = 0.7
TEST_FRAC = 0.1
VALIDATION_FRAC = 0.2
EARLY_STOP_CRITERION = 20
VERBOSE = 10
DUMP_HIST = 50
LR = 1e-3  #5e-3 # 5e-4
VAL_INTERVAL = 1
W_EVA = 1e2 #1000  # 1e4
W_DIP = 2e1 #0.1  # 1e3
W_POL = 1 #0.01  # 1e2
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
molecule_data = MoleculeDataset(
    mol_name="ethane",
    use_precomputed=False,
    path="data/ethane/",
    aux_path="data/ethane/sto-3g",
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
# Compute Features
# ----------------
#

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


# %%
# Prepare training
# ----------------
#

train_dl, val_dl, test_dl = get_dataloader(
    ml_data, model_return="blocks", batch_size=BATCH_SIZE
)

ml_data.target_train, ml_data.target_val, ml_data.target_test = drop_zero_blocks(
    ml_data.target_train, ml_data.target_val, ml_data.target_test
)

ml_data.feat_train, ml_data.feat_val, ml_data.feat_test = drop_zero_blocks(
    ml_data.feat_train, ml_data.feat_val, ml_data.feat_test
)

model = LinearTargetModel(
    dataset=ml_data, nlayers=1, nhidden=16, bias=False, device=DEVICE #16
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
pred_fock_ridge_test = model.forward(
    ml_data.feat_test,
    return_type="tensor",
    batch_indices=ml_data.test_idx,
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

var_eva = torch.cat([ref_eva_lb[i].flatten() for i in range(len(ref_eva_lb))]).var()
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

# %%
# Training parameters and training
# --------------------------------
#

loss_fn = mlmetrics.mse_per_atom
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=20,  # 20,
)

# Initialize trainer
trainer = Trainer(model, optimizer, scheduler, DEVICE)

# Define necessary arguments for the training and validation process
fit_args = {
    "ml_data": ml_data,
    "all_mfs": all_mfs,
    "loss_fn": loss_fn,
    "ref_eva": ref_eva_lb,
    "ref_dipole": ref_dip_lb,
    "ref_polar": ref_polar_lb,
    "var_eva": var_eva,
    "var_dipole": var_dipole,
    "var_polar": var_polar,
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


np.save(f"{FOLDER_NAME}/model_output/loss_stats.npy", history)

plot_losses(history, save=True, savename=f"{FOLDER_NAME}/loss_vs_epoch.pdf")


# Parity plot

eva_test = []
eva_test_pred = []
eva_test_pred_ridge = []
f_pred = model.forward(
        ml_data.feat_test, return_type="tensor", batch_indices=ml_data.test_idx,
    )

for j, i in enumerate(test_dl.dataset.test_idx):
    f = molecule_data.lb_target["fock"][i]
    s = molecule_data.lb_aux_data["overlap"][i]
    eig = scipy.linalg.eigvalsh(f, s)[:molecule_data.target["fock"][i].shape[0]]
    eva_test.append(torch.from_numpy(eig))
    
    eig_pred = torch.linalg.eigvalsh(f_pred[j])
    eva_test_pred.append(eig_pred)



    eig_pred_ridge=scipy.linalg.eigvalsh(pred_fock_ridge_test[j].detach().numpy(),s)
    eva_test_pred_ridge.append(torch.from_numpy(eig_pred_ridge))
    

# print("testrev", eva_test)
# print("testpred", eva_test_pred)
a, b = np.polyfit(np.array(eva_test).flatten(),np.array(eva_test_pred).flatten(), 1)
print('linefit',a,b)
#plt.plot(np.array(eva_test).flatten(), a*np.array(eva_test).flatten()+b,label='linefit')
#plt.loglog(np.array(eva_test).flatten(), a*np.array(eva_test).flatten()+b)

plt.loglog(
#plt.plot(
    [np.amin(eva_test_pred), np.amax(eva_test_pred)],
    [np.amin(eva_test_pred), np.amax(eva_test_pred)],
    "k",
)
plt.loglog(np.array(eva_test).flatten(), np.array(eva_test_pred).flatten(), "o")
#plt.plot(np.array(eva_test).flatten(), np.array(eva_test_pred).flatten(), "o", label='training')
#plt.plot(np.array(eva_test).flatten(), np.array(eva_test_pred_ridge).flatten(), "o",label='ridge')
plt.legend()
plt.xlabel("Eigenvalue$_{ DFT}$")
plt.ylabel("Eigenvalue$_{ ML}$")
=======
plt.figure()
plt.plot([-25,20], [-25,20], "k--")
plt.plot(torch.cat(eva_test).detach().numpy()*Hartree, torch.cat([ref_eva[i] for i in ml_data.test_idx]).detach().numpy()*Hartree, "o",
            alpha=0.7,
            color="gray",
            markeredgecolor="black",
            markeredgewidth=0.2,
            label="STO-3G",)
plt.plot(torch.cat(eva_test).detach().numpy()*Hartree, torch.cat(eva_test_pred).detach().numpy()*Hartree, "o",
            alpha=0.7,
            color="royalblue",
            markeredgecolor="black",
            markeredgewidth=0.2,
            label="ML")
plt.ylim(-25, 20)
plt.xlim(-25,20)
plt.xlabel("Reference eigenvalues")
plt.ylabel("Predicted eigenvalues")
>>>>>>> 43629bd0aa43085024c97779b79dbd65e0a5ed0a
plt.savefig(f"{FOLDER_NAME}/parity_eva.pdf")
plt.legend()
plt.show()
