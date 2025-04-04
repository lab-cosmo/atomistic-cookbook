"""
Hamiltonian Learning for Molecules with Indirect Targets
========================================================

:Authors: Divya Suman `@DivyaSuman14 <https://github.com/DivyaSuman14>`__,
          Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`__

This tutorial introduces a machine learning (ML) framework
that predicts Hamiltonians for molecular systems. Another 
one of our 'cookbook examples 
<https://atomistic-cookbook.org/examples/periodic-hamiltonian/periodic-hamiltonian.html>`__
demonstrates an ML model that predicts real-space Hamiltonians
for periodic systems. While we use the same model here to predict 
a molecular Hamiltonians, we further finetune these models to 
optimise predictions of different quantum mechanical (QM)
properties of interest, thereby treating the Hamiltonian 
predictions as an intermediate component of the ML 
framework. More details on this hybrid or indirect learning 
framework can be found in `ACS Cent. Sci. 2024, 10, 637−648.
<https://pubs.acs.org/doi/full/10.1021/acscentsci.3c01480>`_ .


"""

# %%
# Within a Hamiltonian learning framework, one could chose to learn
# a target that corresponds to the matrix representation for an
# existing electronic structure method, but in a finite AO basis,
# such a representation will lead to a finite basis set error. 
# The parity plot below illustrates this error by showing the
# discrepancy between molecular orbital (MO) energies of an ethane
# molecule obtained from a self-consistent calculation on a minimal 
# STO-3G basis and the larger def2-TZVP basis, especially for the
# high energy, virtuals MOs.

# %%
# .. figure:: minimal_vs_lb.png
#    :alt: Parity plot comparing the MO energies of ethane from a 
#           DFT calculation with the STO-3G and the def2-TZVP basis.
#    :width: 600px

# %%
# The choice of basis set plays a crucial role in determining 
# the accuracy of the observables derived from the predicted 
# Hamiltonian. Although the larger basis sets generally provide
# more reliable results, the computational cost to train such a
# model is very high. Using the indirect learning framework, one
# could instead learn a reduced effective Hamiltonian that reproduces
# calculations from a much larger basis while using a significantly
# simpler and smaller model consistent with a smaller basis.
#

# %%
# We first show an example where we predict the reduced effective
# Hamiltonians for a homogenous dataset of ethane molecule while 
# targeting the MO energies of the def2-TZVP basis. 

# %%
# Python environment and used packages
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %%
# We start by creating a virtual environment and installing
# all necessary packages. The required packages are provided
# in the environment.yml file that can be dowloaded at the end.
# We can then import the necessary packages.
#

import os

import matplotlib.pyplot as plt
import numpy as np
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
from mlelec.utils.property_utils import (  # noqa: E402
    compute_dipole_moment,
    compute_eigvals,
    compute_polarisability,
    instantiate_mf,
)

torch.set_default_dtype(torch.float64)

# %%
# Set parameters for training
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Before we begin our training we can decide on a set the parameters,
# including the dataset set size, splitting fractions, the batch size, 
# learning rate, number of epochs, the early stop criterion in case of
# early stopping and we can save the parameters for our reference later.
#

NUM_FRAMES = 100
BATCH_SIZE = 4
NUM_EPOCHS = 100  # 100
SHUFFLE_SEED = 42
TRAIN_FRAC = 0.7
TEST_FRAC = 0.1
VALIDATION_FRAC = 0.2
EARLY_STOP_CRITERION = 20
VERBOSE = 10
DUMP_HIST = 50
LR = 1e-1  # 5e-4
VAL_INTERVAL = 1
DEVICE = "cpu"

ORTHOGONAL = True  # set to 'FALSE' if working in the non-orthogonal basis
FOLDER_NAME = "FPS/ML_orthogonal_eva"
NOISE = False

# %%
# Create folders and save parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    DEVICE=DEVICE,
    ORTHOGONAL=ORTHOGONAL,
    FOLDER_NAME=FOLDER_NAME,
)

# %%
# Generate reference data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In principle one can generate the training data of reference
# Hamiltonians from a given set of structures, using any  
# electronic structure code. Here we provide a pre-computed,
# homogenous dataset that contains 100 different
# configurations of ethane molecule. For all structures, we 
# performed Kohn-Sham density functional theory (DFT)
# calculations with `PySCF <https://github.com/pyscf/pyscf>`_, 
# using the B3LYP functional. For each molecular geometry, we 
# computed the Fock and overlap matrices along with other
# molecular properties of interest, using both STO-3G and def2-TZVP 
# basis sets.

# %%
# Prepare the dataset for ML training
# ~~~~~~~~~~~~~~~
#
# In this section, we will prepare the dataset required to
# train our machine learning model using the ``MoleculeDataset`` 
# and ``MLDataset`` classes. These classes help format and store
# the DFT data in a way compatible with our ML package, `mlelec  <https://github.com/curiosity54/mlelec/tree/qm7>`_.
# In this section we initialise the ``MoleculeDataset`` where 
# we specify the molecule name, file paths and the desired targets
# and auxillary data to be used for training for the minimal
# (STO-3G), as well as a larger basis (lb, def2-TZVP).
# Once the molecular data is prepared, we wrap it into an 
# ``MLDataset`` instance. This class structures the dataset 
# into a format that is optimal for ML the Hamiltonians. The 
# Hamiltonian matrix elements depend on specific pairs of 
# orbitals involved in the interaction. When these orbitals
# are centered on atoms, as is the case for localized AO
# bases, the Hamiltonian matrix elements can be viewed
# as objects labeled by pairs of atoms, as well as multiple
# quantum numbers, namely the radial (`n`) and the angular
# (`l`, `m`) quantum numbers characterizing each AO. These
# angular functions are typically chosen to be real spherical 
# harmonics, and determine the equivariant behavior of the 
# matrix elements under rotations and inversions. ``MLDataset`` 
# leverages this equivariant structure of the Hamiltonians, 
# which is discussed in further detail in the `Periodic Hamiltonian Model Example
# <https://atomistic-cookbook.org/examples/periodic-hamiltonian/periodic-hamiltonian.html>`__
# . Finally, we split the loaded dataset into training, 
# validation and test datasets using ``_split_indices``.

molecule_data = MoleculeDataset(
    mol_name="ethane",
    use_precomputed=False,
    path="data/ethane/",
    aux_path="data/ethane/sto-3g",
    frame_slice=slice(0, NUM_FRAMES),
    device=DEVICE,
    aux=["overlap", "orbitals"],
    lb_aux=["overlap", "orbitals"],
    target=["fock", "eigenvalues"],
    lb_target=["fock", "eigenvalues"],
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
# Computing features that can learn Hamiltonian targets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As discussed above the Hamiltonian matrix elements
# are dependent on single as well as two centers. So, 
# to address this we extend the equivariant, SOAP based
# features for the atom-centered desciptors 
# to a descriptor capable of describing multiple atomic centers and
# their connectivities, giving rise to the equivariant pairwise descriptor
# which simultaneously characterizes the environments for pairs of atoms
# in a given system. Our `Periodic Hamiltonian Model Example
# <https://atomistic-cookbook.org/examples/periodic-hamiltonian/periodic-hamiltonian.html>`__
# discusses the construction of these descriptors in greater detail.
# To construct these atom- and pair-centered features we use our 
# in house library `featomic <https://github.com/metatensor/featomic/>`_
# for each structure in the our dataset using the hyperparameters
# defined below. The features are constructed starting from a
# description of a structure in terms of atom density. The 
# ``density`` hyperparameter lets us chose that. the features are
# discretized on a basis of radial functions and spherical harmonics, 
# which are given by the ``basis`` hyperparameter. The ``cutoff``
# hyperparameter controls the extent of the atomic environment.
# The atom and pairwise features have very similar hyperparameters, 
# except for the cutoff radius, which is larger for pairwise
# features to include as many pairs.
# 

hypers = {
    "cutoff": {"radius": 2.5, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
    "density": {"type": "Gaussian", "width": 0.3},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 5},
    },
}

hypers_pair = {
    "cutoff": {"radius": 2.5, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
    "density": {"type": "Gaussian", "width": 0.3},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 5},
    },
}

features = compute_features_for_target(
    ml_data, device=DEVICE, hypers=hypers, hypers_pair=hypers_pair
)
ml_data._set_features(features)

# %%
# Prepare Dataloaders
# ~~~~~~~~~~~~~~~~~~~
# To efficiently feed data into the model during training, 
# we use data loaders. These handle batching and shuffling 
# to optimize training performance. ``get_dataloader``
# creates data loaders for training, validation and testing.
# The ``model_return="blocks"`` argument determines that the 
# model targets the different blocks that the Hamiltonian is
# decomposed into and the ``batch_size`` argument defines the 
# number of samples per batch for the batch-wise training.
# 

train_dl, val_dl, test_dl = get_dataloader(
    ml_data, model_return="blocks", batch_size=BATCH_SIZE
)

# %%
# Prepare training
# ~~~~~~~~~~~~~~~~
#
# Next, we set up our linear model that predicts the Hamiltonian
# matrices, using ``LinerTargetModel``. To improve the model 
# convergence, we first start with a symmetry-adapted ridge regression
# targeting the Hamiltonian matrices from the STO-3G basis QM 
# calculation using the ``fit_ridge_analytical`` function.
# This provides us a more reliable set of weights to initialise the 
# fine-tuning rather than starting from any random guess.


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

with io.capture_output() as captured:
    all_mfs, fockvars = instantiate_mf(
        ml_data,
        fock_predictions=None,
        batch_indices=list(range(len(ml_data.structures))),
    )

# %%
# Indirect learning of the MO energies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now rather than explicitly targeting the Hamiltonian matrix,
# we instead treat it as an intermediate layer in our framework
# and the model weights are subsequently fine-tuned by backpropagating
# a loss on a derived molecular property of the Hamiltonian such as
# the MO energies but from the larger def-TZVP basis instead of the
# STO-3G basis. 
#
# Before fine-tuning, we set up the loss function, optimizer
# and the learning rate scheduler for our model. We use a customized
# mean squared error (MSE) loss function that guides the learning and
# ``Adam`` optimizer that performs a stochastic gradient descent 
# that minimizes the error and the scheduler reduces the learning 
# rate by the given factor if the validation loss plateaus.

loss_fn = mlmetrics.mse_per_atom
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=3,  # 20,
)

# %%
# We use a``Trainer`` class to encapsulate all training logic.
# It manages the training and validation loops.
trainer = Trainer(model, optimizer, scheduler, DEVICE)

# %%
# Define necessary arguments for the training and validation process.
fit_args = {
    "ml_data": ml_data,
    "all_mfs": all_mfs,
    "loss_fn": loss_fn,
    "weight_eva": 1,
    "weight_dipole": 0,
    "weight_polar": 0,
    "ORTHOGONAL": ORTHOGONAL,
    "upscale": True,
}

# %%
# With these steps complete, the model begins training and validation 
# using the structured molecular data, features, and defined parameters.
# The ``fit`` function returns the training and validation losses for 
# each epoch, which we can then use to plot the `loss versus epoch` curve.

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

np.save(f"{FOLDER_NAME}/model_output/loss_stats.npy", history)

# %%
# Plot loss
# ~~~~~~~~~
#
# With the help of ``plot_losses`` function we can conveniently
# plot the training and validation losses.

plot_losses(history, save=True, savename=f"{FOLDER_NAME}/loss_vs_epoch.pdf")

# %%
# Parity plot
# ~~~~~~~~~~~
# We then evaluate the prediction of the fine-tuned model on 
# the MO energies by comparing it the with the MO energies 
# from the reference def2-TZVP calculation. The parity plot
# below shows the performance of our model on the test
# dataset. ML predictions are shown with blue points and 
# the corresponding MO energies from the STO-3G basis are
# are shown in grey. One can clearly see that even with a 
# minimal basis parametrisation, the model is able to reproduce
# the large basis MO energies with good accuracy. Thus, using 
# an indirect model, makes it possible to promote the model
# accuracy to a higher level of theory, at no additional cost.


def plot_parity_property(molecule_data, propert="eigenvalues", orthogonal=True):
    plt.figure()
    plt.plot([-25, 20], [-25, 20], "k--")
    plt.plot(
        torch.cat(
            [
                molecule_data.lb_target[propert][i][
                    : molecule_data.target[propert][i].shape[0]
                ]
                for i in ml_data.test_idx
            ]
        )
        .detach()
        .numpy()
        .flatten()
        * Hartree,
        torch.cat([molecule_data.target[propert][i] for i in ml_data.test_idx])
        .detach()
        .numpy()
        .flatten()
        * Hartree,
        "o",
        alpha=0.7,
        color="gray",
        markeredgecolor="black",
        markeredgewidth=0.2,
        label="STO-3G",
    )
    f_pred = model.forward(
        ml_data.feat_test,
        return_type="tensor",
        batch_indices=ml_data.test_idx,
    )

    if propert == "eigenvalues":
        prop = compute_eigvals(
            ml_data, f_pred, range(len(ml_data.test_idx)), orthogonal=orthogonal
        )
        prop = torch.tensor([p for pro in prop for p in pro]).detach().numpy()
    elif propert == "dipole_moment":
        prop = compute_dipole_moment(
            [molecule_data.structures[i] for i in ml_data.test_idx],
            f_pred,
            orthogonal=orthogonal,
        )
        prop = torch.tensor([p for pro in prop for p in pro]).detach().numpy()
    elif propert == "polarisability":
        prop = compute_polarisability(
            [molecule_data.structures[i] for i in ml_data.test_idx],
            f_pred,
            orthogonal=orthogonal,
        )
        prop = prop.flatten().detach().numpy()
    else:
        print("Property not implemented")

    plt.plot(
        torch.cat(
            [
                molecule_data.lb_target[propert][i][
                    : molecule_data.target[propert][i].shape[0]
                ]
                for i in ml_data.test_idx
            ]
        )
        .detach()
        .numpy()
        .flatten()
        * Hartree,
        prop * Hartree,
        "o",
        alpha=0.7,
        color="royalblue",
        markeredgecolor="black",
        markeredgewidth=0.2,
        label="ML",
    )
    plt.ylim(-25, 20)
    plt.xlim(-25, 20)
    plt.xlabel(f"Reference {propert}")
    plt.ylabel(f"Predicted {propert}")
    plt.savefig(f"{FOLDER_NAME}/parity_{propert}.pdf")
    plt.legend()
    plt.show()


plot_parity_property(molecule_data, propert="eigenvalues", orthogonal=ORTHOGONAL)

# %%
# Targeting the Multiple Properties
# ---------------------------------
#
# In principle we can also target multiple properties for
# the indirect training. While MO energies can be computed
# by simply diagonalizing the Hamiltonian matrix, some 
# properties like the dipole moment require
# the position operator integral and it's derivative if 
# we want to bacpropagate the loss. We therefore interface
# our ML model with an electronic structure code that supports
# automatic differentiation, `PySCFAD <https://github.com/fishjojo/pyscfad>_`,
# an end-to-end auto-differentiable version of PySCF. By
# doing so we delegate the computation of properties to PySCFAD,
# which provides automatic differentiation of observables with
# respect to the intermediate Hamiltonian. In particular, we will now
# indirectly target the dipole moment and polarisability along 
# with the MO energies from a large basis reference calculation


# %%
# Get Data and Prepare Data Set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In our last example even though we show an indirect ML model
# that was trained on a homogenous dataset of different 
# configurations of ethane, we can also easily extend the 
# framework to use  a much diverse dataset such as the 
# `QM7 dataset <http://quantum-machine.org/datasets/>_.
# For our next example we select a subset 150 structures from
# this dataset that consists of only C, H, N and O atoms.

# %%
# Set parameters for training
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Set the parameters for the training, including the dataset set size
# and split, the batch size, learning rate and weights for the individual
# components of eigenvalues, dipole and polarisability.
# We additionally define a folder name, in which the results are saved.
# Optionally, noise can be added to the ridge regression fit.
# Here, we now need to provide different weights
# for the different targets (eigenvalues
# :math:`\epsilon`, the dipole moment
# :math:`\mu`, and polarisability
# :math:`\alpha`), which we will use when computing the loss :math:`\mathcal{L}`.
#
# .. math::
#    \mathcal{L}_{\epsilon,\mu,\alpha} = & \;
#    \frac{\omega_{\epsilon}}{N} \sum_{n=1}^{N} \frac{1}{O_n} \
#    \sum_{o=1}^{O_n} \left( \epsilon_{no} - \tilde{\epsilon}_{no} \right)^2 \
#     + \frac{\omega_{\mu}}{N} \sum_{n=1}^{N} \frac{1}{N_A^2} \
#    \sum_{m=1}^{N_A} \left( \mu_{nm} - \tilde{\mu}_{nm} \right)^2 \\
#    & + \frac{\omega_{\alpha}}{N} \sum_{n=1}^{N} \frac{1}{N_A^2} \
#    \sum_{m=1}^{N_A} \left( \alpha_{nm} - \tilde{\alpha}_{nm} \right)^2
#
# where
# :math:`N` is the number of training points,
# :math:`O_n` is the number of MO orbitals in the nth molecule,
# :math:`N_A` is the number of atoms :math:`i`.

# %%
# The weights :math: `\omega` are based on the magnitude of errors
# for different properties, in the end we want each of them to 
# contribute equally to the loss.
# The following values worked well for the QM7 example, but of
# course depending on the system that one investigates another set
# of weights might work better.

NUM_FRAMES = 150
LR = 1e-3
W_EVA = 1e4
W_DIP = 1e3
W_POL = 1e2

FOLDER_NAME = "FPS/ML_orthogonal_eva_dip_pol"

# %%
# Create folders and save parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
# ~~~~~~~~~~~~~~~
#
# We use the dataloader of the
# `mlelec package (qm7 branch) <https://github.com/curiosity54/mlelec/tree/qm7>`_,
# and load the ethane
# dataset for the defined number of slices.
# First, we load all relavant data (geometric structures,
# auxiliary matrices (overlap and orbitals), and
# targets (fock, dipole moment and polarisablity)) into a molecule dataset.
# We do this for the minimal (STO-3G), as well as a larger basis (lb, def2-TZVP).
# The larger basis has additional basis functions on the valence electrons.
# The dataset, we can then load into our dataloader ml_data, together with some
# settings on how to get data from the dataloader.
# Finally, we define the desired dataset split for training, validation,
# and testing from the parameters defined above.

molecule_data = MoleculeDataset(
    mol_name="qm7",
    use_precomputed=True,
    path="data/qm7",
    aux_path="data/qm7/sto-3g",
    frame_slice=slice(0, NUM_FRAMES),
    device=DEVICE,
    aux=["overlap", "orbitals"],
    lb_aux=["overlap", "orbitals"],
    target=["fock", "eigenvalues", "dipole_moment", "polarisability"],
    lb_target=["fock", "eigenvalues", "dipole_moment", "polarisability"],
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
# ~~~~~~~~~~~~~~~~
# TODO
#

hypers = {
    "cutoff": {"radius": 3, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
    "density": {"type": "Gaussian", "width": 0.3},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 5},
    },
}

hypers_pair = {
    "cutoff": {"radius": 5, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
    "density": {"type": "Gaussian", "width": 0.3},
    "basis": {
        "type": "TensorProduct",
        "max_angular": 4,
        "radial": {"type": "Gto", "max_radial": 5},
    },
}


features = compute_features_for_target(
    ml_data, device=DEVICE, hypers=hypers, hypers_pair=hypers_pair
)
ml_data._set_features(features)


train_dl, val_dl, test_dl = get_dataloader(
    ml_data, model_return="blocks", batch_size=BATCH_SIZE
)
# %%
# Depending on the diversity of the structures in the datasets, it may
# happen that some blocks are empty, because certain structural 
# features are only present in certain structures (e.g. if we 
# would have some organic molecules with oxygen and some without).
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
# ~~~~~~~~~~~~~~~~
#
# Here, we also first fit a ridge regression model to the data.

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

with io.capture_output() as captured:
    all_mfs, fockvars = instantiate_mf(
        ml_data,
        fock_predictions=None,
        batch_indices=list(range(len(ml_data.structures))),
    )

# %%
# Training parameters and training
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Set loss function, optimizer and scheduler

loss_fn = mlmetrics.mse_per_atom
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=10,  # 20,
)

# Initialize trainer
trainer = Trainer(model, optimizer, scheduler, DEVICE)

# Define necessary arguments for the training and validation process
fit_args = {
    "ml_data": ml_data,
    "all_mfs": all_mfs,
    "loss_fn": loss_fn,
    "weight_eva": W_EVA,
    "weight_dipole": W_DIP,
    "weight_polar": W_POL,
    "ORTHOGONAL": ORTHOGONAL,
    "upscale": True,
}


# Train and validate
history = trainer.fit(
    train_dl,
    val_dl,
    200,
    EARLY_STOP_CRITERION,
    FOLDER_NAME,
    VERBOSE,
    DUMP_HIST,
    **fit_args,
)

# Save the loss history
np.save(f"{FOLDER_NAME}/model_output/loss_stats.npy", history)

# %%
# Plot loss
# ~~~~~~~~~
#

# We can conveniently use the plotting function we imported from
# mlelec above. At the same time we can save the pdf.

plot_losses(history, save=True, savename=f"{FOLDER_NAME}/loss_vs_epoch.pdf")

# %%
# Parity plot
# ~~~~~~~~~~~
# We can now compare the properties derived from the predicted hamiltonian
# to their actual values
# in a parity plot.
# The 'STO-3G' illustrates the difference of the two computed datasets,
# the STO-3G eigenvalues from DFT against the ones obtained with the large basis
# def2-TZVP.
# The results of the upscale Hamiltonian model are plotted with the label 'ML'.

plot_parity_property(molecule_data, propert="eigenvalues", orthogonal=ORTHOGONAL)
plot_parity_property(molecule_data, propert="dipole_moment", orthogonal=ORTHOGONAL)
plot_parity_property(molecule_data, propert="polarisability", orthogonal=ORTHOGONAL)

# %%
