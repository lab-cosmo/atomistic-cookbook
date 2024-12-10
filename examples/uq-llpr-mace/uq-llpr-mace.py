"""
Uncertainty Quantification: LLPR-based approach for MACE
========================================================

:Authors: Sanggyu "Raymond" Chong `@SanggyuChong <https://github.com/SanggyuChong/>`_;

In this tutorial, we demonstrate how uncertainty quantification
for MACE can be done using our last-layer prediction rigidity (LLPR)
implementation. Demonstration will be made for an already-trained MACE
model for silicon 10-mers. Interested readers can learn more about the
LLPR method `here <https://iopscience.iop.org/article/10.1088/2632-2153/ad805f>`_.

First, we import the necessary packages and functions:
"""

# %%
import numpy as np
import torch
import torch.nn.functional
from mace import modules, data, tools
from mace.tools import torch_geometric
from mace.tools.scripts_utils import get_dataset_from_xyz
from mace.tools.llpr import calibrate_llpr_params
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float64)
device = tools.init_device('cuda')

# %%
# Load pre-trained model and silicon 10-mer data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# We download the dataset and model associated with this example
# then load them for UQ demonstration. Notice that after reading the
# vanilla model (i.e. taken directly from training result), we apply
# the `LLPRModel` wrapper to gain access to all the necessary UQ functions.


model = torch.load("model_and_data/silicon_10mer.model").to(device)
model_llpr = modules.LLPRModel(model)
model_llpr.to(device);

z_table = tools.get_atomic_number_table_from_zs([14])
config_type_weights = {"Default": 1.0}

collections, atomic_energies_dict = get_dataset_from_xyz(
    working_dir="",
    train_path="model_and_data/Si10_train.xyz",
    valid_path="model_and_data/Si10_valid.xyz",
    config_type_weights=config_type_weights,
)

valid_loader = torch_geometric.dataloader.DataLoader(
    dataset=[
        data.AtomicData.from_config(config, z_table=z_table, cutoff=5.5)
        for config in collections.validation_set
    ],
    batch_size=10,
    shuffle=False,
    drop_last=False,
)

# %%
# Compute covariance matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Central to the LLPR-based UQ method is the computation of the
# covariance matrix of last-layer features (before the linear readout)
# over the training set. An internal function of our MACE implementation
# supports this computation step. Note that in cases where the gradients
# (forces, virials, stresses) have also been used for model training,
# last-layer feature gradients also make their way into this covariance
# matrix. However, these computations can be quite costly, and users have
# the choice of omitting the gradient contributions and relying on the
# calibration step to recover accuracy UQ estimates. While this recipe will
# omit the gradients, gradient functionalities are still exposed below.


print("covariance matrix before computation: ", model_llpr.covariance)

model_llpr.compute_covariance(
    valid_loader,
    include_energy=True,  # by default
    include_forces=False,
    include_virials=False,
    include_stresses=False,
)

print("covariance matrix after computation: ", model_llpr.covariance)

# %%
# Calibration of UQ metrics to validation set
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Now that the covariance matrix has been computed, we are ready to
# perform UQ metric calibration to the validation set. Calibration is
# performed to obtain two parameters, `C` and `sigma`, which appear in
# Eq. 24 of `Bigi et al.  <https://iopscience.iop.org/article/10.1088/2632-2153/ad805f>`_.
# Calibration is done to the energy predictions on some calibration set,
# which is fixed to be the validation set in this recipe. Two utility
# functions are provided in our MACE implementation: ``sum of squared
# log errors'' and ``negative log-likelihood''. Note that the first 
# approach performs binning of the data points for robust calibration
# and takes in `n_samples_per_bin` as input.

calibrate_llpr_params(
    model_llpr,
    valid_loader,
    function="ssl",  # sum of squared log errors
    n_samples_per_bin=10,
)

calibrate_llpr_params(
    model_llpr,
    valid_loader,
    function="nll",  # negative log-likelihood
)

# %%
# Compute inverse covariance matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Having obtained the calibrated parameters, now the inverse covariance matrix
# can be computed to prepare the model for actual UQ:

model_llpr.compute_inv_covariance(
    C=1e-2,
    sigma=5e-6,
)

# %%
# Checking the calibration quality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Now, the model is ready to perform UQ. Uncertainties for energies
# and their gradients can be readily obtained along with the predictions.
# Before it is deployed, however, it is important to verify the quality
# of the UQ metrics. To do this, we consider true error vs. predicted error
# for the validation set.

refs = []
preds = []
uncertainties = []

for batch in iter(test_loader):
    batch.to(device)
    outputs = model_llpr(batch)
    preds.append(outputs['energy'].cpu().detach().numpy())
    uncertainties.append(outputs['energy_uncertainty'].cpu().detach().numpy())
    batch_dict = batch.to_dict()
    refs.append(batch_dict['energy'].cpu().detach().numpy())

refs = np.concatenate(refs).flatten()
preds = np.concatenate(preds).flatten()
uncertainties = np.concatenate(uncertainties).flatten()

true_errors = (refs - preds) ** 2

n_samples = 1000
n_samples_per_bin = 25

sort_indices = np.argsort(uncertainties)
actual_errors_sorted = true_errors[sort_indices]
predicted_errors_sorted = uncertainties[sort_indices]

actual_error_bins = []
predicted_error_bins = []

for i_bin in range(n_samples // n_samples_per_bin):
    actual_error_bins.append(
        actual_errors_sorted[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
    )
    predicted_error_bins.append(
        predicted_errors_sorted[i_bin*n_samples_per_bin:(i_bin+1)*n_samples_per_bin]
    )

actual_error_bins = np.stack(actual_error_bins)
predicted_error_bins = np.stack(predicted_error_bins)

actual_error_means = actual_error_bins.mean(axis=1)
predicted_error_means = predicted_error_bins.mean(axis=1)

plt.plot(predicted_error_means, actual_error_means, ".")
plt.yscale('log')
plt.xscale('log')

plt.xlabel(f'predicted errors [eV$^2$] \n binned by {n_samples_per_bin} points')
plt.ylabel(f'actual errors [eV$^2$] \n binned by {n_samples_per_bin} points')

plt.plot([4e-3, 4e-2], [4e-3, 4e-2], "k--", zorder=0)


# %%
# Final 
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# Now, the model is ready to perform UQ. Uncertainties for energies
# and their gradients can be readily obtained along with the predictions.
# Before it is deployed, however, it is important to verify the quality
# of the UQ metrics. To do this, we consider true error vs. predicted error
# for the validation set.

