"""
Sample and Feature Selection with FPS and CUR
=============================================

:Authors: Davide Tisi `@DavideTisi <https://github.com/DavideTisi>`_

In this tutorial we generate descriptors using rascaline, then select a subset
of structures using both the farthest-point sampling (FPS) and CUR algorithms
implemented in scikit-matter. Finally, we also generate a selection of
the most important features using the same techniques.

First, import all the necessary packages
"""

# %%

import ase.io
import chemiscope
import metatensor
import numpy as np
from equisolve.numpy import feature_selection, sample_selection
from matplotlib import pyplot as plt
from rascaline import SoapPowerSpectrum
from sklearn.decomposition import PCA
from skmatter import feature_selection as skfeat_selection


# %%
# Load molecular data
# -------------------
#
# Load 500 example BTO structures from file, reading them using
# `ASE <https://wiki.fysik.dtu.dk/ase/>`_.

# Load a subset of :download:`structures <input-fps.xyz>` of the example dataset
n_frames = 500
frames = ase.io.read("input-fps.xyz", f":{n_frames}", format="extxyz")

# %%
# Compute SOAP descriptors using rascaline
# ----------------------------------------
#
# First, define the rascaline hyperparameters used to compute SOAP.


# rascaline hyperparameters
hypers = {
    "cutoff": 6.0,
    "max_radial": 8,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {"accuracy": 1e-6}},
    "radial_scaling": {"Willatt2018": {"exponent": 4, "rate": 1, "scale": 3.5}},
    "center_atom_weight": 1.0,
}

# Generate a SOAP power spectrum
calculator = SoapPowerSpectrum(**hypers)
rho2i = calculator.compute(frames)


# Makes a dense block
atom_soap = rho2i.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

atom_soap_single_block = atom_soap.keys_to_samples(keys_to_move=["species_center"])

# Sum over atomic centers to compute structure features
struct_soap = metatensor.sum_over_samples(
    atom_soap_single_block, sample_names=["center", "species_center"]
)


print("atom feature descriptor shape:", atom_soap.block(0).values.shape)
print(
    "atom feature descriptor (all in one block) shape:",
    atom_soap_single_block.block(0).values.shape,
)
print("structure feature descriptor shape:", struct_soap.block(0).values.shape)


# %%
# Perform atomic environment (i.e. sample) selection
# ---------------------------------------------------
#
# Using FPS and CUR algorithms, we can perform selection of atomic environments.
# These are implemented in equisolve, which provides a wrapper around
# scikit-matter to allow for interfacing with data stored in the metatensor
# format.
#
# Suppose we want to select the 10 most diverse environments for each chemical
# species.
#
# First, we can use the `keys_to_properties` operation in metatensor to move the
# neighbour species indices to the properties of the TensorBlocks. The resulting
# descriptor will be a TensorMap comprised of three blocks, one for each
# chemical species, where the chemical species indices are solely present in the
# keys.


print("----Atomic environment selection-----")
# Define the number of structures to select using FPS/CUR
n_envs = 25

print(atom_soap)
print(atom_soap.block(0))

# %% Now let's perform sample selection on the atomic environments. We want to
# select 10 atomic environments for each chemical species.

# Define the number of structures *per block* to select using FPS
n_envs = 10

# FPS sample selection
selector_atomic_fps = sample_selection.FPS(n_to_select=n_envs, initialize="random").fit(
    atom_soap
)

# Print the selected envs for each block
print("atomic envs selected with FPS:\n")
for key, block in selector_atomic_fps.support.items():
    print("species_center:", key, "\n(struct_idx, atom_idx)\n", block.samples.values)

selector_atomic_cur = sample_selection.CUR(n_to_select=n_envs).fit(atom_soap)
# Print the selected envs for each block
print("atomic envs selected with CUR:\n")
for key, block in selector_atomic_cur.support.items():
    print("species_center:", key, "\n(struct_idx, atom_idx)\n", block.samples.values)


# %%
# Selecting from a combined pool of atomic environments
# -----------------------------------------------------
#
# One can also select from a combined pool of atomic environments and
# structures, instead of selecting an equal number of atomic environments for
# each chemical species. In this case, we can move the 'species_center' key to samples
# such that our descriptor is a TensorMap consisting of a single block. Upon
# sample selection, the most diverse atomic environments will be selected,
# regardless of their chemical species.
print("----All atomic environment selection-----")

print("keys", atom_soap.keys)
print("blocks", atom_soap[0])
print("samples in first block", atom_soap[0].samples)

# Using the original SOAP descriptor, move all keys to properties.


# Define the number of structures to select using FPS
n_envs = 10

# FPS sample selection
selector_atomic_fps = sample_selection.FPS(n_to_select=n_envs, initialize="random").fit(
    atom_soap_single_block
)
print(
    "atomic envs selected with FPS: \n (struct_idx, atom_idx, species_center) \n",
    selector_atomic_fps.support.block(0).samples.values,
)


# %%
# Perform structure (i.e. sample) selection with FPS/CUR
# ---------------------------------------------------------
#
# Instead of atomic environments, one can also select diverse structures. We can
# use the `sum_over_samples` operation in metatensor to define features in the
# structural basis instead of the atomic basis. This is done by summing over the
# atomic environments, labeled by the 'center' index in the samples of the
# TensorMap.
#
# Alternatively, one could use the `mean_over_samples` operation, depending on
# the specific inhomogeneity of the size of the structures in the training set.

print("----Structure selection-----")

# Define the number of structures to select *per block* using FPS
n_structures = 10

# FPS structure selection
selector_struct_fps = sample_selection.FPS(
    n_to_select=n_structures, initialize="random"
).fit(struct_soap)
struct_fps_idxs = selector_struct_fps.support.block(0).samples.values.flatten()

print("structures selected with FPS:\n", struct_fps_idxs)

# CUR structure selection
selector_struct_cur = sample_selection.CUR(n_to_select=n_structures).fit(struct_soap)
struct_cur_idxs = selector_struct_cur.support.block(0).samples.values.flatten()
print("structures selected with CUR:\n", struct_cur_idxs)


# Slice structure descriptor along axis 0 to contain only the selected structures
struct_soap_fps = struct_soap.block(0).values[struct_fps_idxs, :]
struct_soap_cur = struct_soap.block(0).values[struct_cur_idxs, :]
assert struct_soap_fps.shape == struct_soap_cur.shape

print("Structure descriptor shape before selection ", struct_soap.block(0).values.shape)
print("Structure descriptor shape after selection (FPS)", struct_soap_fps.shape)
print("Structure descriptor shape after selection (CUR)", struct_soap_cur.shape)


# %%
# Visualize selected structures
# -----------------------------
#
# sklearn can be used to perform PCA dimensionality reduction on the SOAP
# descriptors. The resulting PC coordinates can be used to visualize the the
# data alongside their structures in a chemiscope widget.
#

# Generate a structure PCA
struct_soap_pca = PCA(n_components=2).fit_transform(struct_soap.block(0).values)
assert struct_soap_pca.shape == (n_frames, 2)


# %%
# Plot the PCA map
# ~~~~~~~~~~~~~~~~
#
# Notice how the selected points avoid the densely-sampled area, and cover
# the periphery of the dataset

# Matplotlib plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
scatter = ax.scatter(struct_soap_pca[:, 0], struct_soap_pca[:, 1], c="red")
ax.plot(
    struct_soap_pca[struct_cur_idxs, 0],
    struct_soap_pca[struct_cur_idxs, 1],
    "ko",
    fillstyle="none",
    label="FPS selection",
)
ax.set_xlabel("PCA[1]")
ax.set_ylabel("PCA[2]")
ax.legend()
fig.show()


# %%
# Creates a chemiscope viewer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Selected level
selection_levels = []
for i in range(len(frames)):
    level = 0
    if i in struct_cur_idxs:
        level += 1
    if i in struct_fps_idxs:
        level += 2
    if level == 0:
        level = "Not selected"
    elif level == 1:
        level = "CUR"
    elif level == 2:
        level = "FPS"
    else:
        level = "FPS+CUR"
    selection_levels.append(level)

properties = chemiscope.extract_properties(frames)

properties.update(
    {
        "PC1": struct_soap_pca[:, 0],
        "PC2": struct_soap_pca[:, 1],
        "selection": np.array(selection_levels),
    }
)

widget = chemiscope.show(
    frames,
    properties=properties,
    settings={
        "map": {
            "x": {"property": "PC1"},
            "y": {"property": "PC2"},
            "color": {"property": "energy"},
            "symbol": "selection",
            "size": {"factor": 50},
        },
        "structure": [{"unitCell": True}],
    },
)
widget.save("sample-selection.json.gz")

# display, if in notebook or sphinx
widget

# %%
# Perform feature selection
# -------------------------
#
# Now perform feature selection. In this example we will go back to using the
# descriptor decomposed into atomic environments, as opposed to the one
# decomposed into structure environments, but only use FPS for brevity.
print("----Feature selection-----")

# Define the number of features to select
n_features = 200

# FPS feature selection
feat_fps = feature_selection.FPS(n_to_select=n_features, initialize="random").fit(
    atom_soap_single_block
)

# Slice atomic descriptor along axis 1 to contain only the selected features
# atom_soap_single_block_fps = atom_soap_single_block.block(0).values[:, feat_fps_idxs]
atom_soap_single_block_fps = metatensor.slice(
    atom_soap_single_block,
    axis="properties",
    labels=feat_fps.support.block(0).properties,
)

print(
    "atomic descriptor shape before selection ",
    atom_soap_single_block.block(0).values.shape,
)
print(
    "atomic descriptor shape after selection ",
    atom_soap_single_block_fps.block(0).values.shape,
)

# %%

# %%
# Perform feature selection (skmatter)
# ------------------------------------
#
# Now perform feature selection. In this example we will go back to using the
# descriptor decomposed into atomic environments, as opposed to the one
# decomposed into structure environments, but only use FPS for brevity.

print("----Feature selection (skmatter)-----")

# Define the number of features to select
n_features = 200

# FPS feature selection
feat_fps = skfeat_selection.FPS(n_to_select=n_features, initialize="random").fit(
    atom_soap_single_block.block(0).values
)
feat_fps_idxs = feat_fps.selected_idx_

print("Feature indices obtained with FPS ", feat_fps_idxs)

# Slice atomic descriptor along axis 1 to contain only the selected features
atom_dscrptr_fps = atom_soap_single_block.block(0).values[:, feat_fps_idxs]

print(
    "atomic descriptor shape before selection ",
    atom_soap_single_block.block(0).values.shape,
)
print("atomic descriptor shape after selection ", atom_dscrptr_fps.shape)

# %%
