"""
Sample Selection with FPS and CUR (librascal)
=============================================

.. start-body

In this tutorial we generate descriptors using librascal, then select a subset
of structures using both the farthest-point sampling (FPS) and CUR algorithms
implemented in scikit-matter.
"""
# %%
# First, import all the necessary packages
import ase.io
import numpy as np
from sklearn.decomposition import PCA

import chemiscope
from rascal.representations import SphericalInvariants
from rascal.utils import FPSFilter
from skmatter import sample_selection, feature_selection


# %%
# Load molecular data
# -------------------
#
# Load 100 example BTO structures from file, reading them using
# `ASE <https://wiki.fysik.dtu.dk/ase/>`_.

# Load a subset of structures of the example dataset
n_frames = 250
frames = ase.io.read("./dataset/input-fps.xyz", f":{n_frames}", format="extxyz")

# librascal requires the atomic positions to be wrapped in the cell
for frame in frames:
    frame.wrap(eps=1.0e-12)


# %%
# Compute SOAP descriptor using librascal
# ---------------------------------------
#
# First, define the librascal hyperparameters used to compute SOAP.

# librascal hyperparameters
soap_hypers = {
    "soap_type": "PowerSpectrum",
    "interaction_cutoff": 6.0,
    "max_radial": 8,
    "max_angular": 6,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_function_type": "RadialScaling",
    "cutoff_smooth_width": 0.5,
    "cutoff_function_parameters": {
        "rate": 1,
        "scale": 3.5,
        "exponent": 4,
    },
    "radial_basis": "GTO",
    "normalize": True,
    "optimization": {
        "Spline": {
            "accuracy": 1.0e-05,
        },
    },
    "compute_gradients": False,
}

# Generate a SOAP spherical expansion
soap = SphericalInvariants(**soap_hypers)

# Perform a data trasnformation and get the descriptor with samples as atomic environments
atom_dscrptr = soap.transform(frames).get_features(soap)

# Calculate the stucture features as the mean over the atomic features for each
# structure
struct_dscrptr = np.zeros((len(frames), atom_dscrptr.shape[1]))
atom_idx_start = 0
for i, frame in enumerate(frames):
    atom_idx_stop = atom_idx_start + len(frame.numbers)
    struct_dscrptr[i] = atom_dscrptr[atom_idx_start:atom_idx_stop].mean(axis=0)
    atom_idx_start = atom_idx_stop

print("atom feature descriptor shape:", atom_dscrptr.shape)
print("structure feature descriptor shape:", struct_dscrptr.shape)


# %%
# Perform structure (i.e. sample) selection
# -----------------------------------------
#
# Using FPS and CUR algorithms implemented in scikit-matter, select a subset of
# the structures. skmatter assumes that our descriptor is represented as a 2D
# matrix, with the samples along axis 0 and features along axis 1.
#
# For more info on the functions: `skmatter
# <https://scikit-cosmo.readthedocs.io/en/latest/selection.html>`_

# Define the number of structures to select using FPS/CUR
n_structures = 25

# FPS sample selection
struct_fps = sample_selection.FPS(n_to_select=n_structures, initialize="random").fit(
    struct_dscrptr
)
struct_fps_idxs = struct_fps.selected_idx_

# CUR sample selection
struct_cur = sample_selection.CUR(n_to_select=n_structures).fit(struct_dscrptr)
struct_cur_idxs = struct_cur.selected_idx_

print("Structure indices obtained with FPS ", struct_fps_idxs)
print("Structure indices obtained with CUR ", struct_cur_idxs)

# Slice structure descriptor along axis 0 to contain only the selected structures
struct_dscrptr_fps = struct_dscrptr[struct_fps_idxs, :]
struct_dscrptr_cur = struct_dscrptr[struct_cur_idxs, :]
assert struct_dscrptr_fps.shape == struct_dscrptr_cur.shape

print("Structure descriptor shape before selection ", struct_dscrptr.shape)
print("Structure descriptor shape after selection ", struct_dscrptr_fps.shape)


# %%
# Visualize selected structures with chemiscope
# ---------------------------------------------
#
# sklearn can be used to perform PCA dimensionality reduction on the SOAP
# descriptors. The resulting PC coordinates can be used to visualize the the
# data alongside their structures in a chemiscope widget.
#
# Note: chemiscope widgets are not currently integrated into our sphinx gallery:
# coming soon.

# Generate a structure PCA
struct_dscrptr_pca = PCA(n_components=2).fit_transform(struct_dscrptr)
assert struct_dscrptr_pca.shape == (n_frames, 2)

# Selected level
selection_levels = []
for i in range(len(frames)):
    level = 0
    if i in struct_cur_idxs:
        level += 1
    if i in struct_fps_idxs:
        level += 2
    selection_levels.append(level)


properties = {
    "PC1": struct_dscrptr_pca[:, 0],
    "PC2": struct_dscrptr_pca[:, 1],
    "Selection: (1) CUR, (2) FPS, (3) both": np.array(selection_levels),
}


# Display with chemiscope. This currently does not work - as raised in issue #8
# https://github.com/lab-cosmo/software-cookbook/issues/8
# chemiscope.show(frames, properties=properties)


# %%
# Perform feature selection
# -------------------------
#
# Now perform feature selection. In this example we will go back to using the
# descriptor decomposed into atomic environments, as opposed to the one
# decomposed into structure environments, but only use FPS for brevity.

# Define the number of features to select
n_features = 200

# FPS feature selection
feat_fps = feature_selection.FPS(n_to_select=n_features, initialize="random").fit(
    atom_dscrptr
)
feat_fps_idxs = feat_fps.selected_idx_

print("Feature indices obtained with FPS ", feat_fps_idxs)

# Slice atomic descriptor along axis 1 to contain only the selected features
atom_dscrptr_fps = atom_dscrptr[:, feat_fps_idxs]

print("atomic descriptor shape before selection ", atom_dscrptr.shape)
print("atomic descriptor shape after selection ", atom_dscrptr_fps.shape)

# %%
