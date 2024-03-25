"""
Sample and Feature Selection with FPS and CUR
=============================================

.. start-body

In this tutorial we generate descriptors using rascaline, then select a subset
of structures using both the farthest-point sampling (FPS) and CUR algorithms
implemented in scikit-matter. Finally, we also generate a selection of
the most important features using the same techniques.

First, import all the necessary packages
"""

# %%

import ase.io
import chemiscope
import numpy as np
from matplotlib import pyplot as plt
from metatensor import mean_over_samples
from rascaline import SoapPowerSpectrum
from sklearn.decomposition import PCA
from skmatter import feature_selection, sample_selection


# %%
# Load molecular data
# -------------------
#
# Load 500 example BTO structures from file, reading them using
# `ASE <https://wiki.fysik.dtu.dk/ase/>`_.

# Load a subset of structures of the example dataset
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
rho2i = rho2i.keys_to_samples(["species_center"]).keys_to_properties(
    ["species_neighbor_1", "species_neighbor_2"]
)
# Averages over atomic centers to compute structure features
rho2i_structure = mean_over_samples(rho2i, sample_names=["center", "species_center"])

atom_dscrptr = rho2i.block(0).values
struct_dscrptr = rho2i_structure.block(0).values

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
# Visualize selected structures
# -----------------------------
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


# %%
# Plot the PCA map
# ~~~~~~~~~~~~~~~~
#
# Notice how the selected points avoid the densely-sampled area, and cover
# the periphery of the dataset

# Matplotlib plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
scatter = ax.scatter(struct_dscrptr_pca[:, 0], struct_dscrptr_pca[:, 1], c="red")
ax.plot(
    struct_dscrptr_pca[struct_cur_idxs, 0],
    struct_dscrptr_pca[struct_cur_idxs, 1],
    "kx",
    label="CUR selection",
)
ax.plot(
    struct_dscrptr_pca[struct_fps_idxs, 0],
    struct_dscrptr_pca[struct_fps_idxs, 1],
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
# Interactive viewer (only works in notebooks)

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
        "PC1": struct_dscrptr_pca[:, 0],
        "PC2": struct_dscrptr_pca[:, 1],
        "selection": np.array(selection_levels),
    }
)

print(properties)

# Display with chemiscope. This currently does not work - as raised in issue #8
# https://github.com/lab-cosmo/software-cookbook/issues/8
cs = chemiscope.show(
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

if chemiscope.jupyter._is_running_in_notebook():
    from IPython.display import display

    display(cs)
else:
    cs.save("sample_selection.json.gz")


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
