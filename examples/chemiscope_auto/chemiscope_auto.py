"""
Chemiscope Auto
~~~~~~~~~~~~~~~
"""

import ase.io
from mace.calculators import mace_off, mace_mp
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import chemiscope


# %%
# Load QM9
# 

from load_atoms import load_dataset

frames = load_dataset("QM9")


# %%
# Computation of features
# ~~~~~~~~~~~~~~~~~~~~~~~
# 

def compute_mace_features(frames, calculator, invariants_only=True):
    descriptors = []
    for frame in tqdm(frames):
        structure_avg = np.mean(
            (calculator.get_descriptors(frame, invariants_only=invariants_only)),
            axis=0,
        )
        descriptors.append(structure_avg)
    return np.array(descriptors)

def save_descriptors(file_name, descriptors):
    np.save(file_name, descriptors)

import numpy as np


def load_or_compute_descriptors(file_name, frames, calculator, invariants_only=False):
    if os.path.exists(file_name):
        print(f"Loading descriptors from {file_name}")
        descriptors = np.load(file_name)
    else:
        print(f"Computing descriptors and saving to {file_name}")
        descriptors = compute_mace_features(frames, calculator, invariants_only)
        np.save(file_name, descriptors)
    return descriptors


# %%
# Initialize calculators
# 

descriptor_opt = {"model": "small", "device": "cpu"}
calculator_mace_off = mace_off(**descriptor_opt)
calculator_mace_mp = mace_mp(**descriptor_opt)


# %%
# Load or calculate MACE-OFF and MACE-MP features
# 

mace_mp_features_file = "data/descriptors_MACE_MP0_all.npy"
mace_mp_features = load_or_compute_descriptors(
    mace_mp_features_file, frames, calculator_mace_mp
)

mace_off_features_file = "data/descriptors_MACE_OFF_all.npy"
mace_off_features = load_or_compute_descriptors(
    mace_off_features_file, frames, calculator_mace_off
)


# %%
# Perform deminsionality reduction technics on the MACE features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import umap
from sklearn.manifold import TSNE
import numpy as np

def dimensionality_reduction_analysis(descriptors, method="PCA", use_gpu=False):
    start_time = time.time()

    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "UMAP":
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            target_metric="categorical",
        )
    elif method == "TSNE":
        perplexity = min(30, descriptors.shape[0] - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity)
    elif method == "ICA":
        reducer = FastICA(n_components=2)
    else:
        raise ValueError("Invalid method name.")

    X_reduced = reducer.fit_transform(descriptors)

    execution_time = time.time() - start_time
    print(f"{method} execution time: {execution_time:.2f} seconds")

    return X_reduced, execution_time

methods = ["PCA", "UMAP", "TSNE", "ICA"]
use_gpu = [False, False, True, False]
descriptors = [mace_off_features, mace_mp_features]

fig, axes = plt.subplots(2, len(methods), figsize=(15, 8))

for i, descriptors in enumerate(descriptors):
    for j, method in enumerate(methods):
        ax = axes[i, j]
        X_reduced, execution_time = dimensionality_reduction_analysis(
            descriptors, method=method, use_gpu=use_gpu[j]
        )
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
        ax.set_title(f"{method} ({execution_time:.2f} seconds)")
        if i == 1:
            ax.set_xlabel("Component 1")
        if j == 0:
            ax.set_ylabel("Component 2")

plt.tight_layout()
plt.show()


# %%
# Methods separately
# ~~~~~~~~~~~~~~~~~~
# 

def show_plt(
    data, method_name, features_name, x_label="Component 1", y_label="Component 2"
):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{method_name} of {features_name} Features")
    plt.show()


# %%
# PCA
# ^^^
# 

from sklearn.decomposition import PCA


def apply_pca(descriptors):
    return PCA(n_components=2).fit_transform(descriptors)

X_pca_mace_off = apply_pca(mace_off_features)

show_plt(X_pca_mace_off, method_name="PCA", features_name="MACE OFF")

X_pca_mace_mp = apply_pca(mace_mp_features)

show_plt(X_pca_mace_mp, method_name="PCA", features_name="MACE MP0")


# %%
# UMAP
# ^^^^
# 

import umap


def apply_umap(descriptors):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        target_metric="categorical",
    )
    return reducer.fit_transform(descriptors)

X_umap_mace_off = apply_umap(mace_off_features)

show_plt(X_umap_mace_off, method_name="UMAP", features_name="MACE OFF")

X_umap_mace_mp = apply_umap(mace_off_features)

show_plt(X_umap_mace_mp, method_name="UMAP", features_name="MACE MP0")


# %%
# TSNE
# ^^^^
# 

from sklearn.manifold import TSNE


def apply_tsne(descriptors):
    return TSNE(n_components=2).fit_transform(descriptors)

X_tsne_mace_off = apply_tsne(mace_off_features)

show_plt(X_tsne_mace_off, method_name="TSNE", features_name="MACE OFF")

X_tsne_mace_mp = apply_tsne(mace_mp_features)

show_plt(X_tsne_mace_mp, method_name="TSNE", features_name="MACE MP0")


# %%
# ICA
# ^^^
# 

from sklearn.decomposition import FastICA


def apply_ica(descriptors):
    reducer = FastICA(n_components=2)
    return reducer.fit_transform(descriptors)

X_ica_mace_off = apply_ica(mace_off_features)

show_plt(X_ica_mace_off, method_name="ICA", features_name="MACE OFF")

X_ica_mace_mp = apply_ica(mace_mp_features)

show_plt(X_ica_mace_mp, method_name="ICA", features_name="MACE MP0")


# %%
# Chemiscope Visualisation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 


# %%
# Extracting all properties
# 

def extract_properties(frames):
    properties = {prop: [] for prop in frames[0].info.keys()}
    for frame in frames:
        for prop, value in frame.info.items():
            properties[prop].append(value)
    return properties


properties = extract_properties(frames)

EVERY_N = 25


def display_chemiscope(X_pca, properties, meta):
    return chemiscope.show(
        frames=frames[::EVERY_N],
        properties={
            "PCA 1": {"target": "structure", "values": X_pca[:, 0][::EVERY_N].tolist()},
            "PCA 2": {"target": "structure", "values": X_pca[:, 1][::EVERY_N].tolist()},
            "homo": {"target": "structure", "values": properties["homo"][::EVERY_N]},
            "lumo": {"target": "structure", "values": properties["lumo"][::EVERY_N]},
            "gap": {"target": "structure", "values": properties["gap"][::EVERY_N]},
        },
        meta=meta,
    )


# %%
# MACE OFF
# 

display_chemiscope(X_pca_mace_off, properties, meta={"name": "QM9 MACE OFF features"})


# %%
# MACE MP
# 

display_chemiscope(X_pca_mace_mp, properties, meta={"name": "QM9 MACE MP features"})