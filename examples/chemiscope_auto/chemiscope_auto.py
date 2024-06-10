"""
Chemiscope Auto
==========================================

This example performs dimensionality reduction and visualization analysis on
molecular data from the QM9 dataset using techniques such as PCA, UMAP, t-SNE,
and ICA. It calculates features like MACE-OFF, MACE-MP, and SOAP for the
molecular structures, then applies dimensionality reduction methods to explore
their intrinsic structures.

First, we import all the necessary packages:
"""

# %%
import os
import time
from itertools import isli

import chemiscope
import joblib
import matplotlib.pyplot as plt
import numpy as np
import umap
from load_atoms import load_dataset
from mace.calculators import mace_mp, mace_off
from metatensor import mean_over_samples
from rascaline.calculators import SoapPowerSpectrum
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


# %% [markdown]
# Load QM9

# %%
frames = load_dataset("QM9")
# frames = frames[:20]

# %% [markdown]
# ### Deminsionality reduction technics

# %%
methods = ["PCA", "UMAP", "TSNE", "ICA"]


# %%
def dimensionality_reduction_analysis(descriptors, method="PCA"):
    if method not in methods:
        raise ValueError("Invalid method name.")

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

    X_reduced = reducer.fit_transform(descriptors)

    execution_time = time.time() - start_time
    print(f"{method} execution time: {execution_time:.2f} seconds")
    return X_reduced, execution_time


# %% [markdown]
# ### Dimensionality Reduction on every 25 structures

# %% [markdown]
# #### Computation of features

# %% [markdown]
# Initialize MACE calculators

# %%
descriptor_opt = {"model": "small", "device": "cpu", "default_dtype": "float64"}
calculator_mace_off = mace_off(**descriptor_opt)
calculator_mace_mp = mace_mp(**descriptor_opt)

# %% [markdown]
# Initialize SOAP calculator

# %%
hypers = {
    "cutoff": 4,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.7,
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_basis": {"Gto": {"accuracy": 1e-6}},
    "center_atom_weight": 1.0,
}
calculator_soap = SoapPowerSpectrum(**hypers)

# %% [markdown]
# Calculate MACE-OFF and MACE-MP features


# %%
def compute_mace_features(frames, calculator, invariants_only=False):
    descriptors = []
    for frame in tqdm(frames):
        structure_avg = np.mean(
            (calculator.get_descriptors(frame, invariants_only=invariants_only)),
            axis=0,
        )
        descriptors.append(structure_avg)
    return np.array(descriptors)


# %%
mace_mp_features = compute_mace_features(frames, calculator_mace_mp)
mace_off_features = compute_mace_features(frames, calculator_mace_off)

# %% [markdown]
# Calculate SOAP features


# %%
def compute_soap_features(frames, calculator):
    reducer = calculator.compute(frames)
    feat = reducer.keys_to_samples(["center_type"])
    feat = feat.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])
    X_reduced = mean_over_samples(feat, sample_names=["atom", "center_type"])
    return X_reduced.block(0).values


# %%
soap_features = compute_soap_features(frames, calculator_soap)

# %% [markdown]
# ### Permorm the dimensionality reduction

# %%
descriptors = [mace_off_features, mace_mp_features, soap_features]
descriptor_names = ["MACE OFF", "MACE MP", "SOAP"]

# %%
fig, axes = plt.subplots(len(descriptors), len(methods), figsize=(15, 8))

for i, descriptors in enumerate(descriptors):
    descriptor_name = descriptor_names[i]
    print(descriptor_name)

    for j, method in enumerate(methods):
        ax = axes[i, j]
        X_reduced, execution_time = dimensionality_reduction_analysis(
            descriptors, method=method
        )

        ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
        ax.set_title(f"{method} ({execution_time:.3f} seconds)")
        if i == 2:  # Last row
            ax.set_xlabel("Component 1")
        if j == 0 and i == 1:  # First column
            ax.set_ylabel(f"Component 2\n{descriptor_name}")
        elif j == 0:
            ax.set_ylabel(descriptor_name)

    print("")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Dimensionality reduction on the concatenated features

# %%
concatenated_features = np.concatenate(
    (mace_off_features, mace_mp_features, soap_features), axis=1
)

# %%
fig, axes = plt.subplots(1, len(methods), figsize=(20, 5))

for j, method in enumerate(methods):
    ax = axes[j]
    X_reduced, execution_time = dimensionality_reduction_analysis(
        concatenated_features, method
    )
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1])
    ax.set_title(f"{method} ({execution_time:.3f} seconds)")
    ax.set_xlabel("Component 1")
    if j == 0:
        ax.set_ylabel("Component 2")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Dimensionality Reduction on whole dataset

# %% [markdown]
# Load precomputed descriptors

# %%
mace_mp_features_file = os.path.join("data", "descriptors_MACE_MP0_all.npy")
mace_mp_features = np.load(mace_mp_features_file)

mace_off_features_file = os.path.join("data", "descriptors_MACE_OFF_all.npy")
mace_off_features = np.load(mace_off_features_file)

soap_features_file = os.path.join("data", "descriptors_SOAP_all.npy")
soap_features = np.load(soap_features_file)

# %%
descriptors = [mace_off_features, mace_mp_features, soap_features]
descriptor_names = ["mace_off", "mace_mp", "soap"]

fig, axes = plt.subplots(len(descriptors), len(methods), figsize=(15, 8))

for i, descriptors in enumerate(descriptors):
    for j, method in enumerate(methods):
        ax = axes[i, j]

        print(f"Loading existing reducer and points for {method}...")
        descriptor = descriptor_names[i]
        reducer_path = os.path.join("data", f"{method}_{descriptor}_reducer.pkl")
        points_path = os.path.join("data", f"{method}_{descriptor}_points.npy")

        reducer = joblib.load(reducer_path)
        X_reduced = np.load(points_path)

        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.3, s=1)
        ax.set_title(method)
        if i == 1:
            ax.set_xlabel("Component 1")
        if j == 0:
            ax.set_ylabel("Component 2")

plt.tight_layout()
plt.show()

######################################################################
# Chemiscope visualization
# ------------------------
#
# Visualizes the structure-property map using a chemiscope widget (and
# generates a .json file that can be viewed on
# `chemiscope.org <https://chemiscope.org>`__).
#

# %% [markdown]
# Extracting all properties

# %%


def get_properties(frames):
    properties_structure = {}

    for frame in frames:
        for prop, value in frame.info.items():
            # if prop in ['index', 'A', 'B', 'C', 'mu', 'alpha', 'homo']:
            if prop != "frequencies":
                structure_entry = properties_structure.setdefault(
                    prop, {"target": "structure", "values": []}
                )
                structure_entry["values"].append(value)

    return properties_structure


# %%
properties = get_properties(frames)

# %%
chemiscope.show(frames, properties, meta={"name": "QM9 MACE OFF features"})

# %% [markdown]
# ### Methods separately


# %%


def show_plt(data, method_name, features_name):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"{method_name} of {features_name} Features")
    plt.show()


# %% [markdown]
# #### PCA

# %%
from sklearn.decomposition import PCA


def apply_pca(descriptors):
    return PCA(n_components=2).fit_transform(descriptors)


# %%
X_pca_mace_off = apply_pca(mace_off_features)

show_plt(X_pca_mace_off, method_name="PCA", features_name="MACE OFF")

# %%
X_pca_mace_mp = apply_pca(mace_mp_features)

show_plt(X_pca_mace_mp, method_name="PCA", features_name="MACE MP0")

# %% [markdown]
# #### UMAP

# %%
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


# %%
X_umap_mace_off = apply_umap(mace_off_features)

show_plt(X_umap_mace_off, method_name="UMAP", features_name="MACE OFF")

# %%
X_umap_mace_mp = apply_umap(mace_off_features)

show_plt(X_umap_mace_mp, method_name="UMAP", features_name="MACE MP0")

# %% [markdown]
# #### TSNE

# %%
from sklearn.manifold import TSNE


def apply_tsne(descriptors):
    return TSNE(n_components=2).fit_transform(descriptors)


# %%
X_tsne_mace_off = apply_tsne(mace_off_features)

show_plt(X_tsne_mace_off, method_name="TSNE", features_name="MACE OFF")

# %%
X_tsne_mace_mp = apply_tsne(mace_mp_features)

show_plt(X_tsne_mace_mp, method_name="TSNE", features_name="MACE MP0")

# %% [markdown]
# #### ICA

# %%
from sklearn.decomposition import FastICA


def apply_ica(descriptors):
    reducer = FastICA(n_components=2)
    return reducer.fit_transform(descriptors)


# %%
X_ica_mace_off = apply_ica(mace_off_features)

show_plt(X_ica_mace_off, method_name="ICA", features_name="MACE OFF")

# %%
X_ica_mace_mp = apply_ica(mace_mp_features)

show_plt(X_ica_mace_mp, method_name="ICA", features_name="MACE MP0")

# %% [markdown]
# #### SOAP


# %%
def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


# %%
hypers_ps = {
    "cutoff": 5.0,
    "max_radial": 6,
    "max_angular": 6,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 0.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width": 0.5},
    },
    "radial_scaling": {"Willatt2018": {"exponent": 7.0, "rate": 1.0, "scale": 2.0}},
}

calculator = SoapPowerSpectrum(**hypers_ps)

# %%
for batch in batched(frames, 100):
    feat = calculator.compute(batch)

# %%
# feat = calculator.compute(frames)

# %%
feat = feat.keys_to_samples(["center_type"])
feat = feat.keys_to_properties(["neighbor_1_type", "neighbor_2_type"])

feat = mean_over_samples(feat, sample_names=["atom", "center_type"])

Xfeat = feat.block(0).values

# %%
import numpy as np


# Assuming `calculator` is your SoapPowerSpectrum instance
# and `Xfeat` is your descriptor array

# Save the descriptors (features)
# np.save('data/descriptors_SOAP_all.npy', Xfeat)
