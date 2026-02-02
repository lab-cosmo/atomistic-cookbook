"""
Computing NMR shielding tensors using ShiftML
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

:Authors: Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

This example shows how to compute NMR shielding tensors
using a point-edge transformer model trained on the ShiftML
dataset.
"""

import os
import zipfile

import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import requests

# %%
from ase.io import read
from shiftml.ase import ShiftML
from urllib3.util.retry import Retry


# %%
# Create a ShiftML calculator and fetch a dataset
# ===============================================


def fetch_dataset(filename, base_url, local_path=""):
    """Helper function to load data with retries on errors."""

    local_file = local_path + filename
    if os.path.isfile(local_file):
        return

    # Retry strategy: wait 1s, 2s, 4s, 8s, 16s on 429/5xx errors
    retry_strategy = Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry_strategy))

    # Fetch with automatic retry and error raising
    response = session.get(base_url + filename)
    response.raise_for_status()

    with open(local_file, "wb") as file:
        file.write(response.content)


calculator = ShiftML("ShiftML3")

filename = "ShiftML_poly.zip"
fetch_dataset(filename, "https://archive.materialscloud.org/records/j2fka-sda13/files/")

with zipfile.ZipFile(filename, "r") as zip_ref:
    for file in ["ShiftML_poly/Cocaine/cocaine_QuantumEspresso.xyz"]:
        target = os.path.basename(file)
        with zip_ref.open(file) as source, open(target, "wb") as dest:
            dest.write(source.read())

frames_cocaine = read("cocaine_QuantumEspresso.xyz", index=":16")
reference = [frame.arrays["CS"] for frame in frames_cocaine]

# %%
# Predicts isotropic chemical shielding tensors, including uncertainty
# ====================================================================
predicted = [calculator.get_cs_iso_ensemble(frame) for frame in frames_cocaine]


# %%
# Make a plot for all the H shielding values

h_shieldings = np.hstack(
    [
        [
            r[f.symbols == "H"],
            p[f.symbols == "H"].mean(axis=1),
            p[f.symbols == "H"].std(axis=1),
        ]
        for r, p, f in zip(reference, predicted, frames_cocaine)
    ]
)

# %%
# and of the uncertainty
fig, ax = plt.subplots(figsize=(4, 3))

ax.plot(h_shieldings[0], h_shieldings[1], "o", markersize=2, alpha=0.5)
# %%
fig, ax = plt.subplots(figsize=(4, 3))
ax.loglog(
    h_shieldings[2],
    np.abs(h_shieldings[0] - h_shieldings[1]),
    "o",
    markersize=2,
    alpha=0.5,
)
ax.plot([0.2, 0.8], [0.2, 0.8], "k--", lw=0.5)

# %%
# Anisotropic shielding tensors
# =============================

tensors = [calculator.get_cs_tensor(frame) for frame in frames_cocaine]

# %%

h_tensors = np.array(
    [
        (t[iat] if f.symbols[iat] == "H" else np.zeros((3, 3)))
        for t, f in zip(tensors, frames_cocaine)
        for iat in range(len(f))
    ]
)

# %%


chemiscope.show(
    frames_cocaine,
    shapes={
        "cs_ellipsoid": {
            "kind": "ellipsoid",
            "parameters": {
                "global": {},
                "atom": [
                    chemiscope.ellipsoid_from_tensor(cs_h * 0.05) for cs_h in h_tensors
                ],
            },
        }
    },
    mode="structure",
    settings=chemiscope.quick_settings(
        periodic=True,
        structure_settings={
            "shape": ["cs_ellipsoid"],
        },
    ),
)
