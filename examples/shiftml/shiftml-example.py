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


if hasattr(__import__("builtins"), "get_ipython"):
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821

# %%
# Create a ShiftML calculator and fetch a dataset
# ===============================================

calculator = ShiftML("ShiftML3")

filename = "ShiftML_poly.zip"
if not os.path.exists(filename):
    url = (
        "https://archive.materialscloud.org/records/j2fka-sda13/files/ShiftML_poly.zip"
    )
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)


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
