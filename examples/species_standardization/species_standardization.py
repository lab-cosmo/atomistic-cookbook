"""
Multi-Species Standardization
========================================================

.. start-body

In this tutorial we generate feature vectors using rascaline then perform 
standardization for a dataset of multi-species (gallium arsenide) structures.
"""

# %%
# First, we import all the necessary packages:
import ase.io
import numpy as np
from sklearn.decomposition import PCA
from time import time

from rascaline import SoapPowerSpectrum
from skmatter.preprocessing import StandardFlexibleScaler

# %%
# Load structures
# -------------------
#
# Load 100 GaAs structures from file, reading them using
# `ASE <https://wiki.fysik.dtu.dk/ase/>`_. Note that first
# half of the structure file is 1:1 ratio structures, and 
# second half is pure Ga structures.

# Load a subset of structures of the example dataset
n_frames = 100
frames = ase.io.read("./dataset/GaAs.xyz", f":{n_frames}", format="extxyz")

# atomic positions are wrapped prior to calculating the feature vectors
for frame in frames:
    frame.wrap(eps=1.0e-12)


# %%
# Compute SOAP feature vectors using rascaline
# ---------------------------------------
#
# First, define the rascaline hyperparameters used to compute SOAP,
# taken from `Lopanitsyna et al. <https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.045802>`.

# rascaline hyperparameters
HYPER_PARAMETERS = {
    "cutoff": 5.0,
    "max_radial": 8,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "radial_scaling": {"Willatt2018": {"scale": 2.0, "rate": 0.8, "exponent": 2}},
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)
# we pass `species_center`, `species_neighbor_1`, and `species_neighbor_2`
# all to properties. This results in the description of the two atomic species
# in strictly disonnected, separate feature spaces.
raw_featvec = calculator.compute(frames).keys_to_properties(["species_center", "species_neighbor_1", "species_neighbor_2"])


