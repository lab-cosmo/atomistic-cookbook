
"""
Sample Selection with FPS and CUR (rascaline and equisolve)
=============================================

.. start-body

In this tutorial we generate descriptors using rascaline, then select a subset
of structures using both the farthest-point sampling (FPS) and CUR algorithms
with equisolve.
"""
# %%
# First, import all the necessary packages
import ase.io  # we need to add ASE to the requirements.txt
import equistore
import equisolve
import rascaline

from equisolve.numpy import sample_selection

# %%
# Load molecular data
# -------------------
#
# Load 100 example BTO structures from file, reading them using
# `ASE <https://wiki.fysik.dtu.dk/ase/>`_.

# Load a subset of structures of the example dataset
n_frames = 250
frames = ase.io.read("./dataset/input-fps.xyz", f":{n_frames}", format="extxyz")


# %%
# Compute SOAP descriptor using librascal
# ---------------------------------------
#
# First, define the rascaline hyperparameters used to compute SOAP.

# Define SOAP hyperparameters in rascaline format
soap_hypers = {
    "cutoff": 3.0,  # Angstrom
    "max_radial": 8,  # Exclusive
    "max_angular": 5,  # Inclusive
    "atomic_gaussian_width": 0.3,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0,
}


# Generate a SOAP spherical expansion

calculator = rascaline.SoapPowerSpectrum(**soap_hypers)
soap = calculator.compute(frames)

print(soap)
# %%
# Perform structure (i.e. sample) selection
# -----------------------------------------
#
# Using FPS and CUR algorithms implemented in equisolve using scikit-matter, select a subset of
# the structures. equisolve sample selection wraps scikit-matter sample selection algorithms 
# but handles them in the equistore tensor format.
#
# For more info on the functions: `skmatter
# <https://scikit-cosmo.readthedocs.io/en/latest/selection.html>`_

# Define the number of structures to select using FPS/CUR
n_structures = 5

#FPS sample selection
selector=sample_selection.FPS(n_to_select=n_structures, initialize='random').fit(soap)

struct_fps_idxs=selector.support[0].samples['structure'] #samples selected in of first block?

print("Structure indices obtained with FPS ", struct_fps_idxs)



