"""
Sample Selection with FPS and CUR (rascaline and equisolve)
=============================================

.. start-body

TODO: 
- Fill out context + motivation here
- Does structure selection for blocks of different chemical species make sense?
- Chemiscope
- Plot a histogram (or just report numbers) of the chemical species selected
when using a single block descriptor vs the equal number of samples for each
species when using a multi-block descriptor.

# Motivations

1. Selecting reference atomic envs for kernel construction
2. Selecting diverse samples for training:
    * Diverse structures, i.e. configurations from an MD trajectory
       to compute with high fidelity DFT and used as the training data.
    * Atomic environments - i.e. for selecting atomic environments used as the
       training data. Used when the target property is locally decomposed (i.e.), but
       not appropriate for globally defined properties (i.e. total energies).
"""
# %%
# First, import all the necessary packages
import ase.io  # we need to add ASE to the requirements.txt
import metatensor
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
# Compute SOAP descriptor using rascaline
# ---------------------------------------
#
# First, define the rascaline hyperparameters used to compute SOAP.
# Then the SOAP descriptor can be computed using rascaline.

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
soap_atomic = calculator.compute(frames)

# %%
# Perform atomic environment (i.e. sample) selection
# --------------------------------------------------

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

# Move the neighbour species indices from keys to properties.
soap_atomic = soap_atomic.keys_to_properties(
    keys_to_move=["species_neighbor_1", "species_neighbor_2"]
)
print(soap_atomic)
print(soap_atomic.block(0))

# %% Now let's perform sample selection on the atomic environments. We want to
# select 10 atomic environments for each chemical species.

# Define the number of structures *per block* to select using FPS
n_envs = 10

# FPS sample selection
selector_atomic_fps = sample_selection.FPS(n_to_select=n_envs, initialize="random").fit(
    soap_atomic
)

#selector=sample_selection.FPS(n_to_select=n_envs, initialize='random').fit(soap_atomic)

# Print the selected envs for each block
print("atomic envs selected with FPS:\n")
for key, block in selector_atomic_fps.support.items():
    print("species_center:", key, "(struct_idx, atom_idx)", block.samples)

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

# Sum over atomic environments.
soap_struct = metatensor.sum_over_samples(soap_atomic, "center")
print(soap_struct)
print(soap_struct.block(0))

# Define the number of structures to select *per block* using FPS
n_structures = 5

# FPS structure selection
selector_struct_fps = sample_selection.FPS(
    n_to_select=n_structures, initialize="random"
).fit(soap_struct)
print("structures selected with FPS:", selector_struct_fps.support.block(0).samples)

# FPS structure selection
selector_struct_cur = sample_selection.FPS(n_to_select=n_structures).fit(soap_struct)
print("structures selected with CUR:", selector_struct_cur.support.block(0).samples)

# %%
# Selecting from a combined pool of atomic environments
# -----------------------------------------------------
#
# One can also select from a combined pool of atomic environments and
# structures, instead of selecting an equal number of atomic environments for
# each chemical species. In this case, we can move all the keys to properties
# such that our descriptor is a TensorMap consisting of a single block. Upon
# sample selection, the most diverse atomic environments will be selected,
# regardless of their chemical species.

print('keys',soap_atomic.keys)
print('blocks',soap_atomic[0])

# Using the original SOAP descriptor, move all keys to properties.
soap_atomic_single_block = soap_atomic.keys_to_samples(
    keys_to_move=["species_center"]#, "species_neighbor_1", "species_neighbor_2"]
)
print(soap_atomic_single_block)
print(soap_atomic_single_block.block(0))  # There is only one block now!

# Define the number of structures to select using FPS
n_envs = 10

# FPS sample selection
selector_atomic_fps = sample_selection.FPS(n_to_select=n_envs, initialize="random").fit(
    soap_atomic_single_block
)
print(
    "atomic envs selected with FPS (struct_idx, atom_idx):\n",
    selector_atomic_fps.support.block(0).samples,
)
