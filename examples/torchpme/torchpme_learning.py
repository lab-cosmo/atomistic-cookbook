"""
Learning Capabilities with torchpme
=======================================

:Authors: Egor Rumiantsev `@E-Rum <https://github.com/E-Rum/>`_; Philip Loche
   `@PicoCentauri <https://github.com/PicoCentauri>`_

This example demonstrates the capabilities of the `torchpme` package, focusing on
learning target charges and utilizing the :class:`CombinedPotential` class to evaluate
potentials that combine multiple pairwise interactions with optimizable ``weights``.

The ``weights`` are optimized to reproduce the energy of a system interacting purely
through Coulomb forces.
"""

# %%
import os
import urllib.request
from typing import Dict

import ase.io
import ase.visualize.plot
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchpme import CombinedPotential, EwaldCalculator, InversePowerLawPotential
from vesin import NeighborList


# %%
# Select computation device
device = "cpu"

if torch.cuda.is_available():
    device = "cuda"

dtype = torch.float32

prefactor = 0.5292  # Unit conversion prefactor.

# %%
# Download and load the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
dataset_url = (
    "https://archive.materialscloud.org/record/file?"
    + "record_id=1924&filename=point_charges_Training_set_p1.xyz"
)
dataset_path = os.path.join(data_dir, "point_charges_Training_set.xyz")

if not os.path.isfile(dataset_path):
    print(f"Downloading dataset from {dataset_url} ...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download complete.")

# The dataset consists of atomic configurations with reference energies.
frames = ase.io.read(dataset_path, ":10")

# %%
# Define model parameters
cell = frames[0].get_cell().array
cell_dimensions = np.linalg.norm(cell, axis=1)
cutoff = np.min(cell_dimensions) / 2 - 1e-6  # Define the cutoff distance.
smearing = cutoff / 6.0  # Smearing parameter for interaction potentials.
lr_wavelength = 0.5 * smearing  # Wavelength for long-range interactions.

params = {"lr_wavelength": lr_wavelength}

# %%
# Build the neighbor list
# ~~~~~~~~~~~~~~~~~~~~~~~~
# The neighbor list is used to identify interacting pairs within the cutoff distance.
nl = NeighborList(cutoff=cutoff, full_list=False)

l_positions = []
l_cell = []
l_neighbor_indices = []
l_neighbor_distances = []
l_ref_energy = torch.zeros(len(frames), device=device, dtype=dtype)

for i_atoms, atoms in enumerate(frames):
    # Compute neighbor indices and distances
    i, j, d = nl.compute(
        points=atoms.positions, box=atoms.cell.array, periodic=True, quantities="ijd"
    )
    i = torch.from_numpy(i.astype(int))
    j = torch.from_numpy(j.astype(int))

    # Store atom positions, cell information, neighbor indices, and distances
    l_positions.append(torch.tensor(atoms.positions, device=device, dtype=dtype))
    l_cell.append(torch.tensor(atoms.cell.array, device=device, dtype=dtype))
    l_neighbor_indices.append(torch.vstack([i, j]).to(device=device).T)
    l_neighbor_distances.append(torch.from_numpy(d).to(device=device, dtype=dtype))

    # Store reference energy
    l_ref_energy[i_atoms] = torch.tensor(
        atoms.get_potential_energy(), device=device, dtype=dtype
    )


# %%
# Function to assign charges to atoms
def assign_charges(atoms, charge_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Assign charges to atoms based on their chemical symbols."""
    chemical_symbols = np.array(atoms.get_chemical_symbols())
    charges = torch.zeros(len(atoms), dtype=dtype, device=device)

    for chemical_symbol, charge in charge_dict.items():
        charges[chemical_symbols == chemical_symbol] = charge

    return charges.reshape(-1, 1)


# %%
# Define the energy computation
def compute_energy(charge_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute the total energy based on assigned charges and potentials."""
    energy = torch.zeros(len(frames), device=device, dtype=dtype)
    for i_atoms, atoms in enumerate(frames):
        charges = assign_charges(atoms, charge_dict)

        potential = calculator(
            charges=charges,
            cell=l_cell[i_atoms],
            positions=l_positions[i_atoms],
            neighbor_indices=l_neighbor_indices[i_atoms],
            neighbor_distances=l_neighbor_distances[i_atoms],
        )
        energy[i_atoms] = (charges * potential).sum()

    return energy


# %%
# Define the loss function
def loss(charge_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Calculate the loss as the mean squared error between computed and reference
    energies."""
    energy = compute_energy(charge_dict)
    mse = torch.sum((energy - l_ref_energy) ** 2)

    return mse.sum()  # Optionally add charge_penalty for strict neutrality enforcement.


# %%
# Fit charge model
# ~~~~~~~~~~~~~~~~~

# %%
# Set initial values for the potential
potential = CombinedPotential(
    potentials=[InversePowerLawPotential(exponent=1.0, smearing=smearing)],
    smearing=smearing,
)
calculator = EwaldCalculator(potential=potential, **params, prefactor=prefactor)
calculator.to(device=device, dtype=dtype)

q_Na = torch.tensor(1e-5).to(device=device, dtype=dtype)
q_Na.requires_grad = True

q_Cl = -torch.tensor(1e-5 + 0.2).to(device=device, dtype=dtype)
q_Cl.requires_grad = True

charge_dict = {"Na": q_Na, "Cl": q_Cl}

# %%
# Learning loop:
# optimize charges to minimize the loss function
optimizer = torch.optim.Adam([q_Na, q_Cl], lr=0.1)

q_Na_timeseries = []
q_Cl_timeseries = []
loss_timeseries = []

for step in range(1000):
    optimizer.zero_grad()

    charge_dict = {"Na": q_Na, "Cl": q_Cl}

    loss_value = loss(charge_dict)
    loss_value.backward()
    optimizer.step()

    if step % 10 == 0:
        print(
            f"Step: {step:>5}, Loss: {loss_value.item():>5.2e}, "
            + ", ".join([f"q_{k}: {v:>5.2f}" for k, v in charge_dict.items()]),
            end="\r",
        )

    loss_timeseries.append(float(loss_value.detach().cpu()))
    q_Na_timeseries.append(float(q_Na.detach().cpu()))
    q_Cl_timeseries.append(float(q_Cl.detach().cpu()))

    if loss_value < 1e-10:
        break

# %%
# Fit kernel model
# ~~~~~~~~~~~~~~~~
# The second phase involves optimizing the weights of the combined potential kernels.

# %%
# Set initial values for the kernel model
potential = CombinedPotential(
    [
        InversePowerLawPotential(exponent=1.0, smearing=smearing),
        InversePowerLawPotential(exponent=2.0, smearing=smearing),
    ],
    smearing=smearing,
)

calculator = EwaldCalculator(potential=potential, **params, prefactor=prefactor)
calculator.to(device=device, dtype=dtype)

# %%
# Kernel optimization loop:
# optimize kernel weights to minimize the loss function
optimizer = torch.optim.Adam(calculator.parameters(), lr=0.1)

weights_timeseries = []
loss_timeseries = []

for step in range(1000):
    optimizer.zero_grad()

    # Fix charges to their ideal values for this phase
    loss_value = loss({"Na": 1.0, "Cl": -1.0})
    loss_value.backward()
    optimizer.step()

    if step % 10 == 0:
        print(
            f"Step: {step:>5}, Loss: {loss_value.item():>5.2e} "
            + ", ".join(
                [
                    f"w_{i}: {float(v):>5.2f}"
                    for i, v in enumerate(
                        calculator.potential.weights.detach().cpu().tolist()
                    )
                ]
            ),
            end="\r",
        )

    loss_timeseries.append(float(loss_value.detach().cpu()))
    weights_timeseries.append(calculator.potential.weights.detach().cpu().tolist())

    if loss_value < 1e-10:
        break

# %%
# Plot results
# ~~~~~~~~~~~~~~~~~~~
# Visualize the learning process for charges and kernel weights.

palette = [
    "#EE7733",  # Orange
    "#0077BB",  # Blue
    "#33BBEE",  # Light Blue
    "#EE3377",  # Pink
    "#CC3311",  # Red
    "#009988",  # Teal
    "#BBBBBB",  # Grey
    "#000000",  # Black
]


def plot_results(fname=None, show_snapshot=True):
    """
    Plot the learning process for charges and kernel weights.

    Args:
        fname (str): File name to save the plot. If None, the plot is not saved.
        show_snapshot (bool): Whether to show a snapshot of the atomic configuration.
    """
    fig, ax = plt.subplots(
        2,
        sharex=True,
        layout="constrained",
        dpi=200,
    )

    if show_snapshot:
        ax_in = fig.add_axes([0.12, 0.14, 0.27, 0.27])
        ase.visualize.plot.plot_atoms(atoms, ax=ax_in, radii=0.75)
        ax_in.set_axis_off()

    # Plot charge learning
    ax[0].plot(q_Na_timeseries, c=palette[0], label=r"Na")
    ax[0].plot(np.array(q_Cl_timeseries), c=palette[1], label=r"Cl")

    ax[0].set_ylim(-1.3, 1.3)
    ax[0].axhline(1, ls="dotted", c=palette[0])
    ax[0].axhline(-1, ls="dotted", c=palette[1])
    ax[0].legend()
    ax[0].set_ylabel(r"Charge / e")

    # Plot kernel weight learning
    ax[1].axhline(1, c=palette[2], ls="dotted")
    ax[1].axhline(0, c=palette[3], ls="dotted")
    weights_timeseries_array = np.array(weights_timeseries)
    ax[1].plot(weights_timeseries_array[:, 0], label="p=1", c=palette[2])
    ax[1].plot(weights_timeseries_array[:, 1], label="p=2", c=palette[3])

    ax[1].set_ylim(-0.2, 1.2)
    ax[1].legend()
    ax[1].set_ylabel("Kernel weight")

    for a in ax:
        a.set_xscale("log")

    ax[1].set_xlabel("Learning epoch")

    fig.align_labels()

    if fname is not None:
        fig.savefig(fname, transparent=True, bbox_inches="tight")

    plt.show()


# Call the plot function to visualize results
plot_results("toy_model_learning.pdf", show_snapshot=True)

# %%
