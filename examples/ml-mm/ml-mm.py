"""
ML–MM Simulations with GROMACS and Metatomic
============================================

In this tutorial we will simulate alanine dipeptide in water using a machine learning
potential for the solute, while the solvent is treated with a classical force field.
This setup is commonly referred to as an ML/MM simulation and follows very similar ideas
to QM/MM.

.. hint ::

    **ML/MM vs QM/MM**

    In QM/MM simulations, a small region of the system (typically the chemically active
    part of a biomolecule) is treated with quantum mechanics, while the rest of the
    environment is described using a classical force field. The idea is that only part
    of the system requires high accuracy, and using an expensive method everywhere would
    be unnecessary.

    ML/MM follows exactly the same principle. Instead of a QM Hamiltonian, however, we
    use a machine learning (ML) potential, trained on high-level reference data, to
    provide accurate energies and forces for the solute. This retains
    near–first-principles accuracy at a fraction of the cost. Meanwhile, the surrounding
    water molecules behave perfectly well with a classical model like TIP3P, so we keep
    those as MM.

We use the *metatomic* plugin to couple a pretrained ML model to
GROMACS. The ML region is alanine dipeptide (the “protein” group), and the water is kept
as standard classical MM. We will use a pretrained PET-MAD model for ML part

.. attention ::

    The PET-MAD is trained on the PBE-sol functional and trained for solid-state
    materials and is *not* ideal for biomolecular systems. It is used here only for
    demonstration of the workflow.

We begin by loading the required Python packages.
"""

# %%
#

import subprocess

import chemiscope
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.rms import RMSD


# %%
#
# Next, we load the initial structure with ``MDAnalysis``. We select all non-water atoms
# (the protein) and visualize the structure with chemiscope. This helps confirm that the
# selections are correct and gives an overview of the starting configuration.

u_initial = mda.Universe("data/conf.gro")
ala_initial = u_initial.select_atoms("not resname SOL")

chemiscope.show(ala_initial, mode="structure")

# %%
#
# Before running the simulation, we prepare the MD input files. Below we show the MD
# parameter (:download:`grompp.mdp`) file used for this run:
#
# .. literalinclude:: grompp.mdp
#    :language: ini
#
# As you will notice the settings are very standard for a GROMACS simulation. The
# last section describes the **Metatomic interface**.
#
# To follow this tutorial, we download the PET-MAD model from HuggingFace that we list
# as ``metatomic-modelfile`` in the MD parameters.

# _ =subprocess.check_call(
#     [
#         "mtt",
#         "export",
#         "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt",
#         "-o",
#         "pet-mad-v1.0.2.pt",
#     ]
# )

# %%
#
# Next, we run the GROMACS preprocessor to generate the .tpr file, which combines the
# topology, coordinates, and MDP settings into a single binary input required for MD.

_ = subprocess.check_call(
    [
        "gmx",
        "grompp",
        "-f",
        "grompp.mdp",
        "-c",
        "data/conf.gro",
        "-p",
        "data/topol.top",
    ]
)

# %%
#
# Now we can run the MD simulation. This will perform the ML/MM calculation using the
# Metatomic plugin for the solute and classical MD for the solvent.

_ = subprocess.check_call(["gmx", "mdrun"])

# %%
#
# After the simulation finishes, we analyze the trajectory. Here we compute the RMSD of
# the solute (alanine dipeptide) relative to the initial structure.
#
# .. hint ::
#
#   RMSD measures the average positional deviation of atoms from a reference structure.
#   It is commonly used to monitor structural stability and conformational changes in
#   biomolecular simulations. A low RMSD indicates that the structure remains close to
#   the starting conformation; larger RMSD values reflect changes in backbone or
#   side-chain orientation.
#
# Using ``MDAnalysis``, we load the trajectory and compute RMSD for the selected solute
# atoms.

u = mda.Universe("topol.tpr", "traj.trr")
ala = u.select_atoms("not resname SOL")

rmsd = RMSD(atomgroup=ala, reference=ala_initial)
_ = rmsd.run()

# %%
#
# Visualize RMSD and trajectory-driven properties with ``chemiscope``.
#
# We first prepare a dictionary containing the time (in ps) and the RMSD values. The
# `RMSD results attribute
# <https://docs.mdanalysis.org/stable/documentation_pages/analysis/rms.html>`_ has shape
# ``(n_frames, 3)``, and the actual RMSD values appear in column 2.

properties = {
    "time": u.trajectory.dt * np.arange(u.trajectory.n_frames),
    "rmsd": rmsd.results["rmsd"][:, 2],
}

chemiscope.show(ala, properties=properties)

# %%
