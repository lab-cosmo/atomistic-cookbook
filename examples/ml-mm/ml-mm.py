"""
ML/MM Simulations with GROMACS and Metatomic
=============================================

:Authors: Philip Loche `@PicoCentauri <https://github.com/PicoCentauri>`_,
          Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_

In this tutorial we simulate and analyse a alanine dipeptide in water using a machine
learning potential for the solute while the solvent is treated with a classical force
field. This setup is commonly referred to as an ML/MM simulation and follows very
similar ideas to QM/MM.

.. hint ::

    **ML/MM vs QM/MM**

    In QM/MM simulations, a small region of the system (typically the chemically active
    part of a biomolecule) is treated with quantum mechanics, while the rest of the
    environment is described using a classical force field. The idea is that only part
    of the system requires high accuracy, and using an expensive method everywhere would
    be unnecessary.

    ML/MM follows exactly the same principle.  Instead of a QM Hamiltonian, however, we
    use a machine learning (ML) potential, trained on high-level reference data, to
    provide accurate energies and forces for the solute.  This retains
    near-first-principles accuracy at a fraction of the cost.  Meanwhile, the
    surrounding water molecules behave perfectly well with a classical model like TIP3P,
    so we keep those as MM.

We use the *Metatomic* plugin to couple a pretrained ML model to GROMACS. The ML region
consists of an alanine dipeptide (the "protein" group), and the water is kept as
standard classical MM.

We will use the **PET-MAD XS** model (v1.5.0), a small but capable universal potential
from the `uPET <https://huggingface.co/lab-cosmo/upet>`_ family.

.. attention ::

    PET-MAD is trained on a broad materials dataset (r2SCAN functional) and is *not*
    specifically optimized for biomolecular systems.  It is used here to demonstrate the
    ML/MM workflow.  For production work, consider a model fine-tuned on relevant
    biochemical data.
"""

# %%
# Setup
# -----
#
# We begin by loading the required Python packages.

import subprocess
from pathlib import Path

import ase.io
import chemiscope
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.dihedrals import Ramachandran
from MDAnalysis.analysis.rms import RMSD


# %%
# Initial structure
# -----------------
#
# We load the initial alanine dipeptide + water structure.  We read it with both ASE
# (for chemiscope visualization) and MDAnalysis (for trajectory analysis later). We
# select the non-water atoms (the protein) so we can confirm the selections are correct.

initial_atoms = ase.io.read("data/conf.gro")
u_initial = mda.Universe("data/conf.gro")
ala_initial = u_initial.select_atoms("not resname SOL")

print(f"System: {len(initial_atoms)} atoms total, {len(ala_initial)} solute atoms")
chemiscope.show([initial_atoms], mode="structure")

# %%
# Model export
# ------------
#
# Before running the simulation, we need to export the ML model into the TorchScript
# format that GROMACS can load. We download the PET-MAD XS checkpoint from HuggingFace
# and export it using ``Metatrain``.

repo_id = "lab-cosmo/upet"
tag = "v1.5.0"
url_path = f"models/pet-mad-xs-{tag}.ckpt"
fname = Path(f"models/pet-mad-xs-{tag}.pt")
url = f"https://huggingface.co/{repo_id}/resolve/main/{url_path}"
fname.parent.mkdir(parents=True, exist_ok=True)

subprocess.run(
    [
        "mtt",
        "export",
        url,
        "-o",
        str(fname),
    ],
    check=True,
)
print(f"Successfully exported {fname}.")

# %%
# Running the simulation
# ----------------------
#
# The MD parameter file (:download:`grompp.mdp`) controls the simulation. The key
# section for ML/MM is the **Metatomic interface** at the bottom:
#
# .. literalinclude:: grompp.mdp
#    :language: ini
#    :lines: 26-31
#
# This tells GROMACS to load the exported PET-MAD model and apply ML forces to the
# ``protein`` group.  All other atoms (water) use the classical force field as usual.
#
# We run the GROMACS preprocessor (``grompp``) to combine the topology, coordinates, and
# MDP settings into a single binary input (``.tpr``), then execute the simulation with
# ``mdrun``.

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

_ = subprocess.check_call(["gmx", "mdrun"])

# %%
# RMSD analysis
# -------------
#
# After the simulation finishes, we analyze the trajectory.  We start by
# computing the RMSD (root mean square deviation) of the solute relative
# to the initial structure.
#
# .. hint ::
#
#   RMSD measures the average positional deviation of atoms from a reference structure.
#   It is commonly used to monitor structural stability and conformational changes in
#   biomolecular simulations. A low RMSD indicates the structure remains close to the
#   starting conformation; larger RMSD values reflect changes in backbone or side-chain
#   orientation.

u = mda.Universe("data/conf.gro", "traj.trr")
ala = u.select_atoms("not resname SOL")

rmsd = RMSD(atomgroup=ala, reference=ala_initial)
_ = rmsd.run()

time_ps = u.trajectory.dt * np.arange(u.trajectory.n_frames)

plt.figure(figsize=(6, 3))
plt.plot(time_ps, rmsd.results["rmsd"][:, 2], linewidth=0.8)
plt.xlabel("Time (ps)")
plt.ylabel("RMSD (A)")
plt.title("Solute RMSD")
plt.tight_layout()

# %%
# Ramachandran plot
# -----------------
#
# The Ramachandran plot shows the backbone dihedral angles phi and psi, which
# characterize the conformational state of the peptide backbone. These two angles
# determine the local geometry of each residue and are a classic analysis target for
# peptide and protein simulations.
#
# For this short ML/MM trajectory we can see which region of Ramachandran space the
# alanine dipeptide explores under the ML potential.

protein = u.select_atoms("protein")
rama = Ramachandran(protein).run()

plt.figure(figsize=(5, 5))
plt.scatter(
    rama.results.angles[:, :, 0].flatten(),
    rama.results.angles[:, :, 1].flatten(),
    s=3,
    alpha=0.5,
)
plt.xlabel(r"$\phi$ (degrees)")
plt.ylabel(r"$\psi$ (degrees)")
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.title("Ramachandran plot")
plt.gca().set_aspect("equal")
plt.tight_layout()

# %%
# Trajectory visualization
# ------------------------
#
# Finally, we convert the trajectory to PDB format so that ASE can read it, and
# visualize it interactively with chemiscope. Each frame is colored by its RMSD value,
# letting us see how the structure evolves over the course of the simulation.

subprocess.run(
    ["gmx", "trjconv", "-f", "traj.trr", "-s", "data/conf.gro", "-o", "traj.pdb"],
    input=b"0\n",
    check=True,
)

trajectory = ase.io.read("traj.pdb", index=":")

properties = {
    "time": time_ps,
    "rmsd": rmsd.results["rmsd"][:, 2],
}

chemiscope.show(structures=trajectory, properties=properties)
