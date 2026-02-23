# -*- coding: utf-8 -*-
r"""
Finding Reaction Paths with eOn and a Metatomic Potential
=========================================================

:Authors: Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_,
          Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`_,
          Arslan Mazitov `@abmazitov <https://github.com/abmazitov>`_,
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example describes how to find the reaction pathway for oxadiazole
formation from N₂O and ethylene. We will use the **PET-MAD** `metatomic
model <https://docs.metatensor.org/metatomic/latest/overview.html>`__ to
calculate the potential energy and forces.

The primary goal is to contrast a standard Nudged Elastic Band (NEB) calculation
using the `atomic simulation environment (ASE)
<https://databases.fysik.dtu.dk/ase/>`__ with more sophisticated methods
available in the `eOn package <https://eondocs.org/>`__. For even a relatively
simple reaction like this, a basic NEB implementation can struggle to converge
or may time out. We will show how eOn's advanced features, such as
**energy-weighted springs** and mixing in **single-ended dimer search steps**,
can efficiently locate and refine the transition state along the path.

Our approach will be:

1. Set up the **PET-MAD metatomic calculator**.
2. Use ASE to generate an initial IDPP reaction path.
3. Illustrate the limitations of a standard NEB calculation in ASE.
4. Refine the path and locate the transition state saddle point using
   eOn's optimizers, including energy-weighted springs and the dimer
   method.
5. Visualize the final converged pathway.
6. Demonstrate endpoint relaxation with eOn


Importing Required Packages
---------------------------
First, we import all the necessary python packages for this task.
By convention, all ``import``
statements are at the top of the file.
"""

import contextlib
import os
import subprocess
import sys
from pathlib import Path

import ase
import ase.io as aseio
import ira_mod
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ase.mep import NEB
from ase.optimize import LBFGS
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from metatomic.torch.ase_calculator import MetatomicCalculator
from rgpycrumbs.eon.helpers import write_eon_config
from rgpycrumbs.run.jupyter import run_command_or_exit


# sphinx_gallery_thumbnail_number = 4


# %%
# Obtaining the Foundation Model - PET-MAD
# ----------------------------------------
#
# ``PET-MAD`` is an instance of a point edge transformer model trained on
# the diverse `MAD dataset <https://arxiv.org/abs/2506.19674>`__
# which learns equivariance through data driven measures
# instead of having equivariance baked in [1]. In turn, this enables
# the PET model to have greater design space to learn over. Integration in
# Python and the C++ eOn client occurs through the ``metatomic`` software [2],
# which in turn relies on the atomistic machine learning toolkit build
# over ``metatensor``. Essentially using any of the metatomic models involves
# grabbing weights off of HuggingFace and loading them with
# ``metatomic`` before running the
# `engine of choice <https://docs.metatensor.org/metatomic/latest/engines/index.html>`_.
#

repo_id = "lab-cosmo/upet"
tag = "v1.1.0"
url_path = f"models/pet-mad-s-{tag}.ckpt"
fname = Path(url_path.replace(".ckpt", ".pt"))
url = f"https://huggingface.co/{repo_id}/resolve/main/{url_path}"
fname.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "mtt",
        "export",
        url,
        "-o",
        fname,
    ],
    check=True,
)
print(f"Successfully exported {fname}.")


# %%
# Nudged Elastic Band (NEB)
# -------------------------
#
# Given two known configurations on a potential energy surface (PES),
# often one wishes to determine the path of highest probability between
# the two. Under the harmonic approximation to transition state theory,
# connecting the configurations (each point representing a full molecular
# structure) by a discrete set of images allows one to evolve the path
# under an optimization algorithm, and allows approximating the reaction to
# three states: the reactant, product, and transition state.
#
# The location of this transition state (≈ the point with the highest energy
# along this path) determines the barrier height of the reaction. This saddle
# point can be found by transforming the second derivatives (Hessian) to step
# along the softest mode. However, an approximation which is free from
# explicitly finding this mode involves moving the highest image of a NEB path:
# the "climbing" image.
#
# Mathematically, the saddle point has zero first derivatives and a single
# negative eigenvalue. The climbing image technique moves the highest energy
# image along the reversed NEB tangent force, avoiding the cost of full
# Hessian diagonalization used in single-ended methods [3].
#
# Concretely, in this example, we will
# consider a reactant and product state, for oxadiazole
# formation, namely N₂O and ethylene.
#

reactant = aseio.read("data/min_reactant.con")
product = aseio.read("data/min_product.con")

# %%
# We can visualize these structures using ASE.

fig, (ax1, ax2) = plt.subplots(1, 2)
plot_atoms(reactant, ax1, rotation=("-90x,0y,0z"))
plot_atoms(product, ax2, rotation=("-90x,0y,0z"))
ax1.text(0.3, -1, "reactant")
ax2.text(0.3, -1, "product")
ax1.set_axis_off()
ax2.set_axis_off()

# %%
# Endpoint minimization
# ^^^^^^^^^^^^^^^^^^^^^
#
# For finding reaction pathways, the endpoints should be minimized. We provide
# initial configurations which are already minimized, but in order to see how to
# relax endpoints with eOn, please have a look at the end of this tutorial.


# %%
# Initial path generation
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# To begin an NEB method, an initial path is required, the optimal construction
# of which still forms an active area of research. The simplest approximation to
# an initial path for NEB methods linearly interpolate between the two known
# configurations building on intuition developed from "drag coordinate" methods.
# This may break bonds or otherwise also unphysically pass atoms through each
# other, similar to the effect of incorrect permutations. To ameliorate this
# effect, the NEB algorithm is often started from the linear interpolation but
# then the path is optimized on a surrogate potential energy surface, commonly
# something cheap and analytic, like the IDPP (Image dependent pair potential,
# [5]) which provides a surface based on bond distances, and thus preventing
# atom-in-atom collisions.
#
# Here we use the IDPP from ASE to setup the initial path. You can find
# more information about this method in the corresponding
# `ASE tutorial <https://ase-lib.org/examples_generated/tutorials/neb_idpp.html>`_
# or in the original IDPP publication by
# `S. Smidstru et al. <https://doi.org/10.1063/1.4878664>`_ .
# A brief pedagogical discussion of the transition state methods may be found on
# the `Rowan blog <https://rowansci.com/blog/guessing-transition-states>`_,
# though the software is proprietary there.

N_INTERMEDIATE_IMGS = 10
# total includes the endpoints
TOTAL_IMGS = N_INTERMEDIATE_IMGS + 2
images = [reactant]
images += [reactant.copy() for img in range(N_INTERMEDIATE_IMGS)]
images += [product]

neb = NEB(images)
neb.interpolate("idpp")

# %%
# We don't cover subtleties in setting the number of images, typically too many
# intermediate images may cause kinks but too few will be unable to resolve the
# tangent to any reasonable quality.
#
# For eOn, we write the initial path to a file called ``idppPath.dat``.
#

output_dir = "path"
os.makedirs(output_dir, exist_ok=True)

output_files = [f"{output_dir}/{num:02d}.con" for num in range(TOTAL_IMGS)]

for outfile, img in zip(output_files, images):
    ase.io.write(outfile, img)

print(f"Wrote {len(output_files)} IDPP images to '{output_dir}/'.")

summary_file_path = "idppPath.dat"

with open(summary_file_path, "w") as f:
    for filepath in output_files:
        abs_path = os.path.abspath(filepath)
        f.write(f"{abs_path}\n")

print(f"Wrote absolute paths to '{summary_file_path}'.")

# %%
# Running NEBs
# ------------
#
# We will now consider actually running the Nudged Elastic Band with different
# codes.
#
# ASE and Metatomic
# ^^^^^^^^^^^^^^^^^
#
# We first consider using metatomic with the ASE calculator.


# define the calculator
def mk_mta_calc():
    return MetatomicCalculator(
        fname,
        device="cpu",
        non_conservative=False,
        uncertainty_threshold=0.001,
    )


# set calculators for images
ipath = [reactant] + [reactant.copy() for img in range(10)] + [product]
for img in ipath:
    img.calc = mk_mta_calc()

print(img.calc._model.capabilities().outputs)

neb = NEB(ipath, climb=True, k=5, method="improvedtangent")
neb.interpolate("idpp")

# store initial path guess for plotting
initial_energies = [img.get_potential_energy() for img in ipath]

# setup the NEB clalculation
optimizer = LBFGS(neb, trajectory="A2B.traj", logfile="opt.log")
conv = optimizer.run(fmax=0.01, steps=100)

print("Check if calculation has converged:", conv)

if conv:
    print(neb)

final_energies = [i.get_potential_energy() for i in ipath]

# Plot initial and final path
plt.figure(figsize=(8, 6))
# Initial Path (Blue)
plt.plot(
    initial_energies - initial_energies[0],
    "o-",
    label="Initial Path (IDPP)",
    color="xkcd:blue",
)
# Final Path (Orange)
plt.plot(
    final_energies - initial_energies[0],
    "o-",
    label="Optimized Path (LBFGS)",
    color="xkcd:orange",
)
# Metadata
plt.xlabel("Image number")
plt.ylabel("Potential Energy (eV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("NEB Path Evolution")
plt.show()

# %%
# In the 100 NEB steps we took, the structure did unfortunately not converge.
# The metatomic calculator for PET-MAD v1.0.2 provides `LLPR based energy
# uncertainties <https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html>`_.
# As we obtain a warning that the uncertainty of the path structure sampled is
# very high, we stop after 100 steps.
# The ASE algorithm with LBFGS optimizer does not
# find good intermediate structures and does not converge
# at all. Our test showed that the FIRE optimizer works better in this context,
# but still takes over 500 steps to converge, and since second order methods are
# faster, we consider the LBFGS routine throughout this notebook.
#
# We thus want to
# look at a different code, which manages to compute a NEB for this simple
# system more efficiently.


# %%
# eOn and Metatomic
# ^^^^^^^^^^^^^^^^^
#
# `eOn <https://eondocs.org>`_ has two improvements to accurately locate the
# saddle point.
#
# 1. Energy weighting for improving tangent resolution
#    near the climbing image
# 2. The Off-path climbing image (6) which involves
#    iteratively switching to the dimer method for
#    faster convergence by the climbing image.
#
# To use eOn, we setup a function that writes the desired eOn input for us and
# runs the ``eonclient`` binary. Since we are in a notebook environment, we will
# use several abstractions over raw ``subprocess`` calls. In practice, writing
# and using eOn involves a configuration file, which we define as a dictionary
# to be used with a helper to generate the final output.

# Define configuration as a dictionary for clarity
neb_settings = {
    "Main": {"job": "nudged_elastic_band", "random_seed": 706253457},
    "Potential": {"potential": "Metatomic"},
    "Metatomic": {"model_path": fname.absolute()},
    "Nudged Elastic Band": {
        "images": N_INTERMEDIATE_IMGS,
        # initialization
        "initializer": "file",
        "initial_path_in": "idppPath.dat",
        "minimize_endpoints": "false",
        # CI-NEB settings
        "climbing_image_method": "true",
        "climbing_image_converged_only": "true",
        "ci_after": 0.5,
        "ci_after_rel": 0.8,
        # energy weighing
        "energy_weighted": "true",
        "ew_ksp_min": 0.972,
        "ew_ksp_max": 9.72,
        # OCI-NEB settings
        "ci_mmf": "true",
        "ci_mmf_after": 0.1,
        "ci_mmf_after_rel": 0.5,
        "ci_mmf_penalty_strength": 1.5,
        "ci_mmf_penalty_base": 0.4,
        "ci_mmf_angle": 0.9,
        "ci_mmf_nsteps": 1000,
    },
    "Optimizer": {
        "max_iterations": 1000,
        "opt_method": "lbfgs",
        "max_move": 0.1,
        "converged_force": 0.01,
    },
    "Debug": {"write_movies": "true"},
}


# %%
# Which now let's us write out the final triplet of reactant, product, and
# configuration of the eOn-NEB.

write_eon_config(Path("."), neb_settings)
aseio.write("reactant.con", reactant)
aseio.write("product.con", product)

# %%
# Run the main C++ client
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# This runs 'eonclient' and streams output live.
# If it fails, the notebook execution stops here.
run_command_or_exit(["eonclient"], capture=True, timeout=300)


# %%
# Visual interpretation
# ---------------------
#
# `rgpycrumbs <http://pypi.org/project/rgpycrumbs>`_ is a visualization toolkit
# designed to bridge the gap between raw computational output and physical
# intuition, mapping high-dimensional NEB trajectories onto interpretable 1D
# energy profiles and 2D RMSD landscapes.  As it is normally used from the
# command-line, here we define a helper.


def run_neb_plot(
    mode: str,
    con_file: str = "neb.con",
    output_file: str = "plot.png",
    title: str = "",
    rotation: str = "90x,0y,0z",
) -> list[str]:
    """
    Constructs the CLI command for rgpycrumbs plotting to avoid clutter in notebooks.
    mode: 'profile' (1D) or 'landscape' (2D)
    """
    base_cmd = [
        sys.executable,
        "-m",
        "rgpycrumbs.cli",
        "eon",
        "plt_neb",
        "--con-file",
        con_file,
        "--output-file",
        output_file,
        "--ase-rotation",
        rotation,
        "--facecolor",
        "white",
        "--figsize",
        "5.37",
        "5.37",
        "--plot-structures",
        "crit_points",
    ]

    if title:
        base_cmd.extend(["--title", title])

    if mode == "profile":
        base_cmd.extend(
            [
                "--plot-type",
                "profile",
                "--zoom-ratio",
                "0.15",
            ]
        )
    elif mode == "landscape":
        base_cmd.extend(
            [
                "--plot-type",
                "landscape",
                "--rc-mode",
                "path",
                "--fontsize-base",
                "16",
                "--landscape-mode",
                "surface",
                "--landscape-path",
                "all",
                "--show-pts",
                "--zoom-ratio",
                "0.35",
                "--surface-type",
                "grad_imq",
            ]
        )
    else:
        raise ValueError(f"Unknown plot mode: {mode}")

    # Run the generated command
    run_command_or_exit(base_cmd, capture=False, timeout=60)


# %%
#
# We check both the standard 1D profile against the path reaction
# coordinate, or the distance between intermediate images:

# Clean env to prevent backend conflicts in notebooks
os.environ.pop("MPLBACKEND", None)

# Run the 1D plotting command using the helper
run_neb_plot("profile", title="NEB Path Optimization", output_file="1D_oxad.png")

# Display the result
img = mpimg.imread("1D_oxad.png")
plt.figure(figsize=(5.37, 5.37))
plt.imshow(img)
plt.axis("off")
plt.show()


# %%
# Also, the PES 2D landscape profile as a function of the RMSD [3] which shows
# the relative distance between the endpoints as the optimization takes place:

# Run the 2D plotting command using the helper
run_neb_plot("landscape", title="NEB-RMSD Surface", output_file="2D_oxad.png")

# Display the result
img = mpimg.imread("2D_oxad.png")
plt.figure(figsize=(5.37, 5.37))
plt.imshow(img)
plt.axis("off")
plt.show()

# %%
# Where each black dot represents a configuration at which the energy and forces
# are calculated during the NEB optimization run, and the RMSD is the
# "permutation corrected" distance from the reactant and product. The
# interpolated relative energy surface is generated by a Gaussian approximation
# with a radial basis function kernel interpolation over the energy and forces
# of all the black dots. See [3, Chapter 4] for more details of this
# visualization.
#
# Relaxing the endpoints with eOn
# -------------------------------
#
# In this final part we come back to an essential
# point of performing NEB calculations, and that is the
# relaxation of the initial states. In the tutorials above
# we used directly relaxed structures, and here we are
# demonstrating how these can be relaxed.
# We first load structures which are not relaxed.

reactant = aseio.read("data/reactant.con")
product = aseio.read("data/product.con")


# For compatibility with eOn, we also need to provide
# a unit cell
def center_cell(atoms):
    atoms.set_cell([20, 20, 20])
    atoms.pbc = True
    atoms.center()
    return atoms


reactant = center_cell(reactant)
product = center_cell(product)

# %%
# The resulting reactant has a larger box:
#
fig, (ax1, ax2) = plt.subplots(1, 2)
plot_atoms(reactant, ax1, rotation=("-90x,0y,0z"))
plot_atoms(product, ax2, rotation=("-90x,0y,0z"))
ax1.text(0.3, -1, "reactant")
ax2.text(0.3, -1, "product")
ax1.set_axis_off()
ax2.set_axis_off()

# Reactant setup
dir_reactant = Path("min_reactant")
dir_reactant.mkdir(exist_ok=True)
aseio.write(dir_reactant / "pos.con", reactant)

# Product setup
dir_product = Path("min_product")
dir_product.mkdir(exist_ok=True)
aseio.write(dir_product / "pos.con", product)

# Shared minimization settings
min_settings = {
    "Main": {"job": "minimization", "random_seed": 706253457},
    "Potential": {"potential": "Metatomic"},
    "Metatomic": {"model_path": fname.absolute()},
    "Optimizer": {
        "max_iterations": 2000,
        "opt_method": "lbfgs",
        "max_move": 0.1,
        "converged_force": 0.01,
    },
}

write_eon_config(dir_reactant, min_settings)
write_eon_config(dir_product, min_settings)


# %%
# Run the minimization
# ^^^^^^^^^^^^^^^^^^^^
#
# The 'eonclient' will use the correct configuration within the folder.
#
@contextlib.contextmanager
def work_in_dir(path: Path):
    """
    Context manager to safely change directory and return to previous
    one afterwards. Crucial for notebooks to avoid path drift.
    """
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


with work_in_dir(dir_reactant):
    run_command_or_exit(["eonclient"], capture=True, timeout=300)


with work_in_dir(dir_product):
    run_command_or_exit(["eonclient"], capture=True, timeout=300)


# %%
# Additionally, the relative ordering must be preserved, for which we use
# IRA [4].
#
reactant = aseio.read(dir_reactant / "min.con")
product = aseio.read(dir_product / "min.con")

ira = ira_mod.IRA()
# Default value
kmax_factor = 1.8

nat1 = len(reactant)
typ1 = reactant.get_atomic_numbers()
coords1 = reactant.get_positions()

nat2 = len(product)
typ2 = product.get_atomic_numbers()
coords2 = product.get_positions()

print("Running ira.match to find rotation, translation, AND permutation...")
# r = rotation, t = translation, p = permutation, hd = Hausdorff distance
r, t, p, hd = ira.match(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)

print(f"Matching complete. Hausdorff Distance (hd) = {hd:.6f} Angstrom")

# Apply rotation (r) and translation (t) to the original product coordinates
# This aligns the product's orientation to the reactant's
coords2_aligned = (coords2 @ r.T) + t

# Apply the permutation (p)
# This re-orders the aligned product atoms to match the reactant's atom order
# p[i] = j means reactant atom 'i' matches product atom 'j'
# So, the new coordinate array's i-th element should be coords2_aligned[j]
coords2_aligned_permuted = coords2_aligned[p]

# Save the new aligned-and-permuted structure
# CRUCIAL: Use chemical symbols from the reactant,
# because we have now permuted the product coordinates to match the reactant order.
product = reactant.copy()
product.positions = coords2_aligned_permuted
# %%
# Finally we can visualize these with ASE.
#
view(reactant, viewer="x3d")
view(product, viewer="x3d")
fig, (ax1, ax2) = plt.subplots(1, 2)
plot_atoms(reactant, ax1, rotation=("-90x,0y,0z"))
plot_atoms(product, ax2, rotation=("-90x,0y,0z"))
ax1.text(0.3, -1, "reactant")
ax2.text(0.3, -1, "product")
ax1.set_axis_off()
ax2.set_axis_off()

# %%
# References
# ----------
#
# (1) Mazitov, A.; Bigi, F.; Kellner, M.; Pegolo, P.; Tisi, D.; Fraux, G.;
#     Pozdnyakov, S.; Loche, P.; Ceriotti, M. PET-MAD, a Universal
#     Interatomic Potential for Advanced Materials Modeling. arXiv March
#     18, 2025. https://doi.org/10.48550/arXiv.2503.14118.
#
# (2) Bigi, F.; Abbott, J. W.; Loche, P.; Mazitov, A.; Tisi, D.; Langer,
#     M. F.; Goscinski, A.; Pegolo, P.; Chong, S.; Goswami, R.; Chorna,
#     S.; Kellner, M.; Ceriotti, M.; Fraux, G. Metatensor and Metatomic:
#     Foundational Libraries for Interoperable Atomistic Machine Learning.
#     arXiv August 21, 2025. https://doi.org/10.48550/arXiv.2508.15704.
#
# (3) Goswami, R. Efficient Exploration of Chemical Kinetics. arXiv
#     October 24, 2025. https://doi.org/10.48550/arXiv.2510.21368.
#
# (4) Gunde, M.; Salles, N.; Hémeryck, A.; Martin-Samos, L. IRA: A Shape
#     Matching Approach for Recognition and Comparison of Generic Atomic
#     Patterns. J. Chem. Inf. Model. 2021, 61 (11), 5446–5457.
#     https://doi.org/10.1021/acs.jcim.1c00567.
#
# (5) Smidstrup, S.; Pedersen, A.; Stokbro, K.; Jónsson, H. Improved
#     Initial Guess for Minimum Energy Path Calculations. J. Chem. Phys.
#     2014, 140 (21), 214106. https://doi.org/10.1063/1.4878664.
#
# (6) Goswami, R; Gunde, M; Jónsson, H. Enhanced climbing image nudged elastic
#     band method with hessian eigenmode alignment, Jan. 22, 2026, arXiv:
#     arXiv:2601.12630. doi: 10.48550/arXiv.2601.12630.
#
