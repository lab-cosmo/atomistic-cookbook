# -*- coding: utf-8 -*-
r"""
Finding Reaction Paths with EON and a Metatomic Potential
=========================================================

:Authors: Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_,
          Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`_,
          Arslan Mazitov `@abmazitov <https://github.com/abmazitov>`_,
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example describes how to find the reaction pathway for oxadiazole
formation from N₂O and ethylene. We will use the **PET-MAD** `metatomic
model <https://docs.metatensor.org/metatomic/latest/overview.html>`__ to
calculate the potential energy and forces.

The primary goal is to contrast a standard Nudged Elastic Band (NEB)
calculation using the `atomic simulation environment
(ASE) <https://databases.fysik.dtu.dk/ase/>`__ with more
sophisticated methods available in the `EON
package <https://theochemui.github.io/eOn/>`__. For even a relatively simple
reaction like this, a basic NEB implementation can struggle to converge
or may time out. We will show how EON's advanced features, such as
**energy-weighted springs** and mixing in **single-ended dimer search steps**, can
efficiently locate and refine the transition state along the path.

Our approach will be:

1. Set up the **PET-MAD metatomic calculator**.
2. Use ASE to generate an initial IDPP reaction path.
3. Illustrate the limitations of a standard NEB calculation in ASE.
4. Refine the path and locate the transition state saddle point using
   EON's optimizers, including energy-weighted springs and the dimer
   method.
5. Visualize the final converged pathway.
6. Demonstrate endpoind relaxation with EON


Importing Required Packages
---------------------------
First, we import all the necessary python packages for this task.
By convention, all ``import``
statements are at the top of the file.
"""

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union

import ase
import ase.io as aseio
import ira_mod
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import requests
from ase.mep import NEB
from ase.optimize import LBFGS
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from metatomic.torch.ase_calculator import MetatomicCalculator


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
fname = Path(url_path).name
url = f"https://huggingface.co/{repo_id}/resolve/main/{url_path}"
export_name = fname.replace(".ckpt", ".pt")

if not Path(export_name).exists():
    subprocess.run(
        ["uvx", "--from", "metatrain", "mtt", "export", url], check=True
    )
    print(f"Successfully exported {fname} to {export_name}.")
else:
    print(f"Exported file {export_name} already exists.")


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
# relax endpoints with EON, please have a look at the end of this tutorial.


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
# For EON, we write the initial path to a file called ``idppPath.dat``.
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
        "pet-mad-v1.0.2.pt",
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
# EON and Metatomic
# ^^^^^^^^^^^^^^^^^
#
# `EON <https://eondocs.org>`_ has two improvements to accurately locate the
# saddle point.
#
# 1. Energy weighting for improving tangent resolution
#    near the climbing image
# 2. The Hybrid MMF-NEB-CI which involves
#    iteratively switching to the dimer method for faster convergence by the
#    climbing image.
#
# To use EON, we setup a function that writes the desired EON input for us and
# runs the ``eonclient`` binary. Since we are in a notebook environment, we will
# use several abstractions over raw ``subprocess`` calls. In practice, writing
# and using EON is much simpler.


def write_neb_eon_config(
    run_dir: Path, model_path: Path, ninterm: int = N_INTERMEDIATE_IMGS
):
    """
    Writes the config.ini file for an EON NEB job,
    using the user's provided template.
    """
    config_content = f"""[Main]
job=nudged_elastic_band
random_seed = 706253457

[Potential]
potential = Metatomic

[Metatomic]
model_path = {Path(str(model_path).replace("ckpt", "pt")).absolute()}

[Nudged Elastic Band]
images={ninterm}
energy_weighted=true
ew_ksp_min = 0.972
ew_ksp_max = 9.72
initial_path_in = idppPath.dat
minimize_endpoints = false
climbing_image_method = true
climbing_image_converged_only = true
ci_after = 1.5
ci_mmf = true
ci_mmf_after = 0.5
ci_mmf_nsteps = 20

# for PET-MAD 1.1
# ci_after = 0.5
# ci_mmf_nsteps = 10

[Optimizer]
max_iterations = 100
opt_method = lbfgs
max_move = 0.1
converged_force = 0.01

[Debug]
write_movies=true
"""
    config_path = run_dir / "config.ini"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Wrote EON NEB config to '{config_path}'")


# %%
# Which now let's us write out the final triplet of reactant, product, and
# configuration of the EON-NEB.

write_neb_eon_config(Path("."), Path(fname).absolute())
aseio.write("reactant.con", reactant)
aseio.write("product.con", product)

# %%
# Now we can finally define helpers for easier visualization.


def _run_command_live(
    cmd: Union[str, List[str]],
    *,
    check: bool = True,
    timeout: Optional[float] = None,
    capture: bool = False,
    encoding: str = "utf-8",
) -> subprocess.CompletedProcess:
    """
    Internal: run command and stream stdout/stderr live to current stdout.
    If capture=True, also collect combined output and return it
    in CompletedProcess.stdout.
    """
    shell = isinstance(cmd, str)
    cmd_str = cmd if shell else cmd[0]

    # If list form, ensure program exists before trying to run
    if not shell and shutil.which(cmd_str) is None:
        raise FileNotFoundError(f"{cmd_str!r} is not on PATH")

    # Start the process
    # We combine stderr into stdout so we only have one stream to read
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=encoding,
        shell=shell,
        bufsize=1,  # Line buffered
    )

    collected = [] if capture else None

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # Stream into notebook or terminal live
            print(line, end="")
            sys.stdout.flush()
            if capture:
                collected.append(line)

        # Wait for the process to actually exit after stream closes
        returncode = proc.wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()

    if check and returncode != 0:
        output_str = "".join(collected) if capture else ""
        raise subprocess.CalledProcessError(returncode, cmd, output=output_str)

    return subprocess.CompletedProcess(
        cmd, returncode, stdout="".join(collected) if capture else None
    )


def run_command_or_exit(
    cmd: Union[str, List[str]], capture: bool = False, timeout: Optional[float] = 300
) -> subprocess.CompletedProcess:
    """
    Helper wrapper to run commands, stream output, and exit script/notebook
    cleanly on failure so sphinx-gallery sees the errors appropriately.
    """
    try:
        return _run_command_live(cmd, check=True, capture=capture, timeout=timeout)
    except FileNotFoundError as e:
        print(f"Executable not found: {e}", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except subprocess.TimeoutExpired:
        print("Command timed out", file=sys.stderr)
        sys.exit(124)


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
# energy profiles and 2D RMSD landscapes.
#
# We check both the standard 1D profile against the path reaction
# coordinate, or the distance between intermediate images:

# Clean env to prevent backend conflicts in notebooks
os.environ.pop("MPLBACKEND", None)

oneDprof_oxad = [
    sys.executable,
    "-m",
    "rgpycrumbs.cli",
    "eon",
    "plt_neb",
    "--theme",
    "ruhi",
    "--con-file",
    "neb.con",
    "--plot-structures",
    "crit_points",
    "--facecolor",
    "white",
    "--plot-type",
    "profile",
    "--ase-rotation=90x,0y,0z",
    "--title=NEB Path Optimization",
    "--output-file",
    "1D_oxad.png",
]

# Run the 1D plotting command using the helper
run_command_or_exit(oneDprof_oxad, capture=False, timeout=60)

# Display the result
img = mpimg.imread("1D_oxad.png")
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.show()


# %%
# Also, the PES 2D landscape profile as a function of the RMSD [3] which shows
# the relative distance between the endpoints as the optimization takes place:

twoDprof_oxad = [
    sys.executable,
    "-m",
    "rgpycrumbs.cli",
    "eon",
    "plt_neb",
    "--theme",
    "ruhi",
    "--con-file",
    "neb.con",
    "--plot-structures",
    "crit_points",
    "--facecolor",
    "white",
    "--rc-mode",
    "path",
    "--ase-rotation=90x,0y,0z",
    "--title",
    "NEB-RMSD Surface",
    "--fontsize-base",
    "16",
    "--landscape-mode",
    "surface",
    "--landscape-path",
    "all",
    "--plot-type",
    "landscape",
    "--show-pts",
    "--surface-type",
    "rbf",
    "--output-file",
    "2D_oxad.png",
]

# Run the 2D plotting command using the helper
run_command_or_exit(twoDprof_oxad, capture=False, timeout=60)

# Display the result
img = mpimg.imread("2D_oxad.png")
plt.figure(figsize=(8, 6))
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
# Relaxing the endpoints with EON
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


# For compatibility with EON, we also need to provide
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


def write_eon_min_config(run_dir: Path, model_path: Path):
    """
    Writes the config.ini file for an EON NEB job,
    using the user's provided template.
    """
    config_content = f"""[Main]
job=minimization
random_seed = 706253457

[Potential]
potential = Metatomic

[Metatomic]
model_path = {Path(str(model_path).replace("ckpt", "pt")).absolute()}

[Optimizer]
max_iterations = 2000
opt_method = lbfgs
max_move = 0.1
converged_force = 0.01
"""
    config_path = run_dir / "config.ini"
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Wrote EON NEB config to '{config_path}'")


# Reactant setup
dir_reactant = Path("min_reactant")
dir_reactant.mkdir(exist_ok=True)

aseio.write(dir_reactant / "pos.con", reactant)
write_eon_min_config(dir_reactant, Path(fname).absolute())

# Product setup
dir_product = Path("min_product")
dir_product.mkdir(exist_ok=True)

aseio.write(dir_product / "pos.con", product)
write_eon_min_config(dir_product, Path(fname).absolute())


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
