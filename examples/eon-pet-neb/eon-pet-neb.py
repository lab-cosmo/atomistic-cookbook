"""
Reaction Path Finding with EON and a Metatomic Potential
========================================================

:Authors: Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`__;
Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`__; Guillaume
Fraux `@Luthaf <https://github.com/luthaf/>`__; Arslan Mazitov
`@abmazitov <https://github.com/abmazitov>`__; Michele Ceriotti
`@ceriottim <https://github.com/ceriottim/>`__

This example describes how to find the reaction pathway for oxadiazole
formation from N₂O and ethylene. We will use the **PET-MAD** `metatomic
model <https://docs.metatensor.org/metatomic/latest/overview.html>`__ to
calculate the potential energy and forces.

The primary goal is to contrast a standard Nudged Elastic Band (NEB)
calculation using the `atomic simulation environment
(ASE) <https://databases.fysik.dtu.dk/ase/>`__ with more
sophisticated methods available in the `EON
package <https://theochemui.github.io/eOn/>`__. For a complex
reaction like this, a basic NEB implementation can struggle to converge
or may time out. We will show how EON's advanced features, such as
**energy-weighted springs** and mixing in **single-ended dimer search steps**, can
efficiently locate and refine the transition state along the path.

Our approach will be:

1. Set up the **PET-MAD metatomic calculator**.
2. Illustrate the limitations of a standard NEB calculation in ASE.
3. Use ASE to generate an initial IDPP reaction path.
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

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import ase
import ase.io as aseio
import ira_mod
import matplotlib.pyplot as plt
import requests
from ase.mep import NEB
from ase.visualize.plot import plot_atoms
from IPython.display import Image, display
from metatomic.torch.ase_calculator import MetatomicCalculator


# %%
# Obtaining the Foundation Model - PET-MAD
# ----------------------------------------
#
# ``PET-MAD`` is an instance of a point edge transformer model trained on
# the diverse `MAD dataset <https://arxiv.org/abs/2506.19674>`__
# which learns equivariance through data driven measures 
# instead of having equivariance baked in. In turn, this enables
# the PET model to have greater design space to learn over. Integration in
# Python and the C++ EON client occurs through the ``metatomic`` software,
# which in turn relies on the atomistic machine learning toolkit build
# over ``metatensor``. Essentially using any of the metatomic models involves
# grabbing weights off of HuggingFace and loading them with
# ``metatomic`` before running the `engine of choice <https://docs.metatensor.org/metatomic/latest/engines/index.html>`_.
#

repo_id = "lab-cosmo/pet-mad"
tag = "v1.0.2"
url_path = f"models/pet-mad-{tag}.ckpt"
fname = Path(url_path).name
url = f"https://huggingface.co/{repo_id}/resolve/{tag}/{url_path}"

if not Path(fname).exists():
    response = requests.get(url)
    if response.ok:
        with open(fname, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {fname} from tag {tag}.")
    else:
        print("Failed to download:", response.status_code)
else:
    print(f"{fname} from tag {tag} already present.")

if not Path(fname.replace("ckpt", "pt")).exists():
    subprocess.run(
        [
            "mtt",
            "export",
            "pet-mad-v1.0.2.ckpt",  # noqa: E501
        ]
    )


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
# three states, the reactant, product, and transition state.
# The location of this transition state (≈ the point with the highest
# energy along this path), which is relevant to determine the barrier
# height of the relevant reaction, and can be found
# by applying a transformation to the second
# derivatives, to enable optimization algorithms to step along the
# softest mode to reach a saddle configuration. An approximation which is free
# from finding the actual mode involves moving the highest image of a NEB path,
# the "climbing" image along the reversed NEB tangent force.  Mathematically,
# this is the point with zero first derivatives and a single negative
# eigenvalue, and single ended methods [2] approximate the Hessian and
# softest-mode with different methods.
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
# ~~~~~~~~~~~~~~~~~~~~~
# 
# For finding reaction pathways, the endpoints should be minimized.  We provided
# initial configurations which are already minimized, but in order to see how to
# relax endpoints with EON, please have a look at the end of this tutorial.


# %%
# Initial path generation
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The earliest NEB methods linearly interpolate between the two known
# configurations since these methods build on "drag coordinate" methods. This
# may break bonds or otherwise also unphysically pass atoms through each other,
# similar to the effect of incorrect permutations. To ameliorate this effect,
# the NEB algorithm is often started from the linear interpolation but then the
# path is optimized on a surrogate potential energy surface, commonly something
# cheap and analytic, like the IDPP (Image dependent pair potential, [6]) which
# provides a surface based on bond distances, and thus preventing atom-in-atom
# collisions.
#
# Here we use the IDPP from ASE to setup the initial path. You can find
# more information about this method in the corresponding
# `ASE tutorial <https://ase-lib.org/examples_generated/tutorials/neb_idpp.html>`_
# or in the original IDPP publication by
# `S. Smidstru et al. <https://doi.org/10.1063/1.4878664>`_ .

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


# %%
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
# ASE
# ---
#

import ase.io as aseio
from ase.mep import NEB
from ase.optimize import FIRE, LBFGS
from ase.visualize import view
from metatomic.torch.ase_calculator import MetatomicCalculator


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

# plot the initial energy path guess
energies = [i.get_potential_energy() for i in ipath]
plt.plot(energies)

# setup the NEB clalculation
optimizer = LBFGS(neb, trajectory="A2B.traj", logfile="opt.log")
conv = optimizer.run(fmax=0.01, steps=100)

print("Check if calculation has converged:", conv)

if conv:
    print(neb)

energies = [i.get_potential_energy() for i in ipath]
plt.plot(energies)
plt.show()

# %%
# In the 100 NEB steps we took, the structure did unfortunately not converge.
# The metatomic calculator for PET-MAD v1.0.2 provides `LLPR based energy
# uncertainities <https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html>`_.
# As we obtain a warning that the uncertainty of the path structure sampled is
# very high, we stop after 100 steps.
# The ASE algorithm with LBFGS optimizer does not
# find good intermediate structures and does not converge
# at all. Our test showed that the FIRE
# optimizer works better in this context, but still takes over 500 steps
# to converge.
# We thus want to look at a different code, which manages to compute a
# NEB for this simple system more efficiently.


## %%
## EON and Metatomic
## -----------------
##
## EON has two implementations to accurately locate the saddle point.
##
## 1. Energy weighting for improving tangent resolution
##    near the climbing image
## 2. The Hybrid MMF-NEB-CI which involves
##    iteratively switching to the dimer method for faster convergence by the
##    climbing image.
##
## To use EON, we setup a function that writes us the desired EON input and
## run it.
#
#
# def write_neb_eon_config(
#    run_dir: Path, model_path: Path, ninterm: int = N_INTERMEDIATE_IMGS
# ):
#    """
#    Writes the config.ini file for an EON NEB job,
#    using the user's provided template.
#    """
#    config_content = f"""[Main]
# job=nudged_elastic_band
# random_seed = 706253457
#
# [Potential]
# potential = Metatomic
#
# [Metatomic]
# model_path = {Path(str(model_path).replace("ckpt", "pt")).absolute()}
#
# [Nudged Elastic Band]
# images={ninterm}
# energy_weighted=true
# ew_ksp_min = 0.972
# ew_ksp_max = 9.72
# initial_path_in = idppPath.dat
# minimize_endpoints = true
# climbing_image_method = true
# climbing_image_converged_only = true
# ci_after = 0.5
# ci_mmf = true
# ci_mmf_after = 0.5
# ci_mmf_nsteps = 10
#
# [Optimizer]
# max_iterations = 2000
# opt_method = lbfgs
# max_move = 0.1
# converged_force = 0.01
#
# [Debug]
# write_movies=true
# """
#    config_path = run_dir / "config.ini"
#    with open(config_path, "w") as f:
#        f.write(config_content)
#    print(f"Wrote EON NEB config to '{config_path}'")
#
#
## %%
## Which now let’s us write out the final triplet of reactant, product, and
## configuration of the EON-NEB.
##
#
# write_neb_eon_config(Path("."), Path(fname).absolute())
# aseio.write("reactant.con", reactant)
# aseio.write("product.con", product)
#
#
## %%
## Now we can finally define helpers for easier visualization.
##
#
#
# def _run_command_live(
#    cmd,
#    *,
#    check: bool = True,
#    timeout: Optional[float] = None,
#    capture: bool = False,
#    encoding: str = "utf-8",
# ) -> subprocess.CompletedProcess:
#    """
#    Internal: run command and stream stdout/stderr live to current stdout.
#    If capture=True, also collect combined output and return it
#    in CompletedProcess.stdout.
#    """
#    shell = isinstance(cmd, str)
#
#    # If list form, ensure program exists
#    if not shell and shutil.which(cmd[0]) is None:
#        raise FileNotFoundError(f"{cmd[0]!r} is not on PATH")
#
#    proc = subprocess.Popen(
#        cmd,
#        stdout=subprocess.PIPE,
#        stderr=subprocess.STDOUT,
#        text=True,
#        encoding=encoding,
#        shell=shell,
#    )
#
#    collected = [] if capture else None
#
#    try:
#        assert proc.stdout is not None
#        for line in proc.stdout:
#            # Stream into notebook or terminal live
#            print(line, end="")
#            sys.stdout.flush()
#            if capture:
#                collected.append(line)
#        returncode = proc.wait(timeout=timeout)
#    except subprocess.TimeoutExpired:
#        proc.kill()
#        proc.wait()
#        raise
#    except KeyboardInterrupt:
#        proc.kill()
#        proc.wait()
#        raise
#    finally:
#        if proc.stdout:
#            proc.stdout.close()
#
#    if check and returncode != 0:
#        if capture:
#            raise subprocess.CalledProcessError(
#                returncode, cmd, output="".join(collected)
#            )
#        else:
#            raise subprocess.CalledProcessError(returncode, cmd)
#
#    if capture:
#        return subprocess.CompletedProcess(cmd, returncode, stdout="".join(collected))
#    else:
#        return subprocess.CompletedProcess(cmd, returncode, stdout=None)
#
#
## %%
## While fairly verbose, some helpers are a good investment here.
##
#
#
# def run_eonclient_or_exit(
#    capture: bool = False, timeout: Optional[float] = 300
# ) -> subprocess.CompletedProcess:
#    """
#    One-line call for scripts/examples. Streams live.
#    Exits with non-zero code on failure
#    so sphinx-gallery sees errors.
#    """
#    try:
#        return _run_command_live(
#            ["eonclient"], check=True, capture=capture, timeout=timeout
#        )
#    except FileNotFoundError as e:
#        print("Executable not found:", e, file=sys.stderr)
#        sys.exit(2)
#    except subprocess.CalledProcessError as e:
#        # if output captured, e.output contains the combined output
#        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
#        sys.exit(e.returncode)
#    except subprocess.TimeoutExpired:
#        print("Command timed out", file=sys.stderr)
#        sys.exit(124)
#
#
## %%
## While fairly verbose, the helper is a good investment:
##
#
# run_eonclient_or_exit(capture=True, timeout=300)
#
#
## %%
## Visual interpretation
## ~~~~~~~~~~~~~~~~~~~~~
##
## We check both the standard 1D profile against the path reaction
## coordinate, or the distance between intermediate images:
##
#
# os.environ.pop("MPLBACKEND", None)
#
# oneDprof_oxad = [
#    sys.executable,
#    "-m",
#    "rgpycrumbs.cli",
#    "eon",
#    "plt_neb",
#    "--con-file",
#    "neb.con",
#    "--plot-structures",
#    "crit_points",
#    "--facecolor",
#    "floralwhite",
#    "--plot-type",
#    "profile",
#    "--ase-rotation=90x,0y,0z",
#    "--title=NEB Path Optimization",
#    "--output-file",
#    "1D_oxad.png",
# ]
#
# result = subprocess.run(oneDprof_oxad, capture_output=True, text=True)
# print(result.stdout)
# if result.stderr:
#    print(result.stderr)
#
# display(Image("1D_oxad.png"))
#
#
## %%
## Also, the slice of PES 2D landscape profile () which shows the relative
## distace between edpoints as the optimization takes place:
##
#
## TODO(rg): this needs IRA within the environment which is not pip installable..
## os.environ.pop("MPLBACKEND", None)
#
## twoDprof_oxad = [
##     sys.executable,
##     "-m",
##     "rgpycrumbs.cli",
##     "eon",
##     "plt_neb",
##     "--con-file",
##     "neb.con",
##     "--theme",
##     "ruhi",
##     "--plot-structures",
##     "crit_points",
##     "--facecolor",
##     "floralwhite",
##     "--plot-type",
##     "landscape",
##     "--landscape-mode",
##     "surface",
##     "--ase-rotation=90x,0y,0z",
##     "--title=NEB Path Optimization",
##     "--output-file",
##     "2D_oxad.png",
## ]
#
## result = subprocess.run(oneDprof_oxad, capture_output=True, text=True)
## print(result.stdout)
## if result.stderr:
##     print(result.stderr)
#
# display(Image("2D_oxad.png"))
#
## %%
## Relaxing the endpoints with EON
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
## In this final part we come back to an essential
## point of performing NEB calculations, and that is the
## relaxation of the initial states. In the tutorials above
## we used directly relaxed structures, and here we are
## demonstrating how these can be relaxed.
## We first load structures which are not relaxed.
#
# reactant = aseio.read("data/reactant.con")
# product = aseio.read("data/product.con")
#
## For compatibility with EON, we also need to provide
## a unit cell
#
# def center_cell(atoms):
#    atoms.set_cell([20, 20, 20])
#    atoms.pbc = True
#    atoms.center()
#    return atoms
#
#
# reactant = center_cell(reactant)
# product = center_cell(product)
#
## %%
## The resulting reactant has a larger box:
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# plot_atoms(reactant, ax1, rotation=("-90x,0y,0z"))
# plot_atoms(product, ax2, rotation=("-90x,0y,0z"))
# ax1.text(0.3, -1, "reactant")
# ax2.text(0.3, -1, "product")
# ax1.set_axis_off()
# ax2.set_axis_off()
#
#
# def write_eon_min_config(run_dir: Path, model_path: Path):
#    """
#    Writes the config.ini file for an EON NEB job,
#    using the user's provided template.
#    """
#    config_content = f"""[Main]
# job=minimization
# random_seed = 706253457
#
# [Potential]
# potential = Metatomic
#
# [Metatomic]
# model_path = {Path(str(model_path).replace("ckpt", "pt")).absolute()}
#
# [Optimizer]
# max_iterations = 2000
# opt_method = lbfgs
# max_move = 0.1
# converged_force = 0.01
# """
#    config_path = run_dir / "config.ini"
#    with open(config_path, "w") as f:
#        f.write(config_content)
#    print(f"Wrote EON NEB config to '{config_path}'")
#
#
# Path("min_reactant").mkdir(exist_ok=True)
# aseio.write("min_reactant/pos.con", reactant)
# write_eon_min_config(Path("min_reactant"), Path(fname).absolute())
#
# Path("min_product").mkdir(exist_ok=True)
# aseio.write("min_product/pos.con", product)
# write_eon_min_config(Path("min_product"), Path(fname).absolute())
#
## read the minimized end points
# reactant = aseio.read("min_reactant/pos.con")
# product = aseio.read("min_product/pos.con")
#
#
## %%
## Additionally, the relative ordering must be preserved, for which we use
## IRA [3].
##
#
# ira = ira_mod.IRA()
## Default value
# kmax_factor = 1.8
#
# nat1 = len(reactant)
# typ1 = reactant.get_atomic_numbers()
# coords1 = reactant.get_positions()
#
# nat2 = len(product)
# typ2 = product.get_atomic_numbers()
# coords2 = product.get_positions()
#
# print("Running ira.match to find rotation, translation, AND permutation...")
## r = rotation, t = translation, p = permutation, hd = Hausdorff distance
# r, t, p, hd = ira.match(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)
#
# print(f"Matching complete. Hausdorff Distance (hd) = {hd:.6f} Angstrom")
#
## 1. Apply rotation (r) and translation (t) to the original product coordinates
## This aligns the product's orientation to the reactant's
# coords2_aligned = (coords2 @ r.T) + t
#
## 2. Apply the permutation (p)
## This re-orders the aligned product atoms to match the reactant's atom order
## p[i] = j means reactant atom 'i' matches product atom 'j'
## So, the new coordinate array's i-th element should be coords2_aligned[j]
# coords2_aligned_permuted = coords2_aligned[p]
#
## --- 5. Save the new aligned-and-permuted structure ---
## CRUCIAL: Use chemical symbols from the reactant,
## because we have now permuted the product coordinates to match the reactant order.
# product = reactant.copy()
# product.positions = coords2_aligned_permuted
#
#
## %%
## Finally we can visualize these with ``chemiscope``.
##
#
# settings = {
#    "structure": [
#        {"playbackDelay": 50, "unitCell": True, "bonds": True, "spaceFilling": True}
#    ]
# }
# chemiscope.show([reactant, product], mode="structure", settings=settings)
#
#
## %%
## or with ASE.
##
#
## view(reactant, viewer='x3d')
## view(product, viewer='x3d')
# fig, (ax1, ax2) = plt.subplots(1, 2)
# plot_atoms(reactant, ax1, rotation=("-90x,0y,0z"))
# plot_atoms(product, ax2, rotation=("-90x,0y,0z"))
# ax1.text(0.3, -1, "reactant")
# ax2.text(0.3, -1, "product")
# ax1.set_axis_off()
# ax2.set_axis_off()
#
#
#
## %%
## References
## ==========
##
## (1) Bigi, F.; Abbott, J. W.; Loche, P.; Mazitov, A.; Tisi, D.; Langer,
##     M. F.; Goscinski, A.; Pegolo, P.; Chong, S.; Goswami, R.; Chorna,
##     S.; Kellner, M.; Ceriotti, M.; Fraux, G. Metatensor and Metatomic:
##     Foundational Libraries for Interoperable Atomistic Machine Learning.
##     arXiv August 21, 2025. https://doi.org/10.48550/arXiv.2508.15704.
##
## (2) Goswami, R. Efficient Exploration of Chemical Kinetics. arXiv
##     October 24, 2025. https://doi.org/10.48550/arXiv.2510.21368.
##
## (3) Mazitov, A.; Bigi, F.; Kellner, M.; Pegolo, P.; Tisi, D.; Fraux, G.;
##     Pozdnyakov, S.; Loche, P.; Ceriotti, M. PET-MAD, a Universal
##     Interatomic Potential for Advanced Materials Modeling. arXiv March
##     18, 2025. https://doi.org/10.48550/arXiv.2503.14118.
##
## (4) Gunde, M.; Salles, N.; Hémeryck, A.; Martin-Samos, L. IRA: A Shape
##     Matching Approach for Recognition and Comparison of Generic Atomic
##     Patterns. J. Chem. Inf. Model. 2021, 61 (11), 5446–5457.
##     https://doi.org/10.1021/acs.jcim.1c00567.
##
## (5) Fraux, G.; Cersonsky, R.; Ceriotti, M. Chemiscope: Interactive
##     Structure-Property Explorer for Materials and Molecules. J. Open
##     Source Softw. 2020, 5 (51), 2117.
##     https://doi.org/10.21105/joss.02117.
##
## (6) Smidstrup, S.; Pedersen, A.; Stokbro, K.; Jónsson, H. Improved
##     Initial Guess for Minimum Energy Path Calculations. J. Chem. Phys.
##     2014, 140 (21), 214106. https://doi.org/10.1063/1.4878664.
##
