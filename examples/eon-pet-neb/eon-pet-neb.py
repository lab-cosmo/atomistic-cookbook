# -*- coding: utf-8 -*-
r"""
Finding Reaction Paths with eOn and a Metatomic Potential
=========================================================

:Authors: Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_,
          Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`_,
          Arslan Mazitov `@abmazitov <https://github.com/abmazitov>`_,
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example finds a reaction path for oxadiazole formation from N₂O and
ethylene with the **PET-MAD** `metatomic model
<https://docs.metatensor.org/metatomic/latest/overview.html>`__. Energies and
forces come from that model under two drivers: a climbing-image NEB in the
`atomic simulation environment (ASE)
<https://databases.fysik.dtu.dk/ase/>`__, and an energy-weighted NEB with
optional off-path climbing steps (OCI / MMF) in `eOn
<https://eondocs.org/>`__ via ``pyeonclient``.

Outline:

1. Export PET-MAD and load it for ASE and for eOn
   (``make_backend("rgpot_metatomic", model_path=…)``).
2. Build an IDPP guess and run a short ASE climbing-image NEB.
3. Run eOn ``NudgedElasticBand`` with energy-weighted springs and MMF
   (``NebSpec``), then plot the path.
4. Relax the endpoints with the same potential and check ordering with IRA.


Importing Required Packages
---------------------------
First, we import all the necessary python packages for this task.
By convention, all ``import``
statements are at the top of the file.
"""

import os
from contextlib import chdir
from pathlib import Path

import ase.io as aseio
import ira_mod
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pyeonclient as pyec
import readcon
from ase.mep import NEB
from ase.optimize import LBFGS
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from atomistic_cookbook_utils import run_command
from pyeonclient.backends import make_backend, make_metatomic_ase_calculator
from pyeonclient.models import NebSpec, PathInit

# Library plot_* uses ensure_import (jax, adjustText, …). CLI dispatch sets this
# automatically; gallery imports need it so uv can stage heavies into the cache.
os.environ.setdefault("RGPYCRUMBS_AUTO_DEPS", "1")
from rgpycrumbs.eon import plot_min, plot_neb  # noqa: E402


def write_con(path, atoms_or_list):
    """ASE → ``.con`` via readcon (export for plot tools only)."""
    path = Path(path)
    items = (
        atoms_or_list if isinstance(atoms_or_list, (list, tuple)) else [atoms_or_list]
    )
    frames = [readcon.ConFrame.from_ase(atoms) for atoms in items]
    readcon.write_con(str(path), frames)
    return path


# sphinx_gallery_thumbnail_number = 4


# %%
# Obtaining the Foundation Model - PET-MAD
# ----------------------------------------
#
# ``PET-MAD`` is a point-edge transformer trained on the
# `MAD dataset <https://arxiv.org/abs/2506.19674>`__ [1]. Equivariance is
# learned from data rather than hard-wired into the architecture, which
# leaves a wider design space for the model. Energies and forces are
# evaluated through ``metatomic`` [2] (built on ``metatensor``): export
# weights from HuggingFace, load them, then call the
# `engine of choice <https://docs.metatensor.org/metatomic/latest/engines/index.html>`_.
#

repo_id = "lab-cosmo/upet"
tag = "v1.5.0"
url_path = f"models/pet-mad-xs-{tag}.ckpt"
fname = Path(f"models/pet-mad-xs-{tag}.pt")
url = f"https://huggingface.co/{repo_id}/resolve/main/{url_path}"
fname.parent.mkdir(parents=True, exist_ok=True)
run_command(f"mtt export {url} -o {fname}")
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
# The reactant and product here are N₂O and ethylene forming oxadiazole.
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
# Endpoints should be minimized before a path search. The input geometries
# here are already relaxed; the tutorial end shows endpoint relaxation with eOn.


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
# ASE's IDPP initializer [5]; see also the
# `ASE tutorial
# <https://ase-lib.org/examples_generated/tutorials/neb_idpp.html>`_.
# eOn builds its own IDPP path later with ``neb_idpp_path``.

N_INTERMEDIATE_IMGS = 10
images = [reactant]
images += [reactant.copy() for _ in range(N_INTERMEDIATE_IMGS)]
images += [product]

neb = NEB(images)
neb.interpolate("idpp")

# %%
# Too many images can kink the band; too few under-resolve the tangent.
# Here the path is the ASE list ``images``.

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


def mk_mta_calc():
    """ASE calculator for the ASE-half NEB (same model file as eOn)."""
    return make_metatomic_ase_calculator(
        fname,
        device="cpu",
        non_conservative=False,
        uncertainty_threshold=0.001,
    )


# set calculators for images
ipath = [reactant] + [reactant.copy() for _ in range(10)] + [product]
for img in ipath:
    img.calc = mk_mta_calc()

print(img.calc._model.capabilities().outputs)

neb = NEB(ipath, climb=True, k=5, method="improvedtangent")
neb.interpolate("idpp")

# store initial path guess for plotting
initial_energies = np.array([img.get_potential_energy() for img in ipath])

# setup the NEB calculation
optimizer = LBFGS(neb, trajectory="A2B.traj", logfile="opt.log")
conv = optimizer.run(fmax=0.01, steps=100)

print("Check if calculation has converged:", conv)

if conv:
    print(neb)

final_energies = np.array([img.get_potential_energy() for img in ipath])

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
# After 100 LBFGS steps the band has not converged. PET-MAD v1.5.0 also
# reports large `LLPR energy uncertainties
# <https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html>`__ on
# some images, so we stop the ASE run here. FIRE can do better on this
# system but needs many more steps; we keep LBFGS for the short ASE example.
#
# The next section uses the same model with eOn's energy-weighted NEB and
# OCI-MMF refinements.


# %%
# eOn and Metatomic
# ^^^^^^^^^^^^^^^^^
#
# With `eOn <https://eondocs.org>`__ through ``pyeonclient``, geometries are
# :class:`~pyeonclient.Matter` objects driven by a registered potential.
# Thread-safe empirical pots can share one instance across images; ML backends
# such as metatomic typically set ``needsPerImageInstance()`` so NEB clones
# the potential per image and evaluates forces in parallel when
# ``[Main] parallel`` is on (the default). Paths start from linear
# interpolation, IDPP [5], or sequential IDPP (SIDPP) [8] via
# ``neb_idpp_path`` and related helpers. This run uses energy-weighted springs
# and off-path climbing-image NEB (OCINEB) with minimum-mode following [6]:
#
# 1. **Energy-weighted springs** — larger spring constants near the climb.
# 2. **OCINEB** — dimer-style off-path refinement at the climbing image when
#    the band force drops below a threshold [6].
#
# ``write_movies=True`` records every NEB iteration as ``neb_NNN.dat`` /
# ``neb_path_NNN.con`` so the full band evolution is available for the
# profile plot below (initial IDPP on the true PES is step 0; the converged
# band is the last file).
#
# Load PET-MAD with ``make_backend("rgpot_metatomic", ...)`` and set NEB
# options with ``NebSpec`` on a shared ``Parameters`` object.

neb_spec = NebSpec(
    n_images=N_INTERMEDIATE_IMGS,
    path_init=PathInit.idpp,
    energy_weighted=True,
    ci_mmf=True,
    max_iterations=1000,
    force_tolerance=0.01,
    max_move=0.1,
    write_movies=True,
    random_seed=706253457,
)

params = pyec.Parameters()
params.job = pyec.JobType.Nudged_Elastic_Band
neb_spec.apply_to_parameters(params)
pot = make_backend(
    "rgpot_metatomic",
    model_path=str(Path(fname).resolve()),
    device="cpu",
    params=params,
)

initial = pyec.from_ase(reactant, pot, params)
final = pyec.from_ase(product, pot, params)
path = pyec.neb_idpp_path(initial, final, N_INTERMEDIATE_IMGS, params)
neb = pyec.NudgedElasticBand(path, params, pot)
f0 = pyec.pot_registry_total_force_calls()
status = neb.compute()
f_neb = pyec.pot_registry_total_force_calls() - f0

if status == pyec.NEBStatus.GOOD:
    neb.find_extrema()

written = pyec.write_neb_results(neb, params, f_neb)
energies = np.array([neb.image_energy(i) for i in range(neb.n_path)])
e_react = float(energies[0])
print("NEB status:", status, "  force_calls:", f_neb)
print(f"E_ref = {neb.energy_reference:.6f} eV,  n_path = {neb.n_path}")
print("ΔE vs reactant (eV):", np.round(energies - e_react, 4))
if neb.num_extrema:
    print("extrema positions:", list(neb.extremum_positions)[: neb.num_extrema])
print("written:", written)


# %%
# Visual interpretation
# ---------------------
#
# ``write_movies`` leaves ``neb_NNN.dat`` for every optimizer step. Plot the
# full band evolution (1D) and the reaction-valley landscape (2D) with
# :func:`rgpycrumbs.eon.plot_neb`.


def show_png(path: str, *, max_width: float = 12.0, max_height: float = 9.0) -> None:
    """Display a saved plot in the gallery, preserving the image aspect ratio.

    Fixed ``figsize`` boxes squash landscape+structure-strip PNGs; size the
    axes from the pixel shape instead so each figure stays readable.
    """
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    aspect = w / float(h)
    if aspect >= max_width / max_height:
        fw, fh = max_width, max_width / aspect
    else:
        fh, fw = max_height, max_height * aspect
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    plt.show()


def show_png_row(
    *paths: str,
    labels: list[str] | tuple[str, ...] | None = None,
    max_width: float = 14.0,
    max_height: float = 8.0,
) -> None:
    """Side-by-side gallery panel for several saved PNGs (equal height)."""
    imgs = [mpimg.imread(p) for p in paths]
    n = len(imgs)
    aspects = [im.shape[1] / float(im.shape[0]) for im in imgs]
    total_aspect = sum(aspects)
    # Fit the whole row into the max box, preserving each panel aspect.
    if total_aspect >= max_width / max_height:
        fw, fh = max_width, max_width / total_aspect
    else:
        fh, fw = max_height, max_height * total_aspect
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(fw, fh),
        gridspec_kw={"width_ratios": aspects},
    )
    if n == 1:
        axes = [axes]
    for ax, im, path in zip(axes, imgs, paths, strict=True):
        ax.imshow(im)
        ax.axis("off")
    if labels is not None:
        for ax, lab in zip(axes, labels, strict=True):
            ax.set_title(lab, fontsize=12, pad=4)
    fig.tight_layout(pad=0.2, w_pad=0.35)
    plt.show()


os.environ.setdefault("MPLBACKEND", "Agg")

# Shared style for the gallery NEB figures.
_neb_style = dict(
    con_file="neb.con",
    figsize=(12, 8),
    zoom_ratio=0.35,
    show_pts=True,
    highlight_last=True,
    facecolor="white",
    fontsize_base=14,
    plot_structures="all",
    strip_renderer="xyzrender",
    strip_dividers=True,
    strip_spacing=2.0,
    xyzrender_config="paton",
    rotation="90x,0y,0z",
    show_legend=True,
)

# 1D: every neb_NNN.dat overlaid; step 0 = IDPP on PET-MAD, last = converged.
plot_neb(
    plot_type="profile",
    output_file="1D_oxad.png",
    title="NEB Path Optimization",
    **_neb_style,
)
show_png("1D_oxad.png")


# %%
# The 2D PES landscape is projected onto reaction-valley coordinates [3, 7]:
# *progress* along the path and *orthogonal deviation*, computed from
# permutation-corrected RMSD distances to the reactant and product. The energy
# surface is interpolated using a gradient-enhanced inverse multiquadric (IMQ)
# Gaussian process that incorporates both energies and projected tangential
# forces from the full NEB optimization history.

plot_neb(
    plot_type="landscape",
    output_file="2D_oxad.png",
    title="NEB-RMSD Surface",
    rc_mode="path",
    landscape_mode="surface",
    landscape_path="all",
    surface_type="grad_imq",
    project_path=True,
    **_neb_style,
)
show_png("2D_oxad.png")
# %%
# Each black dot is a configuration evaluated during NEB optimization [7]. The
# horizontal axis measures progress along the converged path; the vertical axis
# measures perpendicular displacement. Both coordinates derive from
# permutation-corrected RMSD (via IRA [4]) to the reactant and product. The
# energy surface is interpolated by a gradient-enhanced inverse multiquadric GP
# that uses both the energy and the tangential NEB force at each evaluated
# configuration. See [3, Chapter 4] for details.
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

# %%
# Run the minimization
# ^^^^^^^^^^^^^^^^^^^^
#
# Endpoints are relaxed in place with ``Matter.relax``. Movies under
# ``min_reactant/`` and ``min_product/`` feed the plots below.

# Same model and backend as the NEB.
params_min = pyec.Parameters()
params_min.job = pyec.JobType.Minimization
params_min.random_seed = 706253457
params_min.opt_max_iterations = 2000
params_min.opt_max_move = 0.1
params_min.opt_converged_force = 0.01
params_min.write_movies = True
pot_min = make_backend(
    "rgpot_metatomic",
    model_path=str(Path(fname).resolve()),
    device="cpu",
    params=params_min,
)

dir_reactant = Path("min_reactant")
dir_product = Path("min_product")
dir_reactant.mkdir(exist_ok=True)
dir_product.mkdir(exist_ok=True)


def relax_endpoint(atoms, out_dir: Path):
    """ASE Atoms → Matter.relax → ASE Atoms (movie written in *out_dir*)."""
    matter = pyec.from_ase(atoms, pot_min, params_min)
    with chdir(out_dir):
        ok = bool(
            matter.relax(
                write_movie=True,
                prefix_movie="minimization",
                prefix_checkpoint="pos",
            )
        )
    print(f"{out_dir.name}: converged={ok},  E = {matter.potential_energy:.6f} eV")
    atoms_out = pyec.to_ase(matter)
    write_con(out_dir / "min.con", atoms_out)
    return atoms_out, matter


reactant, matter_r = relax_endpoint(reactant, dir_reactant)
product, matter_p = relax_endpoint(product, dir_product)


# %%
# Minimization figures
# ^^^^^^^^^^^^^^^^^^^^
#
# Energy profile and optimizer convergence overlay both endpoints. The 2D
# landscapes are **separate** for reactant and product (each trajectory has its
# own RMSD frame). ``auto_thin`` keeps long force-eval movies fit-safe.
# Each figure is its own gallery cell so sphinx-gallery does not pack them into
# a cramped multi-image grid.

_min_style = dict(
    surface_type="grad_imq",
    project_path=True,
    plot_structures="endpoints",
    strip_renderer="xyzrender",
    xyzrender_config="paton",
    rotation="90x,0y,0z",
    strip_dividers=True,
    strip_spacing=2.5,
    auto_thin=True,
    max_surface_points=64,
    dpi=160,
)

# %%
# Reactant landscape (RMSD frame of the reactant movie; strip = initial → minimized):

plot_min(
    job_dir=[dir_reactant],
    label=["reactant"],
    plot_type="landscape",
    output="min_2D_reactant_oxad.png",
    **_min_style,
)
show_png("min_2D_reactant_oxad.png", max_width=11.0, max_height=10.0)

# %%
# Product landscape (separate RMSD frame):

plot_min(
    job_dir=[dir_product],
    label=["product"],
    plot_type="landscape",
    output="min_2D_product_oxad.png",
    **_min_style,
)
show_png("min_2D_product_oxad.png", max_width=11.0, max_height=10.0)

# %%
# Energy profiles for both endpoints on one axis:

plot_min(
    job_dir=[dir_reactant, dir_product],
    label=["reactant", "product"],
    plot_type="profile",
    output="min_1D_oxad.png",
    dpi=160,
)
show_png("min_1D_oxad.png", max_width=11.0, max_height=5.0)

# %%
# Optimizer force convergence for both endpoints:

plot_min(
    job_dir=[dir_reactant, dir_product],
    label=["reactant", "product"],
    plot_type="convergence",
    output="min_conv_oxad.png",
    dpi=160,
)
show_png("min_conv_oxad.png", max_width=11.0, max_height=5.0)

# %%
# Additionally, the relative ordering must be preserved, for which we use
# IRA [4]. ``reactant`` / ``product`` are the relaxed ASE views.

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
# (6) Goswami, R.; Gunde, M.; Jónsson, H. Enhanced Climbing Image Nudged
#     Elastic Band Method with Hessian Eigenmode Alignment. Front. Chem.
#     2026, 14. https://doi.org/10.3389/fchem.2026.1807063.
#
# (7) R. Goswami, Two-dimensional RMSD projections for reaction path
#     visualization and validation, MethodsX, p. 103851, Mar. 2026, doi:
#     10.1016/j.mex.2026.103851.
#
# (8) Schmerwitz, Y. L. A.; Ásgeirsson, V.; Jónsson, H. Improved
#     Initialization of Optimal Path Calculations Using Sequential
#     Traversal over the Image-Dependent Pair Potential Surface. J. Chem.
#     Theory Comput. 2024, 20 (1), 155–163.
#     https://doi.org/10.1021/acs.jctc.3c01111.
#
