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
from rgpycrumbs.eon import plot_min, plot_neb

os.environ.setdefault("MPLBACKEND", "Agg")

# sphinx_gallery_thumbnail_number = 4


def write_con(path, atoms_or_list):
    """ASE → ``.con`` via readcon (for plot tools / on-disk export)."""
    path = Path(path)
    items = (
        atoms_or_list if isinstance(atoms_or_list, (list, tuple)) else [atoms_or_list]
    )
    frames = [readcon.ConFrame.from_ase(atoms) for atoms in items]
    readcon.write_con(str(path), frames)
    return path


def show_png(path: str, *, figsize=(10, 8)) -> None:
    """Display a saved plot PNG in the sphinx-gallery page."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    plt.show()


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
# Initial path (IDPP)
# ^^^^^^^^^^^^^^^^^^^
#
# Endpoints above are already minimized; relaxation with eOn is shown at the
# end. An NEB needs an initial band. Linear interpolation can break bonds or
# pass atoms through each other; a common fix is to refine the band on a cheap
# surrogate such as the image-dependent pair potential (IDPP) [5] (bond-length
# surface). ASE's IDPP initializer is used here; eOn builds its own path later
# with ``neb_idpp_path``. See also the
# `ASE tutorial
# <https://ase-lib.org/examples_generated/tutorials/neb_idpp.html>`_.
# Too many images kink the band; too few under-resolve the tangent.

N_INTERMEDIATE_IMGS = 10


# %%
# Running NEBs
# ------------
#
# ASE climbing-image NEB
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Short ASE + metatomic calculator run on the same PET-MAD export (comparison
# only; full convergence on this system needs many more steps).


def mk_mta_calc():
    """ASE calculator for the ASE-half NEB (same model file as eOn)."""
    return make_metatomic_ase_calculator(
        fname,
        device="cpu",
        non_conservative=False,
        uncertainty_threshold=0.001,
    )


ipath = [reactant] + [reactant.copy() for _ in range(N_INTERMEDIATE_IMGS)] + [product]
for img in ipath:
    img.calc = mk_mta_calc()

print(img.calc._model.capabilities().outputs)

neb = NEB(ipath, climb=True, k=5, method="improvedtangent")
neb.interpolate("idpp")
initial_energies = np.array([img.get_potential_energy() for img in ipath])
optimizer = LBFGS(neb, trajectory="A2B.traj", logfile="opt.log")
conv = optimizer.run(fmax=0.01, steps=100)
print("ASE NEB converged:", conv)
final_energies = np.array([img.get_potential_energy() for img in ipath])

plt.figure(figsize=(8, 6))
plt.plot(
    initial_energies - initial_energies[0],
    "o-",
    label="Initial path (IDPP)",
    color="xkcd:blue",
)
plt.plot(
    final_energies - initial_energies[0],
    "o-",
    label="After 100 LBFGS steps",
    color="xkcd:orange",
)
plt.xlabel("Image number")
plt.ylabel("Potential energy (eV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("ASE NEB path evolution")
plt.show()

# %%
# After 100 LBFGS steps the band has not converged. PET-MAD v1.5.0 also
# reports large `LLPR energy uncertainties
# <https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html>`__ on
# some images. The next section uses the same model with eOn's energy-weighted
# NEB and OCI-MMF refinements.


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
# 2D landscape in reaction-valley coordinates [3, 7]: progress along the path
# and orthogonal deviation from permutation-corrected RMSD to the endpoints
# (IRA [4]). Energies and projected tangential forces feed a gradient-enhanced
# inverse multiquadric GP [7]; black dots are configurations evaluated during
# NEB (see [3, Chapter 4]).

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
# Relaxing the endpoints with eOn
# -------------------------------
#
# The NEB above started from pre-minimized geometries. Here unrelaxed
# structures are boxed, minimized with the same PET-MAD backend, then checked
# with IRA.

reactant = aseio.read("data/reactant.con")
product = aseio.read("data/product.con")


def center_cell(atoms):
    """Assign a cubic cell and center (eOn expects a unit cell)."""
    atoms.set_cell([20, 20, 20])
    atoms.pbc = True
    atoms.center()
    return atoms


reactant = center_cell(reactant)
product = center_cell(product)

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
# ``Matter.relax`` writes movies under ``min_reactant/`` and ``min_product/``
# for the plots below.

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
    """ASE Atoms → Matter.relax → ASE Atoms (movie in *out_dir*)."""
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
    return atoms_out


reactant = relax_endpoint(reactant, dir_reactant)
product = relax_endpoint(product, dir_product)


# %%
# Minimization figures
# ^^^^^^^^^^^^^^^^^^^^
#
# Landscapes use separate RMSD frames per endpoint and are shown side by side.
# Profile and convergence overlay both jobs. ``auto_thin`` keeps long force-eval
# movies fit-safe.

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

plot_min(
    job_dir=[dir_reactant],
    label=["reactant"],
    plot_type="landscape",
    output="min_2D_reactant_oxad.png",
    **_min_style,
)
plot_min(
    job_dir=[dir_product],
    label=["product"],
    plot_type="landscape",
    output="min_2D_product_oxad.png",
    **_min_style,
)
fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(14, 7))
for ax, path, lab in (
    (ax_r, "min_2D_reactant_oxad.png", "reactant"),
    (ax_p, "min_2D_product_oxad.png", "product"),
):
    ax.imshow(mpimg.imread(path))
    ax.set_title(lab)
    ax.axis("off")
fig.tight_layout()
plt.show()

# %%
# Energy profiles for both endpoints:

plot_min(
    job_dir=[dir_reactant, dir_product],
    label=["reactant", "product"],
    plot_type="profile",
    output="min_1D_oxad.png",
    dpi=160,
)
show_png("min_1D_oxad.png", figsize=(10, 5))

# %%
# Optimizer force convergence:

plot_min(
    job_dir=[dir_reactant, dir_product],
    label=["reactant", "product"],
    plot_type="convergence",
    output="min_conv_oxad.png",
    dpi=160,
)
show_png("min_conv_oxad.png", figsize=(10, 5))

# %%
# Atom ordering after relaxation is aligned with IRA [4] (rotation,
# translation, and permutation of the product onto the reactant).

ira = ira_mod.IRA()
kmax_factor = 1.8
nat1 = len(reactant)
typ1 = reactant.get_atomic_numbers()
coords1 = reactant.get_positions()
nat2 = len(product)
typ2 = product.get_atomic_numbers()
coords2 = product.get_positions()

r, t, p, hd = ira.match(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)
print(f"IRA match: Hausdorff distance = {hd:.6f} Å")

# Align product: rotate/translate, then permute so atom *i* matches reactant *i*.
coords2_aligned = (coords2 @ r.T) + t
coords2_aligned_permuted = coords2_aligned[p]
product = reactant.copy()
product.positions = coords2_aligned_permuted

# %%
# Aligned endpoints:

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
