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

1. Download **PET-MAD** and call
   ``make_backend("rgpot_metatomic", model_path=...)`` (or ``metatomic`` /
   ``ase_metatomic``).
2. ASE half: ASE IDPP + short ASE NEB with ``MetatomicCalculator`` (baseline).
3. eOn half: **Matter** + eOn-native IDPP + ``NudgedElasticBand`` (EW springs +
   MMF) — path never goes through ASE after the endpoints.
4. Plot the band and relax endpoints with the same potential.


Importing Required Packages
---------------------------
First, we import all the necessary python packages for this task.
By convention, all ``import``
statements are at the top of the file.
"""

import os
import sys
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
from pyeonclient.backends import (
    ensure_metatomic_load_compat,
    make_backend,
    make_metatomic_ase_calculator,
)
from rgpycrumbs.run.jupyter import run_command_or_exit


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
# ASE baseline: ASE's own IDPP. eOn has the same class of initializer
# natively (``neb_idpp_path`` / ``NEBInit.IDPP``) — use that for the eOn
# band below, not ASE frames re-imported as Matter. IDPP background: [5]
# and the `ASE tutorial <https://ase-lib.org/examples_generated/tutorials/neb_idpp.html>`_.

N_INTERMEDIATE_IMGS = 10
images = [reactant]
images += [reactant.copy() for _ in range(N_INTERMEDIATE_IMGS)]
images += [product]

neb = NEB(images)
neb.interpolate("idpp")  # ASE only — not used for the eOn band

# %%
# We don't cover subtleties in setting the number of images: too many can kink
# the band, too few under-resolve the tangent. The path lives in memory as the
# ASE list ``images`` (same objects eOn will convert to ``Matter`` below).

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


# PET-MAD export: patch load_atomistic_model once, then ASE calculator.
ensure_metatomic_load_compat()


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
# In the 100 NEB steps we took, the structure did unfortunately not converge.
# The metatomic calculator for PET-MAD v1.5.0 provides `LLPR based energy
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
# `eOn <https://eondocs.org>`_ (via ``pyeonclient``) keeps geometries as
# :class:`~pyeonclient.Matter` and a shared potential. Initial paths use
# **eOn** IDPP/SIDPP/linear — the same ``helpers::neb_paths`` engines as
# ``eonclient`` — not ASE ``interpolate("idpp")`` rewrapped as Matter::
#
#     # ASE                                      # eOn (pyeonclient)
#     neb.interpolate("idpp")                    initial/final = from_ase(...)
#     LBFGS(neb).run(fmax=0.01)                  path = neb_idpp_path(R, P, n, params)
#                                                neb = NudgedElasticBand(path, params, pot)
#                                                neb.compute()
#
# Beyond a plain climbing-image NEB, eOn adds:
#
# 1. **Energy-weighted springs** — higher spring constants near the climbing
#    image for better tangent resolution.
# 2. **Off-path climbing image (OCI / MMF)** [6] — hybrid steps that mix in
#    single-ended dimer searches at the climbing image.
#
# Forces via the named backend registry (not a custom wrapper)::
#
#     pot = make_backend("rgpot_metatomic", model_path="pet-mad.pt")
#
# Thin wheel: RGPOT + multi-ABI libmetatomic_engine (from rgpot / env).
# Fat build: make_backend("metatomic", model_path=...).
# ASE wrap as eOn pot: make_backend("ase_metatomic", model_path=...).

def configure_neb_ew_mmf(params: pyec.Parameters, *, n_images: int) -> pyec.Parameters:
    """Energy-weighted CI-NEB + light MMF (cookbook recipe; all live attrs).

    This is the "why eOn" knob set: weighted springs near the climb, then
    off-path dimer-style steps at the climbing image. Defaults match a
    short PET-MAD band — tune here rather than scattering flags.
    """
    params.job = pyec.JobType.Nudged_Elastic_Band
    params.random_seed = 706253457
    params.neb_images = n_images
    params.neb_init_method = pyec.NEBInit.IDPP  # eOn-native IDPP (not ASE)
    params.neb_minimize_endpoints = False
    params.neb_climbing_image = True
    params.neb_climbing_converged_only = True
    params.neb_ci_after = 0.5
    params.neb_ci_after_rel = 0.8
    params.neb_energy_weighted = True
    params.neb_ew_ksp_min = 0.972
    params.neb_ew_ksp_max = 9.72
    params.neb_ci_mmf = True
    params.neb_ci_mmf_after = 0.1
    params.neb_ci_mmf_after_rel = 0.5
    params.neb_ci_mmf_angle = 0.9
    params.neb_ci_mmf_nsteps = 1000
    params.neb_max_iterations = 1000
    params.opt_max_iterations = 1000
    params.opt_max_move = 0.1
    params.opt_converged_force = 0.01  # like ASE fmax
    params.write_movies = True  # path history for landscape plots
    return params


# Shared force engine (same .pt as the ASE calculator above).
# Engine .so: RGPOT_METATOMIC_ENGINE or rgpot.default_metatomic_engine_path().
pot = make_backend("rgpot_metatomic", model_path=str(Path(fname).resolve()), device="cpu")
params = pyec.Parameters()
configure_neb_ew_mmf(params, n_images=N_INTERMEDIATE_IMGS)

# Endpoints only from ASE; path is eOn IDPP.
initial = pyec.from_ase(reactant, pot, params)
final = pyec.from_ase(product, pot, params)
path = pyec.neb_idpp_path(
    initial, final, N_INTERMEDIATE_IMGS, params
)
# Equivalent: NudgedElasticBand(initial, final, params, pot) with
# params.neb_init_method = NEBInit.IDPP and params.neb_images set.
neb = pyec.NudgedElasticBand(path, params, pot)
status = neb.compute()  # like LBFGS(neb).run(...)
print("NEB status:", status)

if status == pyec.NEBStatus.GOOD:
    neb.find_extrema()

# Results stay on the band object (ASE would read image energies / positions).
path = list(neb.path_images())
energies = np.array([neb.image_energy(i) for i in range(neb.n_path)])
print(f"E_ref = {neb.energy_reference:.6f} eV,  n_path = {neb.n_path}")
print("ΔE (eV):", np.round(energies - neb.energy_reference, 4))
if neb.num_extrema:
    print(
        "extrema positions:",
        list(neb.extremum_positions)[: neb.num_extrema],
    )

# Convert back to ASE when you want ASE tools; export .con only for plotters.
path_ase = [pyec.to_ase(m) for m in path]
write_con("neb.con", path_ase)
pyec.neb_write_results(neb, params, 0)


# %%
# Visual interpretation
# ---------------------
#
# `rgpycrumbs <http://pypi.org/project/rgpycrumbs>`_ is a visualization toolkit
# designed to bridge the gap between raw computational output and physical
# intuition, mapping high-dimensional NEB trajectories onto interpretable 1D
# energy profiles and 2D RMSD landscapes.  As it is normally used from the
# command-line, here we define a helper.


def _strip_common_flags() -> list[str]:
    """Shared xyzrender structure-strip flags for all gallery figures."""
    return [
        "--facecolor",
        "white",
        "--fontsize-base",
        "14",
        # Always render every path image (not only R/SP/P).
        "--plot-structures",
        "all",
        "--strip-renderer",
        "xyzrender",
        "--strip-dividers",
        "--strip-spacing",
        "2.0",
        "--xyzrender-config",
        "paton",
        "--rotation",
        "90x,0y,0z",
        "--show-legend",
    ]


def run_neb_plot(
    mode: str,
    con_file: str = "neb.con",
    output_file: str = "plot.png",
    title: str = "",
) -> None:
    """
    Build and run an rgpycrumbs NEB plot.

    Always use the full optimization history and a full structure strip
    (``--plot-structures all``): every image on the path, not only R/SP/P.

    mode: 'profile' (1D) or 'landscape' (2D)
    """
    # Target: in-process chemparseplot/rgpycrumbs library API when the full
    # plot pipeline is exposed as stable functions. Today: CLI + uv PEP 723
    # for plot deps (jax, adjustText,
    # chemparseplot, …). Host env only needs bare rgpycrumbs + readcon.
    base_cmd = [
        sys.executable,
        "-m",
        "rgpycrumbs.cli",
        "eon",
        "plt-neb",
        "--con-file",
        con_file,
        "--output-file",
        output_file,
        "--figsize",
        "12",
        "8",
        "--zoom-ratio",
        "0.35",
        # Full history: all optimizer paths / points.
        "--show-pts",
        "--highlight-last",
        *_strip_common_flags(),
    ]

    if title:
        base_cmd.extend(["--title", title])

    if mode == "profile":
        base_cmd.extend(["--plot-type", "profile"])
    elif mode == "landscape":
        base_cmd.extend(
            [
                "--plot-type",
                "landscape",
                "--rc-mode",
                "path",
                "--landscape-mode",
                "surface",
                # Use all path points for the surface (not last-only).
                "--landscape-path",
                "all",
                "--surface-type",
                "grad_imq",
                "--project-path",
            ]
        )
    else:
        raise ValueError(f"Unknown plot mode: {mode}")

    # Landscape GP (grad_imq, full history) can exceed 3 min on CI runners.
    run_command_or_exit(base_cmd, capture=False, timeout=600)


def thin_min_movie(
    job_dir: Path,
    *,
    max_frames: int = 64,
    prefix: str = "minimization",
) -> int:
    """Thin a dense eOn minimization movie before landscape surface fits.

    ``write_movies`` records every force evaluation, so long LBFGS paths can
    exceed ~150 frames and make gradient surface fits numerically unstable.
    This keeps the first and last frames plus evenly spaced intermediates
    (``max_frames`` default 64).

    Returns the number of frames after thinning (or the original count if no
    thinning was needed).
    """
    job_dir = Path(job_dir)
    movie = None
    for candidate in (job_dir / prefix, job_dir / f"{prefix}.con"):
        if candidate.exists():
            movie = candidate
            break
    if movie is None:
        return 0

    frames = list(readcon.read_con(str(movie)))
    n = len(frames)
    if n <= max_frames:
        return n

    # Inclusive endpoints via linspace; unique keeps order and first/last.
    idx = np.unique(np.linspace(0, n - 1, num=max_frames, dtype=int))
    if idx[-1] != n - 1:
        idx = np.unique(np.append(idx, n - 1))
    thinned = [frames[i] for i in idx]
    readcon.write_con(str(movie), thinned)

    dat_path = job_dir / f"{prefix}.dat"
    if dat_path.exists():
        lines = dat_path.read_text().splitlines()
        if lines:
            header, rows = lines[0], lines[1:]
            if len(rows) == n:
                kept = [rows[i] for i in idx]
                dat_path.write_text(header + "\n" + "\n".join(kept) + "\n")

    print(f"Thinned {movie.name} in {job_dir}: {n} -> {len(thinned)} frames")
    return len(thinned)


def run_min_plot(
    job_dirs: list[Path],
    labels: list[str],
    plot_type: str,
    output_file: str,
) -> None:
    """Plot endpoint minimizations (profile / landscape / convergence) with strips."""
    base_cmd = [
        sys.executable,
        "-m",
        "rgpycrumbs.cli",
        "eon",
        "plt-min",
        "--plot-type",
        plot_type,
        "-o",
        output_file,
        "--surface-type",
        "grad_imq",
        "--project-path",
        # Start/end structures for each minimization trajectory.
        "--plot-structures",
        "endpoints",
        "--strip-renderer",
        "xyzrender",
        "--strip-dividers",
        "--xyzrender-config",
        "paton",
        "--rotation",
        "90x,0y,0z",
    ]
    for d, lab in zip(job_dirs, labels, strict=True):
        base_cmd.extend(["--job-dir", str(d), "--label", lab])
    run_command_or_exit(base_cmd, capture=False, timeout=600)


def show_png(path: str, figsize=(10, 8)) -> None:
    img = mpimg.imread(path)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
#
# NEB figures use the full optimization history and a structure strip for
# **every** image on the path (``--plot-structures all``).

# Prefer Agg for headless/CI; notebooks can still override.
os.environ.setdefault("MPLBACKEND", "Agg")
# Prefer uv PEP 723 isolation for plot scripts so host need not carry
# chemparseplot/jax/adjustText (avoids partial in-env stack).
os.environ.setdefault("RGPKGS_FORCE_UV", "1")
os.environ.setdefault("RGPYCRUMBS_FORCE_UV", "1")  # legacy alias
# chemparseplot 1.9.10-1.9.12 fail to import on Python <= 3.13 (class-body
# annotation without `from __future__ import annotations`). Freeze uv's
# PEP 723 resolution at the last known-good snapshot; remove together with
# the rgpycrumbs cap in environment.yml once a fixed release is published.
os.environ.setdefault("UV_EXCLUDE_NEWER", "2026-07-15T00:00:00Z")

# Run the 1D plotting command using the helper
run_neb_plot("profile", title="NEB Path Optimization", output_file="1D_oxad.png")
show_png("1D_oxad.png")


# %%
# The 2D PES landscape is projected onto reaction-valley coordinates [3, 7]:
# *progress* along the path and *orthogonal deviation*, computed from
# permutation-corrected RMSD distances to the reactant and product. The energy
# surface is interpolated using a gradient-enhanced inverse multiquadric (IMQ)
# Gaussian process that incorporates both energies and projected tangential
# forces from the full NEB optimization history.

run_neb_plot("landscape", title="NEB-RMSD Surface", output_file="2D_oxad.png")
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
# Endpoint relaxation is the ASE ``LBFGS(atoms).run()`` pattern with Matter::
#
#     matter = pyec.from_ase(atoms, pot, params)
#     matter.relax()                 # like dyn.run(fmax=...)
#     atoms = pyec.to_ase(matter)
#
# Movie files under ``min_reactant/`` / ``min_product/`` are only for the
# ``rgpycrumbs`` plots that follow; the optimizer itself is in-memory.

# Same backend key as the NEB — one model, one pot type for the whole example.
pot_min = make_backend(
    "rgpot_metatomic", model_path=str(Path(fname).resolve()), device="cpu"
)
params_min = pyec.Parameters()
params_min.job = pyec.JobType.Minimization
params_min.random_seed = 706253457
params_min.opt_max_iterations = 2000
params_min.opt_max_move = 0.1
params_min.opt_converged_force = 0.01  # like ASE fmax
params_min.write_movies = True

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

# Thin dense force-eval movies so gradient surface fits stay well-conditioned.
for _min_dir in (dir_reactant, dir_product):
    thin_min_movie(_min_dir, max_frames=64)


# %%
# Minimization figures
# ^^^^^^^^^^^^^^^^^^^^
#
# Energy profile and optimizer convergence overlay both endpoints. The 2D
# landscapes are **separate** for reactant and product (each trajectory has its
# own RMSD frame). Structure strips show start/end geometries (xyzrender).
# Trajectories were thinned above before these plots.

min_jobs = [dir_reactant, dir_product]
min_labels = ["reactant", "product"]
# One landscape per endpoint — do not overlay on a shared (s, d) frame.
run_min_plot([dir_reactant], ["reactant"], "landscape", "min_2D_reactant_oxad.png")
show_png("min_2D_reactant_oxad.png")
run_min_plot([dir_product], ["product"], "landscape", "min_2D_product_oxad.png")
show_png("min_2D_product_oxad.png")
run_min_plot(min_jobs, min_labels, "profile", "min_1D_oxad.png")
show_png("min_1D_oxad.png")
run_min_plot(min_jobs, min_labels, "convergence", "min_conv_oxad.png")
show_png("min_conv_oxad.png")

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
# (6) Goswami, R; Gunde, M; Jónsson, H. Enhanced climbing image nudged elastic
#     band method with hessian eigenmode alignment, Jan. 22, 2026, arXiv:
#     arXiv:2601.12630. doi: 10.48550/arXiv.2601.12630.
#
# (7) R. Goswami, Two-dimensional RMSD projections for reaction path
#     visualization and validation, MethodsX, p. 103851, Mar. 2026, doi:
#     10.1016/j.mex.2026.103851.
#
