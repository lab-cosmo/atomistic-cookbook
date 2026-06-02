r"""
Machine-learned electron density and zero-shot properties
==========================================================

:Authors: Joseph W. Abbott `@jwa7 <https://github.com/jwa7>`_

This recipe demonstrates how to predict the electron density of a molecule with a
pretrained machine learning model and use the prediction for two purposes: (1) as an
improved initial guess for a self-consistent field (SCF) calculation that reduces the
number of iterations to convergence, and (2) as a direct source of electronic properties
without running any SCF at all.

The model is a PET (Point Edge Transformer) architecture trained with `metatrain
<https://github.com/lab-cosmo/metatrain>`_ on the `SCFBench
<https://doi.org/10.48550/arXiv.2509.25724>`_ dataset of PBE/def2-SVP density functional
theory calculations on small organic molecules. It predicts the expansion coefficients
of the electron density using an overlap-metric resolution-of-identity (RI) fit onto the
``def2-universal-jfit`` auxiliary basis, and the downstream DFT calculations are carried
out with `PySCF <https://pyscf.org>`_.
"""

# sphinx_gallery_thumbnail_number = 2

# %%

import os
import sys

import ase
import chemiscope
import matplotlib.pyplot as plt
import metatensor.torch as mts
import numpy as np
from metatomic.torch import ModelOutput, load_atomistic_model
from metatomic.torch.ase_calculator import MetatomicCalculator
from pyscf import dft
from atomistic_cookbook_utils import download_with_retry

# Add data/ to sys.path so rho_utils can be imported directly.
# sphinx-gallery runs scripts via exec(), so __file__ is unavailable;
# the CWD is set to the example folder, so a relative path works.
_data_dir = os.path.abspath("data")
sys.path.insert(0, _data_dir)
from rho_utils import (  # noqa: E402
    BOHR_TO_ANG,
    atoms_to_pyscf,
    dm_from_ri_coefficients,
    nmae_percent,
    run_scf,
)

# %%
# The density fitting (RI) approximation
# --------------------------------------
#
# Kohn-Sham DFT represents the electron density through the density matrix
# :math:`D_{\mu\nu}`,
#
# .. math::
#
#    \rho(\mathbf{r}) \;=\; \sum_{\mu\nu} D_{\mu\nu}\,
#    \phi_\mu(\mathbf{r})\,\phi_\nu(\mathbf{r})\;,
#
# where :math:`\{\phi_\mu\}` are the contracted Gaussian orbital basis
# functions. The SCF cycle repeatedly builds the Fock matrix
# :math:`F[D] = h + V_J[D] + V_{xc}[D]` and diagonalises it until
# :math:`D` no longer changes.
#
# The resolution-of-identity (RI) approximation re-expands the density on a
# compact, *atom-centred* auxiliary basis :math:`\{\chi_P\}`,
#
# .. math::
#
#    \tilde\rho(\mathbf{r}) \;=\; \sum_P c_P\,\chi_P(\mathbf{r})\;.
#
# The coefficients :math:`\{c_P\}` are found by minimising the squared
# pointwise error of the expansion,
#
# .. math::
#
#    \min_{\{c_P\}}\;
#    \int [\rho(\mathbf{r}) - \tilde\rho(\mathbf{r})]^2\,\mathrm{d}\mathbf{r}\;.
#
# This is the *overlap-metric* (or *S-fit*) variant of RI. The approximation is not
# exact for a finite auxiliary basis, but it has three properties that make
# :math:`\{c_P\}` an attractive machine-learning target:
#
# * **Compact.** A vector of :math:`N_\text{aux}` numbers rather than an
#   :math:`N_\text{ao}\times N_\text{ao}` matrix.
# * **Local.** The basis functions are atom-centred, so the coefficients
#   naturally decompose by chemical environment and transfer across systems.
# * **Systematically improvable.** Enlarging the auxiliary basis reduces the
#   fitting error towards zero.
#
# Given a predicted :math:`\tilde\rho`, two downstream workflows become cheap:
#
# * **Accelerated SCF.** Diagonalising the Fock matrix built from
#   :math:`\tilde\rho`, :math:`F[\tilde\rho] = h + V_J[\tilde\rho]
#   + V_{xc}[\tilde\rho]`, yields a density matrix :math:`D_0` already
#   close to self-consistency, reducing the number of SCF iterations needed.
# * **Zero-shot properties.** Observables that are linear in :math:`\rho`
#   -- such as the electric dipole moment
#   :math:`\boldsymbol{\mu} = -\int \mathbf{r}\,\rho(\mathbf{r})\,
#   \mathrm{d}\mathbf{r}` --
#   can be evaluated directly from :math:`\tilde\rho` without any SCF at
#   all. The quality depends on how closely :math:`\tilde\rho` approximates
#   the true ground-state density.

# %%
# System setup
# ------------
#
# We use a small organic molecule from the SCFBench dataset (index 3464), a
# seven-atom :math:`\text{C}_2\text{H}_2\text{O}_3` system computed at the
# PBE/def2-SVP level of theory. The DFT functional and orbital basis are set
# to match the training data; the auxiliary basis is
# ``def2-universal-jfit`` -- the auxiliary basis used for the overlap-metric
# RI fit in SCFBench (the ``jfit`` in the name refers to the basis design,
# not the fitting metric).

atoms = ase.Atoms(
    numbers=[8, 6, 6, 1, 8, 8, 1],
    positions=[
        [-1.3058, -1.0321, -1.0321],
        [-1.0843, -0.1924, -0.1924],
        [0.3229, 0.2412, 0.2412],
        [-1.8735, 0.3561, 0.3561],
        [0.4843, 1.0926, 1.0926],
        [1.3055, -0.4059, -0.4059],
        [2.1509, -0.0596, -0.0596],
    ],
)
atoms.center(about=atoms.get_center_of_mass())

BASIS = "def2-svp"
XC = "pbe"
AUXBASIS = "def2-universal-jfit"

chemiscope.show(atoms, mode="structure")

# %%
# Baseline: DFT from the SAD initial guess
# -----------------------------------------
#
# PySCF's default initial density matrix is the *superposition of atomic
# densities* (SAD): pre-computed atomic density matrices are stacked
# block-diagonally, ignoring all bonding. This is a reasonable but crude
# starting point that typically requires 10--15 SCF iterations to converge.
#
# We capture the SAD density matrix *before* the SCF runs so that we can
# later compute properties from it directly and compare with the predicted
# densities.

mol = atoms_to_pyscf(atoms, BASIS)
mf_baseline = dft.RKS(mol)
mf_baseline.xc = XC
dm_sad = mf_baseline.get_init_guess()

mf_conv, n_sad = run_scf(atoms, XC, BASIS)
dm_conv = mf_conv.make_rdm1()

print(f"SAD initial guess → converged in {n_sad} cycles")
print(f"Converged total energy: {mf_conv.e_tot:.6f} Ha")

# %%
# RI reference coefficients
# -------------------------
#
# The ``.mts`` file below contains the overlap-metric RI coefficients for this
# molecule computed directly from the converged DFT density matrix. These represent the
# theoretical best that any ML model trained with this scheme can achieve for
# this auxiliary basis -- the error relative to the true density is purely
# from the RI truncation, not from model inaccuracy.
#
# The helper ``dm_from_ri_coefficients`` builds the Fock matrix
# :math:`F = h + V_J[\tilde\rho] + V_{xc}[\tilde\rho]` from the RI
# coefficients, then diagonalises it to obtain an orbital-consistent density
# matrix. This single-step Fock diagonalisation is not self-consistent, but
# it places the starting density matrix on the correct orbital manifold for
# the chosen Hamiltonian.

ref_coefficients = mts.load(
    os.path.join(_data_dir, "scfbench_test_molecule_3464_jft_density.mts")
)

dm_ri = dm_from_ri_coefficients(atoms, ref_coefficients, XC, BASIS, AUXBASIS)
_, n_ri = run_scf(atoms, XC, BASIS, dm0=dm_ri)

print(f"RI reference initial guess → converged in {n_ri} cycles")

# %%
# ML-predicted RI coefficients
# ----------------------------
#
# We now replace the reference coefficients with the output of the pretrained
# PET model. The model takes a ``metatomic``-format atomistic system and
# returns a :class:`~metatensor.TensorMap` of predicted RI coefficients with
# the same structure as the reference above -- one block per
# :math:`(\ell, \sigma, Z)` triple of angular momentum, inversion parity, and
# atomic species.

download_with_retry(
    "https://github.com/ppegolo/labcosmo_ictp_school/raw/refs/heads/tmp/pet-density.pt",
    "model.pt",
)
model_path = "model.pt"

TARGET = "mtt::rho_c_jfit_overlap"
model = load_atomistic_model(model_path)
calculator = MetatomicCalculator(model)
ml_coefficients = calculator.run_model(atoms, {TARGET: ModelOutput(per_atom=True)})[
    TARGET
]

dm_ml = dm_from_ri_coefficients(atoms, ml_coefficients, XC, BASIS, AUXBASIS)
_, n_ml = run_scf(atoms, XC, BASIS, dm0=dm_ml)

print(f"ML initial guess → converged in {n_ml} cycles")

# %%
# SCF acceleration
# ----------------
#
# Both the RI reference and the ML prediction reduce the SCF iteration count
# compared with the SAD baseline. The RI reference establishes the best
# achievable reduction for this auxiliary basis; the ML model reaches a
# similar reduction without ever having seen this molecule.

labels = ["SAD", "RI reference", "ML prediction"]
counts = [n_sad, n_ri, n_ml]
colors = ["C7", "C0", "C1"]

fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True, dpi=120)
bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.5)
ax.bar_label(bars, padding=3, fontsize=10)
ax.set_ylabel("SCF iterations to convergence")
ax.set_title("Effect of initial density on SCF convergence")
ax.set_ylim(0, max(counts) * 1.3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)

# %%
# Electron density in the molecular plane
# ----------------------------------------
#
# All heavy atoms in this molecule lie in the plane :math:`y = z`, so a 2D
# slice through that plane gives a clear view of the electron density. We
# parametrise the slice by :math:`(x,\; s)` where
# :math:`s = \sqrt{2}\,y = \sqrt{2}\,z` is the in-plane coordinate
# perpendicular to the molecular axis.
#
# The first panel shows the converged PBE/def2-SVP density. The remaining
# panels show the *error* of each initial density relative to the converged
# one, :math:`\Delta\rho = \tilde\rho - \rho_\text{conv}`, on a diverging
# colour scale: positive values (more charge in :math:`\tilde\rho`) are red,
# negative blue. The ML prediction recovers the converged density far more
# accurately than the SAD guess.


def _rho_slice(mol, dm, n_x=100, n_s=80, x_lim=(-3.5, 3.5), s_lim=(-2.5, 2.5)):
    """Electron density on the y=z molecular plane, shaped (n_s, n_x)."""
    x = np.linspace(*x_lim, n_x)
    s = np.linspace(*s_lim, n_s)
    xx, ss = np.meshgrid(x, s)
    # y = z = s / sqrt(2) puts points exactly in the molecular plane
    pts_bohr = (
        np.column_stack([xx.ravel(), ss.ravel() / np.sqrt(2), ss.ravel() / np.sqrt(2)])
        / BOHR_TO_ANG
    )
    ao = mol.eval_gto("GTOval_sph", pts_bohr)  # (npts, nao)
    rho = np.einsum("pi,ij,pj->p", ao, dm, ao).reshape(n_s, n_x)
    return x, s, rho


x_grid, s_grid, rho_conv = _rho_slice(mol, dm_conv)
_, _, rho_ml = _rho_slice(mol, dm_ml)
_, _, rho_sad = _rho_slice(mol, dm_sad)

# Atom positions projected onto (x, s=y*sqrt(2)) plane
pos = atoms.get_positions()
x_atoms = pos[:, 0]
s_atoms = pos[:, 1] * np.sqrt(2)
numbers = atoms.get_atomic_numbers()

_marker = {8: ("o", "red"), 6: ("o", "dimgrey"), 1: ("o", "white")}
_ms = {8: 9, 6: 8, 1: 5}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True, dpi=120)

# Panel 1: converged density
im0 = axes[0].imshow(
    rho_conv,
    extent=[x_grid[0], x_grid[-1], s_grid[0], s_grid[-1]],
    origin="lower",
    cmap="Blues",
    aspect="auto",
    vmin=0,
)
axes[0].set_title(r"Converged density $\rho_\mathrm{conv}$")
plt.colorbar(im0, ax=axes[0], label=r"$\rho$ / $e\,\mathrm{bohr}^{-3}$")

# Panels 2-3: delta densities
delta_vmax = 0.025
for ax, delta_rho, title in [
    (axes[1], rho_ml - rho_conv, r"$\Delta\rho$: ML $-$ converged"),
    (axes[2], rho_sad - rho_conv, r"$\Delta\rho$: SAD $-$ converged"),
]:
    im = ax.imshow(
        delta_rho,
        extent=[x_grid[0], x_grid[-1], s_grid[0], s_grid[-1]],
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        vmin=-delta_vmax,
        vmax=delta_vmax,
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r"$\Delta\rho$ / $e\,\mathrm{bohr}^{-3}$")

for ax in axes:
    for xi, si, Z in zip(x_atoms, s_atoms, numbers):
        m, c = _marker[Z]
        ax.plot(xi, si, m, ms=_ms[Z], color=c, mec="black", mew=0.7, zorder=5)
    ax.set_xlabel(r"$x$ / Å")
    ax.set_ylabel(r"$s = \sqrt{2}\,y$ / Å")
    ax.spines[["top", "right"]].set_visible(False)

# %%
# The SAD delta-density is large and structured -- the superposition of atomic
# densities misses the redistribution of charge due to bonding. The ML
# prediction is visibly much closer to the converged density: residuals are
# small and featureless, concentrated near the nuclear cores where the
# auxiliary basis truncation is most severe.

# %%
# Density parity plots
# --------------------
#
# For a quantitative comparison we evaluate each density on the DFT
# quadrature grid and plot the predicted value against the converged
# reference point-by-point. The normalised mean absolute error (NMAE)
# summarises the agreement over the whole grid with a single number:
#
# .. math::
#
#    \text{NMAE} \;=\;
#    \frac{\int |\tilde\rho(\mathbf{r}) - \rho_\mathrm{conv}(\mathbf{r})|
#    \,\mathrm{d}\mathbf{r}}
#    {\int \rho_\mathrm{conv}(\mathbf{r})\,\mathrm{d}\mathbf{r}}
#    \times 100\%\;.
#
# The RI reference sets the achievable floor for this auxiliary basis;
# the ML prediction should approach it, while the SAD guess will be
# substantially worse.

# The quadrature grid is already built inside mf_conv; reuse it.
grid_coords = mf_conv.grids.coords  # (npts, 3), Bohr
grid_weights = mf_conv.grids.weights  # (npts,)

# AO values at every grid point — evaluated once, shared across all DMs.
ao_at_grid = mol.eval_gto("GTOval_sph", grid_coords)  # (npts, nao)


def _dm_to_rho(dm):
    """ρ(r) = Σ_μν D_μν φ_μ(r) φ_ν(r) evaluated on the pre-built grid."""
    return np.einsum("pi,ij,pj->p", ao_at_grid, dm, ao_at_grid)


rho_conv_grid = _dm_to_rho(dm_conv)
rho_sad_grid = _dm_to_rho(dm_sad)
rho_ri_grid = _dm_to_rho(dm_ri)
rho_ml_grid = _dm_to_rho(dm_ml)

print(
    f"Electrons (converged): {grid_weights @ rho_conv_grid:.4f}  "
    f"(expected {mol.nelectron})"
)

# %%
# Only plot grid points where the converged density exceeds a small threshold
# to avoid crowding in the exponentially decaying tail far from the molecule.

_mask = rho_conv_grid > 1e-4

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True, dpi=120)
_ref = rho_conv_grid[_mask]

for ax, rho_x, color, label in [
    (axes[0], rho_sad_grid, "C7", "SAD"),
    (axes[1], rho_ri_grid, "C0", "RI reference"),
    (axes[2], rho_ml_grid, "C1", "ML prediction"),
]:
    nmae = nmae_percent(rho_x, rho_conv_grid, grid_weights)
    ax.scatter(
        _ref,
        rho_x[_mask],
        s=0.8,
        alpha=0.25,
        color=color,
        rasterized=True,
        linewidths=0,
    )
    lo, hi = _ref.min(), _ref.max()
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=0.9, ls="--", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho_\mathrm{conv}$ / $e\,\mathrm{bohr}^{-3}$")
    ax.set_ylabel(r"$\tilde{\rho}$ / $e\,\mathrm{bohr}^{-3}$")
    ax.set_title(f"{label}\nNMAE = {nmae:.2f}%")
    ax.spines[["top", "right"]].set_visible(False)

# %%
# Zero-shot electronic properties
# --------------------------------
#
# Observables that are linear functionals of the density can be evaluated
# from any density matrix without an SCF. Here we compare the electric dipole
# moment computed from each of the three initial density matrices against the
# converged reference. The dipole is particularly sensitive to the quality of
# the density because it measures the first moment of the charge distribution
# globally across the molecule.
#
# The DFT total energy is *quadratic* in the density error (the energy is
# stationary at the ground-state density), so it is far less discriminating --
# all three initial guesses give virtually the same value when the SCF is run
# to convergence.

mf_props = dft.RKS(mol)
mf_props.xc = XC

dipole_data = {}
for name, dm in [
    ("SAD", dm_sad),
    ("RI reference", dm_ri),
    ("ML prediction", dm_ml),
    ("Converged", dm_conv),
]:
    dipole_data[name] = mf_props.dip_moment(dm=dm, verbose=0)  # Debye

# %%
# Visualise with chemiscope: each frame corresponds to one density-matrix
# choice; the dipole arrow shows its direction and magnitude.

frames = []
for name, _ in [
    ("SAD", dm_sad),
    ("RI reference", dm_ri),
    ("ML prediction", dm_ml),
    ("Converged", dm_conv),
]:
    frame = atoms.copy()
    frame.info["density"] = name
    frame.info["dipole"] = dipole_data[name]
    frames.append(frame)

dipole_arrows = chemiscope.ase_vectors_to_arrows(frames, "dipole", scale=1.0)
dipole_arrows["parameters"]["global"]["color"] = "#e07b00"

chemiscope.show(
    frames,
    shapes={"dipole": dipole_arrows},
    properties={
        "density": [f.info["density"] for f in frames],
        "|dipole| / D": [np.linalg.norm(f.info["dipole"]) for f in frames],
    },
    settings=chemiscope.quick_settings(
        trajectory=True,
        structure_settings={"shape": ["dipole"]},
    ),
)

# %%
# The SAD density matrix has a near-zero dipole moment: the superposition of
# spherical atomic densities is centrosymmetric and entirely misses the
# bond-polarisation physics. The RI reference recovers the correct dipole
# to within the auxiliary basis truncation error. The ML prediction lies
# between the two -- a significant improvement over SAD -- and the residual
# error reflects the finite accuracy of the model on out-of-training-set
# geometries.

print("Dipole magnitudes / Debye:")
conv_mag = np.linalg.norm(dipole_data["Converged"])
for name, d in dipole_data.items():
    mag = np.linalg.norm(d)
    err = abs(mag - conv_mag)
    print(f"  {name:15s}: {mag:.3f} D   (error vs. converged: {err:.3f} D)")

# %%
