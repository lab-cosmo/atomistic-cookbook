r"""
Phonon dispersions with uncertainty quantification
===================================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo/>`_

This recipe demonstrates how to compute phonon band structures with
uncertainty estimates from MLIP ensembles, using BeO in the wurtzite
structure as a test case.

The point-edge transformer (`PET
<https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_)
is an *unconstrained* architecture: it achieves a high degree of symmetry
compliance through data augmentation during training, rather than enforcing
equivariance by construction (see also the `PET-MAD recipe <pet-mad.html>`_
for more details). As a consequence, the predicted potential energy surface
is slightly asymmetric, meaning that its minimum does not sit exactly at the ideal
high-symmetry geometry.

A natural question is whether this small symmetry breaking has any practical
impact on computed phonon dispersions. We address this by comparing two
relaxation strategies:

1. **Symmetry-constrained**: geometry optimized while maintaining
   :math:`P6_3mc` symmetry via ASE's ``FixSymmetry`` constraint, which
   projects forces and stresses onto the symmetry-invariant subspace.
2. **Unconstrained**: geometry optimized freely, allowing the structure to
   settle at the MLIP's (slightly asymmetric) energy minimum.

Both structures are then evaluated along the *same* high-symmetry
:math:`\mathbf{q}`-path and their band structures overlaid, so that any
differences can only arise from the structural relaxation, not from k-path
artifacts. Ensemble uncertainty quantification via
`uqphonon <https://github.com/ppegolo/uqphonon>`_ further provides
confidence intervals on each phonon branch.

The ensemble is based on the *last-layer prediction rigidity* (LLPR) approach
(`Bigi et al., 2024 <https://arxiv.org/abs/2403.02251>`_; see also the `PET-MAD UQ
recipe <pet-mad-uq.html>`_ for more
details on LLPR uncertainty quantification).

The `uqphonon` package wraps `phonopy <https://phonopy.github.io/phonopy/>`_
and `i-PI <https://ipi-code.org>`_ to automate the workflow: generate
finite-displacement supercells, evaluate ensemble forces through the LLPR
committee, and compute the resulting phonon band structures with per-mode
standard deviations.
"""

# %%
#
# We start by importing the required packages. To run this recipe you
# need `upet <https://github.com/lab-cosmo/pet-mad>`_,
# `uqphonon <https://github.com/ppegolo/uqphonon>`_ (ensemble
# phonon workflow), and `spglib <https://spglib.github.io/spglib/>`_
# (symmetry detection).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ase.build import bulk
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from pathlib import Path
import tempfile

import spglib
import upet
from upet.calculator import UPETCalculator
from uqphonon import PhononEnsemble

# sphinx_gallery_thumbnail_number = 1

# %%
# Setup
# -----
#
# We use the extra-small (XS) variant of
# `PET-MAD v1.5.0 <https://arxiv.org/abs/2503.14118>`_, which covers
# 102 elements and is fast enough to run phonon calculations on CPU.
# The ``upet`` package provides a convenient ASE calculator interface and
# a utility to export ``metatomic`` models for use with ``i-PI``.

SYSTEM = "BeO"
CRYSTAL_STRUCTURE = "wurtzite"

FMAX = 1e-4  # eV/Å, force convergence threshold
STEPS = 500  # max optimization steps

SUPERCELL = (3, 3, 3)
DELTA = 0.02  # Å, displacement amplitude for force constants

DEVICE = "cpu"

MODEL_BASE = "pet-mad"
MODEL_SIZE = "xs"
MODEL_VERSION = "1.5.0"
MODEL_NAME = f"{MODEL_BASE}-{MODEL_SIZE}"

# %%
# Explicit :math:`\mathbf{q}`-path definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The wurtzite Brillouin zone has the same topology as HCP. We define
# the standard :math:`\Gamma`-M-K-:math:`\Gamma`-A-L-H-A path using
# explicit fractional reciprocal-space coordinates. This avoids relying
# on ASE's automatic path finder and lets us impose the *same* path on both
# structures, so that the comparison is fair.

G = np.array([0.0, 0.0, 0.0])
M = np.array([0.5, 0.0, 0.0])
K = np.array([1 / 3, 1 / 3, 0.0])
A = np.array([0.0, 0.0, 0.5])
L = np.array([0.5, 0.0, 0.5])
H = np.array([1 / 3, 1 / 3, 0.5])

SPECIAL_LABELS = ["$\\Gamma$", "M", "K", "$\\Gamma$", "A", "L", "H", "A"]


def get_band(q_start, q_stop, N):
    """Linear interpolation between two q-points."""
    return np.array([q_start + (q_stop - q_start) * i / (N - 1) for i in range(N)])


N_KPOINTS = 50

BANDS = [
    get_band(G, M, N_KPOINTS),
    get_band(M, K, N_KPOINTS),
    get_band(K, G, N_KPOINTS),
    get_band(G, A, N_KPOINTS),
    get_band(A, L, N_KPOINTS),
    get_band(L, H, N_KPOINTS),
    get_band(H, A, N_KPOINTS),
]


# %%
# Helper functions
# ^^^^^^^^^^^^^^^^
#
# ``report_symmetry`` detects the space group at two tolerances using
# ``spglib``: a loose one (:math:`10^{-2}`) that is forgiving of small
# numerical noise, and a tight one (:math:`10^{-6}`) that reveals genuine
# symmetry breaking.
#
# ``compute_phonons_with_uncertainty`` wraps the three-step ``uqphonon``
# workflow: (i) generate symmetrically displaced supercells with ``phonopy``,
# (ii) evaluate ensemble forces via ``i-PI``, and (iii) compute the
# phonon band structure from each ensemble member's force constants.
# The resulting ``PhononEnsemble`` object stores one
# `phonopy <https://phonopy.github.io/phonopy/>`_ band structure per
# committee member, plus the mean-force band structure.


def report_symmetry(atoms, label=""):
    """Detect and report space group using spglib."""
    spglib_cell = (
        atoms.get_cell(),
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    sg_loose = spglib.get_spacegroup(spglib_cell, symprec=1e-2)
    sg_tight = spglib.get_spacegroup(spglib_cell, symprec=1e-6)
    print(
        f"{label:20s}  "
        f"loose (1e-2): {str(sg_loose):15s}  "
        f"tight (1e-6): {str(sg_tight)}"
    )


def compute_phonons_with_uncertainty(atoms, model_path, label=""):
    """Compute phonon band structure using uqphonon ensemble."""
    print(f"\n{label}")
    print("-" * len(label))

    ensemble = PhononEnsemble(
        atoms,
        supercell_matrix=np.diag(SUPERCELL),
        model=str(model_path),
        device=DEVICE,
        primitive_matrix=np.eye(3),
    )
    ensemble.compute_displacements(distance=DELTA)
    print("Displacements computed")

    workdir = Path(tempfile.gettempdir()) / f"uqphonon_{label.replace(' ', '_')}"
    ensemble.run_forces(workdir=workdir)
    n_disp, n_ens, n_atoms, _ = ensemble.forces.shape
    print(f"Ensemble forces: {n_ens} members, {n_disp} displacements")

    ensemble.compute_bands(bandpath=BANDS, npoints=N_KPOINTS, labels=SPECIAL_LABELS)
    print("Band structure computed")

    return ensemble, n_ens


# %%
# Building the initial structure
# ------------------------------
#
# We start from a rough BeO wurtzite cell (space group :math:`P6_3mc`,
# #186) built with ``ase.build.bulk``. The lattice parameters ``a=3.0``,
# ``c=5.0`` are intentionally approximate: the optimizer will refine them.

atoms_ref = bulk(SYSTEM, crystalstructure=CRYSTAL_STRUCTURE, a=3.0, c=5.0)

calc = UPETCalculator(
    model=MODEL_BASE + "-" + MODEL_SIZE,
    device=DEVICE,
    dtype="float64",
    version=MODEL_VERSION,
)

# %%
# Constrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The ``FixSymmetry`` constraint projects forces and stresses onto the
# symmetry-invariant subspace at every optimization step, so the structure
# cannot drift away from :math:`P6_3mc`. We wrap the atoms in a
# ``FrechetCellFilter`` to allow simultaneous optimization of atomic
# positions and cell parameters.

atoms_const = atoms_ref.copy()
atoms_const.set_constraint(FixSymmetry(atoms_const))
atoms_const.calc = calc

opt_const = LBFGS(
    FrechetCellFilter(atoms_const, mask=[True] * 3 + [False] * 3), logfile=None
)
opt_const.run(fmax=FMAX, steps=STEPS)

print(f"Converged in {opt_const.nsteps} steps")
print(f"  Energy: {atoms_const.get_potential_energy():.6f} eV")
report_symmetry(atoms_const, "Constrained structure")

atoms_const.set_constraint(None)

# %%
# Unconstrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here we let the optimizer freely minimize the energy without any symmetry
# enforcement. Because the PET PES is slightly asymmetric, the optimizer
# may settle at a geometry that no longer belongs to :math:`P6_3mc` at tight
# tolerance, even though the energy difference is negligible.

atoms_unconst = atoms_ref.copy()
atoms_unconst.calc = calc

opt_unconst = LBFGS(FrechetCellFilter(atoms_unconst), logfile=None)
opt_unconst.run(fmax=FMAX, steps=STEPS)

print(f"Converged in {opt_unconst.nsteps} steps")
print(f"  Energy: {atoms_unconst.get_potential_energy():.6f} eV")
report_symmetry(atoms_unconst, "Unconstrained structure")

# %%
# Exporting the model for ensemble calculations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``uqphonon`` drives force evaluations through ``i-PI``, which requires a
# TorchScript-exported model. The ``upet.save_upet`` utility handles the
# export.

model_path = Path("model.pt")
upet.save_upet(
    model=MODEL_BASE,
    size=MODEL_SIZE,
    version=MODEL_VERSION,
    output=str(model_path),
)
print(f"Model saved to {model_path}")

# %%
# Phonon band structures from the ensemble
# -----------------------------------------
#
# We now compute phonon dispersions for both relaxed structures using the
# LLPR committee model. Each displaced supercell is evaluated by all
# ensemble members in a single forward pass (the members share everything
# except the last linear layer, so the overhead is negligible). This yields
# a distribution of force-constant matrices, and the phonon frequencies at
# every :math:`\mathbf{q}`-point come with a standard-deviation estimate
# that reflects the *epistemic* uncertainty of the model.
#
# Both calculations use the **same** :math:`\mathbf{q}`-path
# (:math:`\Gamma`-M-K-:math:`\Gamma`-A-L-H-A) defined above. This is
# essential for a meaningful comparison: if each structure were evaluated
# along its own auto-detected path, small differences in the reciprocal
# lattice vectors would shift the high-symmetry points and make the curves
# impossible to overlay.

ensemble_const, n_ens_const = compute_phonons_with_uncertainty(
    atoms_const,
    model_path,
    label="Phonons: constrained structure",
)

ensemble_unconst, n_ens_unconst = compute_phonons_with_uncertainty(
    atoms_unconst,
    model_path,
    label="Phonons: unconstrained structure",
)

# %%
# Comparison
# ----------
#
# We overlay the two dispersions on a single plot. Solid lines show the
# mean phonon frequencies (averaged over the ensemble), and shaded regions
# indicate :math:`\pm 1\sigma` uncertainty bands.
#
# If the symmetry-breaking distortions from the unconstrained relaxation
# had a significant effect on the dynamics, the red and blue curves would
# diverge. Instead, they are essentially indistinguishable, confirming
# that the PET symmetry-breaking errors are small enough not to affect
# phonon predictions in a meaningful way.

fig, ax = plt.subplots(figsize=(10, 6))

ensemble_const.plot(
    ax=ax,
    mode="mean+std",
    unit="THz",
    color="tab:blue",
    std_alpha=0.2,
)

ensemble_unconst.plot(
    ax=ax,
    mode="mean+std",
    unit="THz",
    color="tab:red",
    std_alpha=0.2,
)

ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.4)

legend_elements = [
    Patch(facecolor="tab:blue", alpha=0.5, label="Constrained (FixSymmetry)"),
    Patch(facecolor="tab:red", alpha=0.5, label="Unconstrained"),
]
ax.legend(handles=legend_elements, fontsize=11, loc="upper right")

ax.set_ylabel("Frequency (THz)", fontsize=13)
ax.set_xlabel("Wave vector", fontsize=13)

plt.tight_layout()
plt.show()

# %%
# Conclusions
# -----------
#
# The constrained and unconstrained dispersions overlap almost perfectly,
# and the ensemble uncertainty bands are narrow, indicating high model
# confidence. This validates two practical points:
#
# * ``FixSymmetry`` is a useful tool for keeping a structure at a known
#   high-symmetry phase---especially for automated workflows where the
#   :math:`\mathbf{q}`-path is derived from the space group---but it is
#   not *required* for accurate phonon predictions with PET-MAD.
# * Ensemble uncertainty quantification via ``uqphonon`` provides a
#   direct, per-mode measure of the model's epistemic uncertainty,
#   going beyond simple benchmarking against reference data.
#
# For more information on PET-MAD and its uncertainty quantification
# capabilities, see `Mazitov et al., 2025 <https://arxiv.org/abs/2503.14118>`_
# and the `PET-MAD UQ recipe <pet-mad-uq.html>`_.
