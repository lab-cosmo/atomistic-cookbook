r"""
Phonon dispersions with committee uncertainty using kALDo
=========================================================

:Authors: Giuseppe Barbalinardo `@gbarbalinardo <https://github.com/gbarbalinardo/>`_

This recipe computes a phonon dispersion with an uncertainty estimate directly
in `kALDo <https://nanotheorygroup.github.io/kaldo/>`_, using its
``PhononsEnsemble`` API together with the
`PET-MAD <https://arxiv.org/abs/2603.02089>`_ universal machine-learning
potential via the `UPET <https://github.com/lab-cosmo/pet>`_ calculator.

This is a companion to the `phonon dispersions with uncertainty recipe
<https://atomistic-cookbook.org/examples/pet-phonons/pet-phonons.html>`_ by
Paolo Pegolo and Michele Ceriotti, which computes committee-based phonon
uncertainties with `uqphonon <https://github.com/ppegolo/uqphonon>`_ (a wrapper
around `phonopy <https://phonopy.github.io/phonopy/>`_ and
`i-PI <https://ipi-code.org>`_). Here we show the same idea evaluated entirely
within kALDo: kALDo builds the force constants and the phonon spectrum for each
committee member and aggregates them, so no additional phonon backend is
needed.

The uncertainty-quantification methodology is unchanged from that recipe and is
based on the construction of a shallow ensemble
(cf. `Kellner and Ceriotti, 2024
<https://iopscience.iop.org/article/10.1088/2632-2153/ad594a>`_), with the
committee members obtained through the *last-layer prediction rigidity*
framework (LLPR, `Bigi et al., 2024 <https://arxiv.org/abs/2403.02251>`_; see
also the `PET-MAD UQ recipe
<https://atomistic-cookbook.org/examples/pet-mad-uq/pet-mad-uq.html>`_). For a
calibrated committee (e.g. 128 LLPR heads evaluated in a single batched pass),
use ``uqphonon`` as in the companion recipe. Here, to keep the example
self-contained within kALDo, we build a small committee from independent
PET-MAD variants.

The theory and implementation of kALDo are described in
`Barbalinardo et al., J. Appl. Phys. 128, 135104 (2020)
<https://doi.org/10.1063/5.0020443>`_.

We use silicon (diamond) as a test system.

.. note::

   The supercell and displacement settings used here are deliberately small so
   the recipe runs in CI in a few minutes.  Larger supercells give
   better-converged dispersions.
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from ase.constraints import FixSymmetry
from ase.filters import StrainFilter
from ase.optimize import BFGS

from upet.calculator import UPETCalculator
from kaldo.ensemble import PhononsEnsemble
from kaldo.observables.harmonic_with_q import HarmonicWithQ

# %%
# Setup
# -----
#
# We build a small committee from two PET-MAD variants, the extra-small (XS) and
# small (S) models. Treating independent models as committee members gives a
# rougher uncertainty estimate than a calibrated LLPR committee, but it keeps the
# example dependent only on kALDo and UPET. See the companion ``pet-phonons``
# recipe for the calibrated-committee workflow.

DEVICE = "cpu"
SUPERCELL = (3, 3, 3)
KPTS = (5, 5, 5)
DELTA = 3e-2  # Angstrom, finite-difference displacement

members = [
    UPETCalculator(model="pet-mad-xs", device=DEVICE, dtype="float32", version="1.5.0"),
    UPETCalculator(model="pet-mad-s", device=DEVICE, dtype="float32", version="1.5.0"),
]

# %%
# Relaxation
# ----------
#
# We relax the silicon cell with the first model, keeping the diamond symmetry
# with ``FixSymmetry``.  Unconstrained machine-learning potentials only respect
# the crystal symmetry approximately, so constraining the relaxation avoids a
# spuriously symmetry-broken cell.

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = members[0]
atoms.set_constraint(FixSymmetry(atoms))
BFGS(StrainFilter(atoms), logfile=None).run(fmax=1e-4)
atoms.set_constraint(None)
print(f"Optimized lattice parameter: {atoms.cell.cellpar()[0]:.3f} A")

# %%
# Ensemble force constants and phonons
# ------------------------------------
#
# ``PhononsEnsemble.from_calculators`` runs a finite-difference second-order
# calculation for each committee member, projects each set of force constants
# onto the space-group-invariant subspace (``symmetrize=True``), and builds a
# ``Phonons`` object per member.  The symmetrization is important with
# unconstrained models: it removes the small symmetry violations that would
# otherwise break degeneracies near :math:`\Gamma`, so the uncertainty band
# reflects genuine model disagreement rather than symmetry-breaking noise.

ensemble = PhononsEnsemble.from_calculators(
    atoms,
    SUPERCELL,
    members,
    delta_shift=DELTA,
    symmetrize=True,
    kpts=KPTS,
    temperature=300,
    storage="memory",
)

mean, std = ensemble.mean_std("frequency")
print(f"ensemble members: {ensemble.n_members}")
print(f"max frequency std over the k-point mesh: {std.max():.3f} THz")

# %%
# Dispersion with an uncertainty band
# -----------------------------------
#
# Each member is an ordinary kALDo ``Phonons`` object, so we evaluate its
# frequencies at each :math:`\mathbf{q}`-point along a high-symmetry path and
# plot the mean band structure with a shaded plus/minus standard-deviation
# envelope per branch.

X = np.array([0.5, 0.0, 0.5])
G = np.array([0.0, 0.0, 0.0])
L = np.array([0.5, 0.5, 0.5])
n_seg = 60
path = [X + (G - X) * t for t in np.linspace(0, 1, n_seg)] + [
    G + (L - G) * t for t in np.linspace(0, 1, n_seg)[1:]
]
x_axis = np.concatenate([np.linspace(0, 1, n_seg), 1 + np.linspace(0, 1, n_seg)[1:]])


def member_bands(phonons):
    """Frequencies (n_q, n_modes) of one member along the q-path."""
    return np.array(
        [
            HarmonicWithQ(
                np.asarray(q), phonons.forceconstants.second, storage="memory"
            ).frequency.flatten()
            for q in path
        ]
    )


bands = np.array([member_bands(m) for m in ensemble.members])
band_mean = bands.mean(axis=0)
band_std = bands.std(axis=0)

fig, ax = plt.subplots(figsize=(7, 5))
color = "tab:blue"
for b in range(band_mean.shape[1]):
    ax.plot(x_axis, band_mean[:, b], color=color, lw=1.6)
    ax.fill_between(
        x_axis,
        band_mean[:, b] - band_std[:, b],
        band_mean[:, b] + band_std[:, b],
        color=color,
        alpha=0.25,
        linewidth=0,
    )
ax.axhline(0, color="k", lw=0.6, alpha=0.5)
ax.axvline(1, color="k", lw=0.6, alpha=0.5)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["X", r"$\Gamma$", "L"])
ax.set_xlim(0, 2)
ax.set_ylabel("Frequency (THz)")
ax.set_title(r"Si phonon dispersion, PET-MAD committee (mean $\pm$ std)")
plt.tight_layout()
plt.show()

# %%
# The shaded band widens where the two models disagree most (the upper optical
# branches and along :math:`\Gamma`-L), and collapses toward the acoustic sum
# rule at :math:`\Gamma`, where all three acoustic branches go to zero.  For a
# production-quality uncertainty estimate, replace the two independent models
# with a calibrated LLPR committee as shown in the ``pet-phonons`` recipe, and
# increase the supercell size.
