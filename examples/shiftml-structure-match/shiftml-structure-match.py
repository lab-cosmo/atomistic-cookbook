r"""
NMR-driven structure determination with ShiftML3
================================================

:Authors:
    Joseph W. Abbott `@jwa7 <https://github.com/jwa7>`_;
    Matthias Kellner `@bananempampe <https://github.com/bananempampe>`_

This recipe shows how to use a fast machine-learning surrogate for *ab initio*
NMR chemical-shielding calculations --
`ShiftML3 <https://github.com/lab-cosmo/shiftml>`_ -- to pick out an
experimentally observed crystal structure from a pool of candidate polymorphs.
We work through the cocaine benchmark from
`Cordova et al., Nat. Commun. (2018)
<https://www.nature.com/articles/s41467-018-06972-x>`_, where the pool consists
of geometry-optimized candidate structures and a measured solid-state
:sup:`1`\ H spectrum is available for comparison.

The basic workflow of NMR crystallography is:

1. enumerate candidate structures (typically by computational crystal-structure
   prediction workflows);
2. compute :sup:`1`\ H chemical shieldings for each candidate (traditionally
   with GIPAW-DFT, here with the ML-accelerated ShiftML3 model);
3. assign each predicted shielding to one of the experimentally observed
   chemical-shift peaks;
4. score each candidate against experiment via a calibrated RMSE;
5. take the structure with the lowest RMSE as the best match.

The chemical *shielding* is a rank-2 Cartesian tensor; its isotropic part
(the trace divided by three) is rotation-invariant and is the quantity we
use here for structure matching. For background on running ShiftML3 itself
-- calculator setup, ensemble uncertainties, anisotropic tensor predictions
-- see the companion recipe `Computing NMR shielding tensors using ShiftML
<https://atomistic-cookbook.org/examples/shiftml/shiftml-example.html>`_.
For background on rotationally equivariant tensor properties more
generally, see the `polarizability recipe
<https://atomistic-cookbook.org/examples/polarizability/polarizability.html>`_
and the `rotating-equivariants recipe
<https://atomistic-cookbook.org/examples/rotate-equivariants/rotate-equivariants.html>`_.
"""

# %%
# Chemical shifts, chemical shieldings, and structure matching
# ------------------------------------------------------------
#
# A solid-state NMR experiment on a powdered organic solid reports a set of *chemical
# shifts* :math:`\delta_i` (in ppm), one for each magnetically distinct nuclear site
# :math:`i`. Computationally, what is naturally accessible is the *chemical shielding*
# :math:`\sigma_i` -- the response of the local electronic structure to an applied
# magnetic field. The two are related by an approximately linear calibration,
#
# .. math::
#
#    \delta_i \;=\; a\,\sigma_i + b\;,
#
# where the slope :math:`a \approx -1` (since shielding *reduces* the resonance
# frequency, while chemical shift is measured relative to a reference compound such as
# TMS for :sup:`1`\ H), and :math:`b` is the shielding of the reference compound. The
# constants :math:`a` and :math:`b` depend on the level of theory used to compute
# :math:`\sigma`, and are usually obtained by linear regression against a benchmark of
# known experimental shifts.
#
# Given a candidate structure :math:`X`, we predict :math:`\{\sigma_i(X)\}` for the
# relevant nuclei, calibrate via the linear form above to obtain predicted shifts
# :math:`\{\delta_i^{\,\mathrm{pred}}(X)\}`, and compute the RMSE against the assigned
# experimental shifts:
#
# .. math::
#
#    \mathrm{RMSE}(X) \;=\; \sqrt{ \frac{1}{N}\,
#    \sum_{i=1}^{N}\!\left(\delta_i^{\,\mathrm{pred}}(X)
#                          - \delta_i^{\,\mathrm{exp}}\right)^{2} }\;.
#
# The candidate :math:`X^\star` minimising this score is taken as the best match. The
# reason this works as a structural fingerprint is that the chemical shielding at a
# nucleus is highly sensitive to changes in its local environment,  so different
# polymorphs produce visibly different :sup:`1`\ H spectra.
#
# Until recently this workflow relied on GIPAW-DFT shielding calculations, at a cost of
# typically hundreds of CPU-hours per candidate. Instead ShiftML3, an ensemble of 8 PET
# models trained on shieldings computed at the PBE/GIPAW level for ~14k molecular
# crystals from the Cambridge Structural Database (`Kellner et al., J. Phys. Chem. Lett.
# (2025) <https://doi.org/10.1021/acs.jpclett.5c01819>`_), delivers comparable accuracy
# in only a matter of seconds per candidate.

# %%
# The cocaine benchmark dataset
# -----------------------------
#
# We use a curated set of 29 cocaine candidate crystal structures with pre-computed
# GIPAW shieldings, provided together with the original ShiftML release (`Paruzzo et
# al., Nat. Commun. (2018) <https://www.nature.com/articles/s41467-018-06972-x>`_) and
# hosted on Materials Cloud. The same archive is used by the
# `introductory ShiftML recipe
# <https://atomistic-cookbook.org/examples/shiftml/shiftml-example.html>`_.

# %%
import os
import zipfile

import chemiscope
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from atomistic_cookbook_utils import download_with_retry
from shiftml.ase import ShiftML
from sklearn.metrics import root_mean_squared_error

# Download the reference chemical shieldings
filename = "ShiftML_poly.zip"
download_with_retry(
    "https://archive.materialscloud.org/records/j2fka-sda13/files/ShiftML_poly.zip",
    filename,
)

with zipfile.ZipFile(filename, "r") as zip_ref:
    for file in ["ShiftML_poly/Cocaine/cocaine_QuantumEspresso.xyz"]:
        target = os.path.basename(file)
        with zip_ref.open(file) as source, open(target, "wb") as dest:
            dest.write(source.read())

# Load the frames
frames = read("cocaine_QuantumEspresso.xyz", ":")
print(f"Loaded {len(frames)} candidate cocaine structures")
print(f"Atoms per unit cell: {len(frames[0])}")
print(f"Per-atom arrays available: {list(frames[0].arrays.keys())}")

# %%
# 
# Browse the candidate pool interactively with chemiscope

chemiscope.show(
    frames,
    mode="structure",
    settings=chemiscope.quick_settings(trajectory=True, structure_settings={"unitCell": True})
)


# %%
# 
# Each structure carries a ``CS`` array of GIPAW-computed shieldings, one entry per
# atom. We will use those values both as the reference DFT predictions and to illustrate
# that ShiftML3 reproduces DFT-level accuracy on this task at a fraction of the cost.

print("atom types (every tenth atom; 1 = H, 6 = C, 8 = O):", frames[0].arrays["numbers"][::10])
print("chemical shieldings (every tenth atom):", frames[0].arrays["CS"][::10])  # GIPAW shieldings for the first 10 atoms of the first candidate

# %%
# Experimental :sup:`1`\ H shifts and atom labelling
# ---------------------------------------------------
#
# Cocaine has 21 protons, but only 17 chemically distinct proton sites. The experimental
# :sup:`1`\ H spectrum of cocaine, assigned to its 17 chemically distinct proton sites,
# is reproduced in the supplementary information of Cordova et al. 
# 
# The list below pairs an experimental shift (in ppm) with the corresponding 1-based
# atom index in the asymmetric unit. Entries like ``"11,12,13"`` denote chemically
# equivalent protons (e.g. a rotating methyl) whose predicted shieldings should be
# averaged before being compared with the single observed peak.

list_atom = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11,12,13", "14", "15", "16", "17", "18", "19,20,21",
]  # fmt: skip
list_cs_exp = np.array(
    [3.76, 3.78, 5.63, 3.32, 3.49, 3.06, 2.91, 3.38, 2.56, 2.12,
     1.04, 8.01, 8.01, 8.01, 8.01, 8.01, 3.78]
)  # fmt: skip
N_H_PER_MOL = max(int(s) for entry in list_atom for s in entry.split(","))
print(f"Comparing {len(list_atom)} assigned 1H environments")


# %%
# The cocaine crystal contains two molecules per unit cell, so we take the
# first molecule's hydrogens and average shieldings over symmetry-equivalent
# groups (e.g. the three methyl protons ``"11,12,13"`` share a single
# observed NMR peak).


def assign_shieldings(per_h_shieldings, list_atom, n_h_per_mol=N_H_PER_MOL):
    """Pick one cocaine molecule's H shieldings out of the unit cell and
    average over the symmetry-equivalent groups listed in ``list_atom``."""
    per_mol = per_h_shieldings.reshape(n_h_per_mol, -1)[:, 0]
    out = []
    for atom_string in list_atom:
        idx = [int(s) - 1 for s in atom_string.split(",")]
        out.append(per_mol[idx].mean())
    return np.array(out)


# %%
# Predicting shieldings with ShiftML3
# -----------------------------------
#
# We instantiate the ShiftML3 surrogate model as the calculator for the shieldings
# tensors. The ``get_cs_iso`` method computes this tensor and returns the isotropic part
# (the mean trace) of the passed candidate ``frame`` for every atom in the unit cell.
#
# As we are only interested in the hydrogen spectrum in this exercise, we filter the
# redictions only the the H atoms (atomic number 1) before passing them to the
# assignment function above.
#
# Inference on the full set of 29 cocaine candidates takes a matter of seconds on a
# single CPU/GPU. Remember that the GIPAW reference for the same task takes hundreds of
# CPU-hours.

calculator = ShiftML("ShiftML3")

X_sml = []  # ShiftML3-predicted shieldings
X_gipaw = []  # GIPAW reference shieldings

for frame in frames:
    is_h = frame.get_atomic_numbers() == 1  # mask for H atoms

    sml = calculator.get_cs_iso(frame).ravel()[is_h]  # model predictions
    X_sml.append(assign_shieldings(sml, list_atom))

    gipaw = frame.arrays["CS"][is_h]  # GIPAW reference
    X_gipaw.append(assign_shieldings(gipaw, list_atom))

X_sml = np.array(X_sml)  # (n_candidates, n_sites)
X_gipaw = np.array(X_gipaw)


# %%
# Calibrating shielding to shift
# ------------------------------
#
# To compare predictions to experiment we need to map shieldings to shifts via
# :math:`\delta = a\,\sigma + b`. Two parameters, two choices to make.
#
# *Slope.* Chemical shift and chemical shielding are defined so that :math:`\delta
# \approx -\sigma + \sigma_{\mathrm{ref}}`, i.e. the slope is *exactly* :math:`-1` if
# both quantities are perfectly consistent. Letting the slope float effectively rescales
# the predicted spectrum, which lets every candidate cheat its way to a better RMSE and
# erodes the discrimination between polymorphs. We therefore fix :math:`a = -1`.
#
# *Intercept.* :math:`b` is the shielding of the reference compound used experimentally
# (TMS for :sup:`1`\ H). It is not known *a priori* for an arbitrary candidate pool, so
# we fit it per-structure --- equivalent to saying we compare *spectral patterns*, not
# absolute shifts. Concretely, for each candidate the intercept is set to the value that
# makes the predicted and experimental shifts coincide on average, :math:`b =
# \overline{\delta^{\,\mathrm{exp}}} - a\,\overline{\sigma^{\,\mathrm{pred}}}`.
#
# (An alternative is to apply a single *global* calibration -- e.g. the values
# determined by Kellner et al. by regressing predicted ShiftML3 shieldings against
# experimental shifts on a held-out benchmark of organic crystals. That is preferred
# when you also care about absolute shift accuracy; for ranking candidates by spectral
# pattern, the per-structure scheme above gives the same answer with one fewer parameter
# to argue about.)


def calibrated_rmse(X_per_candidate, Y_exp, slope=-1.0):
    """RMSE for each candidate after fitting the intercept that aligns the
    predicted and experimental shift means."""
    rmses = []
    for X in X_per_candidate:
        intercept = np.mean(Y_exp) - slope * np.mean(X)
        Y_pred = slope * X + intercept
        rmses.append(root_mean_squared_error(Y_pred, Y_exp))
    return np.array(rmses)

# Calibrate the shieldings -> shifts
rmse_sml = calibrated_rmse(X_sml, list_cs_exp)
rmse_gipaw = calibrated_rmse(X_gipaw, list_cs_exp)

# Find the best candidate (lowest RMSE vs experiment)
best = int(np.argmin(rmse_sml))
print(f"Best ShiftML3 match: candidate #{best} (RMSE = {rmse_sml[best]:.3f} ppm)")
print(
    f"Best GIPAW   match: candidate #{int(np.argmin(rmse_gipaw))} "
    f"(RMSE = {rmse_gipaw.min():.3f} ppm)"
)


# %%
# The lollipop diagram
# --------------------
#
# The canonical visualisation for this task is the *lollipop diagram*: RMSE vs candidate
# index, plotted as a stem plot. We overlay the ShiftML3 prediction (orange) and the
# GIPAW reference (blue). The shaded band at :math:`0.33 \pm 0.16\,\mathrm{ppm}` marks
# the irreducible spread of GIPAW-vs-experiment errors estimated by `Salager et. al,
# JACS (2010) <https://doi.org/10.1021/ja909449k>`_ across a benchmark of molecular
# solids: candidates with RMSEs in this range are, statistically, indistinguishable from
# experiment.

candidate_idx = np.arange(len(frames))
dx = 0.18  # horizontal offset so the two methods sit side-by-side

fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True, dpi=120)

band_lo, band_hi = 0.33 - 0.16, 0.33 + 0.16
ax.axhspan(
    band_lo,
    band_hi,
    color="0.85",
    alpha=0.7,
    zorder=0,
    label=r"DFT vs experiment noise floor ($0.33 \pm 0.16$ ppm)",
)

ax.vlines(candidate_idx - dx, 0, rmse_gipaw, color="C0", lw=1.4, alpha=0.85, zorder=2)
ax.vlines(candidate_idx + dx, 0, rmse_sml, color="C1", lw=1.4, alpha=0.85, zorder=2)
ax.scatter(
    candidate_idx - dx,
    rmse_gipaw,
    color="C0",
    label="GIPAW (DFT reference)",
    s=28,
    edgecolor="white",
    lw=0.7,
    zorder=3,
)
ax.scatter(
    candidate_idx + dx,
    rmse_sml,
    color="C1",
    label="ShiftML3",
    s=28,
    edgecolor="white",
    lw=0.7,
    zorder=3,
)

ax.set_xlabel("Candidate structure index")
ax.set_ylabel(r"$^1$H shift RMSE / ppm")
ax.set_title("Ranking cocaine polymorph candidates by $^1$H NMR fingerprint")
ax.set_xticks(candidate_idx[::2])
ax.set_xlim(-0.7, len(frames) - 0.3)
ax.set_ylim(0, max(rmse_sml.max(), rmse_gipaw.max()) * 1.15)
ax.grid(axis="y", color="0.92", lw=0.6, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=9)


# %%
# Two things stand out:
#
# 1. **ShiftML3 tracks the DFT reference candidate-by-candidate.** The
#    orange stems closely follow the blue ones, confirming that the model
#    has not just learned the average behaviour of the training set, but
#    also the polymorph-specific deviations from it.
# 2. **The minimum is unambiguous.** The best-matching candidate sits well
#    below the rest of the pool, inside the experimental noise band -- this
#    is the structure that best reproduces the measured spectrum. The other
#    candidates carry markedly larger RMSEs and can be confidently rejected.

# %%
# Parity plots: best vs worst candidate
# -------------------------------------
#
# A useful sanity check is to look at the predicted-vs-experimental
# :sup:`1`\ H shifts site-by-site. For the *best* candidate we expect
# points scattered tightly around :math:`y=x`; for the *worst* candidate
# the scatter is much larger and -- importantly -- the residuals are
# structured rather than random. That structure is what the RMSE detects:
# specific protons sit in chemically wrong environments, so their shifts
# are systematically off.


def calibrate_one(X, Y, slope=-1.0):
    intercept = np.mean(Y) - slope * np.mean(X)
    return slope * X + intercept


def parity(ax, X_sml_i, X_gipaw_i, rmse_sml_i, rmse_gipaw_i, title):
    lo, hi = 0.5, 9.0
    ax.plot([lo, hi], [lo, hi], color="0.55", ls="--", lw=0.9, zorder=1)
    ax.scatter(
        list_cs_exp,
        calibrate_one(X_gipaw_i, list_cs_exp),
        s=60,
        marker="o",
        facecolor="C0",
        edgecolor="white",
        lw=0.8,
        zorder=3,
        label=f"GIPAW   (RMSE {rmse_gipaw_i:.2f} ppm)",
    )
    ax.scatter(
        list_cs_exp,
        calibrate_one(X_sml_i, list_cs_exp),
        s=60,
        marker="s",
        facecolor="C1",
        edgecolor="white",
        lw=0.8,
        zorder=3,
        label=f"ShiftML3 (RMSE {rmse_sml_i:.2f} ppm)",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"experimental $\delta(^1\mathrm{H})$ / ppm")
    ax.set_title(title)
    ax.grid(color="0.92", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


worst = int(np.argmax(rmse_sml))

fig, axes = plt.subplots(
    1, 2, figsize=(9.4, 4.8), sharey=True, constrained_layout=True, dpi=120
)
parity(
    axes[0],
    X_sml[best],
    X_gipaw[best],
    rmse_sml[best],
    rmse_gipaw[best],
    f"Best match (candidate #{best})",
)
parity(
    axes[1],
    X_sml[worst],
    X_gipaw[worst],
    rmse_sml[worst],
    rmse_gipaw[worst],
    f"Worst match (candidate #{worst})",
)
axes[0].set_ylabel(r"predicted $\delta(^1\mathrm{H})$ / ppm")


# %%
# Browsing the candidate pool interactively
# -----------------------------------------
#
# It is instructive to inspect the candidates themselves alongside the RMSE numbers. We
# attach the per-candidate RMSE to each frame and let chemiscope show the pool as a
# structure trajectory linked to a scatter plot of the score.

for f, r_sml, r_gipaw in zip(frames, rmse_sml, rmse_gipaw):
    f.info["rmse_shiftml"] = float(r_sml)
    f.info["rmse_gipaw"] = float(r_gipaw)

chemiscope.show(
    frames,
    properties={
        "ShiftML3 RMSE / ppm": rmse_sml,
        "GIPAW RMSE / ppm": rmse_gipaw,
        "candidate index": candidate_idx,
    },
    settings=chemiscope.quick_settings(
        x="candidate index",
        y="ShiftML3 RMSE / ppm",
        map_color="GIPAW RMSE / ppm",
        trajectory=True,
        structure_settings={"unitCell": True},
    ),
)


# %%
# Calibration choices and their effect
# ------------------------------------
#
# We used a per-structure intercept fit with the slope pinned at :math:`-1`. Two
# alternatives are worth being aware of:
#
# * **Per-structure free fit.** Also let the slope float, fitting it independently for
#   each candidate. Because every candidate gets two free parameters instead of one, the
#   RMSEs shrink overall and the gap between the correct structure and the rest tends to
#   narrow. This makes the ranking less diagnostic and is generally *not* recommended
#   for structure matching.
# * **Global calibration.** Use the slope/intercept obtained by Kellner et al. on a
#   held-out experimental benchmark (:math:`a = -0.9024,\, b = 28.05\,\mathrm{ppm}` for
#   the ShiftML3 :sup:`1`\ H output). Preferred when you also care about absolute
#   :sup:`1`\ H shift accuracy; for the ranking task here it gives the same winner with
#   slightly inflated RMSEs across the pool.
#
#
# Note that in many cases, the regression slope can be attributed to systematic errors
# in the potential energy surface used to generate the candidate pool, or the
# neglectance of finite temperature effects and quantum delocalization of acidic
# protons (`Kellner et al. <https://arxiv.org/abs/2603.06236>`_). Here we use
# geometries relaxed at the PBE-D2 level of theory and at 0K.

# %%
# Outlook
# -------
#
# The same workflow can be repeated for a different cocaine-class molecule, AZD8329,
# whose dataset is included in the same Materials Cloud archive under
# ``ShiftML_poly/AZD/AZD_QuantumEspresso.xyz``. The assignment list and experimental
# shifts for AZD8329 are given in Cordova et al.
#
# Going beyond isotropic shifts, ShiftML3 also predicts the full chemical shielding
# *tensor* for each atom. The introductory ShiftML recipe shows how to extract and
# visualise this tensor with chemiscope ellipsoids; the anisotropic information can be
# incorporated into the structure-matching score whenever experimental CSA values are
# available, typically improving the discrimination between candidates that have similar
# isotropic spectra.
#
# Modern NMR-CSP workflows move to more sophisticated similarity measures between
# computed and experimental values, such as full bayesian treatment, considering also
# prediction uncertainties of the ML model (`Engel et al.
# <https://doi.org/10.1039/C9CP04489B>`_) and metrics that allow for combining NMR
# measurements of different nuclei in the structure selection process
# (`Müller et al. <https://doi.org/10.1039/D4FD00114A>`_).
