"""
Uncertainty-aware universal potentials calibrated on experiment (PET-UAFD)
==========================================================================

:Authors: Matthias Kellner `@bananenpampe <https://github.com/bananenpampe>`_,
          Teitur Hansen `@teih <https://github.com/teih>`_

This recipe builds a **PET-UAFD** model: an ensemble of universal machine-learning
interatomic potentials (uMLIPs), each a fast surrogate for a different
exchange-correlation (xc) functional, *calibrated against experimental data* using
the uncertainty-aware functional distribution (UAFD) framework.

Introduced in `Kellner, Hansen, et al. (2026) <https://arxiv.org/abs/2604.24607>`_
the method builds on the UAFD method of `Hansen et al., Phys. Rev. B 112, 075412 (2025)
<https://doi.org/10.1103/yhly-wxhv>`_. In essence, UAFD builts an uncertainty-estimator
from the spread between different xc-functionals computations on a datapoint.
Importantly can be used to estimate the error of a calculation *relative to
experiment* — usually the error that actually matters. Replacing the (expensive)
self-consistent DFT functionals with universal `PET
<https://lab-cosmo.github.io/upet/>`_ surrogates makes both the calibration and the
subsequent uncertainty propagation affordable, and allows to study properties at
size and length scales inaccessible to DFT, whilst the calibration can
still be performed on reference datasets of small molecules and solids, and
single-point evaluations.

We

1. fetch the experimental reference datasets (molecular atomization energies; solid
   cohesive energies, lattice constants and bulk moduli);
2. compute each property with every member of an ensemble of PET-XS models;
3. assemble the predictions into a design matrix and fit the (regularized) UAFD
   distribution over functional space,

   .. math::

      \\mathcal{P}(\\mathbf{w}) = \\mathcal{N}(\\mathbf{w}|\\mathbf{w}_0, \\mathbf{K})

4. export the calibrated potential and visualize the predictions with their
   calibrated uncertainties.

The benchmark solids follow `Tran, Stelzl and Blaha, J. Chem. Phys. 144, 204120
(2016) <https://doi.org/10.1063/1.4948636>`_.
"""

# %%
#
# Imports
# -------

import csv
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.calculators.mixing import LinearCombinationCalculator, SumCalculator
from ase.db import connect
from ase.eos import EquationOfState
from ase.optimize import LBFGS
from ase.units import GPa
from atomistic_cookbook_utils import download_with_retry
from scipy.optimize import minimize
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from upet.calculator import UPETCalculator


# Universal MLIPs are not meant for truly isolated atoms, and metatomic emits a
# deprecation warning from its neighbor-list helper; silence both so the rendered
# output stays focused on the science.
warnings.filterwarnings("ignore")

# Run on CPU for reproducibility in CI; set DEVICE = "cuda" if you have a GPU.
DEVICE = "cpu"


# %%
#
# The ensemble basis
# ------------------
#
# The UAFD lives in a space spanned by a small *basis* of functionals; following
# the PET-UAFD paper we use one universal PET-XS model per functional:
#
# ============= ===================== ==========
# Member        ``upet`` model        Functional
# ============= ===================== ==========
# ``PBE``       ``pet-omat-xs``       PBE
# ``PBEsol``    ``pet-omad-xs``       PBEsol
# ``r2SCAN``    ``pet-mad-xs`` v1.5.0 r²SCAN
# ``r2SCAN-D3`` ``pet-mad-xs`` + D3   r²SCAN-D3
# ============= ===================== ==========
#
# The ``r2SCAN-D3`` member adds a `DFT-D3 <https://github.com/dft-d3/simple-dftd3>`_
# dispersion correction (Becke-Johnson damping, r²SCAN parameters) on top of the
# r²SCAN PET model using `torch-dftd
# <https://github.com/pfnet-research/torch-dftd>`_.

MEMBERS = {
    "PBE": dict(model="pet-omat-xs", version="latest"),
    "PBEsol": dict(model="pet-omad-xs", version="latest"),
    "r2SCAN": dict(model="pet-mad-xs", version="1.5.0"),
    "r2SCAN-D3": dict(model="pet-mad-xs", version="1.5.0", d3="r2scan"),
}

# if you want to use more expensive models
# (e.g. the "s" instead of "xs" variants),
# just change the model names here:
# MEMBERS = {
#    "PBE": dict(model="pet-omat-s", version="latest"),
#    "PBEsol": dict(model="pet-omad-s", version="latest"),
#    "r2SCAN": dict(model="pet-mad-s", version="1.5.0"),
#    "r2SCAN-D3": dict(model="pet-mad-s", version="1.5.0", d3="r2scan"),
# }
MEMBER_LABELS = list(MEMBERS)
REF = 0  # reference member (PBE) for the sum-rule shift, see the fitting section


def build_calculators(device=DEVICE):
    """One ASE calculator per ensemble member (sharing cached PET checkpoints)."""
    cache, calculators = {}, {}
    for label, spec in MEMBERS.items():
        key = (spec["model"], spec["version"])
        if key not in cache:
            cache[key] = UPETCalculator(
                model=spec["model"], version=spec["version"], device=device
            )
        if spec.get("d3"):
            d3 = TorchDFTD3Calculator(xc=spec["d3"], damping="bj", device=device)
            calculators[label] = SumCalculator([cache[key], d3])
        else:
            calculators[label] = cache[key]
    return calculators


calculators = build_calculators()


# %%
#
# Fetching the experimental reference data
# ----------------------------------------
#
# Most data is downloaded at runtime: the molecular structures live in an `ASE
# database <https://cmr.fysik.dtu.dk/uafd/uafd.html>`_, and the solid-state
# experimental references (with the prototypes needed to build the structures) come
# as CSV tables. The small ``G3-99.csv`` table of experimental molecular atomization
# targets ships with the recipe.

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
UAFD_DB = os.path.join(DATA_DIR, "uafd.db")
CSV = {
    p: os.path.join(DATA_DIR, f"{p}.csv")
    for p in ("lattice_constants", "cohesive_energies", "bulk_moduli")
}

download_with_retry("https://wiki.fysik.dtu.dk/cmr-files/uafd.db", UAFD_DB)
_CSV_URLS = {
    "lattice_constants": "cabb79a118fe2e413470acdd2696c828",
    "cohesive_energies": "e91701ef549872dc9eb06916ed9885a2",
    "bulk_moduli": "13a24458bde291e7cc8ca860d97d5b94",
}
for name, h in _CSV_URLS.items():
    download_with_retry(
        f"https://cmr.fysik.dtu.dk/_downloads/{h}/{name}.csv", CSV[name]
    )


# %%
#
# Referencing atomization and cohesive energies
# ---------------------------------------------
#
# Atomization and cohesive energies are differences between a structure and its
# separated atoms. Universal MLIPs, however, are unreliable on *isolated atoms*
# (far outside their training distribution), and the tabulated DFT atom energies in
# the dataset were computed with different DFT settings than the MLIP training data,
# so neither can be used directly.
#
# Instead we fix the (arbitrary) atomic-energy reference of each surrogate by a
# **per-element least-squares alignment** to the experimental data: for member
# ``j`` we choose atomic references :math:`c_e` so that
# :math:`\sum_e n_e\, c_e - E_\mathrm{struct} \approx E_\mathrm{ref}`. This
# 12-parameter gauge fix removes per-element offsets without using any DFT atom
# energies, while leaving the functional-dependent *bonding* differences — the
# actual UAFD signal — intact.


def fit_reference(composition, energies, targets):
    """Per-element atomic reference so that composition@c - energies ~ targets."""
    c, *_ = np.linalg.lstsq(composition, targets + energies, rcond=None)
    return composition @ c - energies


# %%
#
# Molecular atomization energies
# ------------------------------
#
# The molecular structures come from the database, while the experimental
# (bottom-of-well) atomization energies are read from the bundled ``G3-99.csv``
# table — the formation energies of the G3/99 set as compiled by `Wellendorff et al.
# (BEEF-vdW), Phys. Rev. B 85, 235149 (2012)
# <https://doi.org/10.1103/PhysRevB.85.235149>`_, so the atomization energy is minus
# the tabulated value. Database molecules and table rows are in the same order.
#
# Open-shell radicals and non-singlet spin states are dropped: universal MLIPs
# trained on (mostly closed-shell) data are unreliable for them. They are flagged by
# the nonzero magnetic moment stored in the database, or by an explicit spin-state
# label such as ``(s3B1d)`` in the G3/99 name. Every remaining molecule is relaxed
# with each member and referenced as above; the rare molecule a member fails to relax
# is also skipped, so each retained row is complete.


def relaxed_energy(atoms, calc, fmax=0.05, steps=50):
    """Energy after a loose local relaxation (molecules: with vacuum, no PBC)."""
    atoms = atoms.copy()
    atoms.center(vacuum=6.0)
    atoms.pbc = False
    atoms.calc = calc
    if len(atoms) > 1:
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms.get_potential_energy()


def read_g3_table(path):
    """(name, atomization energy [eV]) rows from G3-99.csv, in file order."""
    rows = []
    with open(path, encoding="utf-8") as fd:
        for line in fd:
            fields = next(csv.reader([line]))
            if fields and fields[0].isdigit():  # data rows start with an index
                rows.append((fields[1], -float(fields[2])))  # name, -formation energy
    return rows


def is_open_shell(row, name):
    """Open-shell radical or explicit non-singlet/spin-state entry."""
    return abs(row.get("magmom", 0.0)) > 0.1 or "(s" in name


g3_table = read_g3_table(os.path.join(DATA_DIR, "G3-99.csv"))
mol_rows = [row for row in connect(UAFD_DB).select() if row.natoms > 1]
assert len(mol_rows) == len(g3_table), "database and G3/99 table are misaligned"

# Relax every closed-shell molecule with every member; keep those that converge.
mol_atoms, mol_targets, mol_raw, mol_names = [], [], [], []
for row, (name, target) in zip(mol_rows, g3_table):
    if is_open_shell(row, name):
        continue
    atoms = row.toatoms()
    row_energies = []
    for label in MEMBER_LABELS:
        try:
            row_energies.append(relaxed_energy(atoms, calculators[label]))
        except Exception:
            row_energies.append(np.nan)
    if np.all(np.isfinite(row_energies)):
        mol_atoms.append(atoms)
        mol_targets.append(target)
        mol_raw.append(row_energies)
        mol_names.append(name)
mol_targets = np.array(mol_targets)
mol_energy = np.array(mol_raw)
mol_elements = sorted({s for a in mol_atoms for s in a.get_chemical_symbols()})
print(f"{len(mol_atoms)} closed-shell G3/99 molecules retained (of {len(mol_rows)})")

# Composition matrix and per-member fitted-reference atomization design matrix.
mol_comp = np.zeros((len(mol_atoms), len(mol_elements)))
for i, atoms in enumerate(mol_atoms):
    for s in atoms.get_chemical_symbols():
        mol_comp[i, mol_elements.index(s)] += 1
atomization = np.column_stack(
    [
        fit_reference(mol_comp, mol_energy[:, j], mol_targets)
        for j in range(len(MEMBER_LABELS))
    ]
)


# %%
#
# Solid-state properties from an equation of state
# ------------------------------------------------
#
# All 44 benchmark solids are cubic with no internal degrees of freedom, so a single
# isotropic energy-vs-volume scan per (solid, member) yields the **lattice constant**
# (from the equilibrium volume :math:`V_0`), the **bulk modulus** (:math:`B_0`) and
# the equilibrium energy :math:`E_0` used for the **cohesive energy**. The scan
# re-centers itself if the minimum is not bracketed, and we fit ASE's robust
# polynomial equation of state.

PROTOTYPE = {
    "diamond": "diamond",
    "zincblende": "zincblende",
    "rocksalt": "rocksalt",
    "bcc": "bcc",
    "fcc": "fcc",
}


def load_csv(path):
    with open(path, encoding="utf-8") as fd:
        return {r["material"]: r for r in csv.DictReader(fd)}


def eos_properties(material, structure, a0, calc, npts=11, drange=0.08):
    """Return (a_relaxed [A], B0 [GPa], E0 [eV], n_atoms) from an EOS fit."""
    atoms0 = bulk(material, PROTOTYPE[structure], a=a0)
    v0_cell, center = atoms0.get_volume(), 1.0
    for _ in range(2):  # re-center once if the minimum is not bracketed
        vols, enes = [], []
        for s in center * np.linspace(1 - drange, 1 + drange, npts):
            atoms = atoms0.copy()
            atoms.set_cell(atoms0.cell * s, scale_atoms=True)
            atoms.calc = calc
            vols.append(atoms.get_volume())
            enes.append(atoms.get_potential_energy())
        imin = int(np.argmin(enes))
        if 0 < imin < npts - 1:
            break
        center *= (vols[imin] / v0_cell) ** (1 / 3)
    v0, e0, bulk_mod = EquationOfState(vols, enes, eos="sj").fit()
    return a0 * (v0 / v0_cell) ** (1 / 3), bulk_mod / GPa, e0, len(atoms0)


lattice_csv = load_csv(CSV["lattice_constants"])
bulk_csv = load_csv(CSV["bulk_moduli"])
cohesive_csv = load_csv(CSV["cohesive_energies"])
materials = list(lattice_csv)

a_expt = np.array([float(lattice_csv[m]["Expt."]) for m in materials])
B_expt = np.array([float(bulk_csv[m]["Expt."]) for m in materials])
coh_expt = np.array([float(cohesive_csv[m]["Expt."]) for m in materials])

sol_elements = sorted(
    {
        s
        for m in materials
        for s in bulk(
            m, PROTOTYPE[lattice_csv[m]["structure"]], a=1.0
        ).get_chemical_symbols()
    }
)
sol_comp = np.zeros((len(materials), len(sol_elements)))
natoms = np.zeros(len(materials), dtype=int)
a_pred = np.zeros((len(materials), len(MEMBER_LABELS)))
B_pred = np.zeros((len(materials), len(MEMBER_LABELS)))
E0 = np.zeros((len(materials), len(MEMBER_LABELS)))

for i, mat in enumerate(materials):
    structure = lattice_csv[mat]["structure"]
    atoms0 = bulk(mat, PROTOTYPE[structure], a=a_expt[i])
    natoms[i] = len(atoms0)
    for s in atoms0.get_chemical_symbols():
        sol_comp[i, sol_elements.index(s)] += 1
    for j, label in enumerate(MEMBER_LABELS):
        a_pred[i, j], B_pred[i, j], E0[i, j], _ = eos_properties(
            mat, structure, a_expt[i], calculators[label]
        )

# Cohesive energy design matrix: per-member fitted reference (per atom).
cohesive = np.column_stack(
    [
        fit_reference(sol_comp, E0[:, j], coh_expt * natoms) / natoms
        for j in range(len(MEMBER_LABELS))
    ]
)
print(f"{len(materials)} solids: lattice constants, bulk moduli, cohesive energies")


# %%
#
# Assembling the design matrix
# ----------------------------
#
# The four datasets are stacked into a single design. Using the sum rule
# :math:`\sum_i w_i = 1` we work with the *reduced* design
# :math:`\Phi_{ni} = \phi_{i,n} - \phi_{0,n}` (differences from the reference
# member ``PBE``) and shifted targets :math:`t_n - \phi_{0,n}`, so there are
# :math:`m = 3` free weights. Each dataset is weighted to contribute equally, and
# rows where all functionals agree (zero spread — e.g. the elemental cohesive
# energies that the per-element reference fit reproduces exactly) carry no
# calibration information and are dropped.

DATASETS = [
    ("atomization", atomization, mol_targets),
    ("lattice", a_pred, a_expt),
    ("bulk modulus", B_pred, B_expt),
    ("cohesive", cohesive, coh_expt),
]
# A human-readable label per row of each dataset, for the outlier report below.
DATASET_LABELS = [mol_names, materials, materials, materials]


def constraining(pred):
    """Rows whose functionals disagree (non-zero spread) — those that inform K."""
    return np.linalg.norm(pred[:, 1:] - pred[:, [REF]], axis=1) > 1e-6


def assemble(masks):
    """Stack the four datasets, each restricted to ``masks[k]``, into one design."""
    phi_b, tprime_b, weight_b, id_b = [], [], [], []
    for k, (_name, pred, target) in enumerate(DATASETS):
        m = masks[k]
        phi_b.append(pred[m][:, 1:] - pred[m][:, [REF]])
        tprime_b.append((target - pred[:, REF])[m])
        weight_b.append(np.full(int(m.sum()), 1.0 / int(m.sum())))
        id_b.append(np.full(int(m.sum()), k))
    return (
        np.vstack(phi_b),
        np.concatenate(tprime_b),
        np.concatenate(weight_b),
        np.concatenate(id_b),
    )


use_masks = [constraining(pred) for _name, pred, _target in DATASETS]
Phi, tprime, weights, dataset_id = assemble(use_masks)


# %%
#
# Fitting the UAFD distribution
# -----------------------------
#
# We fit the Gaussian :math:`\mathcal{P}(\mathbf{w}) =
# \mathcal{N}(\mathbf{w}|\mathbf{w}_0, \mathbf{K})` by minimizing the
# negative-log-likelihood cost of Hansen et al.,
#
# .. math::
#
#    \mathcal{C} = \tfrac12 \sum_n \frac{(t_n - \bar y_n)^2}{\sigma_n^2}
#    + \tfrac12 \sum_n \ln \sigma_n^2
#    - \tfrac12 \lambda_S \ln\det(\mathbf{K} + \lambda_K),
#
# with mean prediction :math:`\bar y_n = \phi_{0,n} + \Phi_n \mathbf{w}_0` and
# variance :math:`\sigma_n^2 = \Phi_n (\mathbf{K} + \lambda_K)\Phi_n^T`. Two
# regularizers are essential: :math:`\lambda_K` adds a width to the mean (and keeps
# the variance positive), while the entropy term :math:`\lambda_S` prevents
# :math:`\mathbf{K}` from collapsing. We parametrize :math:`\mathbf{K} =
# \mathbf{L}\mathbf{L}^T` (Cholesky, guaranteeing positive-semidefiniteness) and,
# for a given :math:`\mathbf{K}`, solve for :math:`\mathbf{w}_0` in closed form.


def solve_w0(phi, target, wsig):
    """Closed-form weighted least-squares mean weights for a given variance."""
    return np.linalg.solve(phi.T @ (wsig[:, None] * phi), phi.T @ (wsig * target))


def variances(phi, Kr):
    return np.maximum(np.einsum("ni,ij,nj->n", phi, Kr, phi), 1e-10)


def uafd_cost(lflat, phi, target, wdata, lam_k, lam_s, m):
    L = np.zeros((m, m))
    L[np.tril_indices(m)] = lflat
    Kr = L @ L.T + lam_k * np.eye(m)
    sigma2 = variances(phi, Kr)
    w0 = solve_w0(phi, target, wdata / sigma2)
    res = target - phi @ w0
    data_term = np.sum(wdata * (0.5 * res**2 / sigma2 + 0.5 * np.log(sigma2)))
    return data_term - 0.5 * lam_s * np.log(np.linalg.det(Kr))


def fit_uafd(phi, target, wdata, lam_k, lam_s):
    m = phi.shape[1]
    l0 = (0.1 * np.eye(m))[np.tril_indices(m)]
    res = minimize(
        uafd_cost,
        l0,
        args=(phi, target, wdata, lam_k, lam_s, m),
        method="Nelder-Mead",
        options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10),
    )
    L = np.zeros((m, m))
    L[np.tril_indices(m)] = res.x
    Kr = L @ L.T + lam_k * np.eye(m)
    return solve_w0(phi, target, wdata / variances(phi, Kr)), Kr


# %%
#
# The regularization strengths :math:`(\lambda_S, \lambda_K)` are selected by
# 5-fold cross-validation, minimizing the held-out negative-log-likelihood.


def cv_score(phi, target, wdata, lam_k, lam_s, nfold=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, nfold, size=len(target))
    total = 0.0
    for f in range(nfold):
        tr, te = fold != f, fold == f
        w0, Kr = fit_uafd(phi[tr], target[tr], wdata[tr], lam_k, lam_s)
        sigma2 = variances(phi[te], Kr)
        res = target[te] - phi[te] @ w0
        total += np.sum(0.5 * res**2 / sigma2 + 0.5 * np.log(sigma2))
    return total


def select_and_fit(phi, target, wdata):
    """Pick (lambda_S, lambda_K) by 5-fold CV, then fit the UAFD on all of ``phi``."""
    grid = [(ls, lk) for ls in (1e-3, 1e-2, 1e-1) for lk in (1e-6, 1e-4)]
    lam_s, lam_k = min(grid, key=lambda p: cv_score(phi, target, wdata, p[1], p[0]))
    w0, K = fit_uafd(phi, target, wdata, lam_k, lam_s)
    return lam_s, lam_k, w0, K


def weight_dict(fw):
    """Full member weights as a rounded, labeled dict for printing."""
    return {MEMBER_LABELS[i]: round(float(fw[i]), 3) for i in range(len(fw))}


lam_s, lam_k, w0, K = select_and_fit(Phi, tprime, weights)
full_weights = np.concatenate([[1 - w0.sum()], w0])
Kr = K + lam_k * np.eye(len(w0))
print(f"first-pass fit on {len(tprime)} points: {weight_dict(full_weights)}")


# %%
#
# Pruning outliers and refitting
# ------------------------------
#
# Calibrating on *all* of the data lets a few pathological points distort the
# covariance: the fit inflates :math:`\mathbf{K}` to "explain" discrepancies that no
# combination of functionals can actually remove. Following Kellner et al. (Sec. III A
# and SI S4) we use the first-pass fit only to *flag* outliers, prune them, and refit a
# single clean UAFD on the remainder. A constraining point is flagged if either
#
# * the calibrated model fails outright, :math:`|\bar y_n - t_n| > 3\,\sigma_n`; or
# * the basis functionals **agree closely yet are collectively wrong** —
#   :math:`|\bar y_n - t_n| > 3\,\mathrm{std}_i\,\phi_{i,n}` while the error is also a
#   robust 3-MAD outlier for that property. A tiny spread makes the UAFD
#   over-confident exactly where it is wrong, and no reweighting can fix it.
#
# Because the second test keys on the functional *spread*, genuinely hard points where
# the functionals disagree widely (ozone, N2, ...) are kept: their large spread already
# yields an honest, large uncertainty.


def detect_outliers(full_weights, Kr, masks, k=3.0):
    """Prune ``masks`` and report points the first-pass fit cannot justify."""
    pruned, report = [], []
    for idx, (name, pred, target) in enumerate(DATASETS):
        m = masks[idx]
        rows = np.where(m)[0]
        err = np.abs((pred @ full_weights)[m] - target[m])
        sigma = np.sqrt(variances(pred[m][:, 1:] - pred[m][:, [REF]], Kr))
        spread = pred[m].std(axis=1)
        med = np.median(err)
        floor = med + k * 1.4826 * np.median(np.abs(err - med))
        fails = err > k * sigma
        agrees = (err > k * spread) & (err > floor)
        drop = fails | agrees
        pmask = m.copy()
        pmask[rows[drop]] = False
        pruned.append(pmask)
        for j in np.where(drop)[0]:
            reason = "UAFD > 3 sigma" if fails[j] else "basis agree, all wrong"
            report.append(
                (
                    name,
                    DATASET_LABELS[idx][rows[j]],
                    err[j],
                    sigma[j],
                    spread[j],
                    reason,
                )
            )
    return pruned, report


# keep the first-pass fit and pre-prune masks to visualize what gets removed
full_weights_first, Kr_first, preprune_masks = full_weights, Kr, use_masks
use_masks, report = detect_outliers(full_weights, Kr, use_masks)
outlier_masks = [pre & ~post for pre, post in zip(preprune_masks, use_masks)]
print(f"\nflagged {len(report)} outliers (dataset, point, |err|, sigma, spread):")
for name, label, err, sig, spr, reason in report:
    print(f"  {name:13s}{str(label):16s}{err:8.3f}{sig:8.3f}{spr:8.3f}  {reason}")

Phi, tprime, weights, dataset_id = assemble(use_masks)
lam_s, lam_k, w0, K = select_and_fit(Phi, tprime, weights)
full_weights = np.concatenate([[1 - w0.sum()], w0])
Kr = K + lam_k * np.eye(len(w0))
print(f"\nrefit on {len(tprime)} points: {weight_dict(full_weights)}")


# %%
#
# Reassuringly, the flagged points cluster into physically sensible groups — they are
# systematic failures of the *functionals / surrogates*, not random noise:
#
# * **atomization** — conjugated and strained hydrocarbons (e.g. ethylene,
#   acrylonitrile, cyclooctatetraene, azulene), where every functional shares a
#   systematic bias in the :math:`\pi`-bonding energetics;
# * **lattice constants** — light, soft metals (Li, LiH, Cu) at the edge of the
#   surrogates' training coverage;
# * **bulk moduli** — very stiff solids (GaN and the 5d metals Ta, W, Au), whose sharp
#   energy-volume curvature is the hardest to capture from a finite scan;
# * **cohesive energies** — III-V / III-nitride compound semiconductors (e.g. BN, AlP,
#   GaP, GaAs).
#
# In each case the four functionals sit close together but away from experiment, so
# their tiny spread would otherwise force the calibration to distort :math:`\mathbf{K}`
# or become over-confident.


# %%
#
# Calibrated predictions and uncertainties
# ----------------------------------------
#
# For any property the calibrated estimate is the weighted member average
# :math:`\bar y = \sum_i w_i \phi_i` and its uncertainty is :math:`\sigma =
# \sqrt{\Phi (\mathbf{K}+\lambda_K)\Phi^T}`. We plot each dataset against
# experiment with the calibrated error bars, and check the calibration through the
# root-mean-square normalized error (RMSNE), which should be close to 1.

fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
for k, (ax, (name, pred, target)) in enumerate(zip(axes.flat, DATASETS)):
    # show only the retained rows that constrain the fit (outliers pruned out)
    use = use_masks[k]
    phi = pred[:, 1:] - pred[:, [REF]]
    mean, tgt = (pred @ full_weights)[use], target[use]
    sigma = np.sqrt(variances(phi, Kr))[use]
    rmse = np.sqrt(np.mean((mean - tgt) ** 2))
    rmsne = np.sqrt(np.mean(((mean - tgt) / sigma) ** 2))
    ax.errorbar(
        tgt,
        mean,
        yerr=sigma,
        fmt="o",
        ms=4,
        alpha=0.6,
        ecolor="gray",
        elinewidth=0.8,
        capsize=0,
    )
    lims = [min(tgt.min(), mean.min()), max(tgt.max(), mean.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(
        xlabel="experiment",
        ylabel="PET-UAFD",
        title=f"{name}\nRMSE={rmse:.3g}, RMSNE={rmsne:.2f}",
    )
fig.suptitle("PET-UAFD calibrated predictions vs experiment")
plt.show()


# %%
#
# Uncertainty calibration
# -----------------------
#
# The signature UAFD diagnostic (Fig. 1 of Kellner et al., Fig. 4 of Hansen et al.)
# plots the *normalized error* :math:`(\bar y_n - t_n)/\sigma_n` against the
# predicted uncertainty :math:`\sigma_n` on a logarithmic axis. To avoid optimistic
# in-sample estimates, the predictions are 5-fold cross-validated. A running
# root-mean-square normalized error (RMSNE, red), evaluated in a sliding window of
# 10 points sorted by :math:`\sigma`, should hover around 1 (dashed band at
# :math:`\pm 1`) when the uncertainties are well calibrated.


def cv_predictions(phi, target, wdata, lam_k, lam_s, nfold=5, seed=0):
    """Held-out sigma and normalized error for every row via k-fold CV."""
    fold = np.random.default_rng(seed).integers(0, nfold, size=len(target))
    sigma = np.zeros_like(target)
    nerr = np.zeros_like(target)
    for f in range(nfold):
        tr, te = fold != f, fold == f
        w0_f, Kr_f = fit_uafd(phi[tr], target[tr], wdata[tr], lam_k, lam_s)
        s2 = variances(phi[te], Kr_f)
        sigma[te] = np.sqrt(s2)
        nerr[te] = (phi[te] @ w0_f - target[te]) / np.sqrt(s2)
    return sigma, nerr


def running_rmsne(sigma, nerr, window):
    """Sliding-window RMSNE versus mean sigma, over points sorted by sigma."""
    order = np.argsort(sigma)
    s, e = sigma[order], nerr[order]
    n = len(s) - window + 1
    xs = np.array([s[j : j + window].mean() for j in range(n)])
    ys = np.array([np.sqrt(np.mean(e[j : j + window] ** 2)) for j in range(n)])
    return xs, ys


cv_sigma, cv_nerr = cv_predictions(Phi, tprime, weights, lam_k, lam_s)

UNITS = {
    "atomization": "eV",
    "lattice": r"$\AA$",
    "bulk modulus": "GPa",
    "cohesive": "eV/atom",
}
fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
for k, (ax, dataset) in enumerate(zip(axes.flat, DATASETS)):
    name = dataset[0]
    mask = dataset_id == k
    sigma, nerr = cv_sigma[mask], cv_nerr[mask]
    xs, ys = running_rmsne(sigma, nerr, window=min(10, len(sigma)))
    ax.semilogx(sigma, nerr, "o", ms=3, alpha=0.5)
    ax.semilogx(xs, ys, "r-", lw=2, label="running RMSNE")
    ax.axhline(1, ls="--", c="gray", lw=0.8)
    ax.axhline(-1, ls="--", c="gray", lw=0.8)
    ax.set(
        xlabel=rf"$\sigma$ ({UNITS[name]})",
        ylabel="normalized error",
        title=name,
        ylim=(-5, 5),
    )
axes.flat[0].legend(fontsize=8, loc="lower right")
fig.suptitle("PET-UAFD uncertainty calibration (cross-validated)")
plt.show()


# %%
#
# Exporting the calibrated potential
# ----------------------------------
#
# The calibrated mean potential is simply the weighted linear combination of the
# member calculators (energies and forces combine linearly), which we wrap in an
# ASE :class:`LinearCombinationCalculator` ready for relaxations or molecular
# dynamics.

pet_uafd_calculator = LinearCombinationCalculator(
    [calculators[label] for label in MEMBER_LABELS], list(full_weights)
)

si = bulk("Si", "diamond", a=5.43)
si.calc = pet_uafd_calculator
print(
    f"calibrated Si: E = {si.get_potential_energy():.3f} eV, "
    f"max|force| = {np.abs(si.get_forces()).max():.1e} eV/A"
)

# %%
#
# Calibrated uncertainties are most meaningful for *properties* (differences in
# which the arbitrary per-functional energy references cancel). For any property the
# error bar is :math:`\sigma = \sqrt{\Phi(\mathbf{K}+\lambda_K)\Phi^T}`
# from the member spread. As an example, the calibrated lattice constant of one
# benchmark solid, reusing the equation-of-state results computed above:

i = materials.index("Cu")
phi_a = a_pred[i, 1:] - a_pred[i, REF]
a_mean = float(full_weights @ a_pred[i])
a_sigma = float(np.sqrt(phi_a @ Kr @ phi_a))
print(
    f"Cu lattice constant: PET-UAFD {a_mean:.3f} +/- {a_sigma:.3f} A "
    f"(experiment {a_expt[i]:.3f} A)"
)


# %%
#
# Evaluating the base functionals
# -------------------------------
#
# Before turning to the calibrated model, it is instructive to see how the individual
# PET surrogates (the basis functionals :math:`\phi_i`) do on each property. We reuse
# the predictions computed above and report the root-mean-square error (RMSE),
# mean-absolute error (MAE) and maximum absolute error (MAX) against experiment, on the
# retained points that constrain the fit (those where the functionals disagree and that
# survived the outlier prune; this also drops the elemental-solid cohesive energies the
# reference fit reproduces exactly).


def error_metrics(predicted, target):
    """RMSE, MAE and maximum absolute error of ``predicted`` against ``target``."""
    err = predicted - target
    return np.sqrt(np.mean(err**2)), np.mean(np.abs(err)), np.max(np.abs(err))


PROP_UNIT = {
    "atomization": "eV",
    "lattice": "Å",
    "bulk modulus": "GPa",
    "cohesive": "eV/atom",
}

for k, (name, prop, target) in enumerate(DATASETS):
    use = use_masks[k]
    print(f"\n{name} [{PROP_UNIT[name]}]   RMSE / MAE / MAX")
    print(f"  {'functional':12s} {'RMSE':>9s} {'MAE':>9s} {'MAX':>9s}")
    for j, label in enumerate(MEMBER_LABELS):
        rmse, mae, maxe = error_metrics(prop[use, j], target[use])
        print(f"  {label:12s} {rmse:9.3f} {mae:9.3f} {maxe:9.3f}")


# %%
#
# Evaluating the calibrated model
# -------------------------------
#
# Doing the same for the calibrated PET-UAFD mean prediction
# :math:`\bar y = \sum_i w_i \phi_i`: whereas the individual functionals trade off
# between properties (r²SCAN is the best for atomization energies but the worst for
# bulk moduli; PBEsol the other way round), the calibrated mean stays close to the
# best functional on every property at once.

print(f"  {'property':13s} {'RMSE':>9s} {'MAE':>9s} {'MAX':>9s}   unit")
for k, (name, prop, target) in enumerate(DATASETS):
    use = use_masks[k]
    rmse, mae, maxe = error_metrics((prop @ full_weights)[use], target[use])
    print(f"  {name:13s} {rmse:9.3f} {mae:9.3f} {maxe:9.3f}   {PROP_UNIT[name]}")


# %%
#
# The functional distribution and its committee
# ---------------------------------------------
#
# The fit returns the full distribution :math:`\mathcal{N}(\mathbf{w}_0, \mathbf{K})`
# over functional space. Its covariance :math:`\mathbf{K}` (here in the reduced basis,
# relative to the ``PBE`` reference) encodes how the functionals co-vary: the diagonal
# sets the per-direction variance, the off-diagonal terms the correlations.

free_labels = MEMBER_LABELS[1:]
print("calibrated covariance K (reduced basis, relative to PBE):")
print(" " * 14 + "".join(f"{m:>12s}" for m in free_labels))
for i, m in enumerate(free_labels):
    row = "".join(f"{Kr[i, j]:12.4f}" for j in range(len(free_labels)))
    print(f"  {m:>12s}{row}")

# %%
#
# Instead of carrying the full distribution around, it can be represented by a small
# *committee* of explicit potentials (Eq. 3 of Kellner et al.): the central model
# :math:`\mathbf{w}_0` plus one perturbed member per eigenvector of :math:`\mathbf{K}`,
#
# .. math::
#
#    \mathbf{w}_\alpha = \mathbf{w}_0 + \sqrt{k_\alpha}\, \mathbf{u}_\alpha ,
#    \qquad \alpha = 1 \dots m ,
#
# with :math:`k_\alpha` and :math:`\mathbf{u}_\alpha` the eigenvalues and eigenvectors
# of :math:`\mathbf{K}`. Running these :math:`m+1` potentials and taking their spread
# reproduces the UAFD uncertainty at a fraction of the cost of explicit calculations.

eigvals, eigvecs = np.linalg.eigh(Kr)
print("eigenvalues k_alpha of K:", " ".join(f"{k:.4f}" for k in eigvals))
committee = [w0] + [w0 + np.sqrt(k) * u for k, u in zip(eigvals, eigvecs.T)]


# %%
#
# Committee weights
# -----------------
#
# Finally, the weights of the calibrated (central) model and of each committee member,
# expanded to the full basis of MLIPs — the ``PBE`` weight is fixed by the sum rule
# :math:`\sum_i w_i = 1`. Each column is one PET surrogate, so one can read off how
# much every functional contributes to each committee member.


def to_full_weights(reduced):
    """Expand the reduced free weights to the full basis via the sum rule."""
    return np.concatenate([[1.0 - reduced.sum()], reduced])


print(f"  {'model':17s}" + "".join(f"{m:>11s}" for m in MEMBER_LABELS))
names = ["w0 (calibrated)"] + [f"committee {a}" for a in range(1, len(committee))]
for label, w in zip(names, committee):
    print(f"  {label:17s}" + "".join(f"{x:11.3f}" for x in to_full_weights(w)))
