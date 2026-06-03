"""
Machine-learned dipoles and infrared spectroscopy of liquid water
=================================================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`_

This recipe shows how to combine a machine-learning interatomic potential (MLIP) with a
machine-learned dipole model to compute the infrared (IR) spectrum of liquid water.

The workflow has four steps:

1. **Point-charge baseline**: compute the dipole time series from a TIP3P trajectory
   with fixed atomic charges; identify where this approach fails.
2. **Training**: fine-tune the PET-MAD foundational MLIP on a small water dataset
   with an added dipole output head, using `metatrain
   <https://metatensor.github.io/metatrain>`_.
3. **Simulation**: run a new NVT trajectory driven by the joint MLIP.
4. **Spectrum**: evaluate ML dipoles along the trajectory and compare against the
   point-charge baseline and experiment.

Along the way we discuss *equivariance*, the symmetry constraint that any dipole model
must satisfy, and quantify how well an unconstrained architecture satisfies it.
"""

# sphinx_gallery_thumbnail_number = 4

# %%

from zipfile import ZipFile

import ase.build
import ase.io
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import yaml
from atomistic_cookbook_utils import download_with_retry, run_command
from scipy.spatial.transform import Rotation
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator

# %%
# Data and model downloads
# ------------------------
#
# All data and model files are included in a single zip archive, that we download and
# extract:

download_with_retry(
    "https://github.com/ppegolo/labcosmo_ictp_school/raw/refs/heads/tmp/data.zip",
    "data.zip",
)

with ZipFile("data.zip", "r") as z:
    z.extractall(".")

# %%
# Fixed point-charge dipole model for liquid water
# ------------------------------------------------
#
# A simple (and simplistic) dipole model assigns fixed partial charges to each atom. We
# use the TIP3P values :math:`q_\mathrm{H} = +0.417\,e` and :math:`q_\mathrm{O} =
# -0.834\,e`, which match the force-field charges used in the MD simulation. The total
# dipole of the simulation cell is then
#
# .. math::
#
#   \boldsymbol{\mu} = \sum_i q_i\,\mathbf{r}_i
#
# Note that, as long as charge neutrality is maintained (:math:`q_\mathrm{O} =
# -2\,q_\mathrm{H}`), rescaling all charges by a factor :math:`k` maps
# :math:`\boldsymbol{\mu}` to :math:`k\boldsymbol{\mu}` and therefore the IR spectrum by
# :math:`k^2`. This is a global rescaling that cannot change the relative intensities
# between peaks, so tuning the charge value is not a meaningful way to improve the
# *shape* of the spectrum. For systems that carry a net ionic current (electrolytes,
# ionic liquids), the zero-frequency limit of the spectrum is proportional to the ionic
# conductivity, which *does* depend on the actual transported charge: in that case `only
# formal (oxidation-number) charges yield the correct DC limit,
# <https://doi.org/10.1038/s41567-019-0562-0>`_ even though they produce incorrect
# intensities. Here, pure water has zero ionic conductivity, so the choice of charge
# value is truly nothing else than an overall scale factor.
#
# Under periodic boundary conditions the raw sum :math:`\sum_i q_i \mathbf{r}_i` is
# `gauge-dependent
# <https://www.physics.rutgers.edu/~dhv/pubs/local_preprint/dv_fchap.pdf>`_: shifting a
# molecule across the simulation box by a lattice vector changes its value. To get a
# physically meaningful cell dipole, we unwrap each hydrogen into the minimum-image
# frame of its bonded oxygen before summing, using ASE's minimum-image convention
# (``mic=True``), which handles any cell shape.


def compute_pc_dipole(
    atoms: ase.Atoms, q_H: float = 0.417, q_O: float = -0.834
) -> np.ndarray:
    """Total point-charge dipole of a periodic water box (e·Å).

    Each H is unwrapped relative to its nearest O using ASE's minimum-image
    convention (``mic=True``), which works for any cell shape.

    :param atoms: ASE Atoms object containing a box of water molecules
    :param q_H: partial charge on H in units of e
    :param q_O: partial charge on O in units of e
    :returns: total cell dipole in e·Å, shape (3,)
    """
    syms = np.array(atoms.get_chemical_symbols())
    i_O = np.where(syms == "O")[0]
    i_H = np.where(syms == "H")[0]
    o_pos = atoms.positions[i_O]  # (n_mol, 3)

    # Minimum-image displacement vectors between every atom pair, from ASE: entry
    # [i, j] is the vector from atom i to atom j in the nearest periodic image
    mic_vectors = atoms.get_all_distances(mic=True, vector=True)

    # O->H vectors; for each H pick the nearest O (the oxygen it is bonded to)
    o_to_h = mic_vectors[np.ix_(i_O, i_H)]  # (n_O, n_H, 3)
    nearest_o = np.argmin(np.linalg.norm(o_to_h, axis=-1), axis=0)  # (n_H,)

    # Rebuild each H next to its bonded O by following that minimum-image bond vector
    bond_vec = o_to_h[nearest_o, np.arange(len(i_H))]  # (n_H, 3)
    h_unwrapped = o_pos[nearest_o] + bond_vec

    return q_O * o_pos.sum(axis=0) + q_H * h_unwrapped.sum(axis=0)


# %%
# By the fluctuation-dissipation theorem, the linear infrared absorption spectrum is
# proportional to the Fourier transform of the equilibrium dipole-dipole autocorrelation
# function. In practice this is most easily computed via the `power spectral density
# <https://en.wikipedia.org/wiki/Spectral_density>`_ of the total cell dipole:
#
# .. math::
#
#   n(\omega)\,\alpha(\omega) = \frac{\omega^2}{6\,V\,\varepsilon_0\,k_\mathrm{B} T\,c}
#   \tilde{S}_{\boldsymbol{\mu}}(\omega)
#
# where :math:`\tilde{S}_{\boldsymbol{\mu}}` is the isotropic two-sided power spectral
# density of :math:`\boldsymbol{\mu}(t)`, :math:`n(\omega)` is the refractive index, and
# :math:`1/6 = (1/3) \times (1/2)` combines the orientational average (liquid water is
# isotropic) with the cosine-transform convention.
# Given a dipole time series sampled at regular intervals, we can compute the power
# spectral density with `scipy.signal.periodogram`, which uses a fast Fourier transform
# (FFT) and is therefore very efficient even for long trajectories:


def ir_spectrum(
    dipoles_eA: np.ndarray,
    dt_fs: float,
    volume_A3: float,
    temperature_K: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """IR absorption from a dipole time series.

    :param dipoles_eA: total cell dipole in e·Å, shape ``(n_frames, 3)``
    :param dt_fs: time between saved frames in fs
    :param volume_A3: simulation box volume in Å³
    :param temperature_K: temperature in K
    :returns: tuple with frequencies in cm⁻¹ and :math:`n(\\omega)\\alpha(\\omega)` in
        10³ cm⁻¹
    """
    c_cms = 2.99792458e10  # cm/s
    kb_J = 1.380649e-23  # J/K
    e_C = 1.602176634e-19  # C
    A_m = 1e-10  # Å to m

    # Convert dipole to SI (C·m)
    mu_Cm = dipoles_eA * e_C * A_m
    vol_m3 = volume_A3 * A_m**3

    f, S = scipy.signal.periodogram(mu_Cm, fs=1.0, detrend="constant", axis=0)
    S = S.sum(axis=1)  # mu.mu dot product: sum over Cartesian components

    S[1:-1] *= 0.5  # undo scipy's one-sided doubling: recover the two-sided PSD
    S *= dt_fs * 1e-15  # normalise to physical time step (s)

    freqs_hz = f / (dt_fs * 1e-15)
    freqs_cm = freqs_hz / c_cms

    omega = 2 * np.pi * freqs_hz
    # n(ω) α(ω) = ω² S(ω) / (6 V k_B T ε₀ c)  [cm⁻¹]
    # the 1/6 = (1/3 orientational average) × (1/2 one-sided cosine transform)
    eps0 = 8.854187817e-12
    prefactor = omega**2 / (6.0 * vol_m3 * kb_J * temperature_K * eps0 * c_cms)
    alpha = prefactor * S * 1e-3  # cm⁻¹ to 10³ cm⁻¹

    return freqs_cm, alpha


# %%
# We compute the spectrum from a 5 ps production trajectory (after 1 ps of
# equilibration) at 300 K (`canonical <https://doi.org/10.1063/1.2408420>`_ NVT, CSVR
# thermostat, flexible TIP3P force field), giving a frequency resolution of ~7 cm⁻¹ (the
# minimum resolvable frequency is the inverse of the total simulation time, i.e. 1/(5
# ps) = 0.2 THz, or around 6.7 cm⁻¹). The trajectory is generated by running LAMMPS with
# ``data/in_tip3p.lmp``:
#
# .. literalinclude:: in_tip3p.lmp
#    :language: text
#
# and it's run with:
#

run_command("lmp -in in_tip3p.lmp", print_output=True)

# %%
# We can now load the trajectory and compute the point-charge dipole time series. The
# LAMMPS dump file contains atomic types (1 for O, 2 for H) but not chemical symbols,
# so we assign those based on the type array before computing the dipole.

type_map_pc = {1: "O", 2: "H"}
traj_pc = ase.io.read("tip3p.lammpstrj", index=":", format="lammps-dump-text")
for atoms in traj_pc:
    atoms.set_chemical_symbols([type_map_pc[int(t)] for t in atoms.arrays["type"]])
    atoms.set_pbc(True)

pc_timeseries = np.array([compute_pc_dipole(f) for f in traj_pc])
volume_A3 = float(np.abs(np.linalg.det(traj_pc[0].cell)))
DT_FS = 2.0  # time between saved frames (MD timestep × dump frequency), in fs

# %%
# We use the ``ir_spectrum`` function defined above to compute the IR spectrum from the
# dipole time series. The resulting frequencies and :math:`n(\omega)\,\alpha(\omega)`
# values are stored in ``freqs_pc`` and ``alpha_pc`` for plotting.

freqs_pc, alpha_pc = ir_spectrum(pc_timeseries, DT_FS, volume_A3)

# %%
# As a benchmark we use the experimental IR spectrum of liquid H₂O at 25°C taken
# from `Bertie & Lan, Appl. Spectrosc. 50, 1047--1057
# (1996) <https://doi.org/10.1366/0003702963905385>`_. The file
# ``data/IR_light_expt.txt`` has two columns: frequency in :math:`\mathrm{cm}^{-1}`
# and :math:`n(\omega)\,\alpha(\omega)` in units of
# :math:`10^{3}\,\mathrm{cm}^{-1}`, which is the same combination we just computed from
# the simulated dipole time series, which makes the two directly comparable.

expt = np.loadtxt("data/IR_light_expt.txt")

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
ax.plot(freqs_pc, alpha_pc, label="Point charges")
ax.plot(expt[:, 0], expt[:, 1], "k-", alpha=0.7, label="Experiment")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"Frequency / cm$^{-1}$")
ax.set_ylabel(r"$n(\omega)\,\alpha(\omega)$ / $10^3\,\mathrm{cm}^{-1}$")
ax.legend()
ax.set_title("Point-charge model vs experiment")

# %%
# The point-charge model captures the main bands (the O--H stretch at ~3400 cm⁻¹, the
# H--O--H bend at ~1600 cm⁻¹, and the librational band at ~650 cm⁻¹) but gets their
# shapes wrong in two visible ways: the O--H stretch is narrower than the experimental
# band, and the bending mode is blue-shifted relative to experiment. The reason is that
# IR intensities and peak shapes are governed by dipole *derivatives* (dynamical, or
# Born effective, charges), not by static charges: a band's intensity scales as
# :math:`|\partial\boldsymbol{\mu}/\partial Q|^2` along its vibrational mode :math:`Q`.
# With fixed charges the only contribution to :math:`\partial\boldsymbol{\mu}/\partial
# Q` is the rigid displacement of the charges, :math:`q\,\partial\mathbf{r}/\partial Q`.
# In reality the partial charges themselves change as a bond stretches, resulting in an
# intramolecular *charge flux* :math:`\partial q/\partial Q` that a fixed-charge model
# omits entirely. The same charge-flux and induced-dipole effects shape the
# low-frequency intermolecular (librational and hindered-translational) bands, which the
# point-charge model therefore also misrepresents.

# %%
# Equivariance and the dipole moment
# ----------------------------------
#
# The charge-flux effect discussed above calls for a model that learns the full
# electronic-structure response rather than relying on fixed charges. Before building
# one, it is worth understanding the key symmetry constraint any dipole model must
# satisfy.
#
# The dipole moment :math:`\boldsymbol{\mu}` of a molecule is a *vector* property: under
# a rotation :math:`R` of the whole system, the dipole must rotate accordingly:
#
# .. math::
#
#   \boldsymbol{\mu}(R\{\mathbf{r}_i\}) = R\,\boldsymbol{\mu}(\{\mathbf{r}_i\})
#
# This is an example of *covariance*, in contrast to *invariant* quantities such as the
# energy, which are unchanged by rotation. Together, invariance and covariance are
# instances of a more general principle known as *equivariance*: a property is
# equivariant to a symmetry operation if it transforms in a well-defined way under that
# operation.
#
# Some ML architectures guarantee equivariance under rotations and inversion by
# construction. Others, including `PET
# <https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_
# (Point-Edge Transformer, a message-passing graph neural network with transformer
# attention), which we will fine-tune below, are *unconstrained*: equivariance is not
# built into the architecture but **learned** during training via data augmentation
# (each frame is shown to the model in random orientations). The resulting equivariance
# is therefore approximate. We will quantify the residual error explicitly later in the
# recipe. In return, unconstrained models are more flexible and can be `very expressive
# <https://iopscience.iop.org/article/10.1088/2632-2153/ae6417>`_, improving accuracy at
# equivalent computational cost.
#
# To build intuition, we verify what *exact* equivariance looks like for a simple
# point-charge dipole: rotating the molecule must rotate the arrow by the same amount.
# This gives a visual reference for the approximate equivariance we will measure on the
# ML model later.

water = ase.build.molecule("H2O")
water.center(about=water.get_center_of_mass())

q_H, q_O = 0.417, -0.834  # TIP3P charges in units of e

np.random.seed(42)
rot_angles = np.random.uniform(0, 360, (12, 3))

rotated = []

for angles in rot_angles:
    mol = water.copy()
    R = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    mol.positions = mol.positions @ R.T
    rotated.append(mol)

    syms = mol.get_chemical_symbols()
    pos = mol.positions
    mu = sum((q_H if s == "H" else q_O) * p for s, p in zip(syms, pos))
    mol.info["dipole"] = mu

arrows = chemiscope.ase_vectors_to_arrows(rotated, "dipole", scale=1)
arrows["parameters"]["global"]["color"] = "green"

chemiscope.show(
    rotated,
    shapes={"dipole": arrows},
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True,
        structure_settings={"shape": ["dipole"]},
    ),
)

# %%
# Joint MLIP + dipole training with metatrain
# -------------------------------------------
#
# We will now train a single neural network with two output heads that
# share the same atomic representation: an energy/forces (MLIP) head, which drives the
# molecular dynamics, and a dipole head, which provides the :math:`\boldsymbol{\mu}(t)`
# time series needed for the IR spectrum. Because both heads are attached to the same
# backbone features, each structure in the dataset simultaneously constrains the
# potential-energy surface and the electric dipole, potentially making joint training
# more data-efficient than fitting two separate models.
#
# `PET-MAD-XS <https://huggingface.co/lab-cosmo/pet-mad>`_ is a foundational MLIP
# pre-trained on a diverse dataset of materials at r2SCAN meta-GGA level. `Fine-tuning
# <https://docs.metatensor.org/metatrain/latest/concepts/fine-tuning.html>`_ starts from
# PET-MAD's pre-trained weights, which already encode good atomic representations from
# a broad training distribution, and continues training on our 654-frame water
# dataset, rather than training from scratch. We also add a ``mtt::dipole`` output head
# alongside the standard energy/forces; the new head is randomly initialized and trained
# from scratch, but it benefits from the shared backbone features that are already
# well-trained on the base dataset.
#
# The ``finetune`` section in ``training`` tells metatrain to start from the pre-trained
# model checkpoint (``finetune.read_from``) and fine-tune the ``energy`` head on the
# new dataset. The ``energy/scan`` key sets the name of the fine-tuned head (called a
# ``variant`` in metatomic). The LAMMPS input must reference this same name via
# ``variant scan`` to select this head at inference time. The ``mtt::dipole`` head is
# trained from scratch, with ``type: cartesian rank 1``, meaning it is a Cartesian
# vector.
#
# .. literalinclude:: options.yaml
#    :language: yaml
#
# .. note::
#
#   Given the small dataset and the need to train a new dipole head, training takes a
#   few hours on a GPU. Therefore we provide the fine-tuned checkpoint in
#   ``pet-mad-xs-v1.5.1_SCAN_dipole.ckpt`` (downloaded above); the ``mtt train`` command
#   is shown for reference only.
#
# .. code-block:: bash
#
#    mtt train options.yaml

# %%
# We can monitor convergence by loading the loss log written during training.

train_log = np.genfromtxt(
    "data/train.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)[1:]

epochs = train_log["Epoch"].astype(float) + 1
train_mae = train_log["training_mttdipole_MAE_per_atom"].astype(float)
val_mae = train_log["validation_mttdipole_MAE_per_atom"].astype(float)

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.loglog(epochs, train_mae, label="Training")
ax.loglog(epochs, val_mae, label="Validation")
ax.set_xlabel("Epoch")
# the per_atom suffix means metatrain divides by the number of atoms for consistent
# loss scaling; here all frames are the same box of water so it is a constant factor
ax.set_ylabel("Dipole MAE / D per atom")
ax.legend()

# %%
# Model evaluation on the test set
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# metatrain holds out 5% of the dataset as a test set (``test_set: 0.05`` in
# ``options.yaml``). Ten representative frames from this test set are listed in
# ``data/test.txt``. We select the corresponding frames from the full dataset and write
# them to disk so that ``data/eval.yaml`` can point to them:

# Before evaluating the model we export the fine-tuned checkpoint to TorchScript.
# The base checkpoint ``pet-mad-xs-v1.5.1.ckpt`` (also downloaded) is only needed to
# re-run fine-tuning; its path is referenced in ``options.yaml`` under
# ``finetune.read_from``.

run_command(
    "mtt export pet-mad-xs-v1.5.1_SCAN_dipole.ckpt -o pet-mad-xs-v1.5.1_SCAN_dipole.pt",
    print_output=True,
)

ref_frames = ase.io.read("water_mlip_dipole_data.xyz", index=":")
test_idx = np.loadtxt("data/test.txt", dtype=int)

test_set = [ref_frames[i] for i in test_idx]
ase.io.write("test_set.xyz", test_set, format="extxyz")

# %%
# ``data/eval.yaml`` points to ``test_set.xyz`` and restricts the output to the dipole:
#

run_command(
    "mtt eval pet-mad-xs-v1.5.1_SCAN_dipole.pt data/eval.yaml -o test_set_dipoles.xyz",
    print_output=True,
)

# We load the predictions and visualize a parity plot. Reference dipoles are under
# ``dft_dipole``; model predictions under the ``mtt::dipole`` key, as seen earlier in
# the options file. Each point is one Cartesian component of the cell dipole.

pred_frames = ase.io.read("test_set_dipoles.xyz", index=":")

# mtt::dipole is stored with shape (3, 1); [..., 0] drops the trailing dim to (3,)
pred_dipoles = np.array([f.info["mtt::dipole"][..., 0] for f in pred_frames])
ref_dipoles = np.array([f.info["dft_dipole"] for f in test_set])

fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
ax.set_aspect("equal")
lim = np.abs(ref_dipoles).max() * 1.05
ax.plot(ref_dipoles.flatten(), pred_dipoles.flatten(), ".", alpha=0.3, ms=10)
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlabel("Reference dipole / D")
ax.set_ylabel("Predicted dipole / D")
ax.set_title("Dipole parity (test set)")

# %%
# We can also visualize the predicted and reference dipole arrows side by side on all
# test frames (green = reference, orange = predicted).

test_frames = [ref_frames[i].copy() for i in test_idx]
for f, mu_ref, mu_pred in zip(test_frames, ref_dipoles, pred_dipoles):
    f.info["dipole_ref"] = mu_ref
    f.info["dipole_pred"] = mu_pred

arrows_ref = chemiscope.ase_vectors_to_arrows(test_frames, "dipole_ref", scale=0.1)
arrows_ref["parameters"]["global"]["color"] = "green"

arrows_pred = chemiscope.ase_vectors_to_arrows(test_frames, "dipole_pred", scale=0.1)
arrows_pred["parameters"]["global"]["color"] = "orange"

chemiscope.show(
    test_frames,
    shapes={"reference": arrows_ref, "predicted": arrows_pred},
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True,
        structure_settings={"shape": ["reference", "predicted"], "unitCell": True},
    ),
)

# %%
# How equivariant is the dipole head?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As discussed above, PET does not enforce rotation equivariance by construction, but it
# *learns* it via data augmentation. The residual error can be quantified
# directly: take a test frame, apply :math:`N` Haar-uniform random rotations :math:`R`,
# predict the dipole on each rotated copy, and back-rotate the predictions to the
# original orientation:
#
# .. math::
#
#    \tilde{\boldsymbol{\mu}}_R \;=\; R^{-1}\,
#    \boldsymbol{\mu}_{\mathrm{ML}}\!\left(R\{\mathbf{r}_i\}\right)
#
# If the model were exactly equivariant, all :math:`\tilde{\boldsymbol{\mu}}_R` would
# coincide. Any spread is the equivariance error.
#
# To evaluate the model directly in Python (rather than through ``mtt eval`` as we did
# above) we wrap it in a ``MetatomicCalculator``, which exposes any model output as an
# ASE calculator quantity. We request ``mtt::dipole`` as an *additional output*
# alongside the energy.

dipole_request = {
    "mtt::dipole": ModelOutput(
        quantity="",  # unused, as the dipole is not a "standard output" in metatomic
        unit="",  # unused, as the dipole is not a "standard output" in metatomic
        per_atom=False,
        explicit_gradients=[],
    )
}
calc_eq = MetatomicCalculator(
    "pet-mad-xs-v1.5.1_SCAN_dipole.pt",
    additional_outputs=dipole_request,
    device="cpu",
)


def predict_dipole(atoms: ase.Atoms) -> np.ndarray:
    """Run the model on ``atoms`` and return the predicted dipole (3,)."""
    atoms.calc = calc_eq
    atoms.get_potential_energy()  # ASE runs one forward pass for any property
    block = calc_eq.additional_outputs["mtt::dipole"].block(0)
    return block.values.detach().cpu().numpy()[0, :, 0]


# %%
# We use a held-out test frame (not seen during training) so the measurement reflects
# how well the learned equivariance generalizes to unseen structures (note that, even
# though the frame is not used during training, it contains the same number of
# molecules, so it is not strictly out of distribution).

frame_eq = ref_frames[test_idx[0]].copy()

n_rot = 32
rots = Rotation.random(n_rot, random_state=0)

back_rotated = []
for R in rots:
    Rmat = R.as_matrix()
    fR = frame_eq.copy()
    fR.positions = fR.positions @ Rmat.T
    fR.cell = np.asarray(fR.cell) @ Rmat.T
    mu_R = predict_dipole(fR)
    back_rotated.append(Rmat.T @ mu_R)

back_rotated = np.array(back_rotated)
mu_mean = back_rotated.mean(axis=0)
err_per_rot = np.linalg.norm(back_rotated - mu_mean, axis=1)
rel_err = 100.0 * err_per_rot / np.linalg.norm(mu_mean)

mu_ref_eq = np.array(frame_eq.info["dft_dipole"])
model_err = np.linalg.norm(mu_mean - mu_ref_eq)

print(f"|<mu_back>|           = {np.linalg.norm(mu_mean):.3f} D")
print(
    f"mean ||delta_mu||    = {err_per_rot.mean():.3f} D "
    f"({rel_err.mean():.1f} % of the mean dipole)"
)
print(f"||mu_mean - mu_ref|| = {model_err:.3f} D  (model error on this frame)")

# %%
# A few-percent residual is consistent with what one expects from an unconstrained
# model trained with rotational data augmentation. To put it in perspective, we also
# compare the spread to the model's actual error on this frame---the distance between
# the mean back-rotated prediction (our proxy for the equivariant result) and the DFT
# reference. If the equivariance error is small compared to the model error, the lack of
# equivariance is not the bottleneck, and enforcing it exactly would not
# meaningfully improve accuracy. Note that the spread measures rotational
# *consistency*, not baseline accuracy: it would be zero for an exactly equivariant
# model regardless of how far that model is from the DFT reference.
#
# To *see* the equivariance error directly, we repeat the same protocol on a single
# (isolated) water molecule and overlay the resulting cloud of back-rotated dipoles on
# the molecule. The blue arrows are the individual :math:`\tilde{\boldsymbol{\mu}}_R`,
# the orange one their mean. The tighter the blue cluster around the orange arrow, the
# closer the model is to exact equivariance.

water_iso = ase.build.molecule("H2O")
water_iso.center(about=water_iso.get_center_of_mass())

back_iso = []
for R in Rotation.random(n_rot, random_state=1):
    Rmat = R.as_matrix()
    fR = water_iso.copy()
    fR.positions = fR.positions @ Rmat.T
    back_iso.append(Rmat.T @ predict_dipole(fR))

back_iso = np.array(back_iso)
mu_iso_mean = back_iso.mean(axis=0)

shapes_eq = {}
for i, mu in enumerate(back_iso):
    shapes_eq[f"sample_{i}"] = {
        "kind": "arrow",
        "parameters": {
            "global": {
                "baseRadius": 0.03,
                "headRadius": 0.05,
                "headLength": 0.06,
                "color": "blue",
            },
            "structure": [{"vector": [float(v) for v in mu]}],
        },
    }
shapes_eq["mean"] = {
    "kind": "arrow",
    "parameters": {
        "global": {
            "baseRadius": 0.06,
            "headRadius": 0.10,
            "headLength": 0.12,
            "color": "orange",
        },
        "structure": [{"vector": [float(v) for v in mu_iso_mean]}],
    },
}

chemiscope.show(
    [water_iso],
    shapes=shapes_eq,
    mode="structure",
    settings=chemiscope.quick_settings(
        structure_settings={"rotation": True, "shape": list(shapes_eq.keys())},
    ),
)

# %%
# MD simulation with the joint model
# ----------------------------------
#
# The point-charge baseline used a TIP3P trajectory, so its peak *positions* reflect the
# TIP3P force field. To obtain dynamics governed by the DFT-quality potential-energy
# surface, we run a new trajectory driven by the fine-tuned joint model.
#
# We use LAMMPS for this. First we generate a LAMMPS data file to be used as initial
# frame for the MLIP run from the last frame of the TIP3P trajectory (a 15.6 Å cubic box
# at ~1 g/cm³). The type map follows the TIP3P dump: type 1 = O, type 2 = H;
# ``pair_coeff * * 8 1`` passes atomic numbers Z=8 (O) and Z=1 (H) to the model
# accordingly.

last_frame_tip3p = traj_pc[-1]
initial_frame_mlip = traj_pc[-1].copy()
initial_frame_mlip.wrap()
ase.io.write(
    "water.data", initial_frame_mlip, format="lammps-data", atom_style="atomic"
)

# %%
# The LAMMPS input mirrors the TIP3P setup: 1 ps NVT equilibration followed by 5 ps
# production at 330 K with the Bussi CSVR thermostat, saving snapshots every 2 fs.
# SCAN-level water is slightly over-structured compared to experiment; running at 330 K
# compensates, bringing the liquid dynamics closer to room-temperature behaviour.
# ``variant scan`` is required to select the fine-tuned energy head of the joint
# model; without it LAMMPS would use the base PET-MAD head instead:
#
# .. literalinclude:: in_metatomic.lmp
#
# .. note::
#
#   The pre-run trajectory is provided in ``pet-xs-scan.lammpstrj`` (downloaded above);
#   the LAMMPS run is shown for reference only (it takes several minutes on a GPU):
#
# .. code-block:: bash
#
#    lmp -in in_metatomic.lmp

# %%
# We load the pre-run trajectory and restore chemical symbols from the type map
# (type 1 = O, type 2 = H).

type_map = {1: "O", 2: "H"}
traj_ml = ase.io.read("pet-xs-scan.lammpstrj", index=":", format="lammps-dump-text")
for atoms in traj_ml:
    atoms.set_chemical_symbols([type_map[int(t)] for t in atoms.arrays["type"]])
    atoms.set_pbc(True)

# %%
# To evaluate dipoles we need to pass the trajectory through ``mtt eval``. That
# command requires a reference target in the dataset to restrict its output to a single
# property (``mtt::dipole``); without one it would compute and store all targets the
# model was trained on (energy, forces, dipole), producing a much larger file. We write
# placeholder zeros, and disregard the accuracy report output by metatrain.

for atoms in traj_ml:
    atoms.info["dft_dipole"] = np.zeros(3)

ase.io.write("pet-xs-scan.xyz", traj_ml, format="extxyz")

# %%
# We also write the evaluation config, which mirrors the dataset section of
# ``options.yaml`` but restricts the target to the dipole only:

traj_eval_yaml = {
    "systems": {
        "read_from": "pet-xs-scan.xyz",
        "reader": "ase",
        "length_unit": "angstrom",
    },
    "targets": {
        "mtt::dipole": {
            "reader": "ase",
            "key": "dft_dipole",  # placeholder: restricts output to dipole only
            "type": {"cartesian": {"rank": 1}},
        }
    },
}
with open("traj_eval.yaml", "w") as fh:
    yaml.dump(traj_eval_yaml, fh)

# %%
# Finally we run the evaluation to get the dipole time series for the whole trajectory.
# The ``-b 64`` flag batches the evaluation on 64 frames at a time, which speeds it up
# compared to the default of one frame at a time. Depending on the  available memory,
# this batch size might be too large, or even larger batch sizes may be possible.
#
# .. warning::
#
#   Dipole evaluation on the whole trajectory takes a few minutes on a GPU. The
#   pre-computed output is provided in ``ml_traj_dipoles.xyz`` (downloaded above).
#
# .. code-block:: bash
#
#    mtt eval pet-mad-xs-v1.5.1_SCAN_dipole.pt traj_eval.yaml -o ml_traj_dipoles.xyz -b
#    64

# %%
# IR spectrum from the ML dipole model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We load the pre-computed dipole predictions and convert from Debye to e·Å (1 D =
# 3.33564×10⁻³⁰ C·m; 1 e·Å = 1.602×10⁻¹⁹ C × 10⁻¹⁰ m).

dipole_frames = ase.io.read("ml_traj_dipoles.xyz", index=":")
# mtt eval stores dipoles as shape (3, 1); drop the trailing axis to get (3,)
ml_timeseries = np.array([f.info["mtt::dipole"][..., 0] for f in dipole_frames])

D_TO_EA = 3.33564e-30 / (1.602176634e-19 * 1e-10)
ml_volume_A3 = dipole_frames[0].get_volume()
freqs_ml, alpha_ml = ir_spectrum(
    ml_timeseries * D_TO_EA, DT_FS, ml_volume_A3, temperature_K=330.0
)

# %%
# To isolate the dipole model's contribution from the force field's, we recompute the
# fixed-charge dipole on the *same* ML trajectory. Because the prefactor carries no
# empirical constants, both spectra are in absolute units and can be compared to
# experiment directly. Any difference between the two curves then reflects the dipole
# model alone. Note that this point-charge curve will look slightly different from the
# baseline earlier in the recipe, as that one used the TIP3P trajectory, which gives
# different peak positions.

pc_timeseries = np.array([compute_pc_dipole(f) for f in dipole_frames])
freqs_pc, alpha_pc = ir_spectrum(
    pc_timeseries, DT_FS, ml_volume_A3, temperature_K=330.0
)

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
ax.plot(freqs_pc, alpha_pc, alpha=0.6, label="Point charges")
ax.plot(freqs_ml, alpha_ml, label="ML dipole")
ax.plot(expt[:, 0], expt[:, 1], "k-", alpha=0.7, label="Experiment")
ax.set_xlim(0, 4000)
ax.set_ylim(0, 40)
ax.set_xlabel(r"Frequency / cm$^{-1}$")
ax.set_ylabel(r"$n(\omega)\,\alpha(\omega)$ / $10^3\,\mathrm{cm}^{-1}$")
ax.legend()
ax.set_title("IR spectrum: point charges vs ML vs experiment")

# %%
# The MLIP dynamics alone shifts and broadens the peaks relative to the TIP3P
# baseline, improving agreement with experimental data. The main contribution of the ML
# dipole model is to correct the intensities, especially for the O--H stretch and the
# bending modes.
#
# The main lesson is that the failure modes of fixed partial charges are *systematic*:
# they follow from the absence of charge flux, not from a poor choice of charge value,
# and cannot be remedied by tuning one scalar parameter. An ML dipole model trained on a
# few hundred DFT snapshots already recovers much of the missing physics.
