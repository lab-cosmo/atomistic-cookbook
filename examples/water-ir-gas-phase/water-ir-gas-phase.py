"""
Vibrational IR spectrum of a single water molecule
==================================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`_

This is a hands-on, laptop-friendly companion to the `liquid-water IR recipe
<water-ir-spectrum.html>`_. Instead of a periodic box of water driven by LAMMPS, we
study a **single water molecule in the gas phase** and run the molecular dynamics (MD)
directly in Python with `ASE <https://wiki.fysik.dtu.dk/ase/>`_. Everything runs on a
laptop CPU in a couple of minutes, with no external MD engine.

The same fine-tuned model as in the liquid recipe is used: a joint potential with a
SCAN-quality energy head (which drives the dynamics) and a ``mtt::dipole`` head (which
gives the molecular dipole :math:`\\boldsymbol{\\mu}(t)` we need for the spectrum).

The workflow has three steps:

1. **Dynamics**: run a short gas-phase MD trajectory and record the dipole at every
   saved frame.
2. **Raw spectrum**: compute the infrared spectrum from the dipole time series. Because
   the isolated molecule is free to *tumble*, this spectrum is dominated by molecular
   rotation, which buries the vibrations we care about.
3. **Eckart frame**: rotate every frame onto a common reference geometry (the *Eckart
   frame*) to project out overall translation and rotation, leaving a clean vibrational
   spectrum with the bend and the two O--H stretches.

The key lesson is *why* and *how* one separates rotational from vibrational motion when
extracting vibrational spectra from a classical trajectory.
"""

# sphinx_gallery_thumbnail_number = 2

# %%

import warnings
from zipfile import ZipFile

import ase.build
import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from ase import units
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from atomistic_cookbook_utils import download_with_retry, run_command
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from scipy.spatial.transform import Rotation

# An isolated, non-periodic molecule has no well-defined stress, so the metatomic
# calculator emits two harmless warnings here (about the missing stress and a deprecated
# neighbor-list helper). We silence them so the exercise output stays clean.
warnings.filterwarnings(
    "ignore", message=".*compute_requested_neighbors_from_options.*"
)
warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar add.*")

# %%
# Getting the model
# -----------------
#
# We reuse the fine-tuned checkpoint distributed with the liquid-water recipe. We only
# need the joint energy+dipole checkpoint, so we extract just that one file from the
# archive and export it to a TorchScript model with ``mtt export`` (the same command the
# liquid recipe uses).

CKPT = "pet-mad-xs-v1.5.0_SCAN_dipole.ckpt"
MODEL = "pet-mad-xs-v1.5.0_SCAN_dipole.pt"

download_with_retry(
    "https://github.com/ppegolo/labcosmo_ictp_school/raw/refs/heads/tmp/water-ir-spectrum.zip",  # noqa: E501
    "water-ir-spectrum.zip",
)
with ZipFile("water-ir-spectrum.zip", "r") as z:
    z.extract(CKPT)

run_command(f"mtt export {CKPT} -o {MODEL}", print_output=True)

# %%
# Setting up the calculator
# -------------------------
#
# We build a single water molecule with ASE and attach a ``MetatomicCalculator``. Two
# details matter:
#
# - ``variants={"energy": "scan"}`` selects the **fine-tuned** SCAN energy head. This is
#   the ASE equivalent of ``variant scan`` in the LAMMPS input of the liquid recipe.
#   Without it the calculator would fall back to the base PET-MAD energy, which has not
#   been adapted to this system and would produce unphysical forces.
# - ``additional_outputs`` requests the ``mtt::dipole`` head alongside the energy, so
#   each force evaluation also returns the molecular dipole for free.
#
# The molecule is non-periodic (no cell, ``pbc=False``), which is exactly what we want
# for a gas-phase molecule.

atoms = ase.build.molecule("H2O")

dipole_request = {
    "mtt::dipole": ModelOutput(
        quantity="",  # unused: the dipole is not a "standard" metatomic output
        unit="",  # unused
        per_atom=False,  # we want the single molecular dipole, not per-atom dipoles
        explicit_gradients=[],
    )
}
calc = MetatomicCalculator(
    MODEL,
    additional_outputs=dipole_request,
    variants={"energy": "scan"},
    device="cpu",
)
atoms.calc = calc


def current_dipole() -> np.ndarray:
    """Read the molecular dipole (in Debye) from the calculator's last evaluation."""
    block = calc.additional_outputs["mtt::dipole"].block(0)
    return block.values.detach().cpu().numpy()[0, :, 0]


# %%
# Running the dynamics
# --------------------
#
# We use a 0.5 fs time step (small enough to integrate the fast O--H stretch) and keep
# the molecule near 300 K with the `CSVR (Bussi) thermostat
# <https://doi.org/10.1063/1.2408420>`_, the same thermostat as the liquid-water recipe.
# CSVR controls the temperature by rescaling *all* velocities by a single stochastic
# factor each step. Unlike a per-atom Langevin thermostat it barely perturbs the
# dynamics, so the dipole autocorrelation -- and hence the spectrum -- is left almost
# untouched while the temperature stays pinned at the target.
#
# A thermostat is important here precisely *because* the system is tiny: with so few
# degrees of freedom, plain NVE gives an ill-defined, wildly fluctuating temperature,
# and at the wrong energy a bond can even run away. We run an equilibration segment
# (discarded) and then record the production frames, with the thermostat on throughout.
#
# We initialize random velocities and remove the overall center-of-mass *translation*
# with ``Stationary`` (CSVR's pure rescaling then keeps the center of mass at rest), but
# we deliberately **keep** the angular momentum: letting the molecule rotate is what
# makes the Eckart projection (step 3) necessary and visible.
#
# A single short trajectory gives a *noisy* spectrum: the periodogram of one finite run
# is a high-variance estimator, so its peaks are spiky and shift between runs. To get
# smoother, reproducible peaks we can average the spectra of several independent
# trajectories (the Bartlett method, with variance falling like one over the number of
# trajectories). The count ``N_TRAJ`` defaults to 1 here for speed. All randomness is
# seeded, so the whole calculation is reproducible.

TIMESTEP = 0.5 * units.fs
TEMPERATURE = 300.0  # K
N_EQUIL = 2000  # equilibration steps (1 ps)
N_FRAMES = 2000  # number of recorded production frames per trajectory
STRIDE = 4  # MD steps between recorded frames -> save every 2 fs
DT_FS = STRIDE * TIMESTEP / units.fs  # time between saved frames, in fs (= 2.0)
TAUT = 100 * units.fs  # CSVR coupling time (gentle: well above the vibrational periods)
N_TRAJ = 1  # number of independent trajectories to average over
SEED = 42  # master seed for reproducibility


def run_single_trajectory(rng):
    """Run one constant-temperature (CSVR) trajectory, recording the production part.

    For every saved production frame we record the geometry and the dipole together.
    Reading them at the top of the loop (before advancing) guarantees that the recorded
    dipole corresponds exactly to the recorded geometry.

    :param rng: a seeded NumPy random generator, used for the initial velocities and the
        thermostat
    :returns: the list of production frames (each carrying its dipole in
        ``info["dipole"]``) and the dipole time series as an ``(N_FRAMES, 3)`` array
    """
    mol = atoms.copy()
    mol.calc = calc

    MaxwellBoltzmannDistribution(mol, temperature_K=TEMPERATURE, rng=rng)
    Stationary(mol)  # remove net translation; keep rotation on purpose

    # The CSVR thermostat stays on for the whole run: it holds the temperature while
    # perturbing the dynamics only weakly, so it is safe to keep on during the
    # production segment we analyze.
    dynamics = Bussi(mol, TIMESTEP, temperature_K=TEMPERATURE, taut=TAUT, rng=rng)
    dynamics.run(N_EQUIL)  # equilibration (discarded)

    frames = []
    dipoles = np.zeros((N_FRAMES, 3))
    for i in range(N_FRAMES):
        mol.get_potential_energy()  # sync the calculator state with the geometry
        mu = current_dipole()
        frame = mol.copy()
        frame.info["dipole"] = mu
        frames.append(frame)
        dipoles[i] = mu
        dynamics.run(STRIDE)
    return frames, dipoles


# Drawing every trajectory's randomness from one seeded generator keeps successive runs
# independent but the whole result reproducible.
rng = np.random.default_rng(SEED)
trajectories = [run_single_trajectory(rng) for _ in range(N_TRAJ)]

print(
    f"ran {N_TRAJ} trajectory(ies) of {N_FRAMES} frames "
    f"({N_FRAMES * DT_FS / 1000:.1f} ps each)"
)

# %%
# From a dipole time series to a spectrum
# ---------------------------------------
#
# As in the liquid recipe, the infrared lineshape is proportional to
# :math:`\omega^2` times the power spectral density (PSD) of the dipole. For a single
# molecule there is no cell volume to normalize by, and we are only interested in peak
# *positions* and *relative* intensities, so we work in arbitrary units. The PSD is
# computed efficiently with ``scipy.signal.periodogram`` (an FFT under the hood).


def dipole_spectrum(
    dipole_series: np.ndarray, dt_fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """IR-like spectrum (arbitrary units) from a dipole time series.

    :param dipole_series: dipole vectors, shape ``(n_frames, 3)``
    :param dt_fs: time between frames, in fs
    :returns: ``(frequencies_cm, intensity)`` with frequencies in cm⁻¹
    """
    c_cms = 2.99792458e10  # speed of light, cm/s
    freq_hz, psd = scipy.signal.periodogram(
        dipole_series, fs=1.0, detrend="constant", axis=0
    )
    psd = psd.sum(axis=1)  # sum the three Cartesian components: mu . mu
    freqs_cm = freq_hz / (dt_fs * 1e-15) / c_cms
    return freqs_cm, freqs_cm**2 * psd


# Average the per-trajectory periodograms (a no-op when ``N_TRAJ == 1``).
lab_spectra = [dipole_spectrum(dipoles, DT_FS) for _, dipoles in trajectories]
freqs = lab_spectra[0][0]
spec_lab = np.mean([intensity for _, intensity in lab_spectra], axis=0)

# %%
# The raw (lab-frame) spectrum below is dominated by a strong band below ~400 cm⁻¹. That
# band is **not** a vibration: it is the slow reorientation of the molecule's permanent
# dipole as the whole molecule tumbles in space. The bend (~1600 cm⁻¹) and O--H stretch
# (~3500 cm⁻¹) features are present but comparatively weak and broadened.

vib = (freqs > 500) & (freqs < 4000)  # the vibrational window

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True, dpi=200)
ax.plot(freqs, spec_lab / spec_lab.max(), label="Lab frame (raw)")
ax.axvspan(0, 400, color="0.85", label="molecular rotation")
ax.set_xlim(0, 4000)
ax.set_xlabel(r"Frequency / cm$^{-1}$")
ax.set_ylabel("IR intensity / arb. units")
ax.set_title("Raw dipole spectrum: rotation dominates")
ax.legend()

# %%
# Projecting out rotation: the Eckart frame
# -----------------------------------------
#
# To recover a clean *vibrational* spectrum we must remove the overall translation and
# rotation of the molecule, keeping only the internal (vibrational) motion. The standard
# construction is the `Eckart frame
# <https://en.wikipedia.org/wiki/Eckart_conditions>`_: for each frame we find the rigid
# rotation that best superimposes the molecule onto a fixed reference geometry, using a
# **mass-weighted** least-squares fit (the Kabsch algorithm). Applying that rotation to
# the coordinates aligns the molecule; applying the *same* rotation to the dipole — a
# spatial vector — expresses it in the body-fixed frame, where it no longer reorients
# with the tumbling.


def apply_eckart_frame(frames, ref_frame=None, dipole_key="dipole"):
    """Rotate a trajectory into the Eckart frame.

    Removes overall translation and rotation by mass-weighted alignment of every frame
    onto a common reference geometry, and rotates any stored dipole accordingly. After
    this transformation the only remaining motion is vibrational.

    :param frames: list of :class:`ase.Atoms`, the trajectory to transform
    :param ref_frame: reference geometry; defaults to the first frame
    :param dipole_key: key under which the dipole is stored in ``info``/``arrays``
    :returns: a new list of :class:`ase.Atoms` in the Eckart frame
    """
    if not frames:
        return []

    # Default to the first frame as the reference structure.
    if ref_frame is None:
        ref_frame = frames[0]

    masses = ref_frame.get_masses()

    # Reference coordinates, centered on the center of mass.
    coords_ref = ref_frame.get_positions() - ref_frame.get_center_of_mass()

    eckart_frames = []
    for frame in frames:
        new_frame = frame.copy()

        # Center the current frame on its center of mass (removes translation).
        coords_curr = new_frame.get_positions() - new_frame.get_center_of_mass()

        # Mass-weighted Kabsch: R best maps the current geometry onto the reference one.
        R, _ = Rotation.align_vectors(coords_ref, coords_curr, weights=masses)

        # Apply the rotation to the centered coordinates.
        new_frame.set_positions(R.apply(coords_curr))

        # A dipole is a spatial vector, so it rotates with the very same R.
        if dipole_key in new_frame.info:
            new_frame.info[dipole_key] = R.apply(new_frame.info[dipole_key])
        # Per-atom vectors (e.g. atomic dipoles) would rotate the same way.
        if dipole_key in new_frame.arrays:
            new_frame.set_array(dipole_key, R.apply(new_frame.arrays[dipole_key]))

        eckart_frames.append(new_frame)

    return eckart_frames


# %%
# We align every trajectory onto the ideal equilibrium geometry and average the spectra
# computed from the resulting body-frame dipoles.

reference = ase.build.molecule("H2O")
eckart_trajectories = [
    apply_eckart_frame(frames, ref_frame=reference, dipole_key="dipole")
    for frames, _ in trajectories
]

eckart_spectra = [
    dipole_spectrum(np.array([f.info["dipole"] for f in eckart_frames]), DT_FS)
    for eckart_frames in eckart_trajectories
]
spec_eckart = np.mean([intensity for _, intensity in eckart_spectra], axis=0)

# Keep the first trajectory's aligned frames for the visualization further down.
eckart_traj = eckart_trajectories[0]

# %%
# Overlaying the two spectra makes the effect of the Eckart projection obvious. The
# low-frequency rotational band collapses, and the vibrational bands stand out cleanly.
# For reference we mark the experimental gas-phase fundamentals of H₂O: the bend
# :math:`\nu_2 = 1595`, the symmetric stretch :math:`\nu_1 = 3657`, and the asymmetric
# stretch :math:`\nu_3 = 3756` cm⁻¹ (`NIST
# <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=800>`_).
#
# Each curve is normalized to its maximum *within the vibrational window* (above
# 500 cm⁻¹), so the lab-frame rotational band runs off the top of the plot — a visual
# reminder of how much intensity it carries.

experimental = {r"$\nu_2$": 1595, r"$\nu_1$": 3657, r"$\nu_3$": 3756}

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True, dpi=200)
ax.plot(freqs, spec_lab / spec_lab[vib].max(), alpha=0.7, label="Lab frame (raw)")
ax.plot(freqs, spec_eckart / spec_eckart[vib].max(), label="Eckart frame (vibrational)")
for label, nu in experimental.items():
    ax.axvline(nu, color="k", ls=":", lw=1)
    ax.text(nu, 1.05, label, ha="center", va="bottom", fontsize=9)
ax.set_xlim(0, 4000)
ax.set_ylim(0, 1.2)
ax.set_xlabel(r"Frequency / cm$^{-1}$")
ax.set_ylabel("IR intensity / arb. units")
ax.set_title("Eckart projection isolates the vibrations")
ax.legend(loc="upper left")

# %%
# A few remarks on the result:
#
# - The computed peaks sit somewhat *below* the experimental fundamentals. This is
#   expected: the dynamics are **classical** (no quantum nuclear effects) and the bands
#   reflect the *anharmonic*, finite-temperature motion of the real potential, not the
#   harmonic frequencies. Classical MD also misses the zero-point energy.
# - The two O--H stretches are nearly degenerate and strongly coupled, so with a single
#   short trajectory they appear as one broad stretch band rather than two resolved
#   lines.
# - The bend is well separated and comes out close to experiment.
#
# To *see* what the Eckart frame did, we visualize the aligned trajectory: the molecule
# now vibrates in place instead of tumbling, and the body-frame dipole (green arrow)
# wiggles around its equilibrium direction. We show a short slice of the trajectory.

slice_traj = eckart_traj[:200]
arrows = chemiscope.ase_vectors_to_arrows(slice_traj, "dipole", scale=0.5)
arrows["parameters"]["global"]["color"] = "green"

chemiscope.show(
    slice_traj,
    shapes={"dipole": arrows},
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True,
        structure_settings={
            "shape": ["dipole"],
            "camera": {
                "eye": {"x": 1, "y": 0, "z": 0},
                "center": {"x": 0, "y": 0, "z": 0},
                "up": {"x": 0, "y": 1, "z": 0},
                "zoom": 140,
            },
        },
    ),
)

# %%
# A harmonic reference: normal-mode analysis
# ------------------------------------------
#
# The remarks above blame the red-shift of the MD bands on anharmonicity and finite
# temperature. We can check this by computing the **harmonic** frequencies of the same
# ML potential: the curvatures of the energy surface at its minimum. This is the static,
# zero-temperature limit -- the opposite of the dynamics above.
#
# We relax the molecule to the model's minimum, then build the Hessian by finite
# differences with :class:`ase.vibrations.Vibrations`.

molecule = ase.build.molecule("H2O")
molecule.calc = calc
BFGS(molecule, logfile=None).run(fmax=1e-3)

vibrations = Vibrations(molecule, name="vib")
vibrations.clean()  # discard any cache left by a previous run
vibrations.run()

# A non-linear triatomic has 3N - 6 = 3 genuine vibrations; the other six modes are
# translations and rotations and come out near zero (the small residual frequencies of
# the rotational modes hint that the unconstrained model is only approximately
# rotation-invariant). We keep the three highest frequencies.
frequencies = np.real(vibrations.get_frequencies())
vib_modes = np.argsort(frequencies)[-3:]
harmonic_freqs = frequencies[vib_modes]

# %%
# Harmonic frequencies give the peak *positions*. For IR *intensities* we reuse the
# dipole head: we displace the molecule along each normal mode and finite-difference the
# dipole. The squared norm of the resulting dipole derivative is the relative IR
# intensity of that mode.

r_eq = molecule.get_positions().copy()
displacement = 0.01  # Å along each unit-normalized normal mode


def dipole_at(positions: np.ndarray) -> np.ndarray:
    """ML dipole (Debye) at a given geometry of ``molecule``."""
    molecule.set_positions(positions)
    molecule.get_potential_energy()
    return current_dipole()


harmonic_intensities = np.zeros(3)
for k, mode in enumerate(vib_modes):
    direction = vibrations.get_mode(mode)
    direction /= np.linalg.norm(direction)
    dmu = (
        dipole_at(r_eq + displacement * direction)
        - dipole_at(r_eq - displacement * direction)
    ) / (2 * displacement)
    harmonic_intensities[k] = np.sum(dmu**2)

molecule.set_positions(r_eq)
vibrations.clean()  # remove the finite-difference cache files

for nu, intensity in zip(harmonic_freqs, harmonic_intensities):
    print(f"harmonic mode {nu:7.1f} cm^-1   IR intensity {intensity:6.2f} (arb.)")

# %%
# Overlaying the three pictures -- classical MD (Eckart frame), the harmonic stick
# spectrum, and the experimental fundamentals -- closes the story. The harmonic sticks
# sit *above* the MD band maxima: that gap is the anharmonic red-shift, which classical
# finite-temperature sampling captures but a harmonic analysis cannot. The ML harmonic
# values (~1660 and ~3870/3910 cm⁻¹) are in fact close to the *experimental harmonic*
# frequencies (1648, 3832, 3943 cm⁻¹); the experimental *fundamentals* marked below lie
# lower because the real, quantum vibrations are themselves anharmonic.

heights = harmonic_intensities / harmonic_intensities.max()

fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True, dpi=200)
ax.plot(freqs, spec_eckart / spec_eckart[vib].max(), label="MD (Eckart, classical)")
ax.vlines(harmonic_freqs, 0, heights, color="C3", lw=2, label="Harmonic (ML)")
ax.plot(harmonic_freqs, heights, "o", color="C3")
for label, nu in experimental.items():
    ax.axvline(nu, color="k", ls=":", lw=1)
    ax.text(nu, 1.05, label, ha="center", va="bottom", fontsize=9)
ax.set_xlim(500, 4000)
ax.set_ylim(0, 1.2)
ax.set_xlabel(r"Frequency / cm$^{-1}$")
ax.set_ylabel("IR intensity / arb. units")
ax.set_title("Classical MD vs harmonic vs experiment")
ax.legend(loc="upper left")

# %%
# Things to try
# -------------
#
# This script is meant as a starting point. A few suggested extensions:
#
# - **Resolution and statistics**: increase ``N_FRAMES`` (a longer trajectory sharpens
#   the peaks, since the frequency resolution is :math:`1/T`) and/or average the
#   spectrum over several independent runs started from different random velocities.
# - **Temperature**: change ``TEMPERATURE`` and watch the bands broaden and shift as
#   anharmonicity grows — a direct probe of the shape of the potential.
# - **Windowing**: replace ``periodogram`` with ``scipy.signal.welch`` (or apply a Hann
#   window) to trade frequency resolution for smoother, less noisy peaks.
# - **Quantum nuclear effects**: classical MD lacks zero-point motion; path-integral MD
#   would shift and broaden the bands toward the experimental fundamentals.
# - **Point charges vs ML dipole**: recompute the spectrum using fixed TIP3P charges (as
#   in the liquid recipe) on the *same* trajectory, and see how much of the intensity
#   pattern is due to the ML dipole's charge flux.
