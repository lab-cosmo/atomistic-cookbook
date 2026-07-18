"""
=========================================================
Approximate quantum vibrational spectra of a Morse mode
=========================================================

This recipe adapts the original teaching notebook into a format that is suitable
for the Atomistic Cookbook. The full i-PI workflows from the tutorial are kept
as optional instructions, while the executable part of the recipe relies on the
small precomputed spectra bundled with the example. This keeps the runtime short
enough for continuous integration while preserving the scientific discussion.

We compare five approximations to the vibrational spectrum of a three-dimensional
Morse oscillator with OH-like parameters:

* classical NVE dynamics;
* ring-polymer molecular dynamics (RPMD);
* thermostatted RPMD (TRPMD);
* GLE-tuned TRPMD;
* centroid molecular dynamics (CMD).

The exact fundamental transition of this model is close to 3568 cm^-1. The
interesting features are the classical blue shift, the spurious RPMD
resonances, and the CMD red shift.

Workflow
--------

1. Build and visualize the OH-like Morse potential.
2. Load the precomputed spectra for NVE, RPMD, TRPMD, TRPMD-GLE and CMD.
3. Compare peak positions against the exact fundamental transition.
4. Estimate the first Matsubara frequency and inspect RPMD sidebands.
5. Contrast CMD centroid and bead spectra.

Optional regeneration with i-PI
-------------------------------

The bundled ``data/inputs`` directory contains the original tutorial XML files.
Running them is intentionally left out of the automated recipe execution because
the corresponding trajectory generation would take much longer than the Cookbook
allows for pull-request validation. If you want to reproduce the spectra locally,
the updated workflow is:

* start ``i-pi`` with one of the XML files;
* let i-PI evaluate the forces directly through ``ffdirect`` and the bundled
    ``morsedia_ffdirect.py`` potential;
* post-process the trajectories with ``i-pi-getacf``.

The ``<prng><seed>...</seed></prng>`` blocks already use the current i-PI input
syntax. The only input fix applied here is a typo in the RPMD socket mode,
changing ``unic`` to ``unix`` before the switch away from sockets.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _resolve_data_dir():
    candidates = []
    file_path = globals().get("__file__")
    if file_path:
        candidates.append(Path(file_path).resolve().parent / "data")

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "data",
            cwd / "examples" / "pi-vibrations" / "data",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


DATA_DIR = _resolve_data_dir()
AU_TO_CM1 = 219474.63
EXACT_FUNDAMENTAL_CM1 = 3568.0
KB_OVER_HC_CM1_PER_K = 0.695034800
METHODS = ["nve", "rpmd", "trpmd", "trpmd-gle", "cmd"]
COLORS = {
    "nve": "#3b6fb6",
    "rpmd": "#cb4b16",
    "trpmd": "#238b45",
    "trpmd-gle": "#7a5195",
    "cmd": "#aa3377",
}


def morse_potential(distance_angstrom, r0_angstrom=0.96966):
    """Returns the Morse potential in eV for the OH-like model used in the tutorial."""
    hbar = 6.582119569e-16  # eV s
    speed_of_light = 2.99792458e10  # cm / s
    hc = hbar * speed_of_light * 2 * np.pi  # eV cm
    we = 3737.76  # cm^-1
    xe = 84.881  # cm^-1
    dissociation_energy = hc * we**2 / (4 * xe)
    inverse_length = 2.1930272  # 1 / A
    displacement = distance_angstrom - r0_angstrom
    return dissociation_energy * (
        np.exp(-2 * inverse_length * displacement)
        - 2 * np.exp(-inverse_length * displacement)
    )


def load_spectrum(method):
    raw = np.loadtxt(DATA_DIR / f"{method}_facf_avg.dat")
    frequencies_cm1 = raw[:, 0] * AU_TO_CM1
    intensity = raw[:, 1]
    return frequencies_cm1, intensity


def peak_position(frequencies_cm1, intensity, lower=2500.0, upper=4200.0):
    window = (frequencies_cm1 >= lower) & (frequencies_cm1 <= upper)
    local_frequencies = frequencies_cm1[window]
    local_intensity = intensity[window]
    return float(local_frequencies[np.argmax(local_intensity)])


def matsubara_frequency_cm1(temperature_kelvin):
    return KB_OVER_HC_CM1_PER_K * temperature_kelvin


def local_maxima(frequencies_cm1, intensity, lower=1500.0, upper=4500.0):
    window = (frequencies_cm1 >= lower) & (frequencies_cm1 <= upper)
    f = frequencies_cm1[window]
    y = intensity[window]
    if len(f) < 3:
        return np.array([]), np.array([])
    mask = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    return f[1:-1][mask], y[1:-1][mask]


distances = np.linspace(0.6, 3.0, 300)
potential = morse_potential(distances)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

axes[0].plot(distances, potential, color="#1f3d5b", lw=2.5)
axes[0].set_xlabel(r"OH distance $r$ / $\AA$")
axes[0].set_ylabel("Energy / eV")
axes[0].set_title("OH-like Morse potential")

peak_table = []
for method in METHODS:
    frequencies_cm1, intensity = load_spectrum(method)
    axes[1].plot(
        frequencies_cm1,
        intensity,
        lw=2,
        color=COLORS[method],
        label=method,
    )
    peak_table.append((method, peak_position(frequencies_cm1, intensity)))

axes[1].axvline(
    EXACT_FUNDAMENTAL_CM1,
    color="0.4",
    ls="--",
    lw=1.5,
    label=r"exact $0 \rightarrow 1$",
)
axes[1].set_xlim(1500, 4500)
axes[1].set_ylim(0, 6e-5)
axes[1].set_xlabel(r"$\omega$ / cm$^{-1}$")
axes[1].set_ylabel(r"$C_{vv}(\omega)$")
axes[1].set_title("Precomputed vibrational spectra")
axes[1].legend(frameon=False)

print("Peak positions near the fundamental transition:")
for method, peak_cm1 in peak_table:
    shift_cm1 = peak_cm1 - EXACT_FUNDAMENTAL_CM1
    print(f"{method:9s} peak = {peak_cm1:7.1f} cm^-1  shift = {shift_cm1:+7.1f} cm^-1")

temperature = 109.0
omega_m_cm1 = matsubara_frequency_cm1(temperature)
print(
    f"\nFirst Matsubara frequency at {temperature:.0f} K: "
    f"{omega_m_cm1:.1f} cm^-1"
)

rpmd_frequencies_cm1, rpmd_intensity = load_spectrum("rpmd")
rpmd_peak_freqs, rpmd_peak_intensities = local_maxima(
    rpmd_frequencies_cm1, rpmd_intensity, lower=2800.0, upper=4300.0
)
if len(rpmd_peak_freqs) > 0:
    order = np.argsort(rpmd_peak_intensities)[::-1]
    main_freq = rpmd_peak_freqs[order[0]]
    print("\nRPMD local maxima near the stretching band:")
    for idx in order[:6]:
        freq = rpmd_peak_freqs[idx]
        delta = freq - main_freq
        ratio = delta / omega_m_cm1 if omega_m_cm1 != 0 else np.nan
        print(
            f"  {freq:7.1f} cm^-1   "
            f"Delta(main) = {delta:+7.1f} cm^-1   "
            f"Delta/omega_M = {ratio:+6.2f}"
        )


cmd_bead_frequencies_cm1, cmd_bead_intensity = load_spectrum("cmd-bead")
cmd_centroid_frequencies_cm1, cmd_centroid_intensity = load_spectrum("cmd")

plt.figure(figsize=(6.5, 4.5), layout="constrained")
plt.plot(
    cmd_centroid_frequencies_cm1,
    cmd_centroid_intensity,
    lw=2,
    color=COLORS["cmd"],
    label="CMD centroid",
)
plt.plot(
    cmd_bead_frequencies_cm1,
    cmd_bead_intensity,
    lw=2,
    color="#555555",
    label="CMD bead",
)
plt.xlim(1500, 6000)
plt.xlabel(r"$\omega$ / cm$^{-1}$")
plt.ylabel(r"$C_{vv}(\omega)$")
plt.title("CMD centroid and bead spectra")
plt.legend(frameon=False)