"""Plotting helpers for the tutorial figures.

These live outside the notebook so the recipe reads as physics + analysis: each
function draws one figure from already-computed results (and the exact reference,
which it recomputes from :mod:`analysis`). They return the Matplotlib axes.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import analysis


def plot_boson_energy_curve(bhw_points, e_points, e_err, bhw_range=(0.5, 5.5)):
    """Boson E(T): PIMD points with error bars on the exact bosonic curve.

    Energies are in units of hbar*omega0; the distinguishable curve and the
    ground-state line (4.5 hbar*omega0) are shown for reference.
    """
    omega0 = analysis.omega0()
    bhw = np.linspace(*bhw_range, 80)
    temps = [analysis.temperature_for(b) for b in bhw]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(
        bhw,
        [analysis.analytical_energy(T, "bosonic") / omega0 for T in temps],
        "-",
        color="C0",
        label="bosons (exact)",
    )
    ax.plot(
        bhw,
        [analysis.analytical_energy(T, "dist") / omega0 for T in temps],
        "--",
        color="gray",
        alpha=0.7,
        label="distinguishable (exact)",
    )
    ax.axhline(4.5, ls=":", color="k", alpha=0.5)
    ax.text(
        0.55,
        4.62,
        r"ground state $4.5\,\hbar\omega_0$",
        fontsize=8,
        color="k",
        alpha=0.7,
    )
    ax.errorbar(
        bhw_points,
        e_points,
        yerr=e_err,
        fmt="o",
        color="C0",
        ms=8,
        capsize=4,
        label="bosons (PIMD)",
    )
    ax.set_xlabel(r"$\beta\hbar\omega_0$")
    ax.set_ylabel(r"$\langle E\rangle\,/\,\hbar\omega_0$")
    ax.set_title("Three bosons in a harmonic trap")
    ax.legend()
    return ax


def plot_statistics_levels(labels, sim, err, ref, title):
    """Energy-level comparison at one temperature: the exact total energy of each
    case is drawn as a horizontal level, with the PIMD value (point + error bar)
    on it. Ordered low to high, this shows how exchange orders the energies."""
    sim = np.array(sim) * 1e3
    err = np.array(err) * 1e3
    ref = np.array(ref) * 1e3
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for i, lab in enumerate(labels):
        ax.hlines(ref[i], i + 0.15, i + 0.85, color="gray", lw=2.5)
        ax.errorbar(i + 0.5, sim[i], yerr=err[i], fmt="o", ms=9, capsize=5, color="C0")
        ax.annotate(
            lab,
            (i + 0.5, -0.07),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=9,
        )
    ax.set_xticks([])
    ax.set_xlim(0, len(labels))
    ax.plot([], [], color="gray", lw=2.5, label="exact")
    ax.plot([], [], "o", color="C0", label="PIMD")
    ax.set_ylabel("total energy / mHa")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def plot_fermion_ensemble(traj_energies, mean, err, exact):
    """One-run-vs-many fermion picture: scattered per-trajectory energies (grey)
    and the sign-weighted mean with its error bar (blue), against the exact line."""
    traj = np.asarray(traj_energies)
    n = len(traj)
    jitter = 0.15 * (np.arange(n) - (n - 1) / 2) / max((n - 1) / 2, 1)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.axhline(exact * 1e3, color="k", ls="--", label=f"exact ({exact * 1e3:.3f} mHa)")
    ax.plot(
        jitter,
        traj * 1e3,
        "o",
        color="gray",
        alpha=0.6,
        label="individual trajectories",
    )
    ax.errorbar(
        [0],
        [mean * 1e3],
        yerr=[err * 1e3],
        fmt="s",
        color="C0",
        ms=9,
        capsize=5,
        label="sign-weighted mean",
    )
    ax.set_xlim(-1, 1)
    ax.set_xticks([])
    ax.set_ylabel("fermionic total energy / mHa")
    ax.set_title("Three fermions at 30 K: one run vs. eight")
    ax.legend()
    return ax


def plot_sign_scatter(cases):
    """Per-trajectory fermionic energies at several temperatures side by side, to
    show the sign problem: colder -> smaller sign -> far wider scatter.

    ``cases`` is a list of ``(label, traj_energies_Ha, exact_Ha)``. Each is drawn
    as a jittered column of grey points with a dashed line at its exact value.
    """
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for x, (_label, traj, exact) in enumerate(cases):
        traj = np.asarray(traj) * 1e3
        n = len(traj)
        jitter = x + 0.12 * (np.arange(n) - (n - 1) / 2) / max((n - 1) / 2, 1)
        ax.plot(jitter, traj, "o", color="gray", alpha=0.6)
        ax.hlines(exact * 1e3, x - 0.28, x + 0.28, color="k", ls="--")
    ax.set_xticks(range(len(cases)), [c[0] for c in cases])
    ax.set_ylabel("per-trajectory energy / mHa")
    ax.set_title("The sign problem: colder = much wider scatter")
    ax.text(
        0.98,
        0.02,
        "dashed = exact",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="gray",
    )
    return ax
