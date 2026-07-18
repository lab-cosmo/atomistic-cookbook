"""Analysis helpers for the bosonic / fermionic PIMD cookbook recipe.

This module is a trimmed and modernised version of the ``analysis.py`` /
``fermion.py`` pair shipped with the original PIQM 2023 tutorial
(https://github.com/i-pi/piqm2023-tutorial).  Two things changed for i-PI 3.x:

* the analytical reference energies (distinguishable / bosonic / fermionic) are
  kept essentially verbatim -- they are exact results for non-interacting
  particles in a harmonic trap and do not depend on the i-PI version;
* the fermionic reweighting no longer needs the custom ``ExchangePotential``
  class or ``MDAnalysis``.  i-PI 3.x writes the ``fermionic_sign`` directly to
  the output file, so recovering fermionic averages is a two-line numpy
  operation (see :func:`reweighted_fermionic_energy`).
"""

import re

import numpy as np


# --------------------------------------------------------------------------- #
#  Reading i-PI output                                                         #
# --------------------------------------------------------------------------- #
def read_ipi_output(filename):
    """Read an i-PI ``*.out`` file into a dict keyed by property name.

    The header lines (``# column N --> name``) are parsed to map each named
    property onto its column, so callers can index by name rather than by a
    hard-coded column number.
    """
    regex = re.compile(r".*column *([0-9]*) *--> ([^ {]*)")

    fields = []
    cols = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#"):
                match = regex.match(line)
                if match is None:
                    continue
                fields.append(match.group(2))
                cols.append(int(match.group(1)) - 1)
            else:
                break  # done with the header

    raw = np.loadtxt(filename)
    if raw.ndim == 1:  # a single output row
        raw = raw[np.newaxis, :]

    return {name: raw[:, col] for name, col in zip(fields, cols)}


def mean_energies(filename, skip_steps=0):
    """Return time-averaged kinetic and potential energies from an i-PI run.

    ``virial_fq`` is minus the quantum kinetic energy in i-PI's convention, so
    we flip its sign here.  The centroid-virial estimator (``kinetic_cv``) is
    only returned when it is present in the file (it is *not* valid under
    bosonic exchange, so we do not request it for exchange runs).
    """
    o = read_ipi_output(filename)
    out = {
        "virial": np.mean(-o["virial_fq"][skip_steps:]),
        "potential": np.mean(o["potential"][skip_steps:]),
    }
    if "kinetic_td" in o:
        out["kinetic_td"] = np.mean(o["kinetic_td"][skip_steps:])
    if "kinetic_cv" in o:
        out["kinetic_cv"] = np.mean(o["kinetic_cv"][skip_steps:])
    # Total energy from the (always-valid for bound systems) virial estimator
    out["total"] = out["virial"] + out["potential"]
    return out


def total_energy_series(filename, skip_steps=0):
    """Time series of the total energy (primitive virial + potential).

    This estimator is valid under bosonic exchange (unlike the centroid virial),
    so it is what the boson temperature sweep uses for its per-point error bars.
    """
    o = read_ipi_output(filename)
    return (-o["virial_fq"] + o["potential"])[skip_steps:]


def block_average(series, n_blocks=10):
    """Mean and block-averaged standard error of a correlated MD time series.

    Consecutive MD steps are correlated, so ``std/sqrt(N)`` on the raw samples
    underestimates the error. Instead the series is split into ``n_blocks``
    contiguous blocks; provided each block is longer than the autocorrelation
    time the block means are approximately independent, and the standard error
    is their spread, ``std(block_means)/sqrt(n_blocks)``.

    Returns ``(mean, error)``.
    """
    series = np.asarray(series, dtype=float)
    block_len = len(series) // n_blocks
    if block_len == 0:
        return float(np.mean(series)), float("nan")
    blocks = series[: block_len * n_blocks].reshape(n_blocks, block_len)
    block_means = blocks.mean(axis=1)
    return float(block_means.mean()), float(block_means.std(ddof=1) / np.sqrt(n_blocks))


def reweighted_fermionic_energy(filename, skip_steps=0):
    """Recover the fermionic total energy from a bosonic run by reweighting.

    Fermionic expectation values are obtained from a bosonic simulation via

        <A>_F = <A * s> / <s>,

    where ``s`` is the fermionic sign recorded by i-PI 3.x in the
    ``fermionic_sign`` column.  This replaces the entire hand-written
    reweighting machinery (``fermion.py`` + ``MDAnalysis``) of the 2023
    tutorial.

    Returns ``(mean_sign, fermionic_total_energy)``.
    """
    o = read_ipi_output(filename)
    sign = o["fermionic_sign"][skip_steps:]
    kinetic = -o["virial_fq"][skip_steps:]
    potential = o["potential"][skip_steps:]
    total = kinetic + potential

    mean_sign = np.mean(sign)
    fermionic_total = np.mean(total * sign) / mean_sign
    return mean_sign, fermionic_total


def fermionic_trajectory_estimate(filename, skip_steps=0):
    """Per-trajectory quantities for the weighted multi-trajectory estimator.

    Following the SI of Hirshberg, Invernizzi & Parrinello (*J. Chem. Phys.*
    152, 171102 (2020)), each trajectory contributes:

    * ``E_j`` -- its reweighted fermionic energy ``<eps*s>_j / <s>_j``;
    * ``W_j`` -- its total weight in the reweighted ensemble, i.e. the *sum*
      (not the mean) of the instantaneous signs over the trajectory.

    Returns ``(E_j, W_j, mean_sign_j)``.
    """
    o = read_ipi_output(filename)
    sign = o["fermionic_sign"][skip_steps:]
    total = -o["virial_fq"][skip_steps:] + o["potential"][skip_steps:]

    w_j = np.sum(sign)  # W_j = sum of instantaneous signs
    e_j = np.sum(total * sign) / w_j  # E_j = <eps*s>_j / <s>_j
    return e_j, w_j, np.mean(sign)


def weighted_average(values, weights=None):
    """Weighted mean and its statistical error over independent trajectories.

    Implements Eqs. 4-6 of the SI referenced in
    :func:`fermionic_trajectory_estimate`. With ``weights=None`` (or all-equal
    weights) it reduces to the ordinary mean with standard error
    ``std/sqrt(M)`` -- the correct treatment for the bosonic energies and for
    the average sign, where every trajectory carries equal weight.

    Returns ``(mean, error, n_eff)`` where ``n_eff`` is the effective sample
    size ``(sum W)^2 / sum W^2`` (equal to the number of trajectories when they
    are all similarly converged, and smaller when a few dominate the weight).
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)

    w_sum = weights.sum()
    mean = np.sum(weights * values) / w_sum
    n_eff = w_sum**2 / np.sum(weights**2)

    # The weighted-error formula assumes positive weights and n_eff > 1; if a
    # trajectory has a non-positive total weight (a badly under-sampled sign)
    # or there are too few effective samples, the variance is not meaningful.
    var = (n_eff / (n_eff - 1)) * np.sum(weights * (values - mean) ** 2) / w_sum
    if n_eff <= 1 or var < 0 or np.any(weights <= 0):
        return mean, float("nan"), n_eff

    error = np.sqrt(var / n_eff)
    return mean, error, n_eff


# --------------------------------------------------------------------------- #
#  Analytical reference energies (exact, non-interacting harmonic trap)        #
# --------------------------------------------------------------------------- #
def getZk(k, bhw, dim):
    return np.power((np.exp(0.5 * k * bhw) / (np.exp(k * bhw) - 1)), dim)


def getdZk(k, bhw, dim):
    return (
        -0.5
        * k
        * dim
        * getZk(k, bhw, dim)
        * (1 + np.exp(-k * bhw))
        / (1 - np.exp(-k * bhw))
    )


def get_harmonic_energy(n, bhw, dim, ptcl_type="dist"):
    """Exact mean energy (in units of hbar*omega) of ``n`` non-interacting
    particles in a ``dim``-dimensional isotropic harmonic trap.

    ``ptcl_type`` is one of ``dist``, ``bosonic`` or ``fermionic``.  The
    indistinguishable cases use the canonical recursion for the ``n``-particle
    partition function ``Z_n``,

        Z_n = (1/n) * sum_{k=1}^{n} xi^{k-1} * Z_1(k*beta) * Z_{n-k},

    with ``xi = +1`` for bosons (complete homogeneous symmetric polynomial /
    Alg. 4.7 in Krauth) and ``xi = -1`` for fermions (elementary symmetric
    polynomial). Here ``Z_1(k*beta) = getZk(k, ...)`` is the single-particle
    partition function at inverse temperature ``k*beta``. The energy follows as
    ``<E>/hbar*omega = -d ln Z_n / d(bhw)`` and is propagated through the same
    recursion via ``dz_arr``.

    This recursion is exact for any ``n`` and replaces the incorrect hard-coded
    three-fermion closed form used in earlier versions of this tutorial (which
    gave 0.912 mHa at 30 K instead of the correct 1.053 mHa).
    """
    if ptcl_type == "dist":
        return -n * getdZk(1, bhw, dim) / getZk(1, bhw, dim)

    if ptcl_type in ("bosonic", "fermionic"):
        xi = 1 if ptcl_type == "bosonic" else -1
        z_arr = np.zeros(n + 1)
        dz_arr = np.zeros(n + 1)
        z_arr[0] = 1.0
        for m in range(1, n + 1):
            sig_z = 0.0
            sig_dz = 0.0
            for j in range(m, 0, -1):
                sign = xi ** (j - 1)
                sig_z += sign * getZk(j, bhw, dim) * z_arr[m - j]
                sig_dz += sign * (
                    getdZk(j, bhw, dim) * z_arr[m - j]
                    + getZk(j, bhw, dim) * dz_arr[m - j]
                )
            z_arr[m] = sig_z / m
            dz_arr[m] = sig_dz / m
        return -dz_arr[n] / z_arr[n]

    return 0.0


# Physical constants / model parameters (atomic units unless noted)
SPRING_CONSTANT = 1.21647924e-8  # k for hbar*omega0 = 3 meV, in Ha/Bohr^2
MASS = 1.0
DIM = 3
KELVIN_TO_HARTREE = 3.1668152e-06  # kB in Ha/K


def omega0():
    """Trap frequency omega0 = sqrt(k / m) in atomic units."""
    return np.sqrt(SPRING_CONSTANT / MASS)


def temperature_for(bhw):
    """Kelvin temperature corresponding to a dimensionless beta*hbar*omega0."""
    return omega0() / (bhw * KELVIN_TO_HARTREE)


def analytical_energy(temp=17.4, sys_type="dist"):
    """Analytical mean total energy (Ha) at temperature ``temp`` (K).

    ``sys_type`` is one of ``dist``, ``bosonic``, ``mixed`` (2 bosons + 1
    distinguishable) or ``fermionic``.
    """
    omega = omega0()
    beta = 1.0 / (temp * KELVIN_TO_HARTREE)
    bhw = beta * omega  # hbar = kB = 1 in these reduced units

    if sys_type == "bosonic":
        e = get_harmonic_energy(3, bhw, DIM, "bosonic")
    elif sys_type == "mixed":
        e = get_harmonic_energy(2, bhw, DIM, "bosonic") + get_harmonic_energy(
            1, bhw, DIM, "dist"
        )
    elif sys_type == "fermionic":
        e = get_harmonic_energy(3, bhw, DIM, "fermionic")
    else:
        e = get_harmonic_energy(3, bhw, DIM, "dist")

    return e * omega  # convert from units of hbar*omega back to Hartree


def mixture_energy(n_bosons, n_dist, temp=17.4):
    """Exact mean total energy (Ha) of ``n_bosons`` bosons plus ``n_dist``
    distinguishable particles in the trap. Non-interacting, so the two groups add.

    Covers the four-particle statistics comparison: ``mixture_energy(0, 4)`` (all
    distinguishable), ``mixture_energy(3, 1)`` (three bosons + one dist), and
    ``mixture_energy(4, 0)`` (all bosons).
    """
    omega = omega0()
    bhw = omega / (temp * KELVIN_TO_HARTREE)
    e = 0.0
    if n_bosons:
        e += get_harmonic_energy(n_bosons, bhw, DIM, "bosonic")
    if n_dist:
        e += get_harmonic_energy(n_dist, bhw, DIM, "dist")
    return e * omega
