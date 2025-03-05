r"""
Free energy methods
==========================

:Authors: Venkat Kapil `@venkatkapil24 <https://github.com/venkatkapil24/>`_
          and Davide Tisi `@DavideTisi <https://github.com/DavideTisi/>`_

This example shows how to perform free energy calculation on a
effective Hamiltonian of a hydrogen atom in an effectively 1D double well
from Ref. `Y. Litman et al., JCP (2022)
<https://pubs.aip.org/aip/jcp/article/156/19/194107/2841188/Dissipative-tunneling-rates-through-the>`_
"""

# %%

import os
import subprocess
import time

import ase.io
import chemiscope
import ipi
import matplotlib.pyplot as plt
import numpy as np


# %%
# Model System
# ------------
#
# We will consider the effective Hamiltonian of a hydrogen atom
# in an effectively 1D double well from
# Ref. `Y. Litman et al., JCP (2022)
# <https://pubs.aip.org/aip/jcp/article/156/19/194107/2841188/Dissipative-tunneling-rates-through-the>`_
# Its potential energy surface (PES) is described by the function
#
# .. math::
#
#    V = A x^2 + B x^4 + \frac{1}{2} m \omega^2 y^2 + \frac{1}{2} m \omega^2 z^2
#
# with:
#
# .. math::
#
#     \begin{align}
#     m &= 1837.107~\text{a.u.}\\
#     \omega &= 3800 ~\text{cm}^{-1} = 0.017314074~\text{a.u.}\\
#     A &= -0.00476705894242374~\text{a.u.}\\
#     B &= 0.000598024968321866~\text{a.u.}\\
#     \end{align}
#
# Let's define a function to visualize the PES!
#


def PES(x, y, z):
    """
    PES from Ref. `Y. Litman et al., JCP (2022)`
    """

    A = -0.00476705894242374
    B = 0.000598024968321866
    k = 1837.107 * 0.017314074**2

    return A * x**2 + B * x**4 + k * y**2 / 2 + k * z**2 / 2


# %%
# The PES, in three dimensions, can be drawn with xy contours
# (each contour corresponds to kB T with T = 300 K), for various z values.
#

x = np.linspace(-3, 3, 21)
y = np.linspace(-0.5, 0.5, 21)

X, Y = np.meshgrid(x, y)
Z = PES(X, Y, 0)


contour_levels = [np.min(Z) + 0.00095004347 * i for i in range(12)]

plt.title(r"$V(x,y,0)$")
plt.xlabel(r"$x$ [a.u.]")
plt.ylabel(r"$y$ [a.u.]")
plt.contour(X, Y, Z, levels=contour_levels)
plt.show()

# %%
# **Questions**:
#
# 1. Looking at the contours can you tell
# if delocalization of the system to its other minimum is
# a rare event with respect to the time scale of
# the system's vibrations?
#
# 2. Feel free to plot it for various z values.
# Does the plot change? What does it tell you
# about the coupling between the modes?
#
#

# %%
# Calculating the free energy of a hamiltonian
# ============================================
#
# The harmonic approximation
# --------------------------
#
# The harmonic approximation to the PES is essentially
# a truncated Taylor series expansion to second
# order around one of its minima.
# :math:`V^{\text{harm}} = V(q_0) +
# \frac{1}{2} \left.\frac{\partial^2 V}{\partial q^2}\right|_{q_0} (q - q_0)^2`
# where :math:`q = (x,y,z)`
# is a position vector and :math:`q_0 = \text{arg min}_{q} V` is
# the position where the PES has a local minimum.
#
# Let's use the i-PI code to optimize the PES with respect to its position
# to find :math:`q_0` and :math:`V^{\text{harm}}`.
#

# %%
# Step 1: fixed cell geometry optimization
# ----------------------------------------
#
# To find :math:`q_0` we will use the fixed-cell geometry
# optimization feature of i-PI! The inputs are
# into the `geop` directory.
#
# You will find the i-PI input file. Have a look at the
# `input.xml` file and see if you can understand its various parts.
#
# The snippet that implements the geometry optimization feature is
#
# .. code-block:: xml
#
#    <motion mode='minimize'>
#      <optimizer mode='bfgs'>
#        <tolerances>
#          <energy> 1e-5 </energy>
#          <force> 1e-5 </force>
#          <position> 1e-5 </position>
#        </tolerances>
#      </optimizer>
#    </motion>
#
#
# In essence, the geometry optimization is implemented as a
# motion class. At every step instead of performing "dynamics" we will
# simply move towards a local minimum of the PES. There are
# many algorithms for locally optimizing high-dimensional functions;
# here we use the robust
# `BFGS` (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton
# method. The tolerances set thershold values for changes in the energy,
# positions and forces, that are sufficient to deem an optimization converged.
#

# %%
# Installing the Python driver
#
# i-PI comes with a FORTRAN driver, which however has to be installed
# from source. We use a utility function to compile it. Note that this requires
# a functioning build system with `gfortran` and `make`.

ipi.install_driver()
# %%
# To perform a fixed-cell geomerty optimization
# the bash script command should look like:
#
# .. code-block:: bash
#
#      i-pi geop/input.xml > log.i-pi $
#      sleep(5)
#      i-pi-driver -u -h geop -m doublewell_1D &
#
# The same can be achieved from Python using `subprocess.Popen`

ipi_process = None
if not os.path.exists("geop.out"):
    ipi_process = subprocess.Popen(["i-pi", "geop/input.xml"])
    time.sleep(5)  # wait for i-PI to start
    driver_process = subprocess.Popen(
        ["i-pi-driver", "-u", "-h", "geop", "-m", "doublewell_1D"]
    )

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

if ipi_process is not None:
    ipi_process.wait()
    driver_process.wait()

# %%
#
# The number of steps taken for an optimization calculation depends on
# the system (the PES), the optimization method,
# and the initial configuration. Here,
# we initialize the system from :math:`q = (2.0, 0.2, -0.5)`
# as seen in the `geop/init.xyz` file.
#
# You can analyze your optimization calculation by plotting
# the potential energy vs steps and confirm if the potential
# energy has indeed converged to the set threshold!
#
# The final frame of the `geop.pos_0.xyz` file gives
# with chemiscope we can wisualize the minimzation
# trajectory and its energies.
#

geop_out, _geop = ipi.read_output("geop.out")
geop_traj = ipi.read_trajectory("geop.pos_0.xyz")

chemiscope.show(
    frames=geop_traj,
    properties={
        "steps": {"values": np.arange(len(geop_traj)), "target": "structure"},
        "U": {"values": geop_out["potential"], "target": "structure"},
    },
    settings=chemiscope.quick_settings(x="steps", y="U", trajectory=True),
    mode="structure",
)

# %%
# Step 2: harmonic calculation
# ----------------------------
#
# To compute the system's Hessian we will use the vibrations feature of i-PI.
# Go through the `input.xml` file.
# The snippet that implements the vibrations feature is
#
# .. code-block:: xml
#
#    <motion mode='vibrations'>
#      <vibrations mode='fd'>
#        <pos_shift> 0.001 </pos_shift>
#        <energy_shift> 0.001 </energy_shift>
#        <prefix> phonons </prefix>
#        <asr> none </asr>
#      </vibrations>
#    </motion>
#
# The system's Hessian is computed using the finite difference method.
# This approach approximates the ij-th element of the Hessian in terms
# of the forces acting at infinitisemally displaced positions around the minimum:
#
# .. math::
#
#      \frac{\partial^2 V}{\partial q^2}_{ij} \approx - \frac{1}{2 \epsilon}
#      \left(\left.f_{i}\right|_{q_{j} + \epsilon} - \left.f_{i}\right|_{q_{j}
#      - \epsilon} \right)
#
# At every step instead of performing "dynamics", we will displace a degree of
# freedom along :math:`\pm \epsilon` and estimate one row of the Hessian.
# `pos_shift` sets
# the value of :math:`\epsilon` while `asr` zeros out the blocks
# of the Hessian due to continuous
# symmetries (translations or rotations for solids or clusters). In this example,
# we set this option to `none` as our system doesn't possess any continuous symmetries.
#

# %%
# The initial configuration for this calculation should correspond to an
# opitimized position. You can obtain this from the last frame of the geop trajectory.

ase.io.write("init_harm.xyz", geop_traj[-1], format="extxyz")

# %%
# The `bash` command for this would be
#
# .. code-block:: bash
#
#      i-pi harm/input.xml > log.i-pi $
#      sleep(5)
#      i-pi-driver -u -h vib -m doublewell_1D &
#
# Again we will use `subprocess.Popen`

ipi_process = None
if not os.path.exists("harm.out"):
    ipi_process = subprocess.Popen(["i-pi", "harm/input.xml"])
    time.sleep(5)  # wait for i-PI to start
    driver_process = subprocess.Popen(
        ["i-pi-driver", "-u", "-h", "vib", "-m", "doublewell_1D"]
    )

# %%
# If you run this in a notebook, you can go ahead and start loading
# output files *before* i-PI and lammps have finished running, by
# skipping this cell

if ipi_process is not None:
    ipi_process.wait()
    driver_process.wait()

# %%
# The Hessian can be recovered from the `harm.phonons.hess` file.
# You can use the snippet below to plot the harmonic approximation
# to the PES
#


def PES_harm(x, y, z):
    """
    Harmonic approximation to the PES from Ref. `Y. Litman et al., JCP (2022)` around
    a local minimum. Note this function is only valid for the example!
    """

    return (
        V0
        + hess[0, 0] * (x - q0[0]) ** 2 / 2
        + hess[1, 1] * (y - q0[1]) ** 2 / 2
        + hess[2, 2] * (z - q0[2]) ** 2 / 2
    )


hess = np.loadtxt("harm.phonons.hess", comments="#")
q0 = ipi.read_trajectory("init_harm.xyz")[0].positions[0] / 0.529177
V0 = np.loadtxt("harm.out", ndmin=2)[0, 1]

x = np.linspace(-3, 3, 21)
y = np.linspace(-0.5, 0.5, 21)

X, Y = np.meshgrid(x, y)

contour_levels = [np.min(Z) + 0.00095004347 * i for i in range(12)]

plt.title(r"$V(x,y,0)$")
plt.xlabel(r"$x$ [a.u.]")
plt.ylabel(r"$y$ [a.u.]")
plt.contour(X, Y, PES(X, Y, 0), levels=contour_levels)
plt.show()

# %%

plt.title(r"$V^{\mathrm{harm}}(x,y,0)$")
plt.xlabel(r"$x$ [a.u.]")
plt.ylabel(r"$y$ [a.u.]")
plt.contour(X, Y, PES_harm(X, Y, 0), levels=contour_levels)
plt.show()

# %%
# The harmonic calculation also  gives us the frequency modes
# from the dynamical matrix (mass scaled hessian)!
#

W2s = np.loadtxt("harm.phonons.eigval")
print("Harmonic Frequencies")
for w2 in W2s:
    print("%10.5f cm^-1" % (w2**0.5 * 219474))

# %%
# These frequencies can be used to calculate the quantum harmonic
# free energy of the system -- an approxiamtion to its true free energy!
#


def quantum_harmonic_free_energy(Ws, T):
    """
    Receives a list of frequencies in atomic units and the temperature
    in Kelvin. Returns the system's harmonic free energy.
    """

    hbar = 1
    kB = 3.1668116e-06

    return V0 + -kB * T * np.log(1 - np.exp(-hbar * Ws / (kB * T))).sum()


print(
    "Quantum Harmonic free energy: %15.8f [a.u.]"
    % (quantum_harmonic_free_energy(W2s**0.5, 300))
)
print(
    "Quantum Harmonic free energy: %15.8f [eV]"
    % (quantum_harmonic_free_energy(W2s**0.5, 300) * 27.211386)
)
print(
    "Quantum Harmonic  free energy: %15.8f [kJ/mol]"
    % (quantum_harmonic_free_energy(W2s**0.5, 300) * 2625.4996)
)

# %%
#
# The exact free energy for the system when localized in one of
# the wells is still around 0.65 kJ/mol off w.r.t the harmonic limit.
#

F = -7.53132797e-05 + quantum_harmonic_free_energy(W2s**0.5, 300)

print("Exact Quantum free energy: %15.8f [a.u.]" % (F))
print("Exact Quantum free energy: %15.8f [eV]" % (F * 27.211386))
print("Exact Quantum free energy: %15.8f [kJ/mol]" % (F * 2625.4996))

# %%
#
# Harmonic to anharmonic
# ----------------------
#
# Calculating free energies beyond the harmonic approximation is non-trivial.
# There exist a familty of methods that can  solve the vibrational Schroedinger
# Equation by approximating the anharmonic component of
# the PES, yielding an amharmonic
# free energy. While highly effective for low-dimensional or mildly anharmonic systems,
# the method of resort for *numerically-exact amharmonicfree energies*
# of solid and clusters
# is the thermodynamic integration method combined with the path-integral method
# ( for applications see Refs.
# `M. Rossi et al, PRL (2016) <https://doi.org/10.1103/PhysRevLett.117.115702>`_,
# `V. Kapil et al, JCTC (2019) <https://doi.org/10.1021/acs.jctc.9b00596>`_,
# `V. Kapil et al, PNAS (2022) <https://doi.org/10.1073/pnas.2111769119>`_).
#
#
# The central idea is to reversibly change the potential from harmonic to anharmonic
# by defining a :math:`\lambda`-dependent Hamiltonian
#
# .. math::
#    \hat{H}(\lambda) = \hat{T} + \lambda
#    \hat{V}^{\text{harm}}  + (1 - \lambda) \hat{V}
#
# The  the anharmonic free energy is calculated as the reversible work done
# along the fictitious path in :math:`\lambda`-space
#
# .. math::
#     F = F^{\text{harm}} + \left< \hat{V} -
#     \hat{V}^{\text{harm}} \right>_{\lambda}
#
#
# where :math:`\left< \hat{O} \right>_{\lambda}`
# is the path-integral estimator for a positon dependent operator
# for :math:`\hat{H}(\lambda)`.

# %%
#
# Step 3: Harmonic to anharmonic thermodynamic integration
# --------------------------------------------------------
#
# A full quantum thermodynamic integration calculation requires
# knowledge of the harmonic reference. Luckily we have just performed these
# calculations!
#

# %%
# Classical Statistical Mechanics
# -------------------------------
#
# Let's first compute the anharmonic free
# energy difference within the classical approximation.
#
# To create the input file for I-PI we need to defines a "mixed"
# :math:`\lambda`-dependent Hamiltonian. Let's see the new most important parts.
#
# This i-PI calculation includes two "forcefield classes"
#
# .. code-block:: xml
#
#    <!-->  defines the anharmonic PES <-->
#    <ffsocket name='driver' mode='unix' matching='any' pbc='false'>
#        <address> f0 </address>
#        <latency> 1e-3 </latency>
#    </ffsocket>
#
# a standard socket
#
# and
#
# .. code-block:: xml
#
#    <!-->  defines the harmonic PES <-->
#    <ffdebye name='debye'>
#            <hessian shape='(3,3)' mode='file'> HESSIAN_FILE </hessian>
#            <x_reference mode='file'> X_REF_FILE
#        <!--> relative path to a file containing the optimized positon vector <-->
#            </x_reference>
#            <v_reference> V_REF
#        <!-->  the value of the PES at its local minimum <-->
#            </v_reference>
#    </ffdebye>
#
# an intrinsic harmonic forcefield that builds the harmonic potential.
# This requires :math:`q_0`, :math:`V(q_0)` and the full Hessian.
# `HESSIAN_FILE` is `harm.phonons.hess` which is the file containing the
# full hessian.
# X_REF_FILE is a file with the optimized positon vector
#


HESSIAN_FILE = "harm.phonons.hess"
X_REF_FILE = "ref.data"
with open(X_REF_FILE, "w") as xref:
    for pos in geop_traj[-1].positions:
        xref.write(f"{pos[0]}  {pos[1]}  {pos[2]} \n")

# %%
# V_REF can be obtained from the potential energy in `harm.out`

V_REF = np.loadtxt("harm.out")[1]
print(V_REF)

# %%
# To model a Hamiltonian with a linear combination of the harmonic and
# the anharmonic potential you can define the weights for the force
# components as
#
# .. code-block:: xml
#
#   <forces>
#      <force forcefield='debye' weight=''> </force>  <!-->  set this to lambda <-->
#      <force forcefield='driver' weight=''> </force> <!-->  set this to 1 - lambda <-->
#   </forces>
#
# You can print out the harmonic and the anharmonic
# components as a <property> in the output class
#
# .. code-block:: xml
#
#     <properties filename='pots' stride='10' flush='10'>
#         [ pot_component_raw(0), pot_component_raw(1) ]
#     </properties>
#
# A typical TI calculation requires multiple simulations,
# one for each lambda and a postprocessing step to integrate the free
# energy difference. In this example, we use six linearly-spaced points i.e.
# :math:`\lambda\in\left[0, 0.2, 0.4, 0.6, 0.8, 1.0\right]`,
# and create a set of files for each calculation.

# %%
# Let's generate the input files for the TI calculations

for idx, i in enumerate(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]):
    with open(f"cti/input_{i}.xml", "w") as f:
        strfile = """<simulation verbosity='low'>
   <output prefix='cti_{i}'>
      <properties filename='out' stride='10' flush='10'> [ step, time{{picosecond}},
      conserved, temperature{{kelvin}}, kinetic_cv, potential, pressure_cv,
      volume, ensemble_temperature ] </properties>
      <properties filename='pots' stride='10' flush='10'>
      [ pot_component_raw(0), pot_component_raw(1) ] </properties>
      <trajectory filename='pos1' stride='10' bead='0' flush='10'>
        positions
      </trajectory>
      <trajectory filename='xc' stride='10' flush='10'> x_centroid </trajectory>
      <checkpoint stride='800'/>
   </output>
   <total_steps> 2000 </total_steps>
   <prng><seed>31415</seed></prng>
   <ffsocket name='driver' mode='unix' matching='any' pbc='false'>
       <address> f{idx} </address>
       <latency> 1e-3 </latency>
   </ffsocket>
   <ffdebye name='debye'>
       <hessian shape='(3,3)' mode='file'> {HESSIAN_FILE} </hessian>
       <x_reference mode='file'> {X_REF_FILE} </x_reference>
       <v_reference> {V_REF}  </v_reference>
   </ffdebye>
   <system>
      <initialize nbeads='1'>
      <file mode='xyz'> ./cti/init.xyz </file>
         <velocities mode='thermal' units='kelvin'> 300 </velocities>
      </initialize>
      <forces>
         <force forcefield='debye' weight='{i}'> </force>
         <force forcefield='driver' weight='{im1:2.1f}'> </force>
      </forces>
      <motion mode='dynamics'>
      <fixcom> False </fixcom>
         <dynamics mode='nvt'>
            <timestep units='femtosecond'> 1.00 </timestep>
            <thermostat mode='pile_l'>
                <tau units='femtosecond'> 100 </tau>
            </thermostat>
         </dynamics>
      </motion>
      <ensemble>
         <temperature units='kelvin'> 300 </temperature>
      </ensemble>
   </system>
</simulation>
""".format(
            HESSIAN_FILE=HESSIAN_FILE,
            X_REF_FILE=X_REF_FILE,
            V_REF=V_REF,
            i=i,
            idx=idx,
            im1=1 - float(i),
        )
        f.write(strfile)

# %%
# You can run i-PI simulataneous in these directories, the bash command would look like:
#
# .. code-block:: bash
#
#    for x in 0.0 0.2 0.4 0.6 0.8 1.0 ; do
#       cd ${x}
#       i-pi cti/input_${x}.xml > log${x}.i-pi &
#       cd ..
#    done
#
#    wait 5
#
#    for x in {0..10}; do
#       i-pi-driver -u -h f${x} -m doublewell_1D &
#    done
#

for idx, i in enumerate(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]):
    ipi_process = None
    if not os.path.exists(f"cti_{i}.out"):
        ipi_process = subprocess.Popen(["i-pi", f"cti/input_{i}.xml"])
        time.sleep(5)  # wait for i-PI to start
        driver_process = subprocess.Popen(
            ["i-pi-driver", "-u", "-h", f"f{idx}", "-m", "doublewell_1D"]
        )
    # in a jupiter notebook comments the next 3 lines to be able to
    # load the outputs before the run ends
    if ipi_process is not None:
        ipi_process.wait()
        driver_process.wait()

# %%
#
# When the calculations have finished,
# you can use the following snippet to analyze the
# mean force along the path and estimat its integral!

du_list = []
duerr_list = []

dir_list = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
l_list = [float(lst) for lst in dir_list]

for i in dir_list:

    filename = f"cti_{i}.pots"
    data, _data = ipi.read_output(filename)

    du = data["pot_component_raw(1)"] - data["pot_component_raw(0)"]
    du_list.append(du.mean())
    duerr_list.append(du.std() / len(du) ** 0.5)


du_list = np.asarray(du_list)
duerr_list = np.asarray(duerr_list)

plt.plot(l_list, (du_list) * 2625.4996, color="blue")
plt.fill_between(
    l_list,
    y1=(du_list - duerr_list) * 2625.4996,
    y2=(du_list + duerr_list) * 2625.4996,
    color="blue",
    alpha=0.2,
)
plt.title("Classical thermodynamic integration")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\left<U - U^{\mathrm{harm}}\right>_{\lambda}$ [kJ/mol]")
plt.show()


# %%
# Since we are working within classical stat mech,
# we should use the classical harmonic reference to estimate
# the classical anharmonic free energy.


def classical_harmonic_free_energy(Ws, T):
    """
    Receives a list of frequencies in atomic units and
    the temperature in Kelvin. Returns the system's harmonic free energy.
    """

    hbar = 1
    kB = 3.1668116e-06

    return V0 + kB * T * np.log(hbar * Ws / (kB * T)).sum()


df, dferr = np.trapz(x=l_list, y=du_list), np.trapz(x=l_list, y=duerr_list**2) ** 0.5

F = classical_harmonic_free_energy(W2s**0.5, 300)

print("Classical harmonic free energy: %15.8f [a.u.]" % (F))
print("Classical harmonic free energy: %15.8f [eV]" % (F * 27.211386))
print("Classical harmonic free energy: %15.8f [kJ/mol]" % (F * 2625.4996))
print("")

F = classical_harmonic_free_energy(W2s**0.5, 300) + df
Ferr = dferr


print("Classical anharmonic free energy: %15.8f +/- %15.8f [a.u.]" % (F, Ferr))
print(
    "Classical anharmonic free energy: %15.8f +/- %15.8f [eV]"
    % (F * 27.211386, Ferr * 27.211386)
)
print(
    "Classical anharmonic free energy: %15.8f +/- %15.8f [kJ/mol]"
    % (F * 2625.4996, Ferr * 2625.4996)
)
print("")
F = -7.53132797e-05 + quantum_harmonic_free_energy(W2s**0.5, 300)

print("Exact free energy: %15.8f [a.u.]" % (F))
print("Exact free energy: %15.8f [eV]" % (F * 27.211386))
print("Exact free energy: %15.8f [kJ/mol]" % (F * 2625.4996))


# %%
# Quantum Statistical Mechanics
# -----------------------------
#
# The quantum anharmonic free energy can be calculated using the path-integral
# method. The main difference in the input file is the number of replicas!
# Let's construct the input file:

for idx, i in enumerate(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]):
    with open(f"qti/input_{i}.xml", "w") as f:
        strfile = """<simulation verbosity='low'>
   <output prefix='qti_{i}'>
      <properties filename='out' stride='10' flush='10'> [ step, time{{picosecond}},
      conserved, temperature{{kelvin}}, kinetic_cv, potential,
      pressure_cv, volume, ensemble_temperature ] </properties>
      <properties filename='pots' stride='10' flush='10'>
      [ pot_component_raw(0), pot_component_raw(1) ] </properties>
      <trajectory filename='pos1' stride='10' bead='0' flush='10'>
         positions
      </trajectory>
      <trajectory filename='xc' stride='10' flush='10'> x_centroid </trajectory>
      <checkpoint stride='800'/>
   </output>
   <total_steps> 2000 </total_steps>
   <prng><seed>31415</seed></prng>
   <ffsocket name='driver' mode='unix' matching='any' pbc='false'>
       <address> f{idx} </address>
       <latency> 1e-3 </latency>
   </ffsocket>
   <ffdebye name='debye'>
       <hessian shape='(3,3)' mode='file'> {HESSIAN_FILE} </hessian>
       <x_reference mode='file'> {X_REF_FILE} </x_reference>
       <v_reference> {V_REF}  </v_reference>
   </ffdebye>
   <system>
      <initialize nbeads='32'>
      <file mode='xyz'> ./qti/init.xyz </file>
         <velocities mode='thermal' units='kelvin'> 300 </velocities>
      </initialize>
      <forces>
         <force forcefield='debye' weight='{i}'> </force>
         <force forcefield='driver' weight='{im1:2.1f}'> </force>
      </forces>
      <motion mode='dynamics'>
      <fixcom> False </fixcom>
         <dynamics mode='nvt'>
            <timestep units='femtosecond'> 1.00 </timestep>
            <thermostat mode='pile_l'>
                <tau units='femtosecond'> 100 </tau>
            </thermostat>
         </dynamics>
      </motion>
      <ensemble>
         <temperature units='kelvin'> 300 </temperature>
      </ensemble>
   </system>
</simulation>
""".format(
            HESSIAN_FILE=HESSIAN_FILE,
            X_REF_FILE=X_REF_FILE,
            V_REF=V_REF,
            i=i,
            idx=idx,
            im1=1 - float(i),
        )
        f.write(strfile)


# %%
# You can run i-PI simulataneous.

for idx, i in enumerate(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]):
    ipi_process = None
    if not os.path.exists(f"qti_{i}.out"):
        ipi_process = subprocess.Popen(["i-pi", f"qti/input_{i}.xml"])
        time.sleep(5)  # wait for i-PI to start
        driver_process = [
            subprocess.Popen(
                ["i-pi-driver", "-u", "-h", f"f{idx}", "-m", "doublewell_1D"]
            )
            for i in range(32)
        ]
    # in a jupiter notebook comments the next 3 lines to be able to
    # load the outputs before the run ends
    if ipi_process is not None:
        ipi_process.wait()
        for i in range(32):
            driver_process[i].wait()

# %%
# The data analysys is done with the following snippets

Q_du_list = []
Q_duerr_list = []

Q_dir_list = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
Q_l_list = [float(lst) for lst in Q_dir_list]

for i in Q_dir_list:
    filename = f"qti_{i}.pots"
    data, _data = ipi.read_output(filename)

    du = data["pot_component_raw(1)"] - data["pot_component_raw(0)"]
    Q_du_list.append(du.mean())
    Q_duerr_list.append(du.std() / len(du) ** 0.5)


Q_du_list = np.asarray(Q_du_list)
Q_duerr_list = np.asarray(Q_duerr_list)

plt.plot(
    Q_l_list,
    (Q_du_list) * 2625.4996,
    color="red",
    label="Quantum thermodynamic integration",
)
plt.fill_between(
    Q_l_list,
    y1=(Q_du_list - Q_duerr_list) * 2625.4996,
    y2=(Q_du_list + Q_duerr_list) * 2625.4996,
    color="red",
    alpha=0.2,
)
plt.plot(
    l_list,
    (du_list) * 2625.4996,
    color="blue",
    label="Classical thermodynamic integration",
)
plt.fill_between(
    l_list,
    y1=(du_list - duerr_list) * 2625.4996,
    y2=(du_list + duerr_list) * 2625.4996,
    color="blue",
    alpha=0.2,
)
plt.legend()
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\left<U - U^{\mathrm{harm}}\right>_{\lambda}$ [kJ/mol]")

# %%
# Then show the free energy

df, dferr = (
    np.trapz(x=Q_l_list, y=Q_du_list),
    np.trapz(x=Q_l_list, y=Q_duerr_list**2) ** 0.5,
)

F = quantum_harmonic_free_energy(W2s**0.5, 300)

print("Quantum harmonic free energy: %15.8f [a.u.]" % (F))
print("Quantum harmonic free energy: %15.8f [eV]" % (F * 27.211386))
print("Quantum harmonic free energy: %15.8f [kJ/mol]" % (F * 2625.4996))
print("")

F = quantum_harmonic_free_energy(W2s**0.5, 300) + df
Ferr = dferr


print("Quantum anharmonic free energy: %15.8f +/- %15.8f [a.u.]" % (F, Ferr))
print(
    "Quantum anharmonic free energy: %15.8f +/- %15.8f [eV]"
    % (F * 27.211386, Ferr * 27.211386)
)
print(
    "Quantum anharmonic free energy: %15.8f +/- %15.8f [kJ/mol]"
    % (F * 2625.4996, Ferr * 2625.4996)
)
print("")
F = -7.53132797e-05 + quantum_harmonic_free_energy(W2s**0.5, 300)

print("Exact free energy: %15.8f [a.u.]" % (F))
print("Exact free energy: %15.8f [eV]" % (F * 27.211386))
print("Exact free energy: %15.8f [kJ/mol]" % (F * 2625.4996))
