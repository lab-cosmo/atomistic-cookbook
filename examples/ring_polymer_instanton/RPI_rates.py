"""
Ring Polymer Instanton Rate Theory: Tunneling Rates 
===================================================

:Authors: Yair Litman `@litman90 <https://github.com/litman90>`_ 

This notebook introduces the calculation of tunneling rates using
ring-polymer instanton rate theory. A comprehensive presentation of the
instanton formalism can be found in the review article by
`J. Richardson, Ring-polymer instanton theory, Int. Rev. Phys. Chem., 37, 171, 2018 <https://doi.org/10.1080/0144235X.2018.1472353>`_,
while the implementation within i-PI is described in
`V. Kapil et al., Comp. Phys. Commun., 236, 214, 2020 <https://doi.org/10.1016/j.cpc.2018.09.020>`_.
Additional practical details are available in Yair Litman's doctoral thesis
<https://pure.mpg.de/rest/items/item_3246769_3/component/file_3246770/content>`_.


In these exercises, i-PI will be used not for molecular dynamics simulations,
but to optimize stationary points on the ring-polymer potential-energy surface.
From these stationary points, the ring-polymer instanton can be obtained and
employed to evaluate thermal reaction rates that include tunneling contributions.



As a working example, we consider the gas-phase bimolecular reaction
H + CH\ :math:`_4`. The calculations are performed using the CBE
potential-energy surface reported in
`J. C. Corchado et al., J. Chem. Phys., 130, 184314, 2009 <https://doi.org/10.1063/1.3132223>`_.

If you are new to path-integral simulations or to the use of
 `i-PI <http://ipi-code.org>`_, which is the
software which will be used to perform simulations, you can see
`this introductory recipe
<https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html>`_.
"""

import os
import subprocess
import time
import warnings
import glob
import re

import chemiscope
import ipi
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cycler
from scipy import constants

plt.rc('text', usetex=True)
plt.rc('ps', usedistiller = 'xpdf')
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
params = {'figure.figsize':(12, 5)}
plt.rcParams['axes.prop_cycle'] = cycler(color=['r','b','k','g','y','c'])

# %%
# Calculation Workflow
# --------------------
# The calculation of tunneling rates using ring-polymer instanton theory
# is a multi-step procedure that requires the optimization of several
# stationary points on the system’s potential-energy surface.
# The workflow is not fully automated and therefore involves multiple
# independent runs of i-PI.
#
# A typical workflow is summarized below:
#
#1. Locate the minima on the physical potential-energy surface to identify
#   the reactant and product states. Compute their Hessians, normal-mode
#   frequencies, and eigenvectors.
#
#2. Find the first-order saddle point corresponding to the transition
#   state and evaluate its Hessian.
#
#3. Obtain an "unconverged" instanton by locating the first-order saddle
#   point of the ring-polymer Hamiltonian using N replicas. The chosen
#   N should be large enough that the instanton provides a reasonable
#   initial guess for a more converged calculation with a larger number
#   of replicas, but small enough to keep the computational cost moderate.
#   This step also provides an approximate Hessian for each replica.
#
#4. Use ring-polymer interpolation to construct an instanton with a
#   larger number of replicas. As a rule of thumb, the number of replicas
#   can be doubled.
#
#5. Re-optimize the instanton by locating the corresponding first-order
#   saddle point.
#
#6. Repeat steps 4 and 5 until the instanton is converged with respect
#   to the number of replicas.
#
#7. Recompute the Hessian for each replica accurately and evaluate
#   the reaction rate.
#
#If rates are required at multiple temperatures, it is recommended to
#start with temperatures close to the crossover temperature and then
#cool sequentially, using the optimized instanton from the previous
#temperature as the initial guess for the next calculation.

# %%
# Installing the Python driver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# i-PI provides a FORTRAN driver that includes the CBE potential-energy
# surface. However, the driver must be compiled from source.
# We use a utility function to compile it.
#
# Note that this requires a working build environment with
# `gfortran` and `make` available.
#
#



ipi.install_driver()

# %%
# Step 1 - Optimizing and Analyzing the Reactant
# ----------------------------------------------
#
# 1. We begin by performing a geometry optimization of the reactant.
#
# The relevant sections of the input file that control the geometry
# optimization are highlighted below.
#
# Open and inspect the XML input file:

with open("data/input_vibrations_reactant.xml", "r") as file:
    lines = file.readlines()

for line in lines[18:26]:
    print(line, end="")



# We can launch i-pi and the driver from a Python script as follows

ipi_process = None
ipi_process = subprocess.Popen(["i-pi", "data/input_geop_reactant.xml"])
time.sleep(5)  # wait for i-PI to start

FF_process =  subprocess.Popen(["i-pi-driver", "-u", "-m", "ch4hcbe"])
FF_process.wait()
ipi_process.wait()

# %%
# The simulation converges in 31 steps, and the optimized geometry can
# be found in the last frame of ``min.xc.xyz``. After completing this
# exercise, you may return to this point and attempt to reduce the number
# of optimization steps, for example by replacing ``sd`` with one of the
# alternative optimizers.
#
# Since this is a bimolecular reaction, the H atom and CH\ :math:`_4`
# molecule must remain well separated. For this reason, a very large
# simulation box has been used.
#
# 2. Extract the final frame of the geometry optimization trajectory
#    and use it to compute the harmonic vibrational frequencies.


with open("min_reactant.xyz", "w") as outfile:
    subprocess.run(
        ["tail", "-n", "8", "geop_reactant.xc.xyz"],
        stdout=outfile,
        check=True
    )

     
time.sleep(5)  # wait for i-PI to start
#    We will now compute the hessian of this  minimum eneryg geometry

ipi_process = None
ipi_process = subprocess.Popen(["i-pi", "data/input_vibrations_reactant.xml"])
time.sleep(5)  # wait for i-PI to start

FF_process =  subprocess.Popen(["i-pi-driver", "-u", "-m", "ch4hcbe"])
FF_process.wait()
ipi_process.wait()

# The hessian is saved in ``vib_reactant.hess`` and its eigenvalues in ``vib_reactant.eigval``.

# Check that it has the required number (9) of  almost zero eigenvalues. Why?

eigvals = np.genfromtxt('vib_reactant.vib.eigval')
plt.plot(eigvals,'x')
plt.xlabel('mode number')
plt.ylabel('Eigenvalues (a.u.)')
plt.show()

# 
# %%
# Step 2 - Optimizing and Analyzing the Transition State
# ------------------------------------------------------
#
# We now optimize the transition-state (TS) geometry.
# The procedure is analogous to that used for the reactant, but we
# employ a different input file configured specifically for a
# transition-state optimization.
#
# The relevant sections of the input file that define the
# transition-state optimization are highlighted below:
#


with open("data/input_geop_TS.xml", "r") as file:
    lines = file.readlines()

for line in lines[18:31]:
    print(line, end="")

# Note that Here i-PI treats the TS search as a one-bead instanton
# optimization. 

# We now launch i-PI together with the driver using commands analogous
# to those used in the previous step:


ipi_process = None
ipi_process = subprocess.Popen(["i-pi", "data/input_geop_TS.xml"])
time.sleep(5)  # wait for i-PI to start

FF_process =  subprocess.Popen(["i-pi-driver", "-u", "-m", "ch4hcbe"])
FF_process.wait()
ipi_process.wait()

# The simulation typically converges in 11–12 steps and writes the
# optimized geometry to ``ts.instanton_FINAL_xx.xyz`` and the corresponding
# Hessian to ``ts.instanton_FINAL.hess_xx``.


ts_energy_ev = np.loadtxt('ts.out')[1,-1]  

# %%
# Step 3 - First instanton optimization
# ------------------------------------------------------

# We now continue with the optimization of the ring-polymer instanton. This is a
# first-order saddle point of the ring-polymer Hamiltonian, and therefore
# the optimization procedure is analogous to that used for the transition state.
# To generate the initial guess for the instanton optimization, we use the
# transition state geometry and hessian obtained in the previous step. The

# Let us first now retrieve the filenames of the Hessian and geometry
# corresponding to the converged instanton.
# the next calculation.

final_step = max(
    int(re.search(r"hess_(\d+)$", f).group(1))
    for f in glob.glob("ts.instanton_FINAL.hess_*")
)

ts_hess_file = f"ts.instanton_FINAL.hess_{final_step}"
ts_xyz_file  = f"ts.instanton_FINAL_{final_step}.xyz"


import shutil

shutil.copy(ts_xyz_file, "init_inst_geop.xyz")
shutil.copy(ts_hess_file, "init_inst_geop.hess")


#    We run i-PI in the usual way, but here we launch four
#    driver instances simultaneously using:

ipi_process = None
ipi_process = subprocess.Popen(["i-pi", "data/input_geop_reactant.xml"])
time.sleep(5)  # wait for i-PI to start

FF_process = [
    subprocess.Popen(["i-pi-driver", "-u", "-m", "ch4hcbe"])
    for i in range(4)
]
#FF_process.wait()
ipi_process.wait()


#    The program first generates an initial instanton guess by placing
#    points around the transition state along the direction of the
#    imaginary mode. It then performs an optimization, which converges
#    in 8 steps. The optimized instanton geometry is written to
#    ``ts.instanton_FINAL_7.xyz`` and can be visualized using ``vmd``.
#    Finally, the Hessians for each bead are computed and saved in
#    ``ts.instanton_FINAL.hess_7`` with dimensions (3n, 3Nn), where n
#    is the number of atoms.



# %%
# Step 4 - Second and subsequent instanton optimizations 
# ------------------------------------------------------

# 1.  Make the folder ``input/instanton/80``.
# 
# 2.  Go to the folder ``input/instanton/80``.
# 
# 3.  Copy the optimized instanton geometry obtained in Exercise 3 and
#     name it ``init0``.
# 
# 4.  Copy the last hessian obtained in Exercise 3 and name it ``hess0``.
# 
# 5.  Interpolate the instanton and the hessian to 80 beads by typing:
# 
#     ``$ python ${ipi-path}/tools/py/Instanton_interpolation.py -m -xyz init0 -hess hess0 -n 80``
# 
# 6.  Rename the new hessian and instanton geometry to ``hessian.dat`` and
#     ``init.xyz`` respectively
# 
# 7.  Copy the ``input.xml`` file from ``input/instanton/40/``.
# 
# 8.  Change the number of beads from 40 to 80 in ``input.xml``.
# 
# 9.  Change the hessian shape from (18,18) to (18,1440) in ``input.xml``.
# 
# 10. Run as before. The program performs a optimization which takes 6
#     steps and then computes a hessian.


# %%
# Step 5 -  Postprocessing for the rate calculation 
# ------------------------------------------------------

# Note that we have been optimizing half ring polymers as we expect the
# forward and backward imaginary time path between the reactant and the
# product to be the same. Thus the actual number of replicas for the
# instanton obtained by optimizing a ring polymer with 80 replicas are
# 160. This is a little confusing currently, and may be changed in the
# future, so when using the Instanton posprocessing tool check use the
# help option (using -h) to find out what is needed. Before starting this
# section, ensure that ``$PYTHONPATH`` includes the i-PI directory and
# that the variable ``$ipi_path`` is correctly set in the postprocessing
# script ``${ipi-path}/tools/py/Instanton_postproc.py`` where
# ``${ipi-path}`` is the location of i-PI directory.
# 
# 1. To compute the CH\ :math:`_4` partition function, go the
#    input/reactant/phonons folder and type
# 
#    ``$ python ${ipi-path}/tools/py/Instanton_postproc.py RESTART -c reactant -t 300 -n 160 -f 5``
# 
#    which computes the ring polymer parition function for CH\ :math:`_4`
#    with N = 160. Look at the output and make a note of the
#    translational, rotational and vibrational partition functions. You
#    may also want to put > data.out after the command to save the text
#    directly to a file.
# 
# 2. To compute the TS partition function, go to ``input/TS`` and type
# 
#    ``$ python ${ipi-path}/tools/py/Instanton_postproc.py RESTART -c TS -t 300 -n 160``
# 
#    which computes the ring polymer parition function for the TS with
#    :math:`N = 160`. Look for the value of the imaginary frequency and
#    use this to compute the crossover temperature defined by
# 
#    .. math:: \beta_c = \dfrac{2 \pi}{\omega_b}.
# 
#    Be careful of units! You should find that it is about 340 K. In the
#    cell below we define a quick function that helps you with the
#    necessary unit conversions.
# 

####### Physical Constants ###########
cm2hartree=1./(constants.physical_constants['hartree-inverse meter relationship'][0]/100)
Boltzmannau = constants.physical_constants['Boltzmann constant in eV/K'][0]*constants.physical_constants['electron volt-hartree relationship'][0]
########## Temperature <-> beta conversion ############
K2beta = lambda T : 1./T/Boltzmannau
beta2K = lambda B : 1./B/Boltzmannau

def omb2Trecr(omega):
    return beta2K(2.*np.pi/(omega*cm2hartree))

omega = 1486.88   # given in reciprocal cm
print('The barrier frequency is %.2f cm^-1. \nThe first recrossing temperature is ~ %.2f K.'%(omega,omb2Trecr(omega)))


# %%
# 3. To compute the instanton partition function, :math:`B_N` and action,
#    go to ``input/instanton/80`` and type
# 
#    ``$ python ${ipi-path}/tools/py/Instanton_postproc.py RESTART -c instanton -t 300 > data.txt``
# 
#    Then it is a simple matter to combine the partition functions,
#    :math:`B_N`, :math:`S`, etc. into the formula given for the rate.
#    Compare the instanton results with those of the transition state in
#    order to compute the tunnelling factor.
# 
#    .. math::  \kappa = f_{\mathrm{trans}}\ f_{\mathrm{rot}}\ f_{\mathrm{vib}}\ e^{-S/\hbar + \beta V^{\dagger}}
# 
#    .. math::  f_{\mathrm{trans}} = \dfrac{Q^{\mathrm{inst}}_{\mathrm{trans}}}{Q^{\mathrm{TS}}_{\mathrm{trans}}} 
# 
#    .. math::  f_{\mathrm{rot}} = \dfrac{Q^{\mathrm{inst}}_{\mathrm{rot}}}{Q^{\mathrm{TS}}_{\mathrm{rot}}} 
# 
#    .. math::  f_{\mathrm{vib}} = \sqrt{\dfrac{2 \pi N B_N}{\beta \hbar^2}}\dfrac{Q^{\mathrm{inst}}_{\mathrm{vib}}}{Q^{\mathrm{TS}}_{\mathrm{vib}}} 
# 
#    Note that it is the log of the vibrational partition function which
#    is printed, so you will have to convert this. In this way, you should
#    find that the rate is about 9.8 times faster due to tunnelling. Which
#    is the major contributing factor?
# 

Q_trn_TS = 10.187982157
Q_rot_TS = 1206.15097078
Q_vib_TS = np.exp(-44.2783849573)
Q_trn_inst = 10.188
Q_rot_inst = 1251.044
Q_vib_inst = np.exp(-43.478)
BN = 14.289
recip_betan_hbar =   0.15201
N = 160
Soverhbar = -25.026
VoverBeta = 25.1656385465

def kappa(Q_trn_TS,Q_rot_TS,Q_vib_TS,Q_trn_inst,Q_rot_inst,Q_vib_inst,BN,recip_betan_hbar,N,Soverhbar,VoverBeta):
    f_trn = Q_trn_inst / Q_trn_TS
    f_rot = Q_rot_inst / Q_rot_TS
    f_vib = np.sqrt(2.*np.pi*BN*recip_betan_hbar)*Q_vib_inst/Q_vib_TS
    
    kappa = f_trn * f_rot * f_vib * np.exp(Soverhbar + VoverBeta)
    
    #printing out the transmission factor and the relevant contributions. 
    print('f_trans = %8.5e'%f_trn)
    print('f_rot = %8.5e'%f_rot)
    print('f_vib = %8.5e'%f_vib)
    print('exp(-S/hbar) = %8.5e'%np.exp(Soverhbar))
    print('exp(V/beta) = %8.5e'%np.exp(VoverBeta))
    print('=============================')
    print('kappa = %8.5e'%kappa)
        
    return kappa

kappa(Q_trn_TS,Q_rot_TS,Q_vib_TS,Q_trn_inst,Q_rot_inst,Q_vib_inst,BN,recip_betan_hbar,N,Soverhbar,VoverBeta)


# %%
# 4. In this tutorial, the number of beads used in the example is much
#    more than necessary. Please try the instanton calculation with 10 or
#    20 beads, and see how the results :math:`\kappa` compare.
# 

kappadat = np.array([
    [10  ,  18.486247675370873],
    [20  ,  11.481103709870357],
    [40  ,  10.127813329699068],
    [80  ,   9.809437521247709]])

plt.plot(kappadat[:,0],kappadat[:,1],'ro--')
plt.xlabel(r'number of instanton beads')
plt.ylabel(r'$\kappa$')
plt.show()



