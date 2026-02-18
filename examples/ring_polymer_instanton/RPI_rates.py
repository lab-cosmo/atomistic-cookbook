"""
Ring Polymer Instanton Rate Theory: Tunneling Rates 
===================================================

:Authors: Yair Litman `@litman90 <https://github.com/litman90>`_ 


This notebook provides an introduction to the calcultion of tunnelling rates using
ring polymer instanton rate theory. The main theory can be check in [CITE], and the implementation in i-PI is described in
[CITE]. Some practical guidelines can be found in my doctoral thesis [CITE]. 


In this exercises we will use i-PI not to run molecular dynamics but
instead to optimize stationary points on the ring-polymer
potential-energy surface. Then by simple extension, the ring-polymer
instanton will be found which can be used to compute the thermal rate of
a chemical reaction including tunnelling effects. The example which we
will use is for the gas-phase bimolecular scattering reaction of H +
CH\ :math:`_4` . We have used the CBE potential-energy surface [1]. This
requires multiple runs of i-PI for the different stationary points on
the surface followed by one postprocessing calculation to combine all
the data to compute the rate. Full details of the instanton approach are
described in [2].


If you need an introduction to path integral simulations,
or to the use of `i-PI <http://ipi-code.org>`_, which is the
software which will be used to perform simulations, you can see
`this introductory recipe
<https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html>`_.
"""

import os
import subprocess
import time
import warnings

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
#
#A typical workflow is summarized as follows:
#1. Find the minima on the potential energy landscape of the physical system to identify the reactant and the product, and estimate their hessian, normal-mode frequencies and eigenvectors.
#
#2. Fing the first-order saddle point to locate the relevant transition state, and estimate the hessian.
#
#3. Estimate an "unconverged" instanton by finding the first-order saddle point for the ring polymer Hamiltonian for N replicas. Note that this N should be large enough for instanton can be a good guess for a converged calculation with large enough N, but small enough that the cost of the "unconverged" calculation isn't too high. This step should also give you the approximate Hessian for each replica. 
#
#4. Using ring-polymer interpolation calculate the ring polymer instanton for a larger number of replicas. A good rule of thumb is to double the number of replicas.
#
#5. Estimate the instanton by locating the first-order saddle point. 
#
#6. Repeat 4 and 5 until you have converged the instanton with respect to the number of replicas. 
#
#7. Recompute the Hessian for each replica accurately and estimate the rate. 
#
#If rates are required at more than one temperature, it is recommended to start with those closest to the crossover temperature and cool sequentially, again using initial guesses from the previous optimizations.

# %%
# Installing the Python driver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# These calculations will be performed using the  ALBERTO r model
# (CITE)
# that contains a described 
#
# i-PI comes with a FORTRAN driver, which however has to be installed
# from source. We use a utility function to compile it. Note that this requires
# a functioning build system with `gfortran` and `make`.
#
#
#


ipi.install_driver()

# %%
# Step 1 - Optimizing and analysing the reactant
# ----------------------------------------------
#
# First, let's run the geometry optimization for the reactant 
#
# The important parts of the input file indicating the geometry optimization are
# Open and read the XML file
with open("data/input_vibrations_reactant.xml", "r") as file:
    lines = file.readlines()

for line in lines[7:10]:
    print(line, end="")

print("\n[...]\n")

for line in lines[15:18]:
    print(line, end="")


# We can launch i-pi and the driver from a Python script as follows
print('Running data/input_geop_reactant.xml\n')

ipi_process = None
ipi_process = subprocess.Popen(["i-pi", "data/input_geop_reactant.xml"])
time.sleep(5)  # wait for i-PI to start

FF_process =  subprocess.Popen(["i-pi-driver", "-u", "-m", "ch4hcbe"])
FF_process.wait()
ipi_process.wait()

# %%
# The simulation takes 31 steps and the final geometry can be seen in
# the last frame in ``min.xc.xyz``. Once you finish this exercise, you
# could come back here and try to get the optimization to finish in
# fewer steps, e.g. by changing ``sd`` to one of the other optimizers.
# 
# Note that as this is a bimolecular reaction, the H and the
# CH\ :math:`_4` should be well separated and we have used a very large
# box size.
# 
# 2. Next, let's extract the last frame of the geomety optimization trajectory

with open("min_reactant.xyz", "w") as outfile:
    subprocess.run(
        ["tail", "-n", "8", "geop_reactant.xc.xyz"],
        stdout=outfile,
        check=True
    )

     
time.sleep(5)  # wait for i-PI to start
#    We will now compute the hessian of this  minimum eneryg geometry

print('Running data/input_vibrations_reactant.xml\n')
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
plt.xlabel('$i$')
plt.ylabel('$\lambda_i$')
plt.show()

# 
# %%
# Step 2 - Optimizing and analysing the transition state
# ------------------------------------------------------
#
# Next, we will optimize the transition state geometry.
# We proceed in the same way as for the reactant, but using a different input file, that is set up to perform a transition state optimization. 
# The important parts of the input file indicating the transition state optimization are 

with open("data/input_geop_TS.xml", "r") as file:
    lines = file.readlines()

for line in lines[7:10]:
    print(line, end="")

print("\n[...]\n")

for line in lines[15:18]:
    print(line, end="")

# Let's launch i-PI and the driver from a Python script as follows



ipi_process = None
print('Running data/input_geop_TS.xml\n')
ipi_process = subprocess.Popen(["i-pi", "data/input_geop_TS.xml"])
time.sleep(5)  # wait for i-PI to start

FF_process =  subprocess.Popen(["i-pi-driver", "-u", "-m", "ch4hcbe"])
FF_process.wait()
ipi_process.wait()

# Here i-PI treats the TS search as a one-bead instanton
# optimization. The simulation takes 12 steps and puts the optimized
# geometry in ``ts.instanton_FINAL_12.xyz`` and its Hessian in
# ``ts.instanton_FINAL.hess_12``. 




# Visualize the geometry using ``vmd``. #ALBERTO


# %%
# Step 3 - First instanton optimization
# ------------------------------------------------------



#ALBERTO 
#    Go to the folder ``input/instanton/40``. Copy the optimized
#    transition state geometry obtained in Exercise 2 and name it
#    ``init.xyz``. Also copy the transition state hessian and name it
#    ``hessian.dat``. Run i-PI in the usual way, but here we can run 4
#    instances of driver simultaneously using:
# 
#    ``$ i-pi-driver -m ch4hcbe -u &`` ``$ i-pi-driver -m ch4hcbe -u &``
#    ``$ i-pi-driver -m ch4hcbe -u &`` ``$ i-pi-driver -m ch4hcbe -u &``
# 
#    The program first generates the initial instanton guess based on
#    points spread around the TS in the direction of the imaginary mode.
#    It then an optimization which takes 8 steps. The instanton geometry
#    is saved in ``ts.instanton_FINAL_7.xyz`` and should be visualized
#    with ``vmd``. Finally hessians for each bead are also computed and
#    saved in ``ts.instanton_FINAL.hess_7`` in the shape (3n,3Nn) where n
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

# ALBERTO visualize the instanton geometry. 

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


# [1] Jose C Corchado, Jose L Bravo, and Joaquin Espinosa-Garcia. The
# hydrogen abstraction reaction H + CH_4 . I. New analytical potential
# energy surface based on fitting to ab initio calculations. J. Chem.
# Phys., 130(18): 184314, 2009. [2] Jeremy O. Richardson. Ring-polymer
# instanton theory. Int. Rev. Phys. Chem., 37:171, 2018.
# 

