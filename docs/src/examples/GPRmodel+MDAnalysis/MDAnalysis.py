"""
A Gaussian approximation potential (GAP) for Barium Titanate
==============================================================

.. start-body

The present notebook is meant to give you an overview of the main ingredients that you need to build an Machine Learning (ML) interatomic potential with librascal (soon to be replaced by rascaline) and use it in connection with i-Pi (https://github.com/lab-cosmo/i-pi) to generate molecular dynamics (MD) trajectories of the system of interest. 
We will start from building a GAP model for Barium Titanate (:math:`BaTiO_3`), using a training set of structures computed via Density Functional Theory, following the approach outlined in `[1] <https://www.nature.com/articles/s41524-022-00845-0#Sec9">`_. Specifically, each structure of the dataset corresponds to an independent DFT (self-consistent) calculation, providing a reference total potential energy and a set of atomic forces. The mathematical framework that we are going to use to fit the model is Gaussian-Process Regression (GPR) using both total potential energies and atomic forces as target properties [2]. Then we will calculate the Root-Mean-Squared-Error (RMSE) of the ML-predicted energies and forces on a test set to check its performance and run a short NVT simulation at T = 250 K. We will also provide some basic analysis to compute the radial distribution function, the total energy autocorrelation function and the volume fluctuations from our MD trajectory.

In many cases, the dataset that you need to feed your favourite Machine Learning algorithm is already published or has been computed by someone else, so we will not treat here the topic on how to construct an appropriate dataset of structures for a specific application. If you have to build a dataset from scratch, however, then a general rule of thumb is to start from some reference structure available in the literature or on some online repository, like MaterialsProject (https://materialsproject.org/) or the MaterialsCloudArchive (https://archive.materialscloud.org/) and generate a set of structures by randomly displacing atoms around their initial positions. 


"""
# %%

from matplotlib import pylab as plt

import ase
from ase.io import read, write
from ase.build import make_supercell
from ase.visualize import view
import numpy as np
# If installed -- not essential, though
try:
    from tqdm.notebook import tqdm
except ImportError:
    tqdm = (lambda i, **kwargs: i)

from time import time

from rascal.models import Kernel, train_gap_model, compute_KNM
from rascal.representations import SphericalInvariants
from rascal.utils import from_dict, to_dict, CURFilter, dump_obj, load_obj

# %%
# Let's first load the dataset
# 
dataset = read('Dataset/BTO_dataset.extxyz', index=':')
energies = []
forces = []

for frame in dataset:
    energies.append(frame.get_total_energy())
    forces.append(frame.get_forces())

energies = np.array(energies)
forces = np.array(forces)

#Keys of the arrays dictionary
print(energies.shape, forces.shape)

#%%
# References
# ----------
# [1] L. Gigli, M. Veit, M. Kotiuga, G. Pizzi, N. Marzari and M. Ceriotti, npj Computational Materials 8, 209, 2022 
#
# [2] V. L. Deringer, A. P. Bartók, N. Bernstein, D. M. Wilkins, M. Ceriotti and G. Csànyi, Chem. Rev. 121, 16, 2021

