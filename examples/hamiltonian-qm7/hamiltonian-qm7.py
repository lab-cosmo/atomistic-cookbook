"""
Hamiltonian learning
=============================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo>`__,
          Jigyasa Nigam `@curiosity54 <https://github.com/curiosity54>`__

This tutorial explains how to train a machine learning model for the
electronic Hamiltonian of a periodic system. Even though we focus on
periodic systems, the code and techniques presented here can be directly
transferred to molecules.
"""

# %%
# First, import the necessary packages
#

import os


warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

# sphinx_gallery_thumbnail_number = 3


# %%
# Get Data and Prepare Data Set
# -----------------------------
#


