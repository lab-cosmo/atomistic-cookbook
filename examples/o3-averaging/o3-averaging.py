r"""
Rotational averaging of non-equivariant models
==========================

:Authors:
Marcel Langer `@sirmarcel <https://github.com/sirmarcel/>`_;
Michele Ceriotti `@ceriottm <https://github.com/ceriottm/>`_

DESCRIPTION
"""

# %%

import os
import subprocess
import time
import xml.etree.ElementTree as ET

import ase
import ase.io
import chemiscope
import ipi


# import matplotlib.pyplot as plt
# import numpy as np


# %%
# Show the problem
# ----------------------------------


# %%
#

# Open and show the relevant part of the input
xmlroot = ET.parse("data/input-noo3.xml").getroot()
print("      " + ET.tostring(xmlroot.find(".//ffrotations"), encoding="unicode"))


# %%
# Installing the Python driver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# i-PI comes with a FORTRAN driver, which however has to be installed
# from source. We use a utility function to compile it. Note that this requires
# a functioning build system with `gfortran` and `make`.

ipi.install_driver()

# %%
# We launch the i-PI and LAMMPS processes, exactly as in the
# ``path-integrals`` example.

# don't rerun if the outputs already exist
ipi_process = None
if not os.path.exists("water-noo3.out"):
    ipi_process = subprocess.Popen(["i-pi", "data/input-noo3.xml"])
    time.sleep(2)  # wait for i-PI to start
    lmp_process = [subprocess.Popen(["lmp", "-in", "data/in.lmp"]) for i in range(1)]
    driver_process = [
        subprocess.Popen(
            ["i-pi-driver", "-u", "-a", "h2o-noo3", "-m", "noo3-h2o"], cwd="data/"
        )
        for i in range(1)
    ]

# %%
# Skip this cell if you want to run in the background
if ipi_process is not None:
    ipi_process.wait()
    lmp_process[0].wait()
    driver_process[0].wait()


# %%
#
# Discuss how molecules get oriented

traj_data = ase.io.read("water-noo3.pos_0.extxyz", ":")

# %%
# then, assemble a visualization
chemiscope.show(
    frames=traj_data,
    settings=chemiscope.quick_settings(trajectory=True),
    mode="structure",
)
