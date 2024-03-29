"""
Path integral molecular dynamics
================================

This example shows how to run a path integral molecular dynamics 
simulation using ``i-PI``, analyze the output and visualize the 
trajectory in ``chemiscope``. It uses `LAMMPS <http://lammps.org>`_
as the driver to simulate the q-TIP4P/f water model. 

"""

import chemiscope
import subprocess
import time
import ipi

# %%
# Quantum nuclear effects and path integral methods
# -------------------------------------------------
# 
# The Born-Oppenheimer approximation separates the joint quantum mechanical
# problem for electrons and nuclei into two independent problems. Even though
# often one makes the additional approximation of treating nuclei as classical
# particles, this is not necessary, and in some cases (typically when H atoms are
# present) can add considerable error. 
#
# 
# .. figure:: pimd-slices-round.png
#    :align: center
#    :width: 600px
# 
#    A representation of ther ring-polymer Hamiltonian for a water molecule.
#
# In order to describe the quantum mechanical nature of light nuclei
# (nuclear quantum effects) one of the most widely-applicable methods uses
# the *path integral formalism*  to map the quantum partition function of a
# set of distinguishable particles onto the classical partition function of
# *ring polymers* composed by multiple beads (replicas) with 
# corresponding atoms in adjacent replicas being connected by harmonic springs.
# `The textbook by Tuckerman <https://global.oup.com/academic/product/statistical-mechanics-theory-and-molecular-simulation-9780198825562>`_ 
# contains a pedagogic introduction to the topic, while  
# `this paper <https://doi.org/10.1063/1.3489925>`_ outlines the implementation
# used in ``i-PI``.
#
# The classical partition function of the path converges to quantum statistics
# in the limit of a large number of replicas. In this example, we will use a 
# technique based on generalized Langevin dynamics, known as 
# `PIGLET <http://doi.org/10.1103/PhysRevLett.109.100604>`_ to accelerate the
# convergence.


# %%
# Running PIMD calculations with ``i-PI``
# ---------------------------------------
#
# `i-PI <http://ipi-code.org>`_ is based on a client-server model, with ``i-PI``
# controlling the nuclear dynamics (in this case sampling the path Hamiltonian using
# molecular dynamics) while the calculation of energies and forces is delegated to
# an external client program, in this example ``LAMMPS``. 
#
# An i-PI calculation is specified by an XML file. 

# Open and read the XML file
with open('input.xml', 'r') as file:
    xml_content = file.read()
print(xml_content)

# 
# Runs i-PI and lammps in a subprocess. This is equivalent to 
# running in the command line 
# 
# .. code-block:: bash
#    
#    i-pi input.xml > log &
#    sleep 2
#    lmp -in in.lmp &

ipi_process = subprocess.Popen(["i-pi", "input.xml"])
time.sleep(2) # wait for i-PI to start 
lmp_process = subprocess.Popen(["lmp", "-in", "in.lmp"])

# %%
# If you run this in a notebook, you can go ahead and start loading 
# output files _before_ i-PI and lammps have finished running, by
# skipping this cell

ipi_process.wait()
lmp_process.wait()

