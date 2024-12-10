"""
Melting point determination with machine learning interatomic potentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


# %%
# This recipe discusses melting point determination of general materials
# with machine learning interatomic potentials (MLIPs). Various methods
# have been proposed in the literature to predict melting points of
# materials from atomistic simulations [REFERENCE] - we find that the
# interface pinning method from Pedersen and coworkers [REFERENCE] is
# particularly well-suited for MLIPs (why reasonably system sizes?).
# 
# The tutorial is structured as follows:
# 
# 1. Introduction to the interface pinning method
# 2. Performing a protoypical interface pinning simulation for a
#    Lennard-Jones solid-liquid interface
# 3. Performing an interface pinning simulation for water-ice and Gallium
#    Arsenide
# 4. Uncertainty quantification
# 5. Troubleshooting
# 6. Summary and outlook
# 
# Ulf Pedersen has a series of Youtube videos on the interface pinning
# method for the Lennerd-Jones system, which can be found here:
# https://www.youtube.com/watch?v=4fftxOIFRm8 as well as a LAMMPS-PLUMED
# tutorial
# https://c4science.ch/source/lammps/browse/master/examples/USER/misc/pinning;35f2cfa0bfafa064325a3eb8acc86d3627c6a33a
# one which we will base part 2 of this tutorial. Make sure to check it
# out!
# 

import ase.build
import ase.io
import chemiscope
import subprocess
from ase.visualize import view


# %%
# 1. Introduction to the interface pinning method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


# %%
# 
# 


# %%
# 2. Performing a protoypical interface pinning simulation for a Lennard-Jones solid-liquid interface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 



# fcc structure of a lennard jones crystal
# build fcc structure with lattice constant 1.

a1 = ase.build.bulk('He', 'fcc', a=0.971)
a1 = a1.repeat((8,8,20))
ase.io.write("init.xyz", a1)

chemiscope.show(frames=[a1], mode="structure")


# %%
# Running isotropic MD
# ^^^^^^^^^^^^^^^^^^^^
# 


# %%
# \```bash
# 
# ::
# 
#    i-pi inputs/melting.xml > log &
#    sleep 2
#    i-pi-driver 
# 

ipi_process = subprocess.Popen(["i-pi", "inputs/LJ/NPT.xml"])

# visualize the trajectory

chemiscope.show(
    ase.io.read("simulation.pos_0.extxyz",":"),
    mode="structure",
    settings=chemiscope.quick_settings(
        trajectory=True, structure_settings={"unitCell": True}
    ),
)


# build fcc LJ

#lennard jones units in lammps (in sigma):

"""
<fflj name='lj' pbc='False'>
    <parameters> { eps:0.0, sigma:1.0 } </parameters>
</fflj>
"""




# %%
# 3. Performing an interface pinning simulation for water-ice and Gallium Arsenide
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

# building Gallium supercell:
# we have downloaded the gallium supercell from: 

frame = ase.io.read("./data/Ga/Ga.cif").repeat((4,2,18))

frame

# building G base structure
view(frame)


# %%
# 4 Uncertainty quantification of the melting point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


# %%
# 4.1 Sampling uncertainty
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 


# %%
# 
# 


# %%
# 4.2 Model uncertainty
# ^^^^^^^^^^^^^^^^^^^^^
# 


# %%
# As in any simulation we might be interested in quantifying the
# simulation uncertainty.
# 


# %%
# 5. Troubleshooting
# ~~~~~~~~~~~~~~~~~~
# 


# %%
# 5.1 My chemical potential vs. temperatur curve is not well behaved
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 


# %%
# The chemical potential never crosses zero?
# 


# %%
# .. figure:: attachment:plot_broken_pinning.png
#    :alt: plot_broken_pinning.png
# 
#    plot_broken_pinning.png
# 


# %%
# .. figure:: Gallium_interface.png
#    :alt: Alt text
# 
#    Alt text
# 


# %%
# See how towards the left side of the simulation box, another crystalline
# phase is growing?
# 


# %%
# .. figure:: Gallium_interface_partial_recryst.png
#    :alt: Alt text
# 
#    Alt text
# 


# %%
# 6 Summary
# ~~~~~~~~~
# 
# MLIPs in combination with the interface pinning method are a great match
# for determing melting points of materials and study interface processes.
# 


# %%
# 
# 