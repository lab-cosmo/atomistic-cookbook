Mendeleev's Nano-Soup
=====================

This example demonstrates how to simulate a complex nanoparticle containing a mixture
of elements using the PET-MAD universal potential, ASE, and `i-PI <https://ipi-code.org>`_. 

It starts by preparing random multi-element configurations (a "nano-soup") and then 
uses Replica Exchange Molecular Dynamics (REMD) coupled with Monte Carlo atomic swaps 
to sample the composition and structure of the nanoparticle across a range of 
temperatures.