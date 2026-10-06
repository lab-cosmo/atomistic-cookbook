The metatomic hourglass design
==============================

This example takes three machine-learning models with very different origins,
PET-MAD trained with metatrain, a MACE foundation model, and a branch of the
multitask DPA-3.1-3M model from deepmd-kit, and exports them all to the
common metatomic format. It then uses them to run the same molecular dynamics
simulation with five different engines: ASE, LAMMPS, GROMACS, i-PI, and
TorchSim.
