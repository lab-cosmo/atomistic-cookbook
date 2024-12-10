Uncertainty Quantification for MLIPs: LLPR approach in MACE
===========================================================

This example demonstrates how last-layer prediction rigidity (LLPR)-based
uncertainty quantification can be done for machine learning interatomic
potentials, for a dataset of silicon 10-mers.

It primarily adopts ``MACE`` for the demonstration, using an implementation
available in a fork (to be merged via PR to `MACE/develop`) found `here
<https://github.com/SanggyuChong/mace/tree/LLPR_loss_based>`. Data loading,
processing, and calibration will all be done with the functions internally
available within this MACE implementation.
