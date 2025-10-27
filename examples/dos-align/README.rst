Optimizing the energy reference of the DOS during training
==========================================================

This example illustrates how one can optimize the energy reference of the
DOS during model training. The dataset consists 104 Silicon diamond structures.
There are 2 atoms in each unit cell.

The example uses ``ase`` to process quantum-espresso output files,
``featomic`` to compute structural descriptors, ``scipy`` to calculate
cubic Hermite splines and ``matplotlib`` for visualisation.
