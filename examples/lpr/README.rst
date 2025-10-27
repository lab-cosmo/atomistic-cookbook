Local Prediction Rigidity (LPR)
===============================

This is an example of how one can calculate the local prediction rigidity
for the atoms of "test" set structures, given two differently composed
"training" set structures.

It uses ``featomic`` to compute descriptors for a database of atomic
structures, and ``scikit-matter`` to compute the LPR. The results are
visualized using ``chemiscope`` widgets.
