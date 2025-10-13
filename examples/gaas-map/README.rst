Structure-property map for a GaAs training set
==============================================

This is an example of the analysis of a training set for ML potentials.
The data consists in a collection of Ga(x)As(1-x) structures, spanning
a broad range of geometries and compositions.

The example uses ``featomic`` to compute structural descriptors, and
``scikit-matter`` to determine an informative low-dimensional representation.
Finally, the resulting map is visualized using ``chemiscope`` widgets,
including also the visualization of force data as vector shapes.
