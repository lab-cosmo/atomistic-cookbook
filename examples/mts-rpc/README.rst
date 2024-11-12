Multiple time stepping and ring-polymer contraction
===================================================

This notebook provides an introduction to two closely-related techniques,
that are geared towards reducing the cost of calculations by separating
slowly-varying (and computationally-expensive) components of the potential
energy from the fast-varying (and hopefully cheaper) ones.

The first is named `multiple time stepping`, and is a well-established technique
to avoid evaluating the slowly-varying components at every time step of a MD simulation.
It was first introduced in `LAMMPS <https://lammps.org>`_.
`M. Tuckerman, B. J. Berne, and G. J. Martyna, JCP 97(3), 1990 (1992) <https://doi.org/10.1063/1.463137>`_
and can be applied to classical simulations,
typically to avoid the evaluation of long-range electrostatics in classical potentials. 

The second is named `ring polymer contraction`, first introduced in 
`T. E. Markland and D. E. Manolopoulos, JCP 129(2), 024105 (2008) <https://doi.org/10.1063/1.2953308>`_
can be seen as performing a similar simplification `in imaginary time`,
evaluating the expensive part of the potential on a smaller number of PI replicas. 

The techniques can be combined, which reduces even further the computational effort.
This dual approach, which was introduced in
`V. Kapil, J. VandeVondele, and M. Ceriotti, JCP 144(5), 054111 (2016) <(https://doi.org/10.1063/1.4941091>`_
and `O. Marsalek and T. E. Markland, JCP 144(5), (2016) <https://doi.org/10.1063/1.4941093>`_,
is the one that we will discuss here, allowing us to showcase two advanced features of i-PI.
It is worth stressing that MTS and/or RPC can be used very conveniently together with
machine-learning potentials
(see e.g. `V. Kapil, J. Behler, and M. Ceriotti, JCP 145(23), 234103 (2016 <https://doi.org/10.1063/1.4971438>`_ 
for an early application). 

