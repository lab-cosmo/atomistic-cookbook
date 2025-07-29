# -*- coding: utf-8 -*-
"""
Reaction Path Finding with EON and a Metatomic Potential
=========================================================

:Authors: Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`_;
          Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`_;
          Guillaume Fraux `@Luthaf <https://github.com/luthaf/>`_;
          Arslan Mazitov `@abmazitov <https://github.com/abmazitov>`_;
          Michele Ceriotti `@ceriottim <https://github.com/ceriottim/>`_

This example describes how to find the reaction pathway for oxadiazole formation
from N₂O and ethylene. We will use the **PET-MAD** `metatomic model
<https://docs.metatensor.org/metatomic/latest/overview.html>`_ to calculate the
potential energy and forces.

The primary goal is to contrast a standard Nudged Elastic Band (NEB) calculation
using the `atomic simulation environment (ASE)
<https://databases.fysik.dtu.dk/ase/>`_ with more sophisticated methods
available in the `EON package <https://theochemui.github.io/eOn/>`_. For a
complex reaction like this, a basic NEB implementation can struggle to converge
or may time out. We will show how EON's advanced features, such as
**energy-weighted springs** and **single-ended dimer searches**, can efficiently
locate and refine the reaction path.

Our approach will be:

1. Set up the **PET-MAD metatomic calculator**.
2. Illustrate the limitations of a standard NEB calculation in ASE.
3. Use EON to find an initial reaction path.
4. Refine the path and locate the transition state saddle point using EON's
   optimizers, including energy-weighted springs and the dimer method.
5. Visualize the final converged pathway.
"""

import linecache
import pathlib
import subprocess
