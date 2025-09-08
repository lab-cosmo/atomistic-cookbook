"""
Reaction Path Finding with EON and a Metatomic Potential
========================================================

:Authors: Rohit Goswami `@HaoZeke <https://github.com/haozeke/>`__;
Hanna Tuerk `@HannaTuerk <https://github.com/HannaTuerk>`__; Guillaume
Fraux `@Luthaf <https://github.com/luthaf/>`__; Arslan Mazitov
`@abmazitov <https://github.com/abmazitov>`__; Michele Ceriotti
`@ceriottim <https://github.com/ceriottim/>`__

This example describes how to find the reaction pathway for oxadiazole
formation from N₂O and ethylene. We will use the **PET-MAD** `metatomic
model <https://docs.metatensor.org/metatomic/latest/overview.html>`__ to
calculate the potential energy and forces.

The primary goal is to contrast a standard Nudged Elastic Band (NEB)
calculation using the `atomic simulation environment
(ASE) <https://databases.fysik.dtu.dk/ase/>`__ with more sophisticated
methods available in the `EON
package <https://theochemui.github.io/eOn/>`__. For a complex reaction
like this, a basic NEB implementation can struggle to converge or may
time out. We will show how EON’s advanced features, such as
**energy-weighted springs** and **single-ended dimer searches**, can
efficiently locate and refine the reaction path.

Our approach will be:

1. Set up the **PET-MAD metatomic calculator**.
2. Illustrate the limitations of a standard NEB calculation in ASE.
3. Use EON to find an initial reaction path.
4. Refine the path and locate the transition state saddle point using
   EON’s optimizers, including energy-weighted springs and the dimer
   method.
5. Visualize the final converged pathway.
"""

import linecache
import pathlib
import subprocess

import ase.io as aseio
from ase.visualize import view

from ase.optimize import LBFGS
from ase.mep import NEB
from ase.mep.neb import NEBTools

from metatomic.torch.ase_calculator import MetatomicCalculator

reactant = aseio.read("data/reactant.con")
product = aseio.read("data/product.con")

# subprocess.run(
#     [
#         "mtt",
#         "export",
#         "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt",  # noqa: E501
#     ]
# )

calculator = lambda _: MetatomicCalculator("pet-mad-v1.1.0.pt", device="cpu", non_conservative=False)

reactant.calc = calculator(_)
product.calc = calculator(_)

relax = LBFGS(reactant)
relax.run(fmax=0.01)

relax = LBFGS(product)
relax.run(fmax=0.01)

#view(reactant, viewer='x3d')

#view(product, viewer='x3d')

ipath = [reactant] + [reactant.copy() for img in range(10)] + [product]

for img in ipath:
    img.calc = calculator(_)

band = NEB(ipath, climb=True, k = 9.7, method='improvedtangent')
band.interpolate('idpp')

band.fmax

relax = LBFGS(band)
nebby = NEBTools(band.images)
while nebby.get_fmax() > 0.01:
    relax.run(fmax=0.01, steps=1)
    print(nebby.get_fmax())