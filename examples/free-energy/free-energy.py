r"""
Free energy methods
==========================

:Authors: Venkat Kapil `@venkatkapil24 <https://github.com/venkatkapil24/>`_
          and Davide Tisi `@DavideTisi <https://github.com/DavideTisi/>`_

This example shows how to perform free energy calculation on a
effective Hamiltonian of a hydrogen atom in an effectively 1D double well
from Ref. `Y. Litman et al., JCP (2022)
<https://pubs.aip.org/aip/jcp/article/156/19/194107/2841188/Dissipative-tunneling-rates-through-the>`_
"""

# %%

import os
import subprocess
import time

import ipi.utils.parsing as pimdmooc
import numpy as np
import matplotlib.pyplot as plt



# %%
# Model System
# ------------
#
# We will consider the effective Hamiltonian of a hydrogen atom
# in an effectively 1D double well from
# Ref. `Y. Litman et al., JCP (2022)
# <https://pubs.aip.org/aip/jcp/article/156/19/194107/2841188/Dissipative-tunneling-rates-through-the>`_
# Its potential energy surface (PES) is described by the function
# 
# .. math::
#
#    V = A x^2 + B x^4 + \frac{1}{2} m \omega^2 y^2 + \frac{1}{2} m \omega^2 z^2
# 
# with:
# 
# .. math::
#     
#     \begin{align}
#     m &= 1837.107~\text{a.u.}\\ 
#     \omega &= 3800 ~\text{cm}^{-1} = 0.017314074~\text{a.u.}\\ 
#     A &= -0.00476705894242374~\text{a.u.}\\
#     B &= 0.000598024968321866~\text{a.u.}\\
#     \end{align}
#
