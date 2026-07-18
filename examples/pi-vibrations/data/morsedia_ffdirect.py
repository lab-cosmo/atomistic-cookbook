"""Morse ffdirect helper
=====================

Custom i-PI ffdirect potential matching the tutorial Morse diatomic model.
"""

from __future__ import annotations

import json

import numpy as np

try:
    from ipi.pes.dummy import Dummy_driver
except ModuleNotFoundError:
    class Dummy_driver:
        """Minimal fallback used when this helper is imported outside i-PI."""

        def __init__(self, *args, **kwargs):
            pass


__DRIVER_NAME__ = "morsedia_custom"
__DRIVER_CLASS__ = "MorseDiatomic_driver"


class MorseDiatomic_driver(Dummy_driver):
    """Exact Python equivalent of the Fortran `morsedia` tutorial driver.

    The potential depends on the distance between two atoms:

        V(r) = D * (exp(-2 a (r-r0)) - 2 exp(-a (r-r0)))

    Parameters are expected in i-PI internal units, matching the original
    Fortran driver defaults used in the tutorial.
    """

    def __init__(self, r0=1.8323918, D=0.18748563, a=1.1605, *args, **kwargs):
        self.r0 = float(r0)
        self.D = float(D)
        self.a = float(a)
        super().__init__(*args, **kwargs)

    def compute_structure(self, cell, pos):
        pos3 = np.asarray(pos, dtype=float).reshape(-1, 3)
        if pos3.shape[0] != 2:
            raise ValueError(
                "MorseDiatomic_driver expects exactly two atoms, got "
                f"{pos3.shape[0]}"
            )

        diff = pos3[0] - pos3[1]
        distance = np.linalg.norm(diff)
        if distance == 0.0:
            raise ValueError("Zero interatomic distance in MorseDiatomic_driver")

        displacement = distance - self.r0
        exp1 = np.exp(-self.a * displacement)
        exp2 = np.exp(-2.0 * self.a * displacement)

        potential = self.D * (exp2 - 2.0 * exp1)
        prefactor = -2.0 * self.a * self.D * (exp1 - exp2) / distance

        force = np.zeros_like(pos3)
        force[0] = prefactor * diff
        force[1] = -force[0]

        virial = np.asarray(cell, dtype=float) * 0.0
        extras = json.dumps({"dipole": [0.0, 0.0, 0.0]})
        return potential, force.reshape(np.asarray(pos).shape), virial, extras