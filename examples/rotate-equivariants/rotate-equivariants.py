"""
Rotating equivariants
=====================

:Authors:
    Filippo Bigi `@frostedoyster <https://github.com/frostedoyster/>`_;
    Michelangelo Domina `@MichelangeloDomina <https://github.com/MichelangeloDomina/>`_

This example shows how to rotate equivariant properties of atomic structures
using the ``scipy`` and ``quaternionic`` libraries. The equivariant properties for
this example are computed by the ``featomic`` library.
"""

# %%

import ase.build
import featomic
import metatensor
import numpy as np
import spherical
from scipy.spatial.transform import Rotation


# %%
# The quaternionic library
# ------------------------
# The `quaternionic <https://quaternionic.readthedocs.io/en/latest/>`_ library
# is a Python library for working with quaternions and much more. In our case,
# we will use it to calculate Wigner D matrices, which are essential to rotate
# equivariant spherical tensors.
#
# Two remarks before beginning:
# - Scipy uses a different notation for quaternions from the one in quaternionic
#     + Scipy quaternion:  [x,y,z,w]
#     + Quaternionic:      [w,x,y,z]
#
# - If the only difference between two quaternions is their sign, they represent the
#   same rotation. When this happens, we can just check their scalar product: if it
#   is negative (and it should be -1 when this happens) then we can invert the sign
#   of one of the two quaternions.

# %%
# Utility functions
# -----------------
# We define a few utility functions


def get_random_rotation():
    return Rotation.random()


def complex_to_real_spherical_harmonics_transform(ell: int):
    # Generates the transformation matrix from complex spherical harmonics
    # to real spherical harmonics for a given l.
    # Returns a transformation matrix of shape ((2l+1), (2l+1)).

    if ell < 0 or not isinstance(ell, int):
        raise ValueError("l must be a non-negative integer.")

    # The size of the transformation matrix is (2l+1) x (2l+1)
    size = 2 * ell + 1
    T = np.zeros((size, size), dtype=complex)

    for m in range(-ell, ell + 1):
        m_index = m + ell  # Index in the matrix
        if m > 0:
            # Real part of Y_{l}^{m}
            T[m_index, ell + m] = 1 / np.sqrt(2) * (-1) ** m
            T[m_index, ell - m] = 1 / np.sqrt(2)
        elif m < 0:
            # Imaginary part of Y_{l}^{|m|}
            T[m_index, ell + abs(m)] = -1j / np.sqrt(2) * (-1) ** m
            T[m_index, ell - abs(m)] = 1j / np.sqrt(2)
        else:  # m == 0
            # Y_{l}^{0} remains unchanged
            T[m_index, ell] = 1

    return T


def scipy_quaternion_to_quaternionic(q):
    # This function is used to convert a quaternion between the format
    # the formats (w, x, y, z) and (x, y, z, w).
    # Note: 'xyzw' is the format used by scipy.spatial.transform.Rotation
    # while 'wxyz' is the format used by quaternionic.
    qx, qy, qz, qw = q
    return np.array([qw, qx, qy, qz])


# %%
# Generating equivariants
# -----------------------
#
# Here, we generate some equivariants for a water molecule using ``featomic``.

structure = ase.build.molecule("H2O")
hypers = {
    "cutoff": {
        "radius": 5.0,
        "smoothing": {"type": "ShiftedCosine", "width": 0.5},
    },
    "density": {
        "type": "Gaussian",
        "width": 0.3,
    },
    "basis": {
        "type": "TensorProduct",
        "max_angular": 2,
        "radial": {"type": "Gto", "max_radial": 3},
    },
}
spherical_expansion_calculator = featomic.SphericalExpansion(**hypers)
equivariants = spherical_expansion_calculator.compute(structure)
print(equivariants)

# %%
# Send the center type labels to the samples and the neighbor type labels to the
# properties

equivariants = equivariants.keys_to_samples("center_type").keys_to_properties(
    "neighbor_type"
)
print(equivariants)

# %%
# Let's focus on rotating the properties with ``o3_lambda=2`` (rank-2 spherical tensor).
# We will therefore drop all the other blocks (with ``o3_lambda=0,1``).

equivariants = metatensor.drop_blocks(
    equivariants, metatensor.Labels(names=["o3_lambda"], values=np.array([[0], [1]]))
)
print(equivariants)

# %%
# Rotating!
# ---------

# generate random rotations
n_rotations = 100
rotations = [get_random_rotation() for _ in range(n_rotations)]

# rotate the structures
rotated_structures = [structure.copy() for _ in range(n_rotations)]
for rotated_structure, rotation in zip(rotated_structures, rotations):
    rotated_structure.positions = (
        rotated_structure.positions @ np.array(rotation.as_matrix()).T
    )

# %%
# Rotate the equivariants
L = 2

wigner = spherical.Wigner(L)
complex_to_real_transform = complex_to_real_spherical_harmonics_transform(L)

rotated_equivariants = []
for rotation in rotations:
    quaternion_scipy = rotation.as_quat()
    quaternion_quaternionic = scipy_quaternion_to_quaternionic(quaternion_scipy)
    wigners_R = wigner.D(quaternion_quaternionic)
    wigner_D_matrix_complex = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)
    for mp in range(-L, L + 1):
        for m in range(-L, L + 1):
            # there is an unexplained conjugation factor in the definition
            # given in the quaternionic library.
            wigner_D_matrix_complex[mp + L, m + L] = (
                wigners_R[wigner.Dindex(L, mp, m)]
            ).conj()
    wigner_D_matrix = (
        complex_to_real_transform.conj()
        @ wigner_D_matrix_complex
        @ complex_to_real_transform.T
    )
    assert np.allclose(wigner_D_matrix.imag, 0.0)  # check that the matrix is real
    wigner_D_matrix = wigner_D_matrix.real
    new_values = (
        equivariants.block().values.swapaxes(-1, -2) @ wigner_D_matrix.T
    ).swapaxes(-1, -2)
    rotated_equivariants.append(
        metatensor.TensorMap(
            keys=equivariants.keys,
            blocks=[
                metatensor.TensorBlock(
                    values=new_values,
                    samples=equivariants.block().samples,
                    components=equivariants.block().components,
                    properties=equivariants.block().properties,
                )
            ],
        )
    )


# %%
# Check the correctness of the rotation
# -------------------------------------
#
# We will now check the correctness of the rotation by comparing the rotated
# equivariants with equivariants computed from the rotated structures.

rotated_equivariants_reference = []
for s in rotated_structures:
    eq = spherical_expansion_calculator.compute(s)
    eq = eq.keys_to_samples("center_type").keys_to_properties("neighbor_type")
    eq = metatensor.drop_blocks(
        eq, metatensor.Labels(names=["o3_lambda"], values=np.array([[0], [1]]))
    )
    rotated_equivariants_reference.append(eq)

for re1, re2 in zip(rotated_equivariants, rotated_equivariants_reference):
    assert np.allclose(re1.block().values, re2.block().values)
