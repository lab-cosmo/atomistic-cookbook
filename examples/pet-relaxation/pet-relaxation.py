r"""
Geometry relaxation with unconstrained MLIPs
=============================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo/>`_

This recipe shows how to perform geometry optimization with *unconstrained*
machine-learning interatomic potentials (MLIPs), and what tools are available
to control symmetry during relaxation.

Unconstrained models, such as the `Point-Edge Transformer (PET)
<https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_
learns symmetry through data augmentation rather than having it encoded by
construction (see also the
`PET-MAD recipe <https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html>`_).
This means that PET's predicted forces and stresses can have a small nonzero
residual even on perfect high-symmetry structures. During optimization, this
residual breaks degeneracies: if a structure sits on a saddle point, the
optimizer will move off it toward a nearby minimum. The practical consequence
is that a plain relaxation does not necessarily preserve the initial symmetry.
When the starting configuration is close to a local minimum, this introduces
small distortions that could be problematic for some applications (e.g. when
automatically building brillouin zone paths for
`phonon calculations
<https://atomistic-cookbook.org/examples/pet-phonons/pet-phonons.html>`_).
When the starting configuration is unstable, the unconstrained model will
naturally find the nearest minimum, which is often desirable but not always
(e.g., when you want to study a specific metastable or high-symmetry phase).

This recipe covers the workflow for handling this behavior:

1. **Unconstrained relaxation**: let the optimizer find the nearest minimum.
   Use ``spglib`` to identify the symmetry of the result and ``standardize_cell``
   to obtain a clean primitive cell.
2. **Constrained relaxation with** ``FixSymmetry``: when you want to study a
   *specific* phase (e.g., a metastable or high-symmetry structure), lock in
   its symmetry before optimizing.
3. **Rotational averaging**: a calculator-level option that reduces the
   symmetry-breaking residual by averaging predictions over a grid of
   rotations, and can be useful when one wants to reduce the residual
   distortion in unconstrained optimizations.

We demonstrate these tools on two systems:

1. **Al (Bain path)**: BCC aluminum is a saddle point. Unconstrained
   relaxation drives the cell to FCC along the
   `Bain path <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.3892>`_.
2. **BaTiO₃ (perovskite)**: unconstrained relaxation of the cubic
   :math:`Pm\bar{3}m` structure converges to the ferroelectric
   :math:`R3m` ground state.
"""

# %%

import warnings

import numpy as np
import matplotlib.pyplot as plt
import ase.io
import chemiscope
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE

import spglib
from upet.calculator import UPETCalculator

# Suppress warnings about matrix logarithm accuracy issued by scipy during geometry
# optimization to avoid cluttering the output
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="logm result may be inaccurate, approximate err",
)

# sphinx_gallery_thumbnail_number = 1

# %%
# Setup
# -----
#
# We use the small (S) variant of
# `PET-MAD v1.5.0 <https://arxiv.org/abs/2603.02089>`_, which is trained on
# a dataset of approximately 200k structures computed at the r2SCAN level of theory.

FMAX = 1e-4  # eV/Å, force convergence threshold
STEPS = 500  # max optimization steps

DEVICE = "cpu"

# We suggest using double precison (``dtype="float64"``) for geometry optimization, as
# it allows smaller values of the ``fmax`` convergence threshold and more reliable
# symmetry detection.

calc = UPETCalculator(
    model="pet-mad-s",
    device=DEVICE,
    dtype="float64",
    version="1.5.0",
)

# %%
# Helper functions
# ^^^^^^^^^^^^^^^^


def report_symmetry(atoms, label="", loose_tol=0.02):
    """Detect and report space group using spglib."""
    spglib_cell = (
        atoms.get_cell(),
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers(),
    )
    sg_loose = spglib.get_spacegroup(spglib_cell, symprec=loose_tol)
    sg_tight = spglib.get_spacegroup(spglib_cell, symprec=1e-6)
    print(
        f"{label:20s}  loose ({loose_tol}): {str(sg_loose):15s}"
        f"  tight (1e-6): {str(sg_tight)}"
    )


# %%
# Al along the Bain path
# -----------------------
#
# The `ground state structure for Aluminum
# <https://next-gen.materialsproject.org/materials/mp-134>`_ has FCC symmetry
# (:math:`Fm\bar{3}m`). The BCC structure is a saddle point along the
# `Bain path <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.3892>`_,
# a continuous tetragonal deformation connecting BCC and FCC.


# %%
# Unconstrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Because PET does not enforce point-group symmetry, the predicted stress on the
# perfect BCC cell has a small anisotropic residual. This is enough to push the
# optimizer off the saddle, and the cell slides down to FCC.

atoms_al = bulk("Al", "bcc", a=3.3, cubic=False)
report_symmetry(atoms_al, "Initial (BCC)")

atoms_al.calc = calc

opt_al = FIRE(FrechetCellFilter(atoms_al), trajectory="al_bain.traj")
opt_al.run(fmax=FMAX, steps=STEPS)

report_symmetry(atoms_al, "After relaxation")

# %%
#
# The optimizer converged from BCC (:math:`Im\bar{3}m`) to FCC (:math:`Fm\bar{3}m`). We
# can track the cell parameters along the trajectory: the BCC primitive cell
# (:math:`a = b = c`, :math:`\alpha = \beta = \gamma \approx 109.5°`) deforms
# continuously into the FCC primitive cell.

traj = ase.io.Trajectory("al_bain.traj")
cell_parameters = np.array([frame.cell.cellpar() for frame in traj])
a, b, c, alpha, beta, gamma = cell_parameters.T
steps = np.arange(len(traj))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

ax1.plot(steps, a, label="a")
ax1.plot(steps, b, label="b", ls="--")
ax1.plot(steps, c, label="c", ls=":")
ax1.set_xlabel("Optimization step")
ax1.set_ylabel("Lattice parameter (Å)")
ax1.legend()

ax2.plot(steps, alpha, label=r"$\alpha$")
ax2.plot(steps, beta, label=r"$\beta$", ls="--")
ax2.plot(steps, gamma, label=r"$\gamma$", ls=":")
ax2.set_xlabel("Optimization step")
ax2.set_ylabel("Angle (°)")
ax2.legend()

fig.suptitle("Al: BCC → FCC along the Bain path")
plt.show()

# %%
#
# The `chemiscope <http://chemiscope.org>`_ widget lets you inspect the
# trajectory frame by frame. The BCC (red) and FCC (blue) conventional cells
# are overlaid: watch the BCC cube deform as the FCC cube becomes regular.

traj_frames = ase.io.read("al_bain.traj", index=":")
al_energies = np.array([frame.get_potential_energy() for frame in traj_frames])


def conventional_cell_edges(cell, kind):
    """Return the 12 edges of a conventional cubic cell as (position, vector) pairs."""
    a1, a2, a3 = np.array(cell)
    # BCC conventional cell vectors from BCC primitive vectors
    A_bcc, B_bcc, C_bcc = a2 + a3, a1 + a3, a1 + a2
    if kind == "bcc":
        A, B, C = A_bcc, B_bcc, C_bcc
        origin = np.zeros(3)
    else:  # fcc
        # Bain correspondence: FCC shares one BCC edge, the other two are
        # face diagonals of the BCC cube
        A = B_bcc
        B = A_bcc + C_bcc
        C = A_bcc - C_bcc
        origin = np.zeros(3)
    corners = [
        origin,
        origin + A,
        origin + B,
        origin + C,
        origin + A + B,
        origin + A + C,
        origin + B + C,
        origin + A + B + C,
    ]
    edge_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return [
        (corners[i].tolist(), (corners[j] - corners[i]).tolist()) for i, j in edge_pairs
    ]


conv_shapes = {}
for edge_idx in range(12):
    for kind, color in [("bcc", "0xFF0000"), ("fcc", "0x0000FF")]:
        structure_params = []
        for frame in traj_frames:
            edges = conventional_cell_edges(frame.cell[:], kind)
            pos, vec = edges[edge_idx]
            structure_params.append({"position": pos, "vector": vec})
        conv_shapes[f"{kind}_edge_{edge_idx}"] = {
            "kind": "cylinder",
            "parameters": {
                "global": {"radius": 0.05, "color": color},
                "structure": structure_params,
            },
        }

chemiscope.show(
    traj_frames,
    properties={
        "step": np.arange(len(traj_frames)),
        "energy": al_energies,
    },
    shapes=conv_shapes,
    settings=chemiscope.quick_settings(
        x="step",
        y="energy",
        map_settings={"joinPoints": True},
        structure_settings={
            "unitCell": True,
            "supercell": (4, 5, 6),
            "shape": [f"bcc_edge_{i}" for i in range(12)]
            + [f"fcc_edge_{i}" for i in range(12)],
            "keepOrientation": True,
        },
    ),
)

# %%
# Detecting the relaxed symmetry
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Even if the relaxed structure is very close to FCC and is converged to a very tight
# force threshold, the small symmetry breaking due to the unconstrained nature of PET
# can make symmetry detection tricky. When using a tight ``symprec`` value of ``1e-6``,
# ``spglib`` detects the lowest-symmetry space group (:math:`P\bar{1}`).
# A looser tolerance of ``0.02`` correctly identifies the high-symmetry
# :math:`Fm\bar{3}m`. We recommend scanning the ``symprec`` parameter to find a "plateau"
# that identifies the true symmetry of the relaxed structure.

spglib_cell = (
    atoms_al.get_cell(),
    atoms_al.get_scaled_positions(),
    atoms_al.get_atomic_numbers(),
)
for symprec in np.logspace(-4, np.log10(0.1), 10):
    print(
        f"  symprec={symprec:.4f}  "
        f"{spglib.get_spacegroup(spglib_cell, symprec=symprec)}"
    )

# %%
#
#  At tight tolerances, ``spglib`` detects the residual shear as a highly asymmetric
# :math:`P\bar{1}` state. As the tolerance is relaxed, it identifies an
# :math:`I4/mmm` (body-centered tetragonal) intermediate state before recognizing the
# "true" :math:`Fm\bar{3}m` (FCC) ground state.
#
# Once the plateau is identified, ``standardize_cell`` snaps the coordinates
# onto ideal Wyckoff positions and the lattice vectors onto the exact
# :math:`Fm\bar{3}m` cell, removing the residual numerical noise.

std_data = spglib.standardize_cell(spglib_cell, to_primitive=True, symprec=0.05)

al_fcc = Atoms(
    numbers=std_data[2],
    scaled_positions=std_data[1],
    cell=std_data[0],
    pbc=True,
)

report_symmetry(al_fcc, "FCC (spglib)")

# %%
# Constrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The standardized cell from ``spglib`` has exact :math:`Fm\bar{3}m` symmetry,
# but its lattice parameters may differ slightly from the model's true minimum.
# We can refine them with ``FixSymmetry``, which projects forces and stresses
# onto the symmetry-invariant subspace at every step, ensuring the cell stays
# exactly FCC throughout the optimization.

al_fcc.set_constraint(FixSymmetry(al_fcc))
al_fcc.calc = calc

opt_al_const = FIRE(FrechetCellFilter(al_fcc), trajectory="al_fcc_const.traj")
opt_al_const.run(fmax=FMAX, steps=STEPS)

report_symmetry(al_fcc, "Constrained FCC")

al_fcc.set_constraint(None)

# %%
# Rotational averaging
# ^^^^^^^^^^^^^^^^^^^^^
#
# A complementary approach to ``FixSymmetry`` is *rotational averaging*: the calculator
# averages its predictions over a `grid
# <https://en.wikipedia.org/wiki/Lebedev_quadrature>`_ of rotations (plus inversion),
# reducing the symmetry-breaking residual at the source. Since the group of rotations is
# continuous, the averaging is approximate, but the residual can be controlled via the
# ``rotational_average_order`` parameter, which sets the order of the integration grid
# (formally, spherical harmonics up to this order are integrated exactly).
# The cheapest choice is ``rotational_average_order=3``, which averages predictions on
# a grid of 24 rotations :math:`\times` 2 inversions.
#
# Rotational averaging and ``FixSymmetry`` address different aspects: ``FixSymmetry``
# projects forces and stresses onto the symmetry-invariant subspace of a
# *specific* space group, while rotational averaging reduces the model's O(3)-breaking
# noise for *any* structure. They can be combined. However, if the target space group is
# known, ``FixSymmetry`` is generally preferred: it is exact, cheap, and sufficient.
# Rotational averaging is most useful in exploratory settings where the
# final symmetry is not known in advance, but it comes at a significant computational
# and memory cost.

calc_symm = UPETCalculator(
    model="pet-mad-s",
    device=DEVICE,
    dtype="float64",
    version="1.5.0",
    rotational_average_order=3,
)

atoms_al = bulk("Al", "bcc", a=3.3, cubic=False)
atoms_al.calc = calc_symm

opt_al = FIRE(FrechetCellFilter(atoms_al))
opt_al.run(fmax=FMAX, steps=STEPS)

print("After relaxation with symmetrized calculator")
spglib_cell = (
    atoms_al.get_cell(),
    atoms_al.get_scaled_positions(),
    atoms_al.get_atomic_numbers(),
)
for symprec in np.logspace(-4, np.log10(0.1), 10):
    print(
        f"  symprec={symprec:.4f}  "
        f"{spglib.get_spacegroup(spglib_cell, symprec=symprec)}"
    )

# %%
#
# The symmetrized model successfully protects the angular symmetry of the cell, keeping
# the internal angles closer to 90 degrees. The resulting structure has higher symmetry
# with space group :math:`Immm` (body-centered orthorhombic) at tight tolerances.
# As the tolerance is increased, it still perfectly retraces the :math:`I4/mmm`
# Bain path before reaching the :math:`Fm\bar{3}m` minimum, which is recovered at tighter
# tolerances than before thanks to the reduced noise.

# %%
# BaTiO₃: spontaneous ferroelectric distortion
# --------------------------------------------------------
#
# Barium titanate is a prototypical ferroelectric perovskite. At 0 K
# the cubic :math:`Pm\bar{3}m` `phase
# <https://next-gen.materialsproject.org/materials/mp-2998/>`_ is a saddle point:
# Ti displaces off-center, breaking cubic symmetry and stabilizing the rhombohedral
# :math:`R3m` `phase <https://next-gen.materialsproject.org/materials/mp-5020/>`_.
# Contrary to the case of Al, which we simulated with a primitive cell so that
# the deformation was entirely driven by the cell degrees of freedom, the
# ferroelectric distortion in BaTiO₃ is primarily an internal displacement of Ti
# relative to the oxygen cage, which involves primarily a distortion within the
# cell, and only a small accompanying shear distortion of the cell itself.
#
# An unconstrained relaxation starting from the cubic cell will naturally
# fall off the saddle and converge to the :math:`R3m` basin. We then use
# ``spglib`` to identify the resulting symmetry and extract a clean primitive
# cell.
#
# We also show how ``FixSymmetry`` can lock in the cubic :math:`Pm\bar{3}m`
# phase when that is the structure of interest (e.g., for computing phonons
# at the saddle point).

a_bto = 4.00
bto_cubic = Atoms(
    symbols="BaTiO3",
    scaled_positions=[
        [0.0, 0.0, 0.0],  # Ba
        [0.5, 0.5, 0.5],  # Ti
        [0.5, 0.5, 0.0],  # O1
        [0.5, 0.0, 0.5],  # O2
        [0.0, 0.5, 0.5],  # O3
    ],
    cell=[a_bto, a_bto, a_bto],
    pbc=True,
)

report_symmetry(bto_cubic, "Initial (cubic)")

# %%
# Constrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# As for the case of Al, we can use `FixSymmetry
# <https://ase-lib.org/ase/constraints.html#the-fixsymmetry-class>`_
# to lock in the cubic :math:`Pm\bar{3}m` phase, by projecting
# forces and stresses onto the symmetry-invariant subspace at
# every optimization step. It must therefore be instantiated from a cell
# that already has the target symmetry---applying it to an already-distorted
# cell would lock in the wrong (lower) symmetry.

bto_const = bto_cubic.copy()
bto_const.set_constraint(FixSymmetry(bto_const))
bto_const.calc = calc

# The mask [True]*3 + [False]*3 allows the three lattice lengths to relax
# while freezing the cell angles, preventing the filter from introducing
# angular distortions that FixSymmetry does not constrain.
opt_const = FIRE(FrechetCellFilter(bto_const, mask=[True] * 3 + [False] * 3))
opt_const.run(fmax=FMAX, steps=STEPS)

report_symmetry(bto_const, "Constrained")

# Remove the constraint so that subsequent energy queries return the raw
# model prediction rather than the symmetry-projected one.
bto_const.set_constraint(None)

# %%
# Unconstrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# By removing the constraint, we can run a relaxation using the raw PET model.
# This allows breaking the cubic symmetry and finding the nearest minimum, triggering
# the ferroelectric distortion.

bto_unconst = bto_cubic.copy()
bto_unconst.calc = calc

opt_unconst = FIRE(FrechetCellFilter(bto_unconst), trajectory="bto_unconst.traj")
opt_unconst.run(fmax=FMAX, steps=STEPS)

report_symmetry(bto_unconst, "Unconstrained")

# %%
# Identifying the ferroelectric phase
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The unconstrained structure has broken cubic symmetry, but the
# detected space group depends on the ``spglib`` tolerance. We scan
# ``symprec`` to find the plateau that identifies the true symmetry.

spglib_cell = (
    bto_unconst.get_cell(),
    bto_unconst.get_scaled_positions(),
    bto_unconst.get_atomic_numbers(),
)
for symprec in np.logspace(-3, np.log10(0.2), 10):
    print(
        f"  symprec={symprec:.4f}  "
        f"{spglib.get_spacegroup(spglib_cell, symprec=symprec)}"
    )

# %%
#
# A clear :math:`R3m` plateau appears. The relaxed cell still carries
# small numerical noise that breaks exact :math:`R3m` symmetry.
# `standardize_cell
# <https://spglib.readthedocs.io/en/stable/api.html#spg-standardize-cell>`_
# snaps coordinates onto ideal Wyckoff positions and lattice vectors
# onto the exact :math:`R3m` cell, giving a clean starting point for
# further calculations (e.g., phonons).

std_data = spglib.standardize_cell(spglib_cell, to_primitive=True, symprec=0.05)

bto_r3m = Atoms(
    numbers=std_data[2],
    scaled_positions=std_data[1],
    cell=std_data[0],
    pbc=True,
)

report_symmetry(bto_r3m, "R3m (spglib)")

# %%
# Energy comparison
# ~~~~~~~~~~~~~~~~~
#
# The unconstrained relaxation converges to a lower-energy structure.

dE = bto_const.get_potential_energy() - bto_unconst.get_potential_energy()
print(f"E(cubic) - E(R3m): {dE * 1000:.1f} meV")

cellpar_c = bto_const.cell.cellpar()
cellpar_u = bto_unconst.cell.cellpar()
print(f"  Cubic:         a = {cellpar_c[0]:.4f} Å")
print(f"  Ferroelectric: a = {cellpar_u[0]:.4f} Å, α = {cellpar_u[3]:.2f}°")

# %%
#
# The energy and cell parameter evolution during the unconstrained
# relaxation shows how the optimizer moves away from the cubic saddle
# point toward the :math:`R3m` minimum.

traj_bto = ase.io.read("bto_unconst.traj", index=":")
energies = np.array([frame.get_potential_energy() for frame in traj_bto])
angles = np.array([frame.cell.cellpar()[3] for frame in traj_bto])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(np.arange(len(energies)), (energies - energies[-1]) * 1000)
ax1.axhline(
    (bto_const.get_potential_energy() - energies[-1]) * 1000,
    color="gray",
    ls="--",
    label=r"$Pm\bar{3}m$ (constrained)",
)
ax1.set_xlabel("Optimization step")
ax1.set_ylabel("Energy (meV)")
ax1.legend()

ax2.plot(np.arange(len(angles)), angles)
ax2.axhline(90, color="gray", ls="--", label="cubic")
ax2.set_xlabel("Optimization step")
ax2.set_ylabel(r"$\alpha$ (°)")
ax2.legend()

fig.suptitle(r"BaTiO$_3$: cubic $Pm\bar{3}m$ → ferroelectric $R3m$")
plt.tight_layout()
plt.show()

# %%
#
# The `chemiscope <http://chemiscope.org>`_ widget below lets you
# inspect the trajectory frame by frame. Notice how Ti gradually
# displaces relative to the oxygen cage as the ferroelectric
# distortion develops.

chemiscope.show(
    structures=traj_bto,
    properties={
        "step": np.arange(len(traj_bto)),
        "energy": energies,
    },
    settings=chemiscope.quick_settings(
        map_settings={
            "x": {"property": "step"},
            "y": {"property": "energy"},
            "joinPoints": True,
        },
        structure_settings={
            "unitCell": True,
            "supercell": (2, 2, 2),
            "keepOrientation": True,
        },
    ),
)

# %%
# Conclusions
# -----------
#
# In summary, geometry relaxation with unconstrained MLIPs requires
# attention to symmetry, but the tools are straightforward:
#
# 1. **Unconstrained relaxation** finds the nearest minimum. Use
#    ``spglib`` to identify the resulting symmetry and
#    ``standardize_cell`` to obtain a clean primitive cell.
# 2. **``FixSymmetry``** locks in a specific space group when you
#    want to study a particular phase (stable or metastable). It is
#    exact, cheap, and the recommended default when the target
#    symmetry is known.
# 3. **Rotational averaging** reduces the model's symmetry-breaking
#    noise at the calculator level. It is useful in exploratory
#    settings but expensive; prefer ``FixSymmetry`` when possible.
#
# To verify that the relaxed structures are true minima, one should
# compute their phonon dispersions. This is the subject of the
# `phonon uncertainty quantification recipe
# <https://atomistic-cookbook.org/examples/pet-phonons/pet-phonons.html>`_.

# %%
