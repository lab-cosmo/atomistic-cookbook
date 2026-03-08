r"""
Geometry relaxation with unconstrained MLIPs
=============================================

:Authors: Paolo Pegolo `@ppegolo <https://github.com/ppegolo/>`_

This recipe explores how *unconstrained* machine-learning interatomic
potentials (MLIPs) behave during geometry optimization.

Equivariant models enforce strict rotational and inversion symmetry: if
a structure sits on a high-symmetry saddle point, predicted forces and
stresses are exactly zero by construction, and the optimizer cannot
escape without a manual perturbation.
The `Point-Edge Transformer (PET)
<https://proceedings.neurips.cc/paper_files/paper/2023/file/fb4a7e3522363907b26a86cc5be627ac-Paper-Conference.pdf>`_
learns symmetry through data augmentation rather than having it encoded by
construction (see also the
`PET-MAD recipe <https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html>`_).
The predicted potential energy surface therefore does not necessarily favour symmetrical
configurations, and the predicted forces and stresses may have a small nonzero residual
even on perfect high-symmetry structures. During optimization this noise acts as a
perturbation that breaks degeneracies and pushes the system off saddle points toward the
true minimum.

We demonstrate two scenarios:

1. **Al (Bain path)**: cell stress drives the structure from BCC to
   FCC.
2. **BaTiO₃ (perovskite)**: unconstrained relaxation of the cubic
   `:math:`Pm\bar{3}m`<https://next-gen.materialsproject.org/materials/mp-2998/>`_
   structure spontaneously discovers the ferroelectric
   `:math:`R3m`<https://next-gen.materialsproject.org/materials/mp-5020/>`_
   (rhombohedral) ground state.
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
# `PET-MAD v1.5.0 <https://arxiv.org/abs/2603.02089>`_.

FMAX = 1e-4  # eV/Å, force convergence threshold
STEPS = 500  # max optimization steps

DEVICE = "cpu"

calc = UPETCalculator(
    model="pet-mad-s",
    device=DEVICE,
    dtype="float64",
    version="1.5.0",
)

# We suggest using double precison (``dtype="float64``) for geometry optimization, as
# it allows smaller values of the ``fmax`` convergence threshold and more reliable
# symmetry detection.

# %%
# Helper
# ^^^^^^


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
# Aluminum's ground state is FCC
# (`:math:`Fm\bar{3}m`<https://next-gen.materialsproject.org/materials/mp-134>`_). The
# BCC structure is a saddle point along the
# `Bain path <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.3892>`_,
# a continuous tetragonal deformation connecting BCC and FCC.
# An equivariant model would predict exactly zero stress on the perfect BCC cell. PET's
# residual stress anisotropy instead triggers a spontaneous shear, and the cell slides
# down to FCC.

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

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
plt.tight_layout()
plt.show()


# %%
#
# Another important point is that a tight ``symprec`` value of ``1e-6`` results in the
# lowest-symmetry space group (:math:`P\bar{1}`) being detected for the final relaxed
# structure, while a looser tolerance of ``0.02`` correctly identifies the high-symmetry
# :math:`Fm\bar{3}m`. This is again a consequence of the unconstrained nature of PET.
# We can also see what's the detected symmetry group for increasing values of
# ``symprec`` to see the "plateau" that identifies the true symmetry of the relaxed
# structure.

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
# :math:`P\bar{1}` or :math:`C2/m` state. As the tolerance expands, it reveals an
# :math:`I4/mmm` (body-centered tetragonal) intermediate state before recognizing the
# "true" :math:`Fm\bar{3}m` (FCC) ground state.

# %%
#
# We can also repeat the optimization with a symmetrized version of the calculator,
# which enforces O(3) symmetry via integration on a grid of rotations (plus inversion).
# Since the group of rotatoins is infinite, the symmetrization is still approximate,
# but the residual asymmetry can be controlled by the ``rotational_average_order``
# parameter, which sets the number of rotations in the grid (formally, one can integrate
# exactly spherical harmonic functions of order less than or equal to
# ``rotational_average_order``). The cheapest choice is ``rotational_average_order=3``,
# which averages predictions on a grid of 24 rotations :math:`\times` 2 inversions.

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
# the internal angles closer to 90 degrees. This replaces the messy triclinic noise with
# a more structured :math:`Immm` (body-centered orthorhombic) footprint at tight
# tolerances. As the tolerance expands, it still perfectly retraces the :math:`I4/mmm`
# Bain path before reaching the :math:`Fm\bar{3}m` minimum, which is recovered at lower
# tolerances than before thanks to the reduced noise.

# %%
# BaTiO\ :math:`_3`: spontaneous ferroelectric distortion
# --------------------------------------------------------
#
# Barium titanate is a prototypical ferroelectric perovskite. At 0 K
# the cubic :math:`Pm\bar{3}m` phase is a saddle point: Ti displaces
# off-center, breaking cubic symmetry and stabilizing the
# rhombohedral :math:`R3m` phase.
#
# An equivariant model would predict zero forces on the perfect cubic
# structure. PET's residual noise provides the perturbation that
# triggers the ferroelectric distortion spontaneously.
#
# We compare unconstrained relaxation (which discovers :math:`R3m`)
# with constrained relaxation (which preserves :math:`Pm\bar{3}m`).

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
# Unconstrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^^^

bto_unconst = bto_cubic.copy()
bto_unconst.calc = calc

opt_unconst = FIRE(FrechetCellFilter(bto_unconst), trajectory="bto_unconst.traj")
opt_unconst.run(fmax=FMAX, steps=STEPS)

report_symmetry(bto_unconst, "Unconstrained")

# %%
# Constrained relaxation
# ^^^^^^^^^^^^^^^^^^^^^^^

bto_const = bto_cubic.copy()
bto_const.set_constraint(FixSymmetry(bto_const))
bto_const.calc = calc

opt_const = FIRE(FrechetCellFilter(bto_const, mask=[True] * 3 + [False] * 3))
opt_const.run(fmax=FMAX, steps=STEPS)

report_symmetry(bto_const, "Constrained")

bto_const.set_constraint(None)

# %%
# Identifying the ferroelectric phase
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
# A clear :math:`R3m` plateau appears. We use a tolerance within this
# plateau to extract the standardized primitive cell.

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
# ^^^^^^^^^^^^^^^^^^
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
# relaxation shows how the optimizer escapes the cubic saddle point.

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
        },
        structure_settings={
            "unitCell": True,
        },
    ),
)

# %%
# Conclusions
# -----------
#
# * **Al (Bain path):** PET's residual stress anisotropy drives the
#   structure from the BCC saddle to the FCC minimum.
# * **BaTiO₃:** starting from cubic :math:`Pm\bar{3}m`, the
#   unconstrained optimizer discovers the ferroelectric :math:`R3m`
#   ground state without manual perturbation. ``FixSymmetry`` can
#   lock in the cubic saddle for analysis.
#
# To verify that the relaxed structures are true minima, one should
# compute their phonon dispersions. This is the subject of the
# `phonon uncertainty quantification recipe
# <https://atomistic-cookbook.org/examples/pet-phonons/pet-phonons.html>`_.

# %%
