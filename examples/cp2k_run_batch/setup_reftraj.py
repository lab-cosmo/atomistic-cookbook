"""
Batch run of CP2K calculations
==============================

.. start-body

This is an example how to perform single point calculations based on list of structures
using `CP2K <https://www.cp2k.org>`_ using its `reftraj
functionality <https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/MD/REFTRAJ.html>`_. The
inputs are a set of structures in :download:`example.xyz` using the DFT parameters
defined in :download:`reftraj_template.cp2k` importing basisset and
pseudopotentials from the local CP2K install. The reference DFT paramaters are taken
from `Cheng et al. Ab initio thermodynamics of liquid and solid water 2019
<https://www.pnas.org/doi/10.1073/pnas.1815117116>`_ but adjusted in way to perform the
quick evaluation of this example

To run this example we use a "hardcoded" ``ssmp`` version of CP2K which is available via
``cp2k.ssmp``. If you want to use another version please adjust the the names
accordingly.
"""

# %%
# We start the example by importing the required packages.


import os
import re
import shutil
import subprocess
import warnings
from os.path import basename, splitext
from pathlib import Path
from typing import List, Union

import ase.io
import ase.visualize.plot
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.cp2k import CP2K


# %%
# Define necessary functions
# ==========================
# Next we below define necessary helper functions to run the example.

subprocess.run("bash return_CP2K_versions.sh", shell=True)


# %%


def write_reftraj(fname: str, frames: Union[ase.Atoms, List[ase.Atoms]]) -> None:
    """Writes a list of ase atoms objects to a reference trajectory.

    A reference trajectory is the CP2K compatible format for the compuation of batches.
    All frames must have the stoichiometry/composition.
    """

    if isinstance(frames, ase.Atoms):
        frames = [frames]

    out = ""
    for i, atoms in enumerate(frames):
        if (
            len(atoms) != len(frames[0])
            or atoms.get_chemical_formula() != frames[0].get_chemical_formula()
        ):
            raise ValueError(
                f"Atom symbols in frame {i},{atoms.get_chemical_formula()} are "
                f"different compared to inital frame "
                f"{frames[0].get_chemical_formula()}. "
                "CP2K does not support changing atom types within a reftraj run!"
            )

        out += f"{len(atoms):>8}\n i = {i+1:>8}, time = {0:>12.3f}\n"
        for atom in atoms:
            pos = atom.position
            out += f"{atom.symbol}{pos[0]:24.15f}{pos[1]:24.15f}{pos[2]:24.15f}\n"
    out += "\n"
    with open(fname, "w") as f:
        f.write(out)


# %%


def write_cellfile(fname: str, frames: Union[ase.Atoms, List[ase.Atoms]]) -> None:
    """Writes a cellfile for a list of ``ase.Atoms``.

    A Cellfile accompanies a reftraj containing the cell parameters.
    """
    if isinstance(frames, ase.Atoms):
        frames = [frames]

    out = (
        "#   "
        "Step   "
        "Time [fs]       "
        "Ax [Angstrom]       "
        "Ay [Angstrom]       "
        "Az [Angstrom]       "
        "Bx [Angstrom]       "
        "By [Angstrom]       "
        "Bz [Angstrom]       "
        "Cx [Angstrom]       "
        "Cy [Angstrom]       "
        "Cz [Angstrom]       "
        "Volume [Angstrom^3]\n"
    )

    for i, atoms in enumerate(frames):
        out += f"{i+1:>8}{0:>12.3f}"
        out += "".join([f"{c:>20.10f}" for c in atoms.cell.flatten()])
        out += f"{atoms.cell.volume:>25.10f}"
        out += "\n"

    with open(fname, "w") as f:
        f.write(out)


# %%


def write_cp2k_in(
    fname: str, project_name: str, last_snapshot: int, cell: List[float]
) -> None:
    """Writes a cp2k input file from a template.

    Importantly, it writes the location of the basis set definitions,
    determined from the path of the system CP2K install to the input file.
    """

    with open("reftraj_template.cp2k", "r") as f:
        cp2k_in = f.read()

    warnings.warn(
        "Due to the small size of the test structure and convergence issues, we have \
        decreased the size of the CUTOFF_RADIUS from 6.0 to 3.0. \
        For actual production calculations adapt the template!",
        stacklevel=2,
    )

    cp2k_in = cp2k_in.replace("//PROJECT//", project_name)
    cp2k_in = cp2k_in.replace("//LAST_SNAPSHOT//", str(last_snapshot))
    cp2k_in = cp2k_in.replace("//CELL//", " ".join([f"{c:.6f}" for c in cell]))

    IDENTFIER_CP2K_INSTALL = "PATH_TO_CP2KINSTALL"

    PATH_TO_CP2K_DATA = str(
        Path(shutil.which("cp2k.ssmp")).parents[1] / "share/cp2k/data/"
    )

    cp2k_in = cp2k_in.replace(IDENTFIER_CP2K_INSTALL, PATH_TO_CP2K_DATA)

    with open(fname, "w") as f:
        f.write(cp2k_in)


# %%


def mkdir_force(*args, **kwargs) -> None:
    """Warpper to ``os.mkdir``.

    The function does not raise an error if the directory already exists.
    """
    try:
        os.mkdir(*args, **kwargs)
    except OSError:
        pass


# %%
# Prepare calculation inputs
# ===============================================
# During this example we will create a directory named ``project_directory`` containing
# the subdirectories for each stoichiometry. This is necessary, because CP2K can only
# run calculations using a fixed stoichiometry at a time, using its ``reftraj``
# functionality.
#
# Below we define the general information for the CP2K run. This includes the reference
# files for the structures, the ``project_name`` used to build the name of the
# trajectory during the CP2K run, the ``project_directory`` where we store all
# simulation output as well as the path ``write_to_file`` which is the name of the file
# containing the computed energies and forces of the sinulation.

frames_full = ase.io.read("example.xyz", ":")
project_name = "test_calcs"  # name of the global PROJECT
project_directory = "production"
write_to_file = "out.xyz"

# %%
# Below we show the initial configuration of two water molecules in a cubic box with a
# sidelength of :math:`\approx 4\,\mathrm{Å}`.

ase.visualize.plot.plot_atoms(frames_full[0])

plt.xlabel("Å")
plt.ylabel("Å")

plt.show()

# %%
# We now extreact the stoichiometry from the input dataset using ASE's
# :py:meth:`ase.symbols.Symbols.get_chemical_formula` method.

frames_dict = {}

for atoms in frames_full:
    chemical_formula = atoms.get_chemical_formula()
    try:
        frames_dict[chemical_formula]
    except KeyError:
        frames_dict[chemical_formula] = []

    frames_dict[chemical_formula].append(atoms)

# %%
# Based on the stoichiometries we create one calculation subdirectories for the
# calculations. (reftraj, input and cellfile). For our example this is only is one
# directory named ``H4O2`` because our dataset consists only of a single structure with
# two water molecules.

mkdir_force(project_directory)

for stoichiometry, frames in frames_dict.items():
    current_directory = f"{project_directory}/{stoichiometry}"
    mkdir_force(current_directory)

    write_cp2k_in(
        f"{current_directory}/in.cp2k",
        project_name=project_name,
        last_snapshot=len(frames),
        cell=frames[0].cell.diagonal(),
    )

    ase.io.write(f"{current_directory}/init.xyz", frames[0])
    write_reftraj(f"{current_directory}/reftraj.xyz", frames)
    write_cellfile(f"{current_directory}/reftraj.cell", frames)

# %%
# Run simulations
# ===============
# Now we have all ingredients to run the simulations. Below we call the bash script
# :download:`run_calcs.sh`.
#
# .. literalinclude:: run_calcs.sh
#   :language: bash
#
# This script will loop through all stoichiometry subdirectories and call the CP2K
# engine.

# run the bash script directly from this script
subprocess.run("bash run_calcs.sh", shell=True)

# %%
# .. note::
#
#    For a usage on an HPC machine you can parallize the loop over the subfolders abd
#    submit and single job per stoichiometry.
#
# Load results
# ============
# After the simulation we load the results and perform a unit version from the default
# CP2K output units (Bohr and Hartree) to Å and eV.

cflength = 0.529177210903  # Bohr -> Å
cfenergy = 27.211386245988  # Hartree -> eV
cfforce = cfenergy / cflength  # Hartree/Bohr -> eV/Å

# %%
# Finally, we store the results as :class:`ase.Atoms` in the ``new_frames`` list and
# write them to the ``project_directory`` using the ``new_fname``. Here it will be
# written to ``production/out_dft.xyz``.

new_frames = []

for stoichiometry, frames in frames_dict.items():
    current_directory = f"{project_directory}/{stoichiometry}"

    frames_dft = ase.io.read(f"{current_directory}/{project_name}-pos-1.xyz", ":")
    forces_dft = ase.io.read(f"{current_directory}/{project_name}-frc-1.xyz", ":")
    cell_dft = np.atleast_2d(np.loadtxt(f"{current_directory}/{project_name}-1.cell"))[
        :, 2:-1
    ]

    for i_atoms, atoms in enumerate(frames_dft):
        frames_ref = frames[i_atoms]

        # Check consistent positions
        if not np.allclose(atoms.positions, frames_ref.positions):
            raise ValueError(f"Positions in frame {i_atoms} are not the same.")

        # Check consistent cell
        if not np.allclose(frames_ref.cell.flatten(), cell_dft[i_atoms]):
            raise ValueError(f"Cell dimensions in frame {i_atoms} are not the same.")

        atoms.info["E"] *= cfenergy
        atoms.pbc = True
        atoms.cell = frames_ref.cell
        atoms.set_array("forces", cfforce * forces_dft[i_atoms].positions)

    new_frames += frames_dft

new_fname = f"{splitext(basename(write_to_file))[0]}_dft.xyz"
ase.io.write(f"{project_directory}/{new_fname}", new_frames)

# %%
# Perform calculations using ASE calculator
# =========================================
# Above we performed the calculations using an external bash script. ASE also provides a
# calculator class that we can use the perform the caclulations with pur input file
# without a detour of writing files to disk.
#
# To use the ASE calculator together with a custom input script this requires some
# adjsutements. First the name of the executable that has the exact name ``cp2k_shell``.
# We create a symlink to follow this requirement.

try:
    os.symlink(shutil.which("cp2k.ssmp"), "cp2k_shell.ssmp")
except OSError:
    pass

# %%
# Next, we load the input file abd remove ``GLOBAL`` section because from it

inp = open("./production/H4O2/in.cp2k", "r").read()
inp = re.sub(
    f"{re.escape('&GLOBAL')}.*?{re.escape('&END GLOBAL')}", "", inp, flags=re.DOTALL
)

# %%
# Afterwards we define the ``CP2K`` calculator. Note that we disable all parameters
# because we want to use all DFT options from our input file

calc = CP2K(
    inp=inp,
    max_scf=None,
    cutoff=None,
    xc=None,
    force_eval_method=None,
    basis_set=None,
    pseudo_potential=None,
    basis_set_file=None,
    potential_file=None,
    stress_tensor=False,
    poisson_solver=None,
    print_level=None,
    command="./cp2k_shell.ssmp --shell",
)

# %%
# We now load a new struture, add the calculator and perform the computation.

atoms = ase.io.read("example.xyz")
atoms.set_calculator(calc)
# atoms.get_potential_energy()
