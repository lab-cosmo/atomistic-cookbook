"""
Batch run of CP2K calculations
=============================================

.. start-body

This is an example of a batch calculation using CP2K.
The inputs are a set of structures in `./data/example.xyz` using the parameters defined 
in `./data/reftraj_template.cp2k` importing basisset and pseudopotentials from `./data/basis/`.

The script will create a directory `./production` containing subdirectories for each stoichiometry.
This is only necessary, because CP2K can only run calculations using a single stoichiometry at a time, using the reftraj functionality.
"""

import shutil
import os
from os.path import basename, splitext
from typing import List, Union

import ase.io
import numpy as np
from ase.build import molecule
from ase.calculators.cp2k import CP2K
from numpy.testing import assert_allclose
from pathlib import Path
import subprocess
import re



# %%
# Define nescassary functions
# ============
# 




def write_reftraj(fname: str, frames: Union[ase.Atoms, List[ase.Atoms]]):
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
        n_atoms = len(atoms)
        out += f"{len(atoms):>8}\n i = {i+1:>8}, time = {0:>12.3f}\n"
        for atom in atoms:
            pos = atom.position
            out += f"{atom.symbol}{pos[0]:24.15f}{pos[1]:24.15f}{pos[2]:24.15f}\n"
    out += "\n"
    with open(fname, "w") as f:
        f.write(out)

def write_cellfile(fname: str, frames: Union[ase.Atoms, List[ase.Atoms]]):
    if isinstance(frames, ase.Atoms):
        frames = [frames]

    out = "#   Step   Time [fs]       Ax [Angstrom]       Ay [Angstrom]       Az [Angstrom]       Bx [Angstrom]       By [Angstrom]       Bz [Angstrom]       Cx [Angstrom]       Cy [Angstrom]       Cz [Angstrom]      Volume [Angstrom^3]\n"
    for i, atoms in enumerate(frames):
        out += f"{i+1:>8}{0:>12.3f}"
        out += "".join([f"{c:>20.10f}" for c in atoms.cell.flatten()])
        out += f"{atoms.cell.volume:>25.10f}"
        out += "\n"

    with open(fname, "w") as f:
        f.write(out)

def write_cp2k_in(fname: str, project: str, last_snapshot: int, cell: List[float]):
    with open("./data/reftraj_template.cp2k", "r") as f:
        cp2k_in = f.read()

    cp2k_in = cp2k_in.replace("//PROJECT//", project)
    cp2k_in = cp2k_in.replace("//LAST_SNAPSHOT//", str(last_snapshot))
    cp2k_in = cp2k_in.replace("//CELL//", " ".join([f"{c:.6f}" for c in cell]))

    IDENTFIER_CP2K_INSTALL = "PATH_TO_CP2KINSTALL"
    PATH_TO_CP2K_DATA = str(Path(shutil.which("cp2k.ssmp")).parents[1] / "share/cp2k/data/")

    cp2k_in = cp2k_in.replace(IDENTFIER_CP2K_INSTALL, PATH_TO_CP2K_DATA)

    with open(fname, "w") as f:
        f.write(cp2k_in)

def mkdir_force(*args, **kwargs):
    try:
        os.mkdir(*args, **kwargs)
    except OSError as e:
        pass

# %%
# Prepare calculation inputs
# ============
# 

project = "test_calcs"
project_directory = "production"
write_to_file = "out.xyz"
frames_full = ase.io.read("./data/example.xyz",":")

frames_dict = {}

for atoms in frames_full:
    chemical_formula = atoms.get_chemical_formula()
    try:
        frames_dict[chemical_formula]
    except KeyError:
        frames_dict[chemical_formula] = []

    frames_dict[chemical_formula].append(atoms)

mkdir_force(project_directory)

for stoichiometry, frames in frames_dict.items():
    current_directory = f"{project_directory}/{stoichiometry}"
    mkdir_force(current_directory)

    write_cp2k_in(
        f"{current_directory}/in.cp2k",
        project=project,
        last_snapshot=len(frames),
        cell=frames[0].cell.diagonal(),
    )

    ase.io.write(f"{current_directory}/init.xyz", frames[0])
    write_reftraj(f"{current_directory}/reftraj.xyz", frames)
    write_cellfile(f"{current_directory}/reftraj.cell", frames)

# %%
# Run simulations
# ===============
#
subprocess.run(f"cd ./production && for i in $(find . -mindepth 1 -type d); do cd \"$i\"; cp2k.ssmp -i in.cp2k ; cd -; done", shell=True)

# %%
# Load results
# ============
# 

cflength = 0.529177210903  # Bohr -> Å
cfenergy = 27.211386245988  # Hartree -> eV
cfforce = cfenergy / cflength  # Hartree/Bohr -> eV/Å

new_frames = []

for stoichiometry, frames in frames_dict.items():
    current_directory = f"{project_directory}/{stoichiometry}"

    frames_dft = ase.io.read(f"{current_directory}/{project}-pos-1.xyz", ":")
    forces_dft = ase.io.read(f"{current_directory}/{project}-frc-1.xyz", ":")
    cell_dft = np.atleast_2d(np.loadtxt(f"{current_directory}/{project}-1.cell"))[:, 2:-1]

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
# Perform calculations using ase calculator
# ==========================================
#
# ASE requires a name of the executable that has the exact name cp2k_shell.
# We create a symlink to follow this requirement.
try:
    os.symlink(shutil.which("cp2k.ssmp"), "cp2k_shell.ssmp")
except OSError:
    pass



# remove GLOBAl section becaus eotherwirse we get an error by the ase claculator
inp = open("./production/H4O2/in.cp2k", "r").read()
inp = re.sub(f"{re.escape('&GLOBAL')}.*?{re.escape('&END GLOBAL')}", "", inp, flags=re.DOTALL)


# set an alias for 
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
    #multiplicity=None,
    poisson_solver=None,
    print_level=None,
    command=f"./cp2k_shell.ssmp --shell",
)

atoms = ase.io.read("./data/example.xyz")
atoms.set_calculator(calc)
#atoms.get_potential_energy()
# %%
