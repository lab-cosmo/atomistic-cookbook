"""
Prepares the initial configuration for the barostats recipe.

Takes a water or ice structure (an i-PI xyz file, using the last frame, or a
LAMMPS data file in ``atomic`` style) and writes a q-TIP4P/f-ready LAMMPS data
file (``full`` style, ``real`` units) and an i-PI-formatted xyz file, as
``data/<prefix>.lmp`` and ``data/<prefix>.xyz``. Atoms must be ordered as O, H, H.

Usage::

    python data/make_structure.py ice_96.xyz  # from i-PI examples/init_files
    python data/make_structure.py scan/<run>/sim.pos_0.xyz  # equilibrated frame
"""

import sys

import ase.io
import numpy as np


fname = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "ice_96"
if fname.endswith(".xyz"):
    import ipi

    atoms = ipi.read_trajectory(fname)[-1]
else:
    atoms = ase.io.read(fname, format="lammps-data", atom_style="atomic", units="metal")

box = np.diag(atoms.cell.array)
pos = atoms.positions - np.floor(atoms.positions / box) * box
nmol = len(pos) // 3

# make molecules whole (O, H, H ordering)
for m in range(nmol):
    o = pos[3 * m]
    for h in (1, 2):
        d = pos[3 * m + h] - o
        pos[3 * m + h] = o + d - box * np.round(d / box)

with open(f"data/{prefix}.lmp", "w") as f:
    f.write(f"LAMMPS data file, {nmol} q-TIP4P/f water molecules (units real)\n\n")
    f.write(f"{3 * nmol} atoms\n{2 * nmol} bonds\n{nmol} angles\n\n")
    f.write("2 atom types\n1 bond types\n1 angle types\n\n")
    for b, ax in zip(box, "xyz"):
        f.write(f"0.0 {b:.8f} {ax}lo {ax}hi\n")
    f.write("\nMasses\n\n1 15.9994\n2 1.0080\n\n")
    f.write("Bond Coeffs\n\n1 0.9419 607.19 -1388.65 1852.58\n\n")
    f.write("Angle Coeffs\n\n1 43.93 107.4\n\nAtoms\n\n")
    for i, p in enumerate(pos):
        typ, q = (1, -1.1128) if i % 3 == 0 else (2, 0.5564)
        f.write(f"{i + 1} {i // 3 + 1} {typ} {q} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    f.write("\nBonds\n\n")
    for m in range(nmol):
        f.write(f"{2 * m + 1} 1 {3 * m + 1} {3 * m + 2}\n")
        f.write(f"{2 * m + 2} 1 {3 * m + 1} {3 * m + 3}\n")
    f.write("\nAngles\n\n")
    for m in range(nmol):
        f.write(f"{m + 1} 1 {3 * m + 2} {3 * m + 1} {3 * m + 3}\n")

with open(f"data/{prefix}.xyz", "w") as f:
    f.write(f"{3 * nmol}\n")
    f.write(
        f"# CELL(abcABC): {box[0]:.8f} {box[1]:.8f} {box[2]:.8f} 90.0 90.0 90.0 "
        "positions{angstrom} cell{angstrom}\n"
    )
    for i, p in enumerate(pos):
        f.write(f"{'O' if i % 3 == 0 else 'H'} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
