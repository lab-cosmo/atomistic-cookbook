import ase
import ase.build
import ase.optimize
import ase.filters
import ase.io
import random
from metatomic.torch.ase_calculator import MetatomicCalculator


# create a fcc cell with 108 atoms, remove missing atom types
def mk_atoms(lattice_parameter=6.0, atomic_types=None):
    if atomic_types is None:
        atomic_types = []

    atoms = ase.build.bulk("Al", "fcc", a=lattice_parameter, cubic=True)
    atoms = atoms * (3, 3, 3)

    # shuffle atomic positions and assign atom types
    atoms.numbers[:] = list(range(1, 109))
    atom_shuffle = random.sample(range(0, 108), len(atoms))
    atoms.positions[:] = atoms.positions[atom_shuffle]

    # remove atoms that are not represented in the model
    for index in range(len(atoms) - 1, -1, -1):
        if atoms.numbers[index] not in atomic_types:
            del atoms[index]

    return atoms


def mk_soup(model, n_soup=16):
    calc = MetatomicCalculator(model=model)
    atomic_types = calc._model.capabilities().atomic_types

    for i in range(n_soup):
        atoms = mk_atoms(6.0, atomic_types)
        atoms.calc = calc
        ucf = ase.filters.FrechetCellFilter(atoms, hydrostatic_strain=True)
        opt = ase.optimize.BFGS(ucf, trajectory="optimization.traj")
        opt.run(fmax=0.05, steps=100)

        print(f"Final lattice vectors:\n{atoms.get_cell()}")
        print(f"Final potential energy: {atoms.get_potential_energy():.3f} eV")

        atoms.wrap()
        ase.io.write(f"soup_relaxed-{i:02d}.xyz", atoms)
        traj_data = ase.io.read("optimization.traj", index=":")
        ase.io.write(f"optimization_traj-{i:02d}.xyz", traj_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a random soup of atoms and optimize them."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the metatomic model file."
    )
    parser.add_argument(
        "--n_soup", type=int, default=16, help="Number of copies to create."
    )
    args = parser.parse_args()

    mk_soup(model=args.model, n_soup=args.n_soup)
