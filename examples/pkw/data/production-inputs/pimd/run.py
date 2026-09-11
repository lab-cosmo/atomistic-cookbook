import argparse

import ase.io
from ipi.scripting import InteractiveSimulation

SIZE2TEMP = {
    "s": 330,
    "s_D3": 330,
    "s-ft": 300,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state")
    parser.add_argument("--size")
    parser.add_argument("-y", type=float)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    return args.state, args.size, args.y, args.seed

if __name__ == "__main__":
    state, size, y, seed = parse_args()
    initial_structure = f"{state}.xyz"
    model_str = f"model:pet_sol-s-ft-best.pt,"
    if size == "s-ft":
        model_str += "energy_variant:pbe0,"
    atoms = ase.io.read(initial_structure)
    new_masses = "[" + ", ".join(map(str, atoms.get_masses() / y ** 2)) + "]"
    with open(
        "input.xml", "r", encoding="utf-8"
    ) as file:
        input_xml = file.read()
    input_xml = (
        input_xml.replace("__STRUCTURE__", initial_structure)
        .replace("__MODEL__", model_str)
        .replace("__TEMPERATURE__", str(SIZE2TEMP[size]))
        .replace("__SEED__", str(seed))
        .replace("__PLUMED_DAT__", f"plumed-{state}.dat")
        .replace("__MASSES__", new_masses)
    )

    sim = InteractiveSimulation(input_xml)
    sim.run(20000000)
