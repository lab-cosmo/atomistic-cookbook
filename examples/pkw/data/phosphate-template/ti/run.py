"""Run one PIMD simulation of the mass thermodynamic integration.

``input.xml`` is complete for the associated state at the physical masses
(y = 1); this script selects the state (``associated``, ``acidic_disso`` or
``basic_disso``, with the starting structure ``<state>.xyz`` and the walls of
``plumed-<state>.dat``) and scales all the masses by 1/y^2.
Usage: python run.py --state <state> -y <y> [--seed <int>]
"""

import argparse

import ase.io
from ipi.scripting import InteractiveSimulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="associated")
    parser.add_argument("-y", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()

    with open("input.xml", "r", encoding="utf-8") as file:
        input_xml = file.read()
    input_xml = input_xml.replace("associated.xyz", f"{args.state}.xyz")
    input_xml = input_xml.replace("plumed-associated.dat", f"plumed-{args.state}.dat")
    input_xml = input_xml.replace("<seed> 31415 </seed>", f"<seed> {args.seed} </seed>")
    masses = ase.io.read(f"{args.state}.xyz").get_masses() / args.y**2
    masses_xml = (
        '<masses mode="manual" units="dalton"> ['
        + ", ".join(f"{m:.6f}" for m in masses)
        + "] </masses>\n      "
    )
    input_xml = input_xml.replace("<velocities mode='thermal'", masses_xml + "<velocities mode='thermal'")
    sim = InteractiveSimulation(input_xml)
    sim.run(20000000)
