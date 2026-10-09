"""Run one of the independent fixed-bias REMD simulations.

``input.xml`` is complete as it is (8 replicas, started from ``seeds/0.xyz``);
this script only replaces the starting structure by ``seeds/<idx>.xyz``.
Usage: python run.py -i <idx>   (idx = 0..9)
"""

import argparse

from ipi.scripting import InteractiveSimulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    idx = parser.parse_args().idx

    with open("input.xml", "r", encoding="utf-8") as file:
        input_xml = file.read().replace("seeds/0.xyz", f"seeds/{idx}.xyz")
    sim = InteractiveSimulation(input_xml)
    sim.run(20000000)
