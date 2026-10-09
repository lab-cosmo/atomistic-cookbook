"""Run one of the independent fixed-bias REMD simulations.

``input.xml`` is complete as it is (8 replicas, started from ``seeds/0.xyz``);
this script only replaces the starting structure by ``seeds/<idx>.xyz``, and
continues from a ``RESTART`` file if one is present.
Usage: python run.py -i <idx>
"""

import argparse
import os
import re

from ipi.scripting import InteractiveSimulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=int, default=0)
    idx = parser.parse_args().idx

    if os.path.exists("RESTART"):
        with open("RESTART", "r", encoding="utf-8") as file:
            input_xml = file.read()
        # the checkpoint does not store total_steps: lift the limit
        input_xml = re.sub(
            r"(<simulation[^>]*>)",
            r"\1\n   <total_steps>1000000000</total_steps>",
            input_xml,
            count=1,
        )
    else:
        with open("input.xml", "r", encoding="utf-8") as file:
            input_xml = file.read().replace("seeds/0.xyz", f"seeds/{idx}.xyz")

    sim = InteractiveSimulation(input_xml)
    sim.run(600000)  # 300 ps per job; resubmit to continue
