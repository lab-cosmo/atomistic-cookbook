import argparse

from ipi.utils.scripting import InteractiveSimulation

template_330 = (
    "<instance> [ 00,  330 ] </instance>\n" +
    "        <instance> [ 01,  344 ] </instance>\n" +
    "        <instance> [ 02,  361 ] </instance>\n" +
    "        <instance> [ 03,  378 ] </instance>\n" +
    "        <instance> [ 04,  396 ] </instance>\n" +
    "        <instance> [ 05,  415 ] </instance>\n" +
    "        <instance> [ 06,  434 ] </instance>\n" +
    "        <instance> [ 07,  456 ] </instance>\n" +
    "        <instance> [ 08,  477 ] </instance>\n" +
    "        <instance> [ 09,  500 ] </instance>"
)

template_300 = (
    "<instance> [ 00,  300 ] </instance>\n" +
    "        <instance> [ 01,  314 ] </instance>\n" +
    "        <instance> [ 02,  330 ] </instance>\n" +
    "        <instance> [ 03,  344 ] </instance>\n" +
    "        <instance> [ 04,  361 ] </instance>\n" +
    "        <instance> [ 05,  378 ] </instance>\n" +
    "        <instance> [ 06,  396 ] </instance>\n" +
    "        <instance> [ 07,  415 ] </instance>\n" +
    "        <instance> [ 08,  434 ] </instance>\n" +
    "        <instance> [ 09,  456 ] </instance>\n" +
    "        <instance> [ 10,  477 ] </instance>\n" +
    "        <instance> [ 11,  500 ] </instance>"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-i", "--idx", type=int, required=True)

    args = parser.parse_args()
    return args.model, args.idx


if __name__ == "__main__":
    model, idx = parse_args()
    model_str = f"model:pet_sol-s-ft-best.pt,"
    if model == "s-ft":
        model_str += "energy_variant:pbe0,"
    with open(
        "input.xml", "r", encoding="utf-8"
    ) as file:
        input_xml = file.read()

    input_xml = (
        input_xml.replace("__MODEL__", model_str).replace("__IDX__", str(idx))
        .replace("__TEMPLATE__", template_330 if model != "s-ft" else template_300)
        .replace("__BATCH_SIZE__", "10" if model != "s-ft" else "12")
        .replace("__SIZE__", model)
    )

    sim = InteractiveSimulation(input_xml)
    sim.run(20000000)
