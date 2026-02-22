r"""
Te PIGS demonstration
====================

:Authors: Venkat Kapil `@venkatkapil24 <https://github.com/venkatkapil24>`_

This recipe follows the `i-PI te-pigs demo
<https://github.com/i-pi/i-pi/tree/main/demos/te-pigs>`_.

The demo workflow has two main parts:

1. training a Te-PIGS effective potential using elevated-temperature PIMD data;
2. running production MD with the physical potential + Te-PIGS correction,
   then computing VDOS / IR / Raman spectra.

All input templates and helper scripts shown below are pulled from the upstream
`i-PI` repository at runtime, so this recipe does not vendor a data payload.
"""

import shutil
import subprocess
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import ipi
import matplotlib.pyplot as plt
import numpy as np


try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    # sphinx-gallery executes code in a context where __file__ is not defined
    ROOT = Path.cwd()
    if (ROOT / "examples" / "pigs").is_dir():
        ROOT = ROOT / "examples" / "pigs"

DEMO_DIR = ROOT / "te-pigs-demo"
BASE_URL = "https://raw.githubusercontent.com/i-pi/i-pi/main/demos/te-pigs"


DEMO_FILES = {
    "reference_input": "0_reference_pimd/input.xml",
    "reference_driver": "0_reference_pimd/run-ase.py",
    "dataset_curation": "1_dataset_curation/get_pigs_dataset.py",
    "training_script": "2_training/train.sh",
    "production_input": "3_production_simulations/input.xml",
    "production_driver_physical": "3_production_simulations/run-ase.py",
    "production_driver_pigs": "3_production_simulations/run-ase-pigs.py",
    "dielectric_script": "4_dielectric_response_prediction/get_dielectric_response.sh",
    "spectra_script": "5_final_spectra/get_spectra.sh",
}


def fetch_demo_file(relative_path):
    destination = DEMO_DIR / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    url = f"{BASE_URL}/{relative_path}"
    if shutil.which("wget") is not None:
        subprocess.run(
            ["wget", "-q", "-O", str(destination), url],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, destination)

    return destination


def fetch_all_demo_files():
    paths = {}
    for name, relative_path in DEMO_FILES.items():
        paths[name] = fetch_demo_file(relative_path)
    return paths


try:
    files = fetch_all_demo_files()
except Exception as err:
    warnings.warn(
        "Could not download te-pigs demo files. "
        "Check network access and re-run this recipe. "
        f"Original error: {err}",
        stacklevel=1,
    )
    files = None


# %%
# 1) Generating reference PIMD training data
# ------------------------------------------
#
# The demo uses `0_reference_pimd/input.xml`.
#
# It performs PIMD at 500 K with 8 beads and a 0.5 fs timestep.
# It writes:
# - centroid positions (`x_centroid{ase}`),
# - centroid forces (`f_centroid`),
# - physical force on centroid geometries (`forces_component_raw(1)`).

if files is not None:
    reference_xml = ET.parse(files["reference_input"]).getroot()

    print("Reference simulation setup:")
    print(
        "   "
        + ET.tostring(reference_xml.find(".//initialize"), encoding="unicode").strip()
    )
    print(
        "   "
        + ET.tostring(reference_xml.find(".//dynamics"), encoding="unicode").strip()
    )

    print("\nReference force components:")
    print(
        "   " + ET.tostring(reference_xml.find(".//forces"), encoding="unicode").strip()
    )

    print("\nReference trajectory outputs:")
    for traj in reference_xml.find("output").findall("trajectory"):
        print("   " + ET.tostring(traj, encoding="unicode").strip())

    print("\nReference ASE client:")
    print(files["reference_driver"].read_text())


# %%
# 2) Curating the dataset
# -----------------------
#
# The demo parser builds a dataset where each frame stores:
# - `centroid_force`
# - `physical_force`
# - `delta_force = centroid_force - physical_force`

if files is not None:
    print(files["dataset_curation"].read_text())


# %%
# 3) Fitting the Te-PIGS potential
# --------------------------------
#
# Training is done with MACE using `delta_force` as the force target.

if files is not None:
    print(files["training_script"].read_text())


# %%
# 4) Production simulations
# -------------------------
#
# The demo uses two sockets in i-PI:
# - one for the physical potential (`maceoff23`)
# - one for the Te-PIGS correction (`maceoff23-pigs`)
#
# and combines both with unit weight.

if files is not None:
    production_xml = ET.parse(files["production_input"]).getroot()

    print("Production sockets:")
    for ff in production_xml.findall("ffsocket"):
        print("   " + ET.tostring(ff, encoding="unicode").strip())

    print("\nProduction force components:")
    print(
        "   "
        + ET.tostring(production_xml.find(".//forces"), encoding="unicode").strip()
    )

    print("\nProduction trajectory output:")
    for traj in production_xml.find("output").findall("trajectory"):
        print("   " + ET.tostring(traj, encoding="unicode").strip())

    print("\nProduction ASE client (physical potential):")
    print(files["production_driver_physical"].read_text())

    print("\nProduction ASE client (Te-PIGS correction):")
    print(files["production_driver_pigs"].read_text())


# %%
# 5) Dielectric response prediction
# ---------------------------------
#
# The demo script evaluates dipole and polarizability time series from
# the production trajectory.

if files is not None:
    print(files["dielectric_script"].read_text())


# %%
# 6) Final spectra
# ----------------
#
# The demo post-processes time series into:
# - `xx_der_facf.data` (VDOS)
# - `mm_der_facf.data` (IR)
# - `L0L0_der_facf.data` (isotropic Raman)
# - `L2L2_der_facf.data` (anisotropic Raman)

if files is not None:
    print(files["spectra_script"].read_text())


# %%
# Optional visualization for completed runs
# -----------------------------------------
#
# If the corresponding output files are present in the working directory,
# this block plots trajectories and spectra.

if Path("simulation.out").exists():
    output_data, output_desc = ipi.read_output("simulation.out")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    ax.plot(output_data["time"], output_data["temperature"], label="temperature")
    ax.set_xlabel(r"$t$ / ps")
    ax.set_ylabel(r"temperature / K")
    ax.legend()
    plt.show()


def load_spectrum(path):
    data = np.loadtxt(path)
    freq_cm = data[:, 0] * 219474.63
    intensity = data[:, 1]
    return freq_cm, intensity


spectra_files = {
    "VDOS": Path("xx_der_facf.data"),
    "IR": Path("mm_der_facf.data"),
    "Raman (isotropic)": Path("L0L0_der_facf.data"),
    "Raman (anisotropic)": Path("L2L2_der_facf.data"),
}

available = [(label, path) for label, path in spectra_files.items() if path.exists()]
if available:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), constrained_layout=True)
    for label, path in available:
        x, y = load_spectrum(path)
        ax.plot(x, y, label=label)
    ax.set_xlabel(r"frequency / cm$^{-1}$")
    ax.set_ylabel("intensity (arb. units)")
    ax.set_xlim(0, 4500)
    ax.legend()
    plt.show()
