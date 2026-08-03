"""
The metatomic hourglass design
==============================

:Authors: Joseph W. Abbott `@jwa7 <https://github.com/jwa7>`_; Filippo Bigi
          `@frostedoyster <https://github.com/frostedoyster>`_; Michele Ceriotti
          `@ceriottm <https://github.com/ceriottm>`_; Pol Febrer `@pfebrer
          <https://github.com/pfebrer>`_; Guillaume Fraux `@Luthaf
          <https://github.com/luthaf>`_; Paolo Pegolo `@ppegolo
          <https://github.com/ppegolo>`_; Johannes Spies `@johannes-spies
          <https://github.com/johannes-spies>`_

Machine-learning interatomic potentials come in many flavors. They use different
architectures, are trained with different frameworks, and are distributed in different
formats. Simulation engines are just as diverse, each with its own input files, plugin
mechanisms, and internal units. Connecting :math:`M` models to :math:`N` engines naively
requires :math:`M \\times N` custom interfaces.

`metatomic <https://docs.metatensor.org/metatomic/>`_ avoids this combinatorial
explosion with what could be called an hourglass design. A model of any origin is
exported once into a common format, a TorchScript module that declares its outputs,
units, cutoff, and supported species in a standardized way. This exported file is the
narrow neck of the hourglass. Any engine with a metatomic interface can then run any
exported model, and the :math:`M \\times N` problem becomes :math:`M + N`.

This recipe walks through the whole hourglass. We first obtain three foundation models
of very different origins. `PET-MAD <https://arxiv.org/abs/2503.14118>`_ is trained with
`metatrain <https://docs.metatensor.org/metatrain/>`_. The `MACE
<https://github.com/ACEsuit/mace>`_ model is trained on the OMat24 dataset with the
independent ``mace`` package. Finally, `DPA3 <https://arxiv.org/abs/2506.01686>`_ comes
from the `deepmd-kit <https://github.com/deepmodeling/deepmd-kit>`_ ecosystem, in the
form of a branch of the multitask DPA-3.1-3M checkpoint. After exporting all three to
the metatomic format, we use them to run the same short molecular dynamics simulation of
an ethanol molecule, in both the NVT and the NVE ensembles, with five different engines.
These are `ASE <https://ase-lib.org>`_, `LAMMPS <https://www.lammps.org>`_, `GROMACS
<https://www.gromacs.org>`_, `i-PI <https://ipi-code.org>`_, and `TorchSim
<https://radical-ai.github.io/torch-sim/>`_.

As usual for the recipes in this cookbook, the simulations are small and short so that
they run quickly. If you use this recipe as a template for production runs, you should
increase system size, simulation time, and equilibration.
"""

# %%
#
# We start by importing the packages needed to define and export the models. The
# engine-specific packages are imported inside the corresponding functions below, to
# make it clear which engine needs what.

import subprocess
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from metatomic.torch import AtomisticModel, load_atomistic_model

# %%
#
# Models from many origins
# ------------------------
#
# The first model is `PET-MAD <https://arxiv.org/abs/2503.14118>`_, a
# universal potential based on the PET architecture and trained on the MAD
# dataset with ``metatrain``. Since ``metatrain`` is part of the metatomic
# ecosystem, a single ``mtt export`` command converts a training checkpoint
# (here downloaded from Hugging Face) into an exported metatomic model.


def get_pet() -> AtomisticModel:
    checkpoint = (
        "https://huggingface.co/lab-cosmo/upet/resolve/main/models/"
        "pet-mad-xs-v1.5.0.ckpt"
    )
    subprocess.run(
        ["mtt", "export", checkpoint, "--output", "pet-mad.pt"],
        check=True,
    )
    return load_atomistic_model("pet-mad.pt")


# %%
#
# The second model comes from outside the metatomic ecosystem. It is a MACE
# model trained on the OMat24 dataset with the `mace
# <https://github.com/ACEsuit/mace>`_ package, and distributed in MACE's
# native format. To bring it into the metatomic world we wrap it with
# ``metatrain``'s MACE architecture, as described in `the metatrain
# documentation
# <https://docs.metatensor.org/metatrain/latest/architectures/generated/mace.html>`_.
# A "training" run with zero epochs re-packages the foundation model, and the
# usual export then produces the same kind of metatomic model as before. The
# dummy dataset is never used for actual training, and only needs to declare
# the right target names.


def get_mace() -> AtomisticModel:
    import textwrap
    import urllib.request

    # download the foundation model in its native format
    urllib.request.urlretrieve(
        "https://github.com/ACEsuit/mace-foundations/releases/download/"
        "mace_omat_0/mace-omat-0-small.model",
        "mace-omat-0-small.model",
    )

    # a metatrain options file wrapping the MACE model, with zero
    # training epochs
    train_yaml = """\
    architecture:
        name: experimental.mace
        model:
            mace_model: mace-omat-0-small.model
            mace_head_target: energy
        training:
            num_epochs: 0
            batch_size: 1

    # declaring the units here is important: they end up in the
    # capabilities of the exported model, and the engines rely on them
    # to convert to their internal units
    training_set:
        systems:
            read_from: mace_dummy_dataset.xyz
            length_unit: angstrom
        targets:
            energy:
                key: energy
                unit: eV
    validation_set: 0.0
    """

    # a dummy dataset with a single H2 molecule; only the names of the
    # targets ("energy", "forces") matter
    dummy_dataset = """\
    2
    Properties=species:S:1:pos:R:3:forces:R:3 energy=-2.1
    H 0.0 0.0 0.0 0.0 0.0 0.0
    H 1.0 0.0 0.0 0.0 0.0 0.0
    """

    with open("mace_train.yaml", "w") as f:
        f.write(textwrap.dedent(train_yaml))
    with open("mace_dummy_dataset.xyz", "w") as f:
        f.write(textwrap.dedent(dummy_dataset))

    subprocess.run(
        ["mtt", "train", "mace_train.yaml", "--output", "mace.pt"],
        check=True,
    )
    return load_atomistic_model("mace.pt")


# %%
#
# The third model stretches the hourglass the furthest. DPA-3.1-3M is a
# multitask model trained with `deepmd-kit
# <https://github.com/deepmodeling/deepmd-kit>`_ on the OpenLAM datasets,
# with one output branch per training domain. We extract the branch trained
# on drug-like molecules and hand it to ``metatrain``'s DPA3 architecture,
# this time through the Python API rather than the command line. The wrapped
# module calls compiled TorchScript operators from deepmd-kit, a detail that
# will come back when we run the engines below.


def get_dpa3() -> AtomisticModel:
    import urllib.request

    from deepmd.pt.infer.inference import Tester
    from metatrain.experimental.dpa3 import DPA3
    from metatrain.utils.architectures import get_default_hypers
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.target_info import get_energy_target_info

    # download the multitask checkpoint in its native format
    urllib.request.urlretrieve(
        "https://huggingface.co/deepmodelingcommunity/DPA/resolve/main/DPA-3.1-3M.pt",
        "dpa3-base.pt",
    )

    # extract the branch trained on drug-like molecules
    import ase.data
    import torch

    branch = Tester("dpa3-base.pt", head="Domains_Drug").model

    # the atomic types must follow the order of the type map the model
    # was trained with, so that species are mapped to the right outputs
    checkpoint = torch.load("dpa3-base.pt", map_location="cpu", weights_only=False)
    model_params = checkpoint["model"]["_extra_state"]["model_params"]
    type_map = model_params["model_dict"]["Domains_Drug"]["type_map"]
    atomic_types = [ase.data.atomic_numbers[symbol] for symbol in type_map]

    # wrap the deepmd-kit module with metatrain's DPA3 architecture,
    # declaring units and atomic types as we did for MACE
    hypers = get_default_hypers("experimental.dpa3")["model"]
    hypers["dpa3_model"] = branch
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )

    DPA3(hypers, dataset_info).export().save("dpa3.pt")
    return load_atomistic_model("dpa3.pt")


# %%
#
# The neck of the hourglass
# -------------------------
#
# Both functions return an ``AtomisticModel``, a TorchScript module bundled with
# metadata and a declaration of its capabilities. This single format is the neck of the
# hourglass, and it is all an engine needs to run the model. No engine ever has to know
# whether it is talking to PET, MACE, or anything else.

models = {
    "PET-MAD": get_pet(),
    "MACE-OMAT": get_mace(),
    "DPA3": get_dpa3(),
}

for name, model in models.items():
    capabilities = model.capabilities()
    print(f"{name}:")
    print(f"  outputs: {list(capabilities.outputs.keys())}")
    print(f"  length unit: {capabilities.length_unit}")
    print(f"  interaction range: {capabilities.interaction_range}")
    print()

# %%
#
# Many engines below the neck
# ---------------------------
#
# We now define one function per engine. Each function takes an ``AtomisticModel`` and
# an ensemble (``"nve"`` or ``"nvt"``), runs 100 steps of MD for a single ethanol
# molecule with a 0.5 fs timestep, and returns the simulation time (in fs) and the
# potential energy (in eV) along the trajectory. NVE runs start from zero velocities, so
# that all engines follow the same deterministic trajectory, while NVT runs use each
# engine's thermostat at 300 K.
#
# Engines differ in what they report by default: some skip the initial configuration,
# some subsample the trajectory. Each function below is written so that all of them
# return the same 101 snapshots, from :math:`t = 0` to :math:`t = 50` fs, which lets us
# compare the trajectories point by point.
#
# Inside each function, the engine loads the very same ``model.pt`` file.
#
# ASE
# ^^^
#
# The metatomic ASE interface provides a standard ASE ``Calculator`` class, so the model
# can be used with everything ASE offers, from structure optimizers to MD integrators
# and vibrational analysis.


def run_ase(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    import ase.io
    import ase.md
    from ase.constraints import FixCom
    from metatomic_ase import MetatomicCalculator

    atoms = ase.io.read("data/ethanol.xyz")
    atoms.calc = MetatomicCalculator(model, device="cpu")
    fixcom = FixCom()  # remove center-of-mass motion
    atoms.set_constraint(fixcom)

    dt_fs = 0.5
    if ensemble == "nve":
        integrator = ase.md.VelocityVerlet(atoms, timestep=dt_fs * ase.units.fs)
    else:
        integrator = ase.md.Langevin(
            atoms,
            timestep=dt_fs * ase.units.fs,
            temperature_K=300,
            friction=0.01 / ase.units.fs,
            fixcm=False,  # deprecated, we use the FixCom constraint instead
        )

    # ASE integrators only report the steps they take, so we record the initial
    # configuration ourselves
    times, energies = [0.0], [atoms.get_potential_energy()]
    for step in range(100):
        integrator.run(1)
        times.append((step + 1) * dt_fs)
        energies.append(atoms.get_potential_energy())

    return times, energies


# %%
#
# LAMMPS
# ^^^^^^
#
# The ``lammps-metatomic`` build of LAMMPS provides a ``pair_style metatomic``, which
# loads the exported model directly in the input file. The ``pair_coeff`` line maps
# LAMMPS atom types to atomic numbers, and the rest is a completely standard LAMMPS
# input.


def run_lammps(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    import ase.io
    import numpy as np

    atoms = ase.io.read("data/ethanol.xyz")
    atoms.set_cell([50, 50, 50])
    # write the structure with an explicit type ordering, matching the
    # atomic numbers listed in the pair_coeff line below
    ase.io.write(
        "ethanol.data",
        atoms,
        format="lammps-data",
        masses=True,
        specorder=["H", "C", "O"],
    )

    model.save("model.pt")

    if ensemble == "nve":
        ensemble_setup = """\
velocity all zero linear
fix 1 all nve
"""
    else:
        ensemble_setup = """\
velocity all create 300 87287 mom yes rot yes
fix 1 all nvt temp 300 300 0.05
"""

    with open("lammps.in", "w") as f:
        f.write(f"""\
units metal
atom_style atomic

read_data ethanol.data

pair_style metatomic model.pt device cpu
pair_coeff * * 1 6 8

neighbor 2.0 bin

timestep 0.0005

{ensemble_setup}
fix 2 all print 1 "$(time) $(pe)" file lammps.out screen no

run 100
""")

    subprocess.run(
        ["lmp", "-in", "lammps.in", "-log", "none"],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    time_ps, pe = np.loadtxt("lammps.out", skiprows=1, unpack=True)
    return (time_ps * 1000).tolist(), pe.tolist()


# %%
#
# GROMACS
# ^^^^^^^
#
# The ``gromacs-metatomic`` build of GROMACS adds a few ``metatomic-*`` keys to the
# ``.mdp`` parameter file, selecting the model file and the index group of atoms it
# applies to. The topology only provides masses, through inert atom types with no
# classical interactions, so that all forces come from the model.
#
# GROMACS' default integrator is leap-frog, which stores velocities half a step behind
# the positions: starting it "at rest" means :math:`v(-\\Delta t/2) = 0` rather than
# :math:`v(0) = 0`, and the resulting half-step offset never goes away. We ask instead
# for ``md-vv``, the velocity Verlet integrator the other engines use.


def run_gromacs(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    import shutil

    import ase.io
    import ase.units
    import numpy as np

    model.save("model.pt")
    gmx = shutil.which("gmx_mpi") or shutil.which("gmx")

    # write a .gro coordinate file with the same geometry as ethanol.xyz,
    # with the atoms reordered to match the topology in data/topol.top
    atoms = ase.io.read("data/ethanol.xyz")
    order = [1, 4, 7, 0, 2, 3, 5, 6, 8]
    names = ["C1", "C2", "O1", "H11", "H12", "H13", "H21", "H22", "HO"]
    with open("ethanol.gro", "w") as f:
        f.write("Ethanol molecule\n    9\n")
        for i, (idx, name) in enumerate(zip(order, names)):
            x, y, z = atoms.positions[idx] / 10  # angstrom -> nm
            f.write(
                f"{1:5d}{'ETOH':<5s}{name:>5s}{i + 1:5d}{x:10.5f}{y:10.5f}{z:10.5f}\n"
            )
        f.write("   5.00000   5.00000   5.00000\n")

    if ensemble == "nve":
        thermostat = "tcoupl = no"
    else:
        thermostat = """\
tcoupl = v-rescale
tc-grps = Ethanol
tau-t = 0.1
ref-t = 300
"""

    with open("grompp.mdp", "w") as f:
        f.write(f"""\
; velocity Verlet, rather than the default leap-frog
integrator = md-vv
dt = 0.0005
nsteps = 100

cutoff-scheme = Verlet
pbc = xyz
; the pairlist must cover the interaction range of the model
rlist = 1.6
rcoulomb = 1.6
rvdw = 1.6

{thermostat}

metatomic-active = yes
metatomic-model = model.pt
metatomic-input-group = Ethanol
metatomic-device = cpu

nstenergy = 1
nstlog = 100
""")

    for command in [
        f"{gmx} grompp -f grompp.mdp -c ethanol.gro -p data/topol.top "
        "-n data/index.ndx -o run.tpr",
        f"{gmx} mdrun -deffnm run",
    ]:
        subprocess.run(
            command.split(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # extract the potential energy from the .edr file
    subprocess.run(
        [gmx, "energy", "-f", "run.edr", "-o", "energy.xvg"],
        input="Potential\n",
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    data = np.loadtxt("energy.xvg", comments=["@", "#"])

    time_fs = data[:, 0] * 1000  # ps -> fs
    energy_ev = data[:, 1] * ase.units.kJ / ase.units.mol  # kJ/mol -> eV
    return time_fs.tolist(), energy_ev.tolist()


# %%
#
# i-PI
# ^^^^
#
# i-PI drives the dynamics and delegates the evaluation of energies and forces to a
# metatomic force field, which here runs in direct mode within the same process. We use
# i-PI's scripting utilities to build the XML input.


def run_ipi(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    import ase.io
    from ipi.utils.parsing import read_output
    from ipi.utils.scripting import (
        InteractiveSimulation,
        forcefield_xml,
        motion_nvt_xml,
        simulation_xml,
    )
    from ipi.utils.softexit import softexit

    model.save("model.pt")

    structure = ase.io.read("data/ethanol.xyz")
    structure.cell = [50, 50, 50]

    if ensemble == "nve":
        motion = """
        <motion mode='dynamics'>
            <dynamics mode='nve'>
                <timestep units='femtosecond'> 0.5 </timestep>
            </dynamics>
        </motion>
        """
        temperature = None
    else:
        motion = motion_nvt_xml(timestep=0.5 * ase.units.fs)
        temperature = 300

    # i-PI's default scripting output only prints every other step; ask for every step
    # instead, so that the trajectory can be compared to the other engines
    output = """
    <output prefix='simulation'>
        <properties stride='1' filename='out'>
            [ step, time{picosecond}, potential{electronvolt} ]
        </properties>
    </output>
    """

    input_xml = simulation_xml(
        structures=structure,
        forcefield=forcefield_xml(
            name="metatomic",
            mode="direct",
            pes="metatomic",
            parameters="{template:data/ethanol.xyz,model:model.pt,device:cpu}",
        ),
        motion=motion,
        temperature=temperature,
        output=output,
        prefix="ethanol-ipi",
    )

    # softexit is global: if anything trips it, every later simulation in this
    # process quits after a few steps without saying anything
    softexit.reset()

    sim = InteractiveSimulation(input_xml)
    sim.run(100)

    results, info = read_output("ethanol-ipi.out")
    return (
        (results["time"] * 1000).tolist(),  # ps -> fs
        results["potential"].tolist(),  # already in eV
    )


# %%
#
# TorchSim
# ^^^^^^^^
#
# TorchSim is a PyTorch-native MD engine designed to run batched simulations on GPUs.
# The ``metatomic_torchsim`` package wraps an exported metatomic model into a TorchSim
# ``ModelInterface``.


def run_torchsim(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    from functools import partial

    import ase.io
    import torch
    import torch_sim as ts
    from metatomic_torchsim import MetatomicModel
    from torch_sim.units import MetalUnits

    model.save("model.pt")
    ts_model = MetatomicModel("model.pt", device="cpu")

    if ensemble == "nve":
        init_fn, step_fn = ts.nve_init, ts.nve_step
    else:
        init_fn = ts.nvt_langevin_init
        step_fn = partial(ts.nvt_langevin_step, gamma=10 / MetalUnits.time)

    atoms = ase.io.read("data/ethanol.xyz")
    sim_state = ts.initialize_state(atoms, device=ts_model.device, dtype=ts_model.dtype)

    dt = 0.0005 * MetalUnits.time  # 0.5 fs
    kt = 300.0 * MetalUnits.temperature
    md_state = init_fn(sim_state, ts_model, kT=kt, seed=42)
    if ensemble == "nve":
        # start from zero velocities, like the other engines
        md_state.momenta = torch.zeros_like(md_state.momenta)

    # like ASE, TorchSim leaves it to us to record the initial configuration
    times, energies = [0.0], [md_state.energy.sum().item()]
    for step in range(100):
        md_state = step_fn(md_state, ts_model, dt=dt, kT=kt)
        times.append((step + 1) * 0.5)
        energies.append(md_state.energy.sum().item())

    return times, energies


# %%
#
# Running through the hourglass
# -----------------------------
#
# We can now run every combination of model and engine with the same code. Almost every
# combination, that is. The DPA3 model calls compiled deepmd-kit operators from inside
# its TorchScript code, and at the time of writing there is no deepmd-kit build
# compatible with the libtorch version used by the LAMMPS and GROMACS packages. Those
# two combinations are skipped below, a reminder that the neck of the hourglass is only
# as portable as the operators a model brings along.
#
# We start with a thermostatted (NVT) trajectory for each combination. The absolute
# energies differ between the models, which use different architectures and are trained
# on different datasets, but each column of the figure behaves consistently across the
# engines.

engines = {
    "ASE": run_ase,
    "LAMMPS": run_lammps,
    "GROMACS": run_gromacs,
    "i-PI": run_ipi,
    "TorchSim": run_torchsim,
}

unsupported = {("LAMMPS", "DPA3"), ("GROMACS", "DPA3")}

fig, axes = plt.subplots(
    len(engines),
    len(models),
    figsize=(11, 12),
    sharex=True,
    constrained_layout=True,
    dpi=200,
)

for col, (model_name, model) in enumerate(models.items()):
    for row, (engine_name, run_engine) in enumerate(engines.items()):
        ax = axes[row, col]
        ax.set_title(f"{engine_name} — {model_name}", fontsize=10)
        if (engine_name, model_name) in unsupported:
            ax.text(
                0.5,
                0.5,
                "not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.axis("off")
            continue
        print(f"Running {engine_name} with {model_name} (NVT)")
        times, energies = run_engine(model, ensemble="nvt")
        ax.plot(times, energies)
        ax.set_ylabel("U / eV")

for ax in axes[-1, :]:
    ax.set_xlabel("t / fs")

plt.show()

# %%
#
# The same trajectory from every engine
# -------------------------------------
#
# The NVT curves above fluctuate differently because each engine uses its own thermostat
# and random numbers. A sharper consistency check is a microcanonical (NVE) trajectory
# started from the same configuration at rest. All engines should then follow the same
# deterministic trajectory, and for each model the five curves should superimpose.

fig, axes = plt.subplots(
    1, len(models), figsize=(12, 4), constrained_layout=True, sharex=True, dpi=200
)

nve_trajectories = {}

for col, (model_name, model) in enumerate(models.items()):
    nve_trajectories[model_name] = {}
    for engine_name, run_engine in engines.items():
        if (engine_name, model_name) in unsupported:
            continue
        print(f"Running {engine_name} with {model_name} (NVE)")
        times, energies = run_engine(model, ensemble="nve")
        nve_trajectories[model_name][engine_name] = (
            np.array(times),
            np.array(energies),
        )
        axes[col].plot(times, energies, label=engine_name)
    axes[col].set_title(model_name)
    axes[col].set_xlabel("t / fs")
    axes[col].set_ylabel("U / eV")
    axes[col].legend()

plt.show()

# %%
#
# The curves are hard to tell apart by eye, so we also check the agreement numerically,
# taking ASE as the reference. The engines do not agree exactly: GROMACS runs in mixed
# precision, so its trajectory slowly drifts away from the others, but 1 meV is a
# generous bound on that, and a small fraction of the range the energy spans here.

TIME_TOLERANCE = 1e-6  # fs
ENERGY_TOLERANCE = 0.001  # eV

for model_name, trajectories in nve_trajectories.items():
    reference_times, reference_energies = trajectories["ASE"]
    for engine_name, (times, energies) in trajectories.items():
        assert len(times) == len(reference_times), (
            f"NVE trajectories are not sampled at the same times across engines: "
            f"{engine_name} reports {len(times)} snapshots for {model_name}, "
            f"ASE reports {len(reference_times)}"
        )
        assert np.abs(times - reference_times).max() < TIME_TOLERANCE, (
            f"NVE trajectories are not sampled at the same times across engines: "
            f"{engine_name} differs from ASE for {model_name}"
        )

        energy_error = np.abs(energies - reference_energies).max()
        print(f"{model_name} / {engine_name}: max |ΔU| = {1000 * energy_error:.3f} meV")
        assert energy_error < ENERGY_TOLERANCE, (
            f"NVE trajectories don't match across engines: {engine_name} differs by "
            f"{1000 * energy_error:.3f} meV for {model_name}"
        )

# %%
#
# Where to go from here
# ---------------------
#
# The five engines used here are not the whole story. The same exported models also run
# in `PLUMED <https://www.plumed.org/doc-master/user-doc/html/METATOMIC>`_, as shown in
# the `recipe on machine-learned collective variables
# <https://atomistic-cookbook.org/examples/metatomic-plumed/metatomic-plumed.html>`_. In
# the other direction, any model trained with ``metatrain``, or wrapped in a custom
# ``torch.nn.Module``, flows through the same neck of the hourglass.
