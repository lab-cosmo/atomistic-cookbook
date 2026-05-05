"""
Metatomic hourglass demonstration
=================================



"""

# %%

from typing import List, Tuple, Literal
import matplotlib.pyplot as plt

from metatomic.torch import AtomisticModel, load_atomistic_model


# %%
def get_pet() -> AtomisticModel:
    # TODO: this should work but fails with a float32 error when saving
    # from upet import get_upet

    # # FIXME: we are missing the name of the model in metadata
    # model = get_upet(model="pet-mad", size="s")
    # return AtomisticModel(model, model.metadata(), model.capabilities())

    import metatrain
    import subprocess

    subprocess.run(
        [
            "mtt",
            "export",
            "https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-xs-v1.5.0.ckpt",
            "--output",
            "model.pt",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return load_atomistic_model("model.pt")


# %%
# Export a MACE model following the instructions in the
# `metatrain documentation <https://docs.metatensor.org/metatrain/latest/architectures/generated/mace.html#exporting-a-foundation-mace-model>`_.


def get_mace() -> AtomisticModel:
    import urllib.request
    import subprocess
    import textwrap

    # Download model.
    path = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-small.model"
    urllib.request.urlretrieve(path, "mace-omat-0-small.model")

    # Training yaml with 0 epochs
    train_yaml = """\
    architecture:
        name: experimental.mace
        model:
            mace_model: mace-omat-0-small.model
            mace_head_target: energy
        training:
            num_epochs: 0
            batch_size: 1

    training_set: mace_dummy_dataset.xyz
    validation_set: 0.0
    """

    # A dummy dataset with a single H2 molecule, it will
    # not really get used. The only thing that has to be
    # correct is the name of the targets ("energy", "forces").
    dummy_dataset = """\
    2
    Properties=species:S:1:pos:R:3:forces:R:3 energy=-2.1
    H 0.0 0.0 0.0 0.0 0.0 0.0
    H 1.0 0.0 0.0 0.0 0.0 0.0
    """

    # Write the training yaml and dummy dataset to disk
    with open("mace_train.yaml", "w") as f:
        f.write(textwrap.dedent(train_yaml))
    with open("mace_dummy_dataset.xyz", "w") as f:
        f.write(textwrap.dedent(dummy_dataset))

    # Run training.
    subprocess.run(
        [
            "mtt",
            "train",
            "mace_train.yaml",
            "--output",
            "model.pt",
        ],
        check=True,
    )

    return load_atomistic_model("model.pt")


# %%
def get_dpa3() -> AtomisticModel:
    pass


# %%
def run_ase(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:

    if ensemble not in ["nve", "nvt"]:
        raise NotImplementedError("only nve and nvt are implemented for ase")

    import ase.io
    import ase.md

    # Integration with ASE calculator for metatomic models
    from metatomic.torch.ase_calculator import MetatomicCalculator

    # load the frame and set up the calculator
    atoms = ase.io.read("data/ethanol.xyz")
    atoms.calc = MetatomicCalculator(model)

    # set up the integrator
    dt = 1.0 * ase.units.fs
    n_steps = 100
    if ensemble == "nve":
        integrator = ase.md.VelocityVerlet(
            atoms,
            timestep=dt,
        )
    else:
        assert ensemble == "nvt"
        integrator = ase.md.Langevin(
            atoms,
            timestep=dt,
            temperature_K=300,
            friction=0.1 / ase.units.fs,
        )

    # run a short MD simulation and store the potential energy
    time = []
    potential_energy = []
    for step in range(n_steps):
        integrator.run(1)
        time.append(step * dt)
        potential_energy.append(atoms.get_potential_energy())

    return time, potential_energy


# %%
def run_lammps(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    if ensemble != "nve":
        raise NotImplementedError("only nve is implemented for lammps")

    import ase.io
    import subprocess
    import numpy as np

    data = ase.io.read("data/ethanol.xyz")
    cell = [50, 50, 50]
    data.set_cell(cell)
    ase.io.write("ethanol.data", data, format="lammps-data", masses=True)

    model_name = "model.pt"
    model.save(model_name)

    timestep = 1e-3  # ps
    nstep = 100

    with open("lammps.in", "w") as f:
        f.write(f"""units metal
atom_style atomic

read_data ethanol.data

pair_style metatomic {model_name} device cpu
pair_coeff * * 1 6 8

neighbor 2.0 bin
neigh_modify one 100000 page 1000000 binsize 5.5

timestep {timestep}

# velocity all create 300 87287 mom yes rot yes
velocity all zero linear

# fix 1 all nvt temp 300 300 0.10
fix 1 all nve
run_style verlet

fix 2 all print 1 "$(time) $(pe)" file lammps.out screen no

run {nstep}
""")

    # run lammps
    subprocess.run(
        ["lmp", "-in", "lammps.in", "-log", "none"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # read the output file
    time, pe = np.loadtxt("lammps.out", skiprows=1, unpack=True)
    return (time * 1000).tolist(), pe.tolist()


# %%
def run_ipi(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    if ensemble != "nvt":
        raise NotImplementedError("only nvt is implemented for i-pi")

    import ase.io
    from ipi.utils.scripting import (
        simulation_xml,
        forcefield_xml,
        motion_nvt_xml,
        InteractiveSimulation,
    )
    from ipi.utils.parsing import read_output

    model.save("model.pt")
    structure = ase.io.read("data/ethanol.xyz")
    structure.cell = [50, 50, 50]
    input_xml = simulation_xml(
        structures=structure,
        forcefield=forcefield_xml(
            name="metatomic",
            mode="direct",
            pes="metatomic",
            parameters="{template:data/ethanol.xyz,model:model.pt,device:cpu}",
        ),
        motion=motion_nvt_xml(timestep=0.5 * ase.units.fs),
        temperature=300,
        prefix="ethanol-ipi",
    )

    sim = InteractiveSimulation(input_xml)
    sim.run(100)
    results, _ = read_output("ethanol-ipi.out")
    print("works!")
    return results["time"], results["potential"]


# %%
def run_torchsim(
    model: AtomisticModel, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    from functools import partial
    import ase.io
    import torch_sim as ts
    from torch_sim.units import MetalUnits

    from metatomic_torchsim import MetatomicModel

    # decide which functions to use based on the ensemble
    fns = {
        "nve": (ts.nve_init, ts.nve_step),
        "nvt": (
            ts.nvt_langevin_init,
            partial(ts.nvt_langevin_step, gamma=10 / MetalUnits.time),
        ),
    }
    init_fn, step_fn = fns[ensemble]

    # save the model to be reloaded by TorchSim
    model.save("model.pt")
    model = MetatomicModel("model.pt")

    atoms = ase.io.read("data/ethanol.xyz")
    sim_state = ts.initialize_state(atoms, device=model.device, dtype=model.dtype)

    # initialize the simulation state
    dt = 1e-3 * MetalUnits.time  # 1 fs time step
    kt = 300.0 * MetalUnits.temperature
    md_state = init_fn(sim_state, model, kT=kt)

    # run 100 fs of MD
    times, energies = [], []
    for step in range(100):
        md_state = step_fn(md_state, model, dt=dt, kT=kt)
        times.append(float(step))
        energies.append(md_state.energy.sum().item())

    return times, energies  # return time in fs and energy in eV


def run_gromacs(
    model, ensemble: Literal["nve", "nvt"]
) -> Tuple[List[float], List[float]]:
    if ensemble != "nvt":
        raise NotImplementedError("only nvt is implemented for gromacs")

    import subprocess
    import numpy as np

    model.save("model.pt")

    # gmx grompp -f grompp.mdp -c ethanol.gro -p topol.top -n index.ndx -o run.tpr
    subprocess.run(
        [
            "gmx",
            "grompp",
            "-f",
            "grompp.mdp",
            "-c",
            "ethanol.gro",
            "-p",
            "topol.top",
            "-n",
            "index.ndx",
            "-o",
            "run.tpr",
        ],
        text=True,
    )

    # gmx mdrun -v -deffnm run -update gpu
    subprocess.run(
        ["gmx", "mdrun", "-v", "-deffnm", "run", "-update", "gpu"], text=True
    )

    subprocess.run(["gmx", "energy", "-f", "energy.edr", "-o", "energy.xvg"], text=True)

    # Load the file, skipping the GROMACS header lines
    data = np.loadtxt("energy.xvg", comments=["@", "#"])

    time = data[:, 0]  # First column is always Time (ps)
    energy = data[:, 1]  # Second column is the first property you selected

    # convert energy from kJ/mol to eV
    energy = energy / 96.485

    return time, energy


# %%

# some metadata for printing later
get_pet.__model_name__ = "PET"
get_mace.__model_name__ = "MACE"
get_dpa3.__model_name__ = "DPA-3"

run_ase.__engine_name__ = "ASE"
run_lammps.__engine_name__ = "LAMMPS"
run_gromacs.__engine_name__ = "GROMACS"
run_ipi.__engine_name__ = "i-PI"
run_torchsim.__engine_name__ = "TorchSim"

# %%

all_models = [
    get_pet,
    get_mace,
    # get_dpa3,
]

all_engines = [
    run_torchsim,
    run_ase,
    run_lammps,
    run_gromacs,
    run_ipi,
]

# %%
fig, ax = plt.subplots(len(all_engines), len(all_models), figsize=(15, 8))

for model_i, model_getter in enumerate(all_models):
    model = model_getter()
    for engine_i, run_engine in enumerate(all_engines):
        print(
            f"Running {run_engine.__engine_name__} with {model_getter.__model_name__}"
        )
        times, energies = run_engine(model, ensemble="nvt")
        ax[engine_i, model_i].plot(times, energies)
        ax[engine_i, model_i].set_title(
            f"{run_engine.__engine_name__} — {model_getter.__model_name__}"
        )
        # TODO: check units
        ax[engine_i, model_i].set_xlabel("Time (fs)")
        ax[engine_i, model_i].set_ylabel("Energy (eV)")

plt.tight_layout()
plt.show()

# %%
# Plot NVE trajectories -- they should be similar across engines, but may differ across
# models. We put them all in the same figure to make it easier to compare.

plt.ylabel("Energy (eV)")
plt.xlabel("Time (fs)")
for model_i, model_getter in enumerate(all_models):
    model = model_getter()
    for engine_i, run_engine in enumerate(all_engines):
        print(
            f"Running {run_engine.__engine_name__} with {model_getter.__model_name__}"
        )
        times, energies = run_engine(model, ensemble="nve")
        plt.plot(
            times,
            energies,
            label=f"{run_engine.__engine_name__} — {model_getter.__model_name__}",
        )
plt.legend()
plt.tight_layout()
plt.show()

# %%
