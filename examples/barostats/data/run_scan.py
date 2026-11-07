"""
Runs the (long) simulations that are used as a reference in the barostats
recipe, and post-processes them into small text files that are stored in
``data/``. This takes a few hours on a ~20-core workstation, and is *not*
executed when building the recipe.

Usage (from the recipe folder)::

    python data/run_scan.py run      --nproc 3 --time 100
    python data/run_scan.py run      --nproc 3 --time 1000 --names NAME1,NAME2
    python data/run_scan.py analyze

Raw outputs are written to ``scan/`` (or to the first of the folders listed,
separated by ``:``, in the ``BAROSCAN_DIR`` environment variable). The analysis
uses, for each run, the longest trajectory found in any of these folders.
"""

import argparse
import os
import re
import subprocess
import sys
import time
import zlib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
SCAN_DIRS = os.environ.get("BAROSCAN_DIR", os.path.join(ROOT, "scan")).split(":")
SCAN = SCAN_DIRS[0]
DT = 0.5  # fs
STRIDE = 4  # output stride (steps), i.e. 2 fs
# avoid oversubscription when running many simulations in parallel
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
PRESS = 1.0  # bar
# systems: initial structure, temperature (K), estimate of the bulk modulus (GPa)
SYSTEMS = {
    "ice": dict(structure="ice_96", temp=270.0, bulk_modulus=9.0),
}


# --------------------------------------------------------------------------------
# list of runs
# --------------------------------------------------------------------------------
def run_list():
    runs = []
    # i-PI BZP barostat, critically damped, with a fast global thermostat or an
    # optimal-sampling GLE on the atoms
    for th in [("svr", 10), ("gle", "smart")]:
        for tb in [10, 30, 100, 300, 1000]:
            tc = round(0.3 * tb, 1)
            runs.append(
                dict(
                    name=f"ice_ipi_b{tb}_c{tc:g}_{th[0]}{th[1]}",
                    code="ipi",
                    system="ice",
                    tau_baro=tb,
                    tau_cell=tc,
                    atoms=th,
                )
            )
    # other codes, and the effect of the thermostats
    for tb in [100, 200, 500, 1000, 2000]:
        for code in ["lmp_mtk", "ase_mtk"]:
            runs.append(
                dict(
                    name=f"ice_{code}_b{tb}",
                    code=code,
                    system="ice",
                    tau_baro=tb,
                    tdamp=100,
                )
            )
    for tb, tc, th in [
        (30, 9, ("langevin", 10)),
        (100, 30, ("langevin", 10)),
        (100, 1000, ("svr", 10)),
        (300, 3000, ("svr", 10)),
    ]:
        runs.append(
            dict(
                name=f"ice_ipi_b{tb}_c{tc:g}_{th[0]}{th[1]}",
                code="ipi",
                system="ice",
                tau_baro=tb,
                tau_cell=tc,
                atoms=th,
            )
        )
    # GLE combined with a fast SVR, with tau given directly as the period
    runs.append(
        dict(
            name="ice_ipi_b100_c8_glesvr10",
            code="ipi",
            system="ice",
            tau_baro=100 * np.sqrt(3) / (2 * np.pi),
            tau_period=100,
            tau_cell=8,
            atoms=("multi", "glesvr10"),
        )
    )
    # independent replicas (different random seeds) of the most relevant setups,
    # to estimate the statistical error and to average the ACF
    base = {r["name"]: r for r in runs}
    for name in REPLICATED:
        for k in range(1, N_REPLICAS.get(name, N_REPLICAS[None])):
            runs.append({**base[name], "name": f"{name}_r{k}"})
    return runs


REPLICATED = [
    "ice_ipi_b100_c30_svr10",
    "ice_ipi_b100_c30_glesmart",
    "ice_ipi_b100_c30_langevin10",
    "ice_ipi_b1000_c300_svr10",
    "ice_lmp_mtk_b200",
    "ice_ase_mtk_b200",
]
# number of replicas (including the original run): more for LAMMPS fix npt, whose
# mean volume is slightly smaller than that of the other setups
N_REPLICAS = {None: 4, "ice_lmp_mtk_b200": 6}


# --------------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------------
def wait_socket(address, proc, timeout=60):
    t0 = time.time()
    while not os.path.exists(f"/tmp/ipi_{address}"):
        if proc.poll() is not None or time.time() - t0 > timeout:
            raise RuntimeError(f"server for {address} did not start")
        time.sleep(0.2)


def lmp_client(address, folder, system="ice"):
    structure = SYSTEMS[system]["structure"]
    return subprocess.Popen(
        ["lmp", "-in", "data/in.lmp", "-var", "address", address]
        + ["-var", "data", f"data/{structure}.lmp"]
        + ["-log", "none", "-screen", os.path.join(folder, "lmp.screen")],
        cwd=ROOT,
    )


def run_ipi(r, nsteps, folder, address):
    xml = ET.parse(os.path.join(ROOT, "data", "input_npt.xml"))
    root = xml.getroot()
    root.find("output").set("prefix", os.path.join(folder, "sim"))
    root.find("output/trajectory").set("stride", "10000")
    root.find("total_steps").text = str(nsteps)
    root.find("prng/seed").text = str(zlib.crc32(r["name"].encode()) % 100000)
    root.find("ffsocket/address").text = address
    system = SYSTEMS[r.get("system", "ice")]
    root.find("system/initialize/file").text = os.path.join(
        ROOT, "data", system["structure"] + ".xyz"
    )
    root.find("system/initialize/velocities").text = str(system["temp"])
    root.find("system/ensemble/temperature").text = str(system["temp"])
    dyn = root.find("system/motion/dynamics")
    th = dyn.find("thermostat")
    th.set("mode", r["atoms"][0])

    def add_gle(parent):
        # optimal-sampling GLE (the second element of r["atoms"] is just a label)
        A = np.loadtxt(os.path.join(ROOT, "data", "gle_smart.A"))
        ET.SubElement(parent, "A", shape=str(A.shape)).text = (
            "[" + ", ".join(f"{a:.12e}" for a in A.flatten()) + "]"
        )

    if r["atoms"][0] == "gle":
        th.remove(th.find("tau"))
        add_gle(th)
    elif r["atoms"][0] == "multi":
        # optimal-sampling GLE combined with a fast global SVR thermostat
        th.remove(th.find("tau"))
        add_gle(ET.SubElement(th, "thermostat", mode="gle"))
        svr = ET.SubElement(th, "thermostat", mode="svr")
        ET.SubElement(svr, "tau", units="femtosecond").text = "10"
    else:
        th.find("tau").text = str(r["atoms"][1])
    baro = dyn.find("barostat")
    # tau_baro follows the old i-PI convention (see LEGACY_TAU), converted to the
    # period of the volume oscillations that recent versions of i-PI expect,
    # unless the period is given directly
    period = r.get("tau_period", r["tau_baro"] * 2 * np.pi / np.sqrt(3))
    baro.find("tau").text = str(period)
    baro.find("bulk_modulus").text = str(system["bulk_modulus"])
    baro.find("thermostat/tau").text = str(r["tau_cell"])
    xml.write(os.path.join(folder, "input.xml"))

    ipi = subprocess.Popen(
        ["i-pi", os.path.join(folder, "input.xml")],
        cwd=folder,
        stdout=open(os.path.join(folder, "ipi.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    wait_socket(address, ipi)
    lmp = lmp_client(address, folder, r.get("system", "ice"))
    ipi.wait()
    lmp.wait()
    out = np.loadtxt(os.path.join(folder, "sim.out"))
    # time (ps), volume (A^3), pressure (bar), temperature (K)
    return np.vstack([out[:, 1], out[:, 6], out[:, 5], out[:, 3]]).T


LMP_HEADER = """
units           real
atom_style      full
pair_style      lj/cut/tip4p/long 1 2 1 1 0.14715 9.0
bond_style      class2
angle_style     harmonic
kspace_style    pppm/tip4p 1.0e-5
read_data       data/{structure}.lmp
pair_coeff      * * 0.0 0.0
pair_coeff      1 1 0.18521 3.1589
neighbor        2.0 bin
timestep        {dt}
velocity        all create {T} {seed} mom yes rot yes dist gaussian
"""

LMP_FOOTER = """
variable        V equal vol
variable        P equal press*1.01325       # atm -> bar
variable        T equal temp
fix             out all ave/time {stride} 1 {stride} v_V v_P v_T file {folder}/sim.out
run             {nsteps}
"""


def run_lmp(r, nsteps, folder, address):
    system = SYSTEMS[r.get("system", "ice")]
    temp = system["temp"]
    seed = zlib.crc32(r["name"].encode()) % 100000 + 1
    pa = PRESS / 1.01325  # bar -> atm
    fixes = (
        f"fix 1 all npt temp {temp} {temp} {r['tdamp']} iso {pa} {pa} {r['tau_baro']}\n"
    )
    inp = (
        LMP_HEADER.format(dt=DT, T=temp, seed=seed, structure=system["structure"])
        + fixes
        + LMP_FOOTER.format(stride=STRIDE, folder=folder, nsteps=nsteps)
    )
    with open(os.path.join(folder, "in.lmp"), "w") as f:
        f.write(inp)
    subprocess.run(
        ["lmp", "-in", os.path.join(folder, "in.lmp"), "-log", "none"]
        + ["-screen", os.path.join(folder, "lmp.screen")],
        cwd=ROOT,
        check=True,
    )
    out = np.loadtxt(os.path.join(folder, "sim.out"))
    return np.vstack([out[:, 0] * DT * 1e-3, out[:, 1], out[:, 2], out[:, 3]]).T


def run_ase(r, nsteps, folder, address):
    # runs in a separate process to avoid GIL contention between runs
    subprocess.run(
        [sys.executable, __file__, "ase-worker", folder, address]
        + [str(nsteps), str(r["tau_baro"]), str(r["tdamp"]), r.get("system", "ice")],
        cwd=ROOT,
        check=True,
    )
    return np.loadtxt(os.path.join(folder, "sim.out"))


def ase_worker(folder, address, nsteps, tau_baro, tdamp, system="ice"):
    import ase.units
    import ipi
    from ase.calculators.socketio import SocketIOCalculator
    from ase.md.nose_hoover_chain import IsotropicMTKNPT
    from ase.md.velocitydistribution import thermalize_momenta

    temp = SYSTEMS[system]["temp"]
    structure = SYSTEMS[system]["structure"]
    atoms = ipi.read_trajectory(os.path.join(ROOT, "data", structure + ".xyz"))[0]
    atoms.pbc = True
    seed = zlib.crc32(os.path.basename(folder).encode()) % 100000
    thermalize_momenta(atoms, temperature_K=temp, rng=np.random.default_rng(seed))

    def launch(atoms, properties, port, unixsocket):
        return lmp_client(unixsocket, folder, system)

    calc = SocketIOCalculator(unixsocket=address, launch_client=launch)
    atoms.calc = calc
    dyn = IsotropicMTKNPT(
        atoms,
        timestep=DT * ase.units.fs,
        temperature_K=temp,
        pressure_au=PRESS * ase.units.bar,
        tdamp=tdamp * ase.units.fs,
        pdamp=tau_baro * ase.units.fs,
    )
    data = []

    def log():
        stress = atoms.get_stress(include_ideal_gas=True)
        data.append(
            [
                dyn.get_time() / (1000 * ase.units.fs),
                atoms.get_volume(),
                -stress[:3].mean() / ase.units.bar,
                atoms.get_temperature(),
            ]
        )

    dyn.attach(log, interval=STRIDE)
    dyn.run(nsteps)
    np.savetxt(os.path.join(folder, "sim.out"), np.array(data))
    calc.close()


RUNNERS = dict(ipi=run_ipi, lmp_mtk=run_lmp, ase_mtk=run_ase)


def run_one(r, nsteps, idx):
    folder = os.path.join(SCAN, r["name"])
    if os.path.exists(os.path.join(folder, "volume.dat")):
        return
    os.makedirs(folder, exist_ok=True)
    address = f"baroscan{idx}"
    t0 = time.time()
    data = RUNNERS[r["code"]](r, nsteps, folder, address)
    walltime = time.time() - t0
    np.savetxt(
        os.path.join(folder, "volume.dat"),
        data,
        header=f"walltime {walltime} nsteps {nsteps}\n"
        "time/ps volume/A3 pressure/bar temperature/K",
    )
    print(f"{r['name']}: done in {walltime:.0f} s", flush=True)


# --------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------
def autocorrelation(x, maxlag, mean):
    """
    Normalized autocorrelation function of the fluctuations of a time series
    around a given (converged) mean, computed with FFT. Using the mean of a
    short trajectory instead biases the ACF towards a faster decay.
    """
    x = np.asarray(x) - mean
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:maxlag] / np.arange(n, n - maxlag, -1)
    return acf / acf[0]


def correlation_time(x, dt, mean, block=10.0):
    """
    Integrated autocorrelation time of the series x (sampled every dt), estimated
    from the fluctuations around ``mean`` of the averages over blocks of length
    ``block``, that should be much longer than the correlation time itself.
    """
    nb = int(round(block / dt))
    nblocks = len(x) // nb
    means = x[: nblocks * nb].reshape(nblocks, nb).mean(axis=1)
    return 0.5 * block * np.mean((means - mean) ** 2) / np.mean((x - mean) ** 2)


def find_output(name):
    """The longest trajectory for the run ``name`` in any of the scan folders"""
    best, best_size = None, 0
    for folder in SCAN_DIRS:
        fname = os.path.join(folder, name, "volume.dat")
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            if not f.readline().startswith("# walltime"):
                continue  # not the output of a completed run
        if os.path.getsize(fname) > best_size:
            best, best_size = fname, os.path.getsize(fname)
    return best


# The i-PI runs in the scan were made before i-PI redefined tau as the period
# of the volume oscillations, i.e. with W = 3 V B tau^2 rather than
# W = 9 V B (tau / 2 pi)^2. In the reference files they are labelled with the
# equivalent value of tau in the new convention, 2 pi / sqrt(3) ~ 3.63 times
# larger, rounded.
LEGACY_TAU = {10: 36, 30: 110, 100: 360, 300: 1100, 1000: 3600}


def reference_label(r):
    """Name of the setup of run ``r`` in the reference files, and its tau"""
    name = re.sub(r"_r\d+$", "", r["name"])  # replicas -> setup
    if r["code"] != "ipi":
        return name, r["tau_baro"]
    if "tau_period" in r:
        return name, r["tau_period"]
    tau = LEGACY_TAU[r["tau_baro"]]
    return name.replace(f"_b{r['tau_baro']}_", f"_b{tau}_"), tau


def analyze(equil=10.0, system="ice", maxlag=20.0, resolution=0.02):
    from ase.units import kB

    # load all the runs, grouping the independent replicas of the same setup
    groups = {}
    for r in run_list():
        if r.get("system", "ice") != system:
            continue
        fname = find_output(r["name"])
        if fname is None:
            continue
        with open(fname) as f:
            walltime = float(f.readline().split()[2])
        d = np.loadtxt(fname)
        d = d[d[:, 0] > equil]
        base, tau_baro = reference_label(r)
        groups.setdefault(base, (r, tau_baro, []))[2].append((d, walltime))

    # the converged mean volume, pooling all the i-PI trajectories (weighted by
    # their length: the estimates of the correlation time are too noisy to be
    # used here), so that a misbehaving barostat cannot bias the reference
    vols = np.concatenate(
        [
            d[:, 1]
            for r, _, runs in groups.values()
            if r["code"] == "ipi"
            for d, _ in runs
        ]
    )
    vmean = vols.mean()

    rows, acfs = [], {}
    for base, (r, tau_baro, runs) in groups.items():
        dt = runs[0][0][1, 0] - runs[0][0][0, 0]
        nlag = int(round(maxlag / dt)) + 1
        # ACF averaged over the replicas, always around the same converged mean
        # ACFs of runs that are too short are padded with NaN
        acf = np.full((len(runs), nlag), np.nan)
        for i, (d, _) in enumerate(runs):
            n = min(nlag, len(d) // 2)
            acf[i, :n] = autocorrelation(d[:, 1], n, vmean)
        acf = acf.mean(axis=0)
        acfs[base] = acf[:: int(round(resolution / dt))]
        vol = np.concatenate([d[:, 1] for d, _ in runs])
        dv2 = (vol - vmean) ** 2
        length = np.mean([d[-1, 0] - d[0, 0] for d, _ in runs])
        tau_v = np.mean([correlation_time(d[:, 1], dt, vmean) for d, _ in runs])
        tau_v2 = np.mean(
            [
                correlation_time((d[:, 1] - vmean) ** 2, dt, np.mean(dv2))
                for d, _ in runs
            ]
        )
        running = np.cumsum(acf) * dt - 0.5 * dt * (1 + acf)  # trapezoidal
        tau_5, tau_20 = running[int(round(5.0 / dt))], running[-1]
        # error on the mean, and correlation time, from the scatter of the replicas
        means = np.array([d[:, 1].mean() for d, _ in runs])
        if len(runs) > 1:
            err_v = means.std(ddof=1) / np.sqrt(len(runs))
            tau_rep = length * means.var(ddof=1) / (2 * np.mean(dv2))
        else:
            err_v = tau_rep = np.nan
        tau_atoms = r.get("atoms", (r["code"], r.get("tdamp", 0)))[1]
        rows.append(
            [base]
            + [
                tau_baro,
                r.get("tau_cell", 0),
                tau_atoms if isinstance(tau_atoms, (int, float)) else 0,
                np.mean(vol),
                np.mean(dv2),
                tau_v,
                tau_v2,
                length,
                np.mean([w for _, w in runs]),
                tau_5,
                tau_20,
                tau_rep,
                err_v,
                len(runs),
            ]
        )
        print(
            f"{base:28s} n={len(runs)} <V>={np.mean(vol):8.2f} "
            f"std(V)={np.sqrt(np.mean(dv2)):6.2f} tau_blk={tau_v:5.2f} "
            f"int(5ps)={tau_5:5.2f} int(20ps)={tau_20:5.2f} tau_rep={tau_rep:5.2f} ps"
        )

    v2 = np.mean((vols - vmean) ** 2)
    kappa = v2 / (kB * SYSTEMS[system]["temp"] * vmean) / 160.21766208  # A^3/eV->1/GPa
    print(f"pooled <V> = {vmean:.2f} A^3; kappa_T = {kappa:.4f} 1/GPa")

    with open(os.path.join(ROOT, "data", "reference_scan.dat"), "w") as f:
        f.write(f"# converged mean volume: {vmean:.2f} A3\n")
        f.write(
            "# name tau_baro/fs tau_cell/fs tau_atoms/fs <V>/A3 <dV^2>/A6 "
            "tau(V)/ps tau(dV^2)/ps length/ps walltime/s "
            "int_0^5ps(c_VV)/ps int_0^20ps(c_VV)/ps tau(V)_replicas/ps "
            "err(<V>)/A3 n_replicas\n"
            "# (length and walltime are per replica; tau(V) from 10 ps blocks)\n"
        )
        for row in rows:
            f.write(row[0] + " " + " ".join(f"{x:.6g}" for x in row[1:]) + "\n")
    names = [row[0] for row in rows]
    np.savetxt(
        os.path.join(ROOT, "data", "reference_vacf.dat.gz"),  # compressed
        np.vstack(
            [np.arange(len(acfs[names[0]])) * resolution] + [acfs[n] for n in names]
        ).T,
        fmt="%.4f",
        header="time/ps " + " ".join(names),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ase-worker":
        f, a, n, tb, td, system = sys.argv[2:8]
        ase_worker(f, a, int(n), float(tb), float(td), system)
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "analyze"])
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--time", type=float, default=200.0, help="ps per run")
    parser.add_argument("--only", default="", help="substring filter on run names")
    parser.add_argument(
        "--names", default="", help="comma-separated run names, run in this order"
    )
    parser.add_argument("--equil", type=float, default=10.0, help="ps to discard")
    parser.add_argument("--system", default="ice", choices=list(SYSTEMS))
    args = parser.parse_args()

    if args.action == "run":
        nsteps = int(args.time * 1000 / DT)
        runs = [r for r in run_list() if args.only in r["name"]]
        if args.names:
            by_name = {r["name"]: r for r in runs}
            runs = [by_name[n] for n in args.names.split(",")]
        with ThreadPoolExecutor(args.nproc) as pool:
            futures = [pool.submit(run_one, r, nsteps, i) for i, r in enumerate(runs)]
            for fut in futures:
                fut.result()
    else:
        analyze(args.equil, args.system)
