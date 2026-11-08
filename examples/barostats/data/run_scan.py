"""
Runs the long simulations that are used as a reference in the barostats
recipe, and post-processes them into small text files that are stored in
``data/``. This is *not* executed when building the recipe: each of the 16
setups is run 4 times (with different random seeds) for 250 ps, and the setups
compared in the last figure of the recipe 8 times. This takes about 110 CPU
hours, i.e. less than a day running 6 simulations at a time.

Usage (from the recipe folder, with ``i-pi`` and ``lmp`` in the ``PATH``)::

    python data/run_scan.py run --nproc 3
    python data/run_scan.py run --nproc 3 --only lmp_mtk
    python data/run_scan.py analyze

Raw outputs are written to ``scan/<setup>_r<k>/`` (use ``--scan`` to change the
folder). Runs whose ``volume.dat`` already exists are skipped, so an interrupted
``run`` can simply be restarted.
"""

import argparse
import glob
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
STRUCTURE = "ice_96"
TEMP = 270.0  # K
PRESS = 1.0  # bar
BULK_MODULUS = 9.0  # GPa, estimate that sets the mass of the i-PI piston
DT = 0.5  # fs
STRIDE = 4  # output stride (steps), i.e. 2 fs
N_REPLICAS = 4
# more replicas for the setups compared in the last figure of the recipe
N_REPLICAS_BEST = 8
BEST = [
    "ice_ipi_b360_c30_svr10",
    "ice_ipi_b360_c30_glesmart",
    "ice_ipi_b110_c9_glesmart",
    "ice_ipi_b360_c30_langevin10",
    "ice_ase_mtk_b200",
    "ice_lmp_mtk_b200",
]
RUN_TIME = 250.0  # ps per run
# avoid oversubscription when running many simulations in parallel
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


# --------------------------------------------------------------------------------
# list of setups
# --------------------------------------------------------------------------------
# i-PI pistons: tau is the period of the volume oscillations, and
# tau_cell ~ tau / 4 pi gives critical damping (both in fs)
PISTONS = [(36, 3), (110, 9), (360, 30), (1100, 90), (3600, 300)]


def ipi_setup(tau, tau_cell, thermostat):
    return dict(
        name=f"ice_ipi_b{tau}_c{tau_cell:g}_{thermostat[0]}{thermostat[1]}",
        code="ipi",
        tau=tau,
        tau_cell=tau_cell,
        thermostat=thermostat,
    )


def mtk_setup(code, pdamp, tdamp=100):
    return dict(name=f"ice_{code}_b{pdamp}", code=code, pdamp=pdamp, tdamp=tdamp)


def setups():
    """The setups used in the recipe"""
    result = []
    # critically damped pistons, with a global SVR thermostat or with an
    # optimal-sampling GLE on the atoms
    for thermostat in [("svr", 10), ("gle", "smart")]:
        for tau, tau_cell in PISTONS:
            result.append(ipi_setup(tau, tau_cell, thermostat))
    # a strong local Langevin thermostat on the atoms
    for tau, tau_cell in PISTONS[1:3]:
        result.append(ipi_setup(tau, tau_cell, ("langevin", 10)))
    # an under-damped piston
    result.append(ipi_setup(360, 1000, ("svr", 10)))
    # the MTK integrators of ASE and LAMMPS (pdamp, tdamp in fs)
    result.append(mtk_setup("ase_mtk", 200))
    result.append(mtk_setup("ase_mtk", 1000))
    result.append(mtk_setup("lmp_mtk", 200))
    for s in result:
        s["replicas"] = N_REPLICAS_BEST if s["name"] in BEST else N_REPLICAS
    return result


def run_list(n_replicas=None):
    """
    Independent runs of all the setups, one replica of each setup at a time.
    ``n_replicas``, if given, replaces the number of replicas of every setup.
    """
    runs = setups()
    if n_replicas is not None:
        for s in runs:
            s["replicas"] = n_replicas
    return [
        {**s, "name": f"{s['name']}_r{k}"}
        for k in range(max(s["replicas"] for s in runs))
        for s in runs
        if k < s["replicas"]
    ]


def seed(name):
    return zlib.crc32(name.encode()) % 100000 + 1


# --------------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------------
def wait_socket(address, proc, timeout=60):
    t0 = time.time()
    while not os.path.exists(f"/tmp/ipi_{address}"):
        if proc.poll() is not None or time.time() - t0 > timeout:
            raise RuntimeError(f"server for {address} did not start")
        time.sleep(0.2)


def lmp_client(address, folder):
    return subprocess.Popen(
        ["lmp", "-in", "data/in.lmp", "-var", "address", address]
        + ["-var", "data", f"data/{STRUCTURE}.lmp"]
        + ["-log", "none", "-screen", os.path.join(folder, "lmp.screen")],
        cwd=ROOT,
    )


def run_ipi(r, nsteps, folder, address):
    xml = ET.parse(os.path.join(ROOT, "data", "input_npt.xml"))
    root = xml.getroot()
    root.find("output").set("prefix", os.path.join(folder, "sim"))
    root.find("output/trajectory").set("stride", "10000")
    root.find("total_steps").text = str(nsteps)
    root.find("prng/seed").text = str(seed(r["name"]))
    root.find("ffsocket/address").text = address
    root.find("system/initialize/file").text = os.path.join(
        ROOT, "data", STRUCTURE + ".extxyz"
    )
    root.find("system/initialize/velocities").text = str(TEMP)
    root.find("system/ensemble/temperature").text = str(TEMP)
    root.find("system/ensemble/pressure").text = str(PRESS)
    dyn = root.find("system/motion/dynamics")
    th = dyn.find("thermostat")
    th.set("mode", r["thermostat"][0])
    if r["thermostat"][0] == "gle":
        # optimal-sampling GLE (the second element of r["thermostat"] is a label)
        th.remove(th.find("tau"))
        A = np.loadtxt(os.path.join(ROOT, "data", "gle_smart.A"))
        ET.SubElement(th, "A", shape=str(A.shape)).text = (
            "[" + ", ".join(f"{a:.12e}" for a in A.flatten()) + "]"
        )
    else:
        th.find("tau").text = str(r["thermostat"][1])
    baro = dyn.find("barostat")
    baro.find("tau").text = str(r["tau"])
    baro.find("bulk_modulus").text = str(BULK_MODULUS)
    baro.find("thermostat/tau").text = str(r["tau_cell"])
    xml.write(os.path.join(folder, "input.xml"))

    ipi = subprocess.Popen(
        ["i-pi", os.path.join(folder, "input.xml")],
        cwd=folder,
        stdout=open(os.path.join(folder, "ipi.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    wait_socket(address, ipi)
    lmp = lmp_client(address, folder)
    ipi.wait()
    lmp.wait()
    out = np.loadtxt(os.path.join(folder, "sim.out"))
    # time (ps), volume (A^3), pressure (bar), temperature (K)
    return np.vstack([out[:, 1], out[:, 6], out[:, 5], out[:, 3]]).T


LMP_HEADER = """
units           real
atom_style      full
pair_style      lj/cut/tip4p/long 1 2 1 1 0.14715 9.0
# quartic O-H bond, tabulated (equivalent to bond_style class2)
bond_style      table spline 1000
angle_style     harmonic
kspace_style    pppm/tip4p 1.0e-5
read_data       data/{structure}.lmp
pair_coeff      * * 0.0 0.0
pair_coeff      1 1 0.18521 3.1589
bond_coeff      1 data/qtip4pf_bond.table QTIP4PF_OH
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
    pa = PRESS / 1.01325  # bar -> atm
    inp = (
        LMP_HEADER.format(dt=DT, T=TEMP, seed=seed(r["name"]), structure=STRUCTURE)
        + f"fix 1 all npt temp {TEMP} {TEMP} {r['tdamp']} iso {pa} {pa} {r['pdamp']}\n"
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
        + [str(nsteps), str(r["pdamp"]), str(r["tdamp"])],
        cwd=ROOT,
        check=True,
    )
    return np.loadtxt(os.path.join(folder, "sim.out"))


def ase_worker(folder, address, nsteps, pdamp, tdamp):
    import ase.units
    import ipi
    from ase.calculators.socketio import SocketIOCalculator
    from ase.md.nose_hoover_chain import IsotropicMTKNPT
    from ase.md.velocitydistribution import thermalize_momenta

    atoms = ipi.read_trajectory(os.path.join(ROOT, "data", STRUCTURE + ".extxyz"))[0]
    rng = np.random.default_rng(seed(os.path.basename(folder)))
    thermalize_momenta(atoms, temperature_K=TEMP, rng=rng)

    def launch(atoms, properties, port, unixsocket):
        return lmp_client(unixsocket, folder)

    calc = SocketIOCalculator(unixsocket=address, launch_client=launch)
    atoms.calc = calc
    dyn = IsotropicMTKNPT(
        atoms,
        timestep=DT * ase.units.fs,
        temperature_K=TEMP,
        pressure_au=PRESS * ase.units.bar,
        tdamp=tdamp * ase.units.fs,
        pdamp=pdamp * ase.units.fs,
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


def run_one(r, nsteps, scan, address):
    folder = os.path.join(scan, r["name"])
    if os.path.exists(os.path.join(folder, "volume.dat")):
        return
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    try:
        data = RUNNERS[r["code"]](r, nsteps, folder, address)
    except Exception as e:
        print(f"{r['name']}: FAILED ({e})", flush=True)
        return
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


def load_runs(scan, name, equil):
    """The completed runs of the setup ``name``, without the first ``equil`` ps"""
    runs = []
    for fname in sorted(glob.glob(os.path.join(scan, f"{name}_r*", "volume.dat"))):
        with open(fname) as f:
            walltime = float(f.readline().split()[2])
        d = np.loadtxt(fname)
        runs.append((d[d[:, 0] > equil], walltime))
    return runs


def analyze(scan, equil=10.0, maxlag=20.0, resolution=0.02):
    from ase.units import kB

    groups = [(s, load_runs(scan, s["name"], equil)) for s in setups()]
    groups = [(s, runs) for s, runs in groups if runs]

    # the converged mean volume, pooling all the i-PI trajectories (weighted by
    # their length: the estimates of the correlation time are too noisy to be
    # used here), so that a misbehaving barostat cannot bias the reference
    vols = np.concatenate(
        [d[:, 1] for s, runs in groups if s["code"] == "ipi" for d, _ in runs]
    )
    vmean = vols.mean()

    rows, acfs = [], {}
    for s, runs in groups:
        dt = runs[0][0][1, 0] - runs[0][0][0, 0]
        nlag = int(round(maxlag / dt)) + 1
        # ACF averaged over the replicas, always around the same converged mean
        acf = np.mean([autocorrelation(d[:, 1], nlag, vmean) for d, _ in runs], axis=0)
        acfs[s["name"]] = acf[:: int(round(resolution / dt))]
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
        if s["code"] == "ipi":
            tau_baro, tau_cell = s["tau"], s["tau_cell"]
            tau_atoms = s["thermostat"][1]
            tau_atoms = tau_atoms if isinstance(tau_atoms, (int, float)) else 0
        else:
            tau_baro, tau_cell, tau_atoms = s["pdamp"], 0, s["tdamp"]
        rows.append(
            [s["name"]]
            + [
                tau_baro,
                tau_cell,
                tau_atoms,
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
            f"{s['name']:28s} n={len(runs)} <V>={np.mean(vol):8.2f}+-{err_v:5.2f} "
            f"std(V)={np.sqrt(np.mean(dv2)):6.2f} tau_blk={tau_v:5.2f} "
            f"int(5ps)={tau_5:5.2f} int(20ps)={tau_20:5.2f} tau_rep={tau_rep:5.2f} ps"
        )

    v2 = np.mean((vols - vmean) ** 2)
    kappa = v2 / (kB * TEMP * vmean) / 160.21766208  # A^3/eV -> 1/GPa
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
        folder, address, nsteps, pdamp, tdamp = sys.argv[2:7]
        ase_worker(folder, address, int(nsteps), float(pdamp), float(tdamp))
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "analyze"])
    parser.add_argument("--scan", default=os.path.join(ROOT, "scan"))
    parser.add_argument("--nproc", type=int, default=3)
    parser.add_argument("--time", type=float, default=RUN_TIME, help="ps per run")
    parser.add_argument(
        "--replicas", type=int, default=None, help="replicas of every setup"
    )
    parser.add_argument("--only", default="", help="substring filter on run names")
    parser.add_argument("--equil", type=float, default=10.0, help="ps to discard")
    args = parser.parse_args()
    scan = os.path.realpath(args.scan)

    if args.action == "run":
        nsteps = int(round(args.time * 1000 / DT))
        runs = [r for r in run_list(args.replicas) if args.only in r["name"]]
        with ThreadPoolExecutor(args.nproc) as pool:
            for r in runs:
                # a socket name per run, so that several runners can share a machine
                pool.submit(run_one, r, nsteps, scan, f"baroref_{r['name']}")
    else:
        analyze(scan, args.equil)
