"""Launch i-PI + a force driver -- the mechanical plumbing behind the tutorial.

This module deliberately lives outside the notebook: *how* one drives i-PI from
Python (client/server sockets, subprocess handling, editing the XML input) is not
the subject of the bosons/fermions tutorial. The recipe only needs two calls:

* :func:`run_ipi` -- run one simulation from a template input in ``data/``,
  overriding a few knobs (temperature, beads, seed, length), and return the path
  to its ``data.out`` for analysis;
* :func:`run_parallel` -- run several independent simulations at once.

i-PI uses a client/server model: the ``i-pi`` server evolves the ring polymers
and a *driver* computes the forces over a socket. We use i-PI's built-in
``harmonic`` driver (pure-Python ``i-pi-py_driver``; the compiled ``i-pi-driver
-m harm3d`` is used automatically if it is on the ``PATH`` -- identical results).
"""

import atexit
import concurrent.futures
import os
import re
import shutil
import subprocess
import tempfile
import time

SPRING_CONSTANT = "1.21647924e-8"  # Ha/Bohr^2, matches hbar*omega0 = 3 meV

# temporary run directories, removed when the process exits (keeps callers clean)
_TMPDIRS = []


@atexit.register
def _cleanup_tmpdirs():
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def driver_command(address):
    """Return the force-driver command, preferring the compiled f90 driver."""
    if shutil.which("i-pi-driver") is not None:  # compiled Fortran driver
        return [
            "i-pi-driver",
            "-m",
            "harm3d",
            "-o",
            SPRING_CONSTANT,
            "-u",
            "-a",
            address,
        ]
    return [
        "i-pi-py_driver",
        "-m",
        "harmonic",
        "-o",
        SPRING_CONSTANT,
        "-u",
        "-a",
        address,
    ]


def _prepare_input(
    src_xml, address, temp=None, nbeads=None, seed=None, total_steps=None
):
    """Read a template input from ``data/`` and apply the requested overrides."""
    text = open(f"data/{src_xml}").read()
    text = re.sub(
        r"<address>.*?</address>", f"<address>{address}</address>", text, flags=re.S
    )
    if temp is not None:
        # update both the ensemble temperature and the thermal velocity init
        text = re.sub(
            r"(<temperature[^>]*>)\s*[\d.eE+-]+\s*(</temperature>)",
            rf"\g<1> {temp} \g<2>",
            text,
        )
        text = re.sub(
            r"(<velocities[^>]*>)\s*[\d.eE+-]+\s*(</velocities>)",
            rf"\g<1> {temp} \g<2>",
            text,
        )
    if nbeads is not None:
        text = re.sub(r'nbeads="\d+"', f'nbeads="{nbeads}"', text)
    if seed is not None:
        text = re.sub(r"<seed>.*?</seed>", f"<seed> {seed} </seed>", text, flags=re.S)
    if total_steps is not None:
        text = re.sub(
            r"<total_steps>.*?</total_steps>",
            f"<total_steps> {total_steps} </total_steps>",
            text,
            flags=re.S,
        )
    return text


def run_ipi(src_xml, address="tutorial", **overrides):
    """Run one i-PI + driver simulation and return the path to its ``data.out``.

    ``src_xml`` is a template file name in ``data/`` (e.g. ``"input_3bosons.xml"``).
    ``overrides`` may set ``temp`` (K), ``nbeads``, ``seed`` or ``total_steps``;
    anything left out keeps the template value. Each run uses its own temporary
    directory (removed at process exit), so many can run concurrently.
    """
    tmp = tempfile.mkdtemp(prefix="ipi_run_")
    _TMPDIRS.append(tmp)
    with open(os.path.join(tmp, "input.xml"), "w") as fh:
        fh.write(_prepare_input(src_xml, address, **overrides))

    sock = f"/tmp/ipi_{address}"
    if os.path.exists(sock):
        os.remove(sock)
    # cap BLAS/OpenMP to one thread per child: we parallelise over runs ourselves,
    # so a multithreaded numpy/BLAS would oversubscribe the CPU
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    ipi = subprocess.Popen(
        ["i-pi", "input.xml"],
        cwd=tmp,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(120):  # wait for the socket before launching the driver
        if os.path.exists(sock) or ipi.poll() is not None:
            break
        time.sleep(0.5)
    drv = subprocess.Popen(
        driver_command(address),
        cwd=tmp,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ipi.wait()
    drv.wait()
    return os.path.join(tmp, "data.out")


def run_parallel(jobs, max_workers=None):
    """Run several simulations concurrently and return their ``data.out`` paths.

    ``jobs`` is a list of ``(src_xml, address, overrides)`` tuples. By default we
    use half the available cores (each job is itself an i-pi + driver pair).
    """
    if max_workers is None:
        cap = os.environ.get("PIMD_MAX_WORKERS")
        cores = int(cap) if cap else max(1, (os.cpu_count() or 2) // 2)
        max_workers = min(len(jobs), cores)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda j: run_ipi(j[0], j[1], **j[2]), jobs))
