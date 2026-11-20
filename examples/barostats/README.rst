Constant-pressure molecular dynamics and barostats
==================================================

This example explains how to run constant-pressure (NPT) simulations,
focusing on how to choose the time scales of the barostat, of the
thermostat attached to the cell, and of the atomic thermostat to
obtain efficient sampling of the volume fluctuations.
It uses `i-PI <http://ipi-code.org>`_ with a `LAMMPS <http://lammps.org>`_
force driver, and compares the results with the barostats implemented in
`ASE <https://ase-lib.org>`_ and natively in LAMMPS.
