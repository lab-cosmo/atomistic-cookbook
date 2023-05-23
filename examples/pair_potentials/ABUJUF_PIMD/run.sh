#!/bin/bash
i-pi i-pi_input.xml > log.ipi & # Read i-PI input, output log to "log.ipi"
sleep 5 # Sleep for 5 secs to allow i-PI to start up

lmp_mpi -in lammps_md.in > log.lammps & # Start the LAMMPS driver, output log to "log.lammps"

i-pi-py_driver -u -a gap -m rascal -o ../GAP_model.json,struct.extxyz > log.gap & # Start the Librascal driver, output log to "log.gap"

echo "Simulating..."
wait # Wait until the simulation has finished
echo "Done."

exit 0
