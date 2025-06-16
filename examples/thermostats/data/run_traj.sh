#!/bin/bash

# This is a simple script to run some reference
# NVE trajectories for the autocorrelation function,
# starting from samples collected from a NVT trajectory

# First, we launch the sampling trajectory. This 
# outputs checkpoint files every 2 ps.

i-pi data/input_cvv_sample.xml & sleep 4
lmp < data/in.lmp

# Then, we run an i-PI input that launches multiple
# NVE trajectories

i-pi data/input_cvv_traj.xml & sleep 4
for i in {1..8}; do lmp < data/in.lmp & done
wait

# Finally, run the i-PI post-processing on the 
# concatenated velocity trajectories. The block size
# is chosen so each trajectory is a separate block.

cat traj-*_traj.vel_0.xyz &> traj-all.xyz
i-pi-getacf -ifile traj-all.xyz -mlag 1000 -bsize 2001 -ftpad 2000 -ftwin cosine-blackman -dt "1 femtosecond" -oprefix traj-all
