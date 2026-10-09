Inputs of the production runs (fine-tuned PET-MAD model, 128 water molecules)
===========================================================================

These are the i-PI and PLUMED inputs used for the production runs of the
paper, with absolute paths replaced by file names, and complete with the
settings of the paper: each stage runs with ``i-pi input.xml`` from its folder,
provided that the folder also contains ``model.pt`` (the fine-tuned model,
PET-MAD-1.5-s fine-tuned to revPBE0-D3, shipped in the data bundle as
``production/model.pt``; it is selected with ``energy_variant:pbe0`` because
the model also carries the energy head of the base model) and ``cv.pt`` with
its ``extensions/`` folder (the collective-variable model exported by the
recipe). All runs: 300 K, 1 bar, 0.5 fs, stochastic velocity rescaling
thermostat (20 fs), isotropic barostat (200 fs).

metadynamics/  well-tempered metadynamics (hills of 1 kT every 0.5 ps, width
               0.3 A, bias factor 12, walls at -2 and 12 A) from the
               equilibrated 15.6482 A box ``initial_structure.xyz``; ~2.7 ns
remd/          fixed-bias replica exchange, 8 replicas between 300 and 400 K,
               exchanges every 50 steps, reading the ``HILLS`` file of the
               metadynamics run; ``input.xml`` starts from ``seeds/0.xyz`` and
               ``run.py -i <idx>`` selects one of the 10 seeds (frames of the
               metadynamics trajectory); 10 independent runs of ~435 ps
pimd/          32-bead PIMD (PILE-L thermostat); ``input.xml`` is the
               associated state at the physical masses, ``run.py --state
               <state> -y <y>`` selects the state and scales the masses by
               1/y^2; at least 170 ps per run. The walls that confine the two
               states follow the Supporting Information of the paper (KAPPA=5
               in PLUMED, whose walls are KAPPA*(s-AT)^2, at +-3 A for the
               associated and 5 A for the dissociated state).

Software: i-PI 3.3.0, PLUMED 2.10, metatomic-torch 0.1.16, torch 2.12. The
GPU device settings (``device:cuda``) reflect the production hardware.
