Inputs of the production runs (fine-tuned PET-MAD model, 128 water molecules)
===========================================================================

These are the actual i-PI and PLUMED inputs used for the production runs, with
absolute paths replaced by file names. ``cv.pt`` is the collective-variable
model exported by the recipe, ``pet_sol-s-ft-best.pt`` the fine-tuned model
(``energy_variant:pbe0``), ``initial_structure.xyz`` the equilibrated box.

metadynamics/  well-tempered metadynamics, 10M steps (i-PI ``ffdirect`` on GPU)
remd/          fixed-bias replica exchange; ``run.py`` fills the template with the
               12-replica ladder and one of the 10 seeds (``seeds/<idx>.xyz``,
               selected from the metadynamics trajectory by farthest point sampling)
pimd/          32-bead PIMD with masses scaled by 1/y^2; ``run.py`` fills the
               template for a given state (associated/dissociated) and y

The GPU device settings (``device:cuda``) reflect the production hardware.
