Thermal conductivity from the Boltzmann transport equation
==========================================================

This recipe demonstrates how to compute lattice thermal conductivity
by solving the phonon Boltzmann transport equation (BTE) with
`kALDo <https://nanotheorygroup.github.io/kaldo/>`_ and the
`UPET <https://github.com/lab-cosmo/pet>`_ universal machine-learning
potential (PET-MAD) as the force engine.  Starting from a relaxed
structure, we compute second- and third-order force constants, phonon
dispersions, scattering rates, and the full thermal-conductivity
tensor of silicon (diamond) at 300 K.
