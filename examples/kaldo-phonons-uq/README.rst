Phonon dispersions with committee uncertainty using kALDo
=========================================================

This recipe computes a phonon dispersion with an uncertainty estimate directly
in `kALDo <https://nanotheorygroup.github.io/kaldo/>`_, using its
``PhononsEnsemble`` API with the `UPET <https://github.com/lab-cosmo/pet>`_
universal machine-learning potential (PET-MAD) as the force engine.  kALDo
builds the force constants and phonon spectrum for each committee member and
aggregates them into a mean band structure with a per-branch standard-deviation
band, so no additional phonon backend is needed.  It is a kALDo-native
companion to the `phonon dispersions with uncertainty recipe
<https://atomistic-cookbook.org/examples/pet-phonons/pet-phonons.html>`_ by
Paolo Pegolo and Michele Ceriotti, which uses
`uqphonon <https://github.com/ppegolo/uqphonon>`_.
