Phonon dispersions with uncertainty quantification
===================================================

This recipe demonstrates phonon band structure calculations with
ensemble uncertainty quantification using PET-MAD and `uqphonon`.

Three systems are explored:

1. **Al (FCC)** — comparing phonons from constrained and unconstrained
   relaxations, showing that automatic q-paths can be misleading but
   the underlying physics is identical.
2. **BaTiO₃ (rhombohedral R3m)** — the ferroelectric phase is
   dynamically stable with all real phonon frequencies.
3. **BaTiO₃ (cubic Pm-3m)** — the paraelectric structure is
   dynamically unstable, with imaginary modes confirming the
   ferroelectric instability.
