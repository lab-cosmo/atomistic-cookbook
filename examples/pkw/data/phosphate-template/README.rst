Template for the dissociation constants of phosphoric acid species
==================================================================

These are the inputs used in the paper for the acidic and basic dissociation
of ``NaH2PO4`` (one ``Na+`` and one ``H2PO4-`` in 128 water molecules, in a
rhombic-dodecahedral box with a 17.659 Å edge), with absolute paths replaced
by file names. The same protocol as for water applies; what changes is the
order parameter, the restraints that keep the solvent from autoionizing, and
a few metadynamics parameters that are listed below for all four species.
The other species only differ by the starting structure (replace water
hydrogens by ``Na+`` and remove protons from the phosphate), the atom index
lists in the PLUMED files, and the parameters of the table.

Files
-----

``cv_phosphate.py``
    Exports ``cv.pt``, the ``metatomic`` model computing the signed
    charge-separation coordinate ``r`` (hydronium at distance d from the P atom
    contributes +d, a hydroxide -d), the apparent charge ``q`` of the
    phosphate oxygens, and ``delta``, which counts hydronium-hydroxide pairs in
    the solvent. Run it once in the folder where the simulations are started.
``NaH2PO4.xyz``
    Equilibrated starting structure (392 atoms). The phosphate oxygen atoms are
    the first four O of the file and the P atom is the only P; the PLUMED
    index lists (``SPECIES1``: O, ``SPECIES2``: H, ``SPECIES3``: P,
    ``SPECIES4``: Na) must match the file.
``metad/``
    Well-tempered metadynamics along ``r`` (``input.xml`` for i-PI,
    ``plumed.dat``), to be run with ``i-pi input.xml``; needs ``model.pt``
    (the fine-tuned model, ``energy_variant:pbe0``) and ``cv.pt`` in the same
    folder.
``remd/``
    Fixed-bias replica exchange (8 replicas between 300 and 400 K, exchanges
    every 50 steps) with the ``HILLS`` file of the metadynamics run;
    ``input.xml`` starts from ``seeds/0.xyz`` and ``run.py -i <idx>`` selects
    another of the independent runs (``seeds/<idx>.xyz`` are 10 frames of
    the metadynamics trajectory chosen by farthest point sampling on
    ``(r, q)``; they are not included and must be generated).
``ti/``
    32-bead PIMD with scaled masses for the quantum correction; ``input.xml``
    is the associated state at the physical masses, and ``run.py --state
    <state> -y <y>`` selects a state (``associated``, ``acidic_disso`` or
    ``basic_disso``, with starting structures ``<state>.xyz`` to be taken from
    metadynamics frames with ``r`` close to 0, +10 or -10 Å) and scales the
    masses. The states are confined by one-sided walls on ``r``
    (``plumed-*.dat``).

Restraints on the solvent
-------------------------

For the two amphiprotic species the bias needed to dissociate the solute is
large enough to induce the autoionization of water. The PLUMED inputs
therefore include two one-sided linear restraints on ``delta`` (labelled
``ssc``): a weak one, 200 kJ/mol per unit beyond 0.15, that is always active,
and a strong one, 500 kJ/mol beyond 0.005, that is only switched on when the
charge of the phosphate ``q`` is within the window of its reference value
(``not_dissociated``), so that it does not interfere with the proton transfer
between the solute and the solvent. ``H3PO4`` and ``Na3PO4`` do not need them
(they are commented out in the corresponding production inputs), but
``delta`` should always be monitored.

Species-dependent parameters
----------------------------

====================  ========  ========  ========  ========
                      H3PO4     NaH2PO4   Na2HPO4   Na3PO4
====================  ========  ========  ========  ========
Na indices            --        11        11 14     11 14 17
hill height / kT      0.5       0.8       0.8       0.5
bias factor           5         8         8         10
metadynamics grid     -10..20   -20..20   -25..25   -25..25
walls on r / Å        ±14       ±14       ±14       ±14
``q`` window          [-0.05,   [-1.05,   [-2.05,   [-3.05,
                      0.02]     -0.98]    -1.98]    -2.98]
``delta`` restraints  no        yes       yes       no
TI states             A, acid   A, acid,  A, acid,  A, base
                                base      base
charges (acid/base)   +1·-1/--  +1·-2/    +1·-3/    --/
                                -1·0      -1·-1     -1·-2
====================  ========  ========  ========  ========

The last row gives the charges of the two products of each branch, which
enter the Coulomb reference used to align the profile: the periodic
point-charge free energy is computed with ``torch-pme`` for those charges in
the simulation cell (for a non-neutral pair the Wigner self-energy of the
neutralizing background, ``E_W (q1+q2)^2``, is subtracted), and the branch with
a neutral product (the basic dissociation of ``NaH2PO4``) is aligned to a flat
reference. Each branch is analyzed exactly like water, with ``r_c = 7`` Å, the
association integral over [-1, 7] Å (acidic) or [-7, 1] Å (basic), and the
matching window from ``r_c`` to 10 Å; the standard-state prefactor is
``1/c°`` (there is a single solute in the box). The TI states are confined by
walls at ±3 Å (associated, 5 kJ/mol/Å²) and at +7 Å or -7 Å (dissociated,
100 kJ/mol/Å²).
