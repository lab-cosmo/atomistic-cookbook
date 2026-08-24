Path integral approximations to real-time correlations
======================================================

This example compares the vibrational spectra that different path-integral
approximations to real-time quantum correlation functions -- classical MD,
RPMD, TRPMD, TRPMD-GLE and partially-adiabatic CMD -- predict for a Morse
oscillator parameterized to resemble an OH radical, and shows where each
approximation breaks down.

The i-PI inputs bundled in ``data/inputs`` use ``ffdirect`` together with a
custom Python PES, so the Morse model is evaluated inside i-PI without
launching external socket drivers.

It is based on the tutorial given at the PIMD schools.
