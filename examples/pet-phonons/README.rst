Phonon uncertainty quantification with MLIP ensembles
=====================================================

This recipe demonstrates phonon uncertainty quantification for BeO in the
wurtzite structure using the PET-MAD-XS model.

The workflow uses the `uqphonon <https://github.com/ppegolo/uqphonon>`__
package to:

1. Relax the structure with a high-quality ML potential
2. Compute force constants from an MLIP ensemble via ``i-PI``
3. Evaluate phonon band structures with epistemic uncertainty
4. Visualize mean frequencies and ±1σ uncertainty bands

This approach directly propagates model uncertainty into phonon predictions,
providing confidence estimates for dynamical properties.
