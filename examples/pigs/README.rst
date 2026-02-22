Te PIGS demonstration
====================

This recipe follows the ``i-PI`` `te-pigs demo <https://github.com/i-pi/i-pi/tree/main/demos/te-pigs>`_.

It reproduces the same six-step workflow:

1. generate reference PIMD data at elevated temperature;
2. curate centroid/physical/delta force training data;
3. train a Te-PIGS correction model with MACE;
4. run production MD with physical + Te-PIGS forces;
5. predict polarization and polarizability time series;
6. compute VDOS, IR, and Raman spectra.

To avoid repository bloat, the recipe downloads the upstream XML/scripts at runtime.

Local preview
-------------

If ``nox -e pigs`` fails with a conda writable-environment error, use a writable
environment path just for this run:

.. code-block:: bash

   CONDA_ENVS_PATH=/Users/venkatkapil24/scratch/codex/atomistic-cookbook-pigs/atomistic-cookbook/.nox/conda-envs nox -e pigs

Then serve the built page:

.. code-block:: bash

   python -m http.server 8000 --directory docs/build/html
