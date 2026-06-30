ML-MM Simulations with GROMACS and Metatomic
============================================

Simulate alanine dipeptide in water using a machine learning potential for the
solute (ML-MM simulation) with GROMACS and the Metatomic interface.

GROMACS Metatomic client and stack pins
---------------------------------------

``environment.yml`` installs **published** ``metatensor::gromacs-metatomic``
from the metatensor channel so cookbook CI can resolve a binary without
building GROMACS. That channel package is still linked against an older
``libmetatomic-torch`` (``<0.1.12``), so the Python pins keep
``metatrain`` on the 2026.2 line and ``metatomic-torch`` in
``[0.1.11, 0.1.12)``.

The **source fix** for current Metatomic ``ModelOutput`` granularity
(``sample_kind == "atom"`` / ``set_sample_kind``, plus CMake floors for
metatomic-torch ≥0.1.15) lives in
`metatensor/gromacs#13 <https://github.com/metatensor/gromacs/pull/13>`_
(branch ``HaoZeke/update/metatomic-latest-stack``). After that PR is
merged and the metatensor **gromacs-metatomic feedstock** is rebuilt
against it, raise the pip pins to metatrain ≥2026.3 and
metatomic-torch ≥0.1.15.

**PR-resolvable client (local / custom CI):** build and install GROMACS
from the #13 head with the HaoZeke/pixi_envs metatomic-cpu workspace
(``pixi run -e metatomic-cpu gromk-mpi Release`` or ``gromk-tmpi``), put
that ``gmx`` / ``gmx_mpi`` first on ``PATH``, and use a Python stack with
``metatomic-torch>=0.1.15,<0.2`` (and a matching metatrain). Do **not**
mix that client with the older channel binary in the same process.

Interim CI path (this recipe as shipped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``environment.yml`` unchanged: channel ``gromacs-metatomic`` + ABI pins
above. That exercises the published ML/MM recipe, not the #13 client.

Fixed-client path (after #13 / feedstock)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Checkout ``https://github.com/HaoZeke/gromacs.git`` branch
   ``update/metatomic-latest-stack`` (or the merged ``metatomic`` branch
   once #13 lands).
2. From ``HaoZeke/pixi_envs`` ``orgs/metatensor/gromacs``, run
   ``pixi run -e metatomic-cpu gromk-mpi Release`` (MPI recipe) so
   ``gmx_mpi`` installs into the pixi env.
3. Prefer that ``gmx_mpi`` on ``PATH`` over the conda package.
4. Install Python deps with ``metatomic-torch>=0.1.15,<0.2`` and
   ``metatrain>=2026.3`` (drop the interim upper bounds in
   ``environment.yml``).
