Shared utilities
================

The `atomistic-cookbook-utils
<https://pypi.org/project/atomistic-cookbook-utils/>`_ package provides
small helpers used across many recipes, so that recipe code can stay
focused on the science. Add it to your recipe's ``environment.yml``
under ``pip:`` (e.g. ``atomistic-cookbook-utils >=0.1,<0.2``) and
import the helpers you need.

The package source lives at ``src/atomistic-cookbook-utils/`` in this
repository.

On CI, the ``ATOMISTIC_COOKBOOK_CACHE`` environment variable makes
:func:`~atomistic_cookbook_utils.download_with_retry` go through a
download cache shared by all recipes, so that each file is only fetched
once. When running a recipe locally the variable is not set, and files are
downloaded as usual.

API reference
-------------

.. autofunction:: atomistic_cookbook_utils.download_with_retry

.. autofunction:: atomistic_cookbook_utils.run_command
