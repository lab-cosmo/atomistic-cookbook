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

API reference
-------------

.. autofunction:: atomistic_cookbook_utils.download_with_retry

.. autofunction:: atomistic_cookbook_utils.run_command
