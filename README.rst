COSMO Software Cookbook
=======================

`COSMO Software Cookbook <http://software-cookbook.rtfd.io/>`_

.. marker-intro

The COSMO cookbook contains recipies for atomistic machine learning and modelling

.. marker-building

Building the cookbook locally
-----------------------------

When you add a new example, you can build the doc and check if your code runs with

.. code-block:: bash

    tox -e docs

To visualize the generated cookbook open in a browser the file 
``<cookbook folder>/docs/build/html/index.html``.

When you generate the examples locally all the notebook will be automatically generated
in the folder ``<cookbook folder>/docs/src/examples/<name of the example>``

Known issues
------------

Sometimes the doc preview from readthedocs is not correcty rendered. If something works in your local build but not in the readthedocs PR preview. It could that the issue is fixed once you merge with the main branch.

Chemiscope widgets are not currently integrated into our sphinx gallery.

Contributors
------------

We welcome and recognize all contributions. You can see a list of current contributors in the `contributors tab <https://github.com/lab-cosmo/software-cookbook/graphs/contributors>`_.
