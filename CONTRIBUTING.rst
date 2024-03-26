Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and
credit will always be given. You can contribute in the ways listed below.

Requirements for new contributions
----------------------------------

All code included in this repository is executed in each pull request. This ensures that
the code in this repository stays executable for a longer time frame. Because of that we
do not want to have examples with heavy calculations that take more than 30 seconds to
execute. If heavy calculations are needed, it might be a better option to put your
example in an external repository and link to it on the `Wiki page
<https://github.com/lab-cosmo/software-cookbook/wiki>`_. If you feel unsure if a
contribution is suitable, feel free to contact one of the `support`_ before.

Adding a new examples
---------------------

To visualize examples on our readthedocs page we use `sphinx-gallery`. When building the
doc the examples are run and compiled automatically into HTML files and moved to the
documentation folder `docs/src <docs/src>`_. You will find all the examples Python
scripts in the `examples/` folder of the repository. Each example is put into one of the
example category folders, e.g. `examples/sample_selection <examples/sample_selection>`_.
If you do not know where to put your example, just put in the `examples/uncategorized
<examples/uncategorized>`_ folder and when doing a pull request, we will figure out
where to put it.

After adding a file, you'll need to update ``tox.ini`` to build your example when
building the documentation. Look how it's done for the ``lode_linear`` example, and
do the same for yours!

Converting a Jupyter notebook to a sphinx-gallery compatible Python script
--------------------------------------------------------------------------

Often it is more convenient to work in a Jupyter notebook and convert in later to
sphinx-gallery example. To convert your Jupyter notebook you can just use the
`ipynb-to-gallery.py <ipynb_to_gallery.py>`_ file that is root folder of the repository

.. code-block:: bash

    python ipynb-to-gallery.py <notebook.ipynb>

Building the cookbook locally
-----------------------------

When you add a new example, you can run the linter (code format checker) and build the
doc to check if your code runs with

.. code-block:: bash

    tox

If there are formatting errors appearing you can format your file automatically with

.. code-block:: bash

    tox -e format

That should fix most of the formatting issues automatically. If there are still
formatting issues remaining, then the reviewer of your pull request can fix them.
To visualize the generated cookbook open in a browser the file
``docs/build/html/index.html``.

Known issues
------------

Sometimes the doc preview from readthedocs is not rendered correctly. If something works
in your local build but not in the readthedocs PR preview. It could that the issue is
fixed once you merge with the main branch.

Chemiscope widgets are not currently integrated into our sphinx gallery.

Support
-------

If you still have problems adding your example to the repository, please feel free to
contact one of the people

`@agoscinski (Alexander Goscinski) <alexander.goscinski@epfl.ch>`_

`@davidetisi (Davide Tisi) <davide.tisi@epfl.ch>`_

Code of Conduct
---------------

Please note that the COSMO cookbook project is released with a `Contributor Code of
Conduct <CONDUCT.md>`_. By contributing to this project you agree to abide by its terms.
