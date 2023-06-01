Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in the ways listed below.

Requirements for new contributions
----------------------------------

All code included in this repository is executed in each pull request.
This ensures that the code in this repository stays executable for a longer time frame.
Because of that we do not want to have examples with heavy calculations that take more
than 30 seconds to execute. If heavy calculations are needed, it might be a better option
to put your example in an external repository and link to it on the `Wiki page <https://github.com/lab-cosmo/software-cookbook/wiki>`_.
If you feel unsure if a contribution is suitable, feel free to contact one of the `support`_ before.

Adding an sphinx-gallery examples
---------------------------------

To visualize examples on our readthedocs page we use `sphinx-gallery`.
When building the doc the examples are run and compiled automatically into HTML files
and moved to the documentation folder `docs/src/`.
You will find all the examples Python scripts in the `examples/` folder of the repository. 
Each example is put into one of the example category folders, e.g. `examples/sample_selection`.
If you do not know where to put your example, just put in the `examples/uncategorized`
folder and when doing a pull request, we will figure out where to put it.

Converting a Jupyter notebook to a sphinx-gallery compatible Python script
--------------------------------------------------------------------------

Often it is more convenient to work in a Jupyter notebook and convert in later to 
sphinx-gallery example. To convert your Jupyter notebook you can just use the 
`ipynb_to_gallery.py` file that is root folder of the repository

.. code-block:: bash

    python ipynb_to_gallery.py <notebook.ipynb>

Building the cookbook locally
-----------------------------

When you add a new example, you can build the doc and check if your code runs with

.. code-block:: bash

    tox

To visualize the generated cookbook open in a browser the file
``<cookbook folder>/docs/build/html/index.html``.

When you generate the examples locally all the notebook will be automatically generated
in the folder ``<cookbook folder>/docs/src/examples/<name of the example>``

Known issues
------------

Sometimes the doc preview from readthedocs is not correcty rendered. If something works in your local build but not in the readthedocs PR preview. It could that the issue is fixed once you merge with the main branch.

Chemiscope widgets are not currently integrated into our sphinx gallery.

Support
-------

If you still have problems adding your example to the repository, please feel free to contact one of the people

`@agoscinski (Alexander Goscinski) <alexander.goscinski@epfl.ch>`_

`@davidetisi (Davide Tisi) <davide.tisi@epfl.ch>`_

Code of Conduct
---------------

Please note that the COSMO cookbook project is released with a `Contributor Code of Conduct <CONDUCT.md>`_. By contributing to this project you agree to abide by its terms.
