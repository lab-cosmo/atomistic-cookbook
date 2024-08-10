Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in the ways listed
below.

Requirements for new contributions
----------------------------------

All code included in this repository is executed in each pull request. This
ensures that the code in this repository stays executable for a longer time
frame. Because of that we do not want to have examples with heavy calculations
that take more than 30 to 1 min seconds to execute. If you feel unsure if a
contribution is suitable, feel free to contact one of the `support`_ person
beforehand.

Adding a new examples
---------------------

The examples in this repository are python files that we render for the website
using `sphinx-gallery`_. In short, these are python files containing comments
formatted as `RestructuredText`_, which are executed, and then the comments,
code and outputs (including plots, ``print`` outputs, etc.) are assembled in a
single HTML webpage.

To add a new example, you'll need to create a new folder in example (substitute
``<example-name>`` with the folder name in the instructions below), and add the
following files inside:

- ``README.rst``, can be empty or can contain a short description of the example;
- ``environment.yml``, a `conda`_ environment file containing the list of
  dependencies needed by your example;
- as many Python files as you want, each one will be converted to a separate
  HTML page.

.. _sphinx-gallery: https://sphinx-gallery.github.io/
.. _RestructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _conda: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually

Finally, you'll need to add your example to the list so it is automatically
build on CI. The list is in the ``.github/workflows/docs.yml`` file, near the
``example-name:`` section.

Converting a Jupyter notebook to a sphinx-gallery compatible Python script
--------------------------------------------------------------------------

Often it is more convenient to work in a Jupyter notebook and convert in later
to sphinx-gallery example. To convert your Jupyter notebook you can use the
`ipynb-to-gallery.py <ipynb_to_gallery.py>`_ file that is root folder of the
repository

.. code-block:: bash

    python ipynb-to-gallery.py <notebook.ipynb>

Running your example and visualizing the HTML
---------------------------------------------

We use `nox`_ as a task runner to run all examples and assemble the final
documentation. You can install it with ``pip install nox``.

To run your example and make sure it conforms to the expected code formatting,
you can use the following commands:

.. code-block:: bash

    # execute the example and render it to HTML
    nox -e <example-name>

    # check the code formatting
    nox -e lint

To visualize the generated cookbook open ``docs/build/html/index.html`` in a web
browser.

If there are formatting errors you can try to fix them automatically with:

.. code-block:: bash

    nox -e format

You can also build all examples (warning, this will take quite some time) with:

.. code-block:: bash

    nox -e docs

.. _nox: https://nox.thea.codes/

Known issues
------------

Chemiscope widgets are not currently integrated into our sphinx gallery.

Support
-------

If you still have problems adding your example to the repository, please feel
free to contact one of the people

`@agoscinski (Alexander Goscinski) <alexander.goscinski@epfl.ch>`_

`@davidetisi (Davide Tisi) <davide.tisi@epfl.ch>`_

Code of Conduct
---------------

Please note that this project is released with a 
`Contributor Code of Conduct <CONDUCT.md>`_. 
By contributing to this project you agree to abide by its terms.
