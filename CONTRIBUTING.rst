Contributing
============

.. marker-contrib-start

Contributions are welcome, and they are greatly appreciated, even it they 
are just reporting bugs in one of the existing recipes. Please use github
to report errors and submit contributions, as this helps with making your 
work visible and giving you credit. 
If you intend to submit a new recipe, please read
the guidelines below to ensure that your contributions
fit within the scope and the general style of the cookbook.

Requirements for new contributions
----------------------------------

All code included in the `cookbook repository
<https://github.com/lab-cosmo/atomistic-cookbook>`_
is executed in each pull request. This
ensures that the code in remains compatible and executable for a longer time
frame. Because of that we do not want to have examples with heavy calculations
that require more than a couple of minutes to execute.
If you are unsure whether a contribution is suitable, or if you want to
discuss how to structure your recipe, feel free to
`open an issue <https://github.com/lab-cosmo/atomistic-cookbook/issues>`_.

Adding a new recipe
-------------------

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

Keep in mind that sphinx-gallery will make it easy to download a Python file and
the notebook generated from it, but it won't give direct access to additional
files. If your example needs such data files, there are a few options available:

- (preferred) have the data file stored in a publicly accessible location, e.g.
  a Zotero record, and download the data file from the script
- if the data files are small (few 10s of Kb) you may also include them in a
  ``data/`` folder within the example folder. A zip file will be generated that
  can be downloaded from the example page.

Each new recipe will be automatically added to the list of new recipes, but 
you should also add it to the relevant software and topical sections. To do 
so, edit the appropriate ``.sec`` file within ``docs/src/software`` and 
``docs/src/topics``. 

.. _sphinx-gallery: https://sphinx-gallery.github.io/
.. _RestructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _conda: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually


Converting a Jupyter notebook to a sphinx-gallery compatible Python script
--------------------------------------------------------------------------

It is often more convenient to work in a Jupyter notebook and convert it
to sphinx-gallery example. To convert your Jupyter notebook you can use the
`ipynb-to-gallery.py <ipynb_to_gallery.py>`_ script, in the ``src`` folder of
the repository

.. code-block:: bash

    python src/ipynb-to-gallery.py <notebook.ipynb>

Running your example and visualizing the HTML
---------------------------------------------

We use `nox`_ as a task runner to run all examples and assemble the final
documentation. You can install it with ``pip install nox``.

To run your example and make sure it conforms to the expected code formatting,
you can use the following commands:

.. code-block:: bash

    # execute the example and render it to HTML
    nox -e <example-name>

    # build web pages for the examples that have been already run
    nox -e build_docs

To visualize the generated cookbook open ``docs/build/html/index.html``
in a web browser. If there are dynamical elements that have to be loaded,
it might be better to serve the website using a HTTP server, e.g.
running

.. code-block:: bash

   python -m http.server PORT_NUMBER

from within the ``docs/build/html/`` folder. The website will be served
on ``localhost:PORT_NUMBER``.

Before committing your recipe, you should check that it complies
with the coding style, which you can check automatically using

.. code-block:: bash

    nox -e lint

Most (but not all) formatting errors can be fixed automatically with:

.. code-block:: bash

    nox -e format

You can also build all examples (warning, this will take quite some time) with:

.. code-block:: bash

    nox -e docs

.. _nox: https://nox.thea.codes/

Chemiscope widgets
------------------

If you want to visualize one or more structures, or an interactive
plot, in your example, you can use a `chemiscope <http://chemiscope.org>`_
widget. To get some ideas on how the widgets can be used to better
explain the recipes, you can start looking at the
:ref:`examples from the cookbook <chemiscope>`.

.. marker-contrib-end

Support
-------

If you still have problems adding your example to the repository, please feel
free to contact one of the developers, e.g.

`@davidetisi (Davide Tisi) <davide.tisi@epfl.ch>`_

Code of Conduct
---------------

Please note that this project is released with a
`Contributor Code of Conduct <CONDUCT.md>`_.
By contributing to this project you agree to abide by its terms.
