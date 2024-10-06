The Atomistic Cookbook
======================

.. include:: ../../README.rst
   :start-after: marker-intro-start
   :end-before: marker-intro-end

Downloading and running the recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each recipe can be viewed online as an interactive HTML page, but 
can also be downloaded as a stand-alone ``.py`` script of 
``.ipynb`` Jupyter notebook. 
To simplify setting up an environment that contains all the dependencies
needed for each recipe, you can also download an ``environment.yml`` file 
that you can use with conda to create a custom environment to run the example.

.. code-block:: bash

   # Pick a name for the environment and replace <environment-name> with it
   conda env create --name <environment-name> --file environment.yml

   # when you want to use the environment
   conda env activate --name <environment-name>

Additional data needed for each example is usually either downloaded
dynamically, or can be found in a ``data`` folder for each example,
or downloaded as a ``data.zip`` file at the end of each recipe in
the website.

Table of contents
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   topics/index
   software/index
   all-examples
   contributing
   
Recipe of the day
~~~~~~~~~~~~~~~~~

Want to try something new? Each day, one of the recipes in the cookbook
is highlighted on the front page. There is one to suit everyone's taste!

.. raw:: html

    <div id="daily-thumbnail"></div>
