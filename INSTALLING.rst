Installing a recipe
===================

The main repository contains the source files of all recipes, but you 
should only fetch the entire repository if you plan on contributing 
one (see also  <CONTRIBUTING.rst>`_). If you are interested in learning
the techniques discussed in a specific recipe, this is not recommended,
as you will also have to understand the build mechanism for the 
website.

.. marker-install-start

Each recipe can be viewed online as an interactive HTML page, but 
can also be downloaded as a stand-alone ``.py`` script, or a  
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

.. marker-install-end




