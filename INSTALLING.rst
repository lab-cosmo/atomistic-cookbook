Installing a recipe
===================

The main repository contains the source files of all recipes, but you 
should only fetch the entire repository if you plan on `contributing one
<CONTRIBUTING.rst>`_. If you are interested in learning
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
If you have never used conda before, you may want to read this
`beginners guide 
<https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

.. code-block:: bash

   # Pick a name for the environment and replace <environment-name> with it
   conda env create --name <environment-name> --file environment.yml

   # when you want to use the environment
   conda env activate --name <environment-name>

You can then execute the script from the command line, or open the
notebook in Jupyter lab.  Additional data needed for each example is usually 
either downloaded dynamically, or can be found in a ``data`` folder for each 
example, or downloaded as a ``data.zip`` file at the end of each recipe in
the website. In the latter case, don't forget to unzip the file in the 
same folder as the ``.py`` or ``.ipynb`` files for the example. 

.. marker-install-end




