Installing a recipe
===================

The main repository contains the source files of all recipes, but you 
should only fetch the entire repository if you plan on 
`contributing a recipe <CONTRIBUTING.rst>`_. 
If you are interested in learning the techniques discussed in a specific 
recipe, this is not recommended, as you will also have to understand the 
build mechanism for the website.

.. marker-install-start

Each recipe of the atomistic cookbook can be viewed online as an interactive
HTML page, but can also be downloaded as a stand-alone ``.py`` script, or a  
``.ipynb`` Jupyter notebook. To simplify setting up an environment that 
contains all the dependencies needed for each recipe, each recipe can
be downloaded as an archive that contains the script and notebook
versions, together with any necessary data file and an ``environment.yml`` file 
that you can use with conda to create a custom environment to run the example.
If you have never used conda before, you may want to read this
`beginners guide 
<https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

.. code-block:: bash

   # Deflate the downloaded archive of the recipe
   unzip recipe.zip
   
   # Pick a path where to create the environment and replace 
   # <recipe-env-path> with it - this creates a local folder 
   # to avoid cluttering the global conda folder
   conda env create --prefix <recipe-env-path> --file environment.yml


Whenever you want to run the recipe you just need to activate the environment
and run the recipe file. 

.. code-block:: bash

   # When you want to use the environment
   conda env activate --name <recipe-env-path>
   
   # You should be able to run the recipe within the environment 
   python recipe.py


The ``environment.yml`` contents should also help you set up your own 
conda or pip environment to run without having to enter the recipe-specific
environment.

Given that many recipes contain post-processing and visualizations that
are used to render the HTML page of the recipe, you may prefer to run the
recipe as a Jupyter notebook. If you want to do so and have not installed 
the dependencies globally, you will also have to create a kernel that uses
the environment. Assuming you have a functioning Jupyter installation globally,
you should run

.. code-block:: bash

   # Activate the environment 
   conda env activate --name <recipe-env-path>
   pip install ipykernel # in case is not part of the environment
   
   # Create the kernel definition
   python -m ipykernel install --user --name recipe-env-path --display-name "Python (recipe)"
   
   # You can launch Jupyter from outside the conda environment,
   # unless you also need conda-installed executables. When you open
   # the notebook, make sure you selected "Python (recipe_env)" as the 
   # kernel to run it
   jupyter lab recipe.ipynb
   
.. marker-install-end

