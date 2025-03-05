Installing a recipe
===================

The main repository contains the source files of all recipes, but you 
should only fetch the entire repository if you plan on 
`contributing a recipe <CONTRIBUTING.rst>`_. 
If you are interested in learning the techniques discussed in a specific 
recipe, this is not recommended, as you will also have to understand the 
build mechanism for the website.

.. marker-install-start

Each recipe can be viewed online as an interactive HTML page, but 
can also be downloaded as a stand-alone ``.py`` script, or a  
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
   unzip recipe_name.zip
   
   # Pick a path where to create the environment and replace 
   # <environment-folder> with it - this creates a local folder 
   # to avoid cluttering the global conda folder
   conda env create --prefix <environment-folder/> --file environment.yml

   # When you want to use the environment
   conda env activate --name <environment-folder/>
   
   # You should be able to run the recipe within the environment 
   python recipe-file.py

If you want to run the notebook, you also have to create a kernel that uses
the environment. Assuming you have a functioning Jupyter installation globally,
you should run

.. code-block:: bash

   # Activate the environment 
   conda env activate --name <environment-folder/>
   pip install ipykernel # in case is not part of the environment
   
   # Create the kernel definition
   python -m ipykernel install --user --name recipe_env --display-name "Python (recipe_env)"
   
   # You can launch Jupyter from outside the conda environment,
   # unless you also need conda-installed executables. When you open
   # the notebook, make sure you selected "Python (recipe_env)" as the 
   # kernel to run it
   jupyter lab recipe-file.ipynb
   
.. marker-install-end

