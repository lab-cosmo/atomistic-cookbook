Installation instructions
=========================

These installation instructions are common to all recipes from the
`atomistic cookbook <http://atomistic-cookbook.org>`_ and should help
you getting started. We refer to generic ``recipe.*`` filenames, but
it should be obvious what files you should use at each step.
When using the recipes keep in mind that examples
are designed to showcase a modeling technique or a software package,
and are constrained to have limited execution time, so the simulation
parameters are pushed to the limits and may require adjustment when
running a production calculation.

You should be reading this from within a folder you obtained as a 
``recipe.zip`` archive, either from the website or from a colleague. 
Each recipe consists of a a stand-alone ``.py`` script, a matching
``.ipynb`` Jupyter notebook, an ``environment.yml`` file that allows
recreating an environment to run the recipe, and possibly one or more 
supporting data files.

The fist step to run your recipe is to create a `conda` custom environment
If you have never used conda before, you may want to read this
`beginners guide <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

.. code-block:: bash

   # Deflate the downloaded archive of the recipe
   unzip recipe.zip # you may have done this already
   
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
   python -m ipykernel install --user --name recipei-env-path --display-name "Python (recipe)"
   
   # You can launch Jupyter from outside the conda environment,
   # unless you also need conda-installed executables. When you open
   # the notebook, make sure you selected "Python (recipe_env)" as the 
   # kernel to run it
   jupyter lab recipe.ipynb
