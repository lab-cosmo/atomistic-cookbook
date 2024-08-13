Atomistic Cookbook
==================

.. include:: ../../README.rst
   :start-after: marker-intro-start
   :end-before: marker-intro-end


Each example provides an ``environment.yml`` file that you can download and
then use with conda to create a new environment with all the required dependencies.

.. code-block:: bash

   # Pick a name for the environment and replace <environment-name> with it
   conda env create --name <environment-name> --file environment.yml

   # when you want to use the environment
   conda env activate --name <environment-name>

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1
   
   analysis
   sampling
   ml-models
   all-examples
