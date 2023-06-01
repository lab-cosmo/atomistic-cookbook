Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in the ways listed below.

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

.. include:: ../README.rst
   :start-after: marker-building-start
   :end-before: marker-building-end


Support
-------

If you still have problems adding your example to the repository, please feel free to contact one of the people

`@agoscinski (Alexander Goscinski) <https://github.com/agoscinski>`_
`@davidetisi (Davide Tisi) <https://github.com/davidetisi>`_

Code of Conduct
---------------

Please note that the COSMO cookbook project is released with a [Contributor Code of Conduct](CONDUCT.md). By contributing to this project you agree to abide by its terms.
