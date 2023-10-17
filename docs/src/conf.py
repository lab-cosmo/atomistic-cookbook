# Sphinx documentation build configuration file
import os


# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.autodoc",  # import the modules you are documenting
    "sphinx.ext.intersphinx",  # generate links to external projects
    "sphinx.ext.viewcode",  # add links to highlighted source code
    "sphinx_gallery.gen_gallery",  # provides a source parser for *.ipynb files
]

examples_root = os.path.join(os.getcwd(), "../../examples/")
examples_subdirs = []
for path in os.listdir(examples_root):
    # ignores files and hidden directories
    if os.path.isdir(os.path.join(examples_root, path)) and path[0] != ".":
        examples_subdirs.append(path)

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": [f"../../examples/{subdir}" for subdir in examples_subdirs],
    "gallery_dirs": [f"{subdir}" for subdir in examples_subdirs],
    "min_reported_time": 60,
    # Make the code snippet for own functions clickable
    "reference_url": {"cosmo-software-cookbook": None},
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "cosmo-software-cookbook"
copyright = "BSD 3-Clause License, Copyright (c) 2023, COSMO software cookbook team"

htmlhelp_basename = "COSMO software-cookbook"
html_theme = "furo"

# We create the index.rst here because sphinx is not able to automatically
# include all subdirectories using regex expression
root_index_rst_content = r"""
COSMO Software Cookbook
=======================

.. include:: ../../README.rst
   :start-after: marker-intro-start
   :end-before: marker-intro-end

.. toctree::
   :caption: Table of Contents

"""
root_index_rst_content += "".join(
    [f"   {subdir}/index\n" for subdir in examples_subdirs]
)
print("Creating index.rst including all examples")
print(root_index_rst_content)

with open("index.rst", "w") as f:
    f.write(root_index_rst_content)

# Configuration for intersphinx: refer to the Python standard library
# and other packages used by the cookbook

intersphinx_mapping = {
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "chemiscope": ("https://chemiscope.org/docs/", None),
    "metatensor": ("https://lab-cosmo.github.io/metatensor/latest/", None),
    "equisolve": ("https://lab-cosmo.github.io/equisolve/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "rascal": ("https://lab-cosmo.github.io/librascal/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": (
        "http://scikit-learn.org/stable",
        (None, "./_intersphinx/sklearn-objects.inv"),
    ),
    "skmatter": ("https://scikit-matter.readthedocs.io/en/latest/", None),
}
