import os
from datetime import datetime

from chemiscope.sphinx import ChemiscopeScraper


ROOT = os.path.abspath(os.path.join("..", ".."))

# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "chemiscope.sphinx",
]
examples_dirs = os.path.join(ROOT, "examples")
sphinx_gallery_conf = {
    "examples_dirs": examples_dirs,
    "gallery_dirs": "examples",
    "filename_pattern": ".*",
    "within_subsection_order": "FileNameSortKey",
    "image_scrapers": (ChemiscopeScraper(examples_dirs)),
}


templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "cosmo-software-cookbook"
copyright = (
    "BSD 3-Clause License, "
    f"Copyright (c) {datetime.now().date().year}, "
    "COSMO software cookbook team"
)

htmlhelp_basename = "COSMO software-cookbook"
html_theme = "furo"


intersphinx_mapping = {
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "metatensor": ("https://lab-cosmo.github.io/metatensor/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
