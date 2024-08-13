from datetime import datetime


# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.load_style",
    "chemiscope.sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "atomistic-simulations-cookbook"
copyright = (
    "BSD 3-Clause License, "
    f"Copyright (c) {datetime.now().date().year}, "
    "Atomistic simulations cookbook team"
)

htmlhelp_basename = "Atomistic simulation cookbook"
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

html_js_files = [
    (
        "https://plausible.io/js/script.js",
        {"data-domain": "lab-cosmo.github.io/atomistic-cookbook", "defer": "defer"},
    ),
]
