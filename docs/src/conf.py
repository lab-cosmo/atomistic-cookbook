# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "cosmo-software-cookbook"
copyright = "BSD 3-Clause License, Copyright (c) 2023, COSMO software cookbook team"

htmlhelp_basename = "COSMO software-cookbook"
html_theme = "furo"
