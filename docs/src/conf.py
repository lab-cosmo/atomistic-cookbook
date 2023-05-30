# Sphinx documentation build configuration file
import os

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.inheritance_diagram",
    "jupyter_sphinx",
]

examples_root = os.path.join(os.getcwd(), '../../examples/')
examples_subdirs = []
for path in os.listdir(examples_root):
    # ignores files and hidden directories
    if os.path.isdir(os.path.join(examples_root, path)) and path[0] != ".":
        examples_subdirs.append(path)

sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": [f"../../examples/{p}" for p in examples_subdirs],
    "gallery_dirs": [f"examples/{p}" for p in examples_subdirs],
    "min_reported_time": 60,
    # Make the code snippet for own functions clickable
    "reference_url": {"cosmo-software-cookbook": None},
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "cosmo-software-cookbook"
copyright = "BSD 3-Clause License, Copyright (c) 2023, cosmo software cookbook team"

htmlhelp_basename = "cosmo-software-cookbook"
