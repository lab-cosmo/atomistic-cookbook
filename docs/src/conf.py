# Sphinx documentation build configuration file

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

example_subdirs = ["mlp_models", "sample_selection"]
sphinx_gallery_conf = {
    "filename_pattern": "/*",
    "examples_dirs": [f"../../examples/{p}" for p in example_subdirs],
    "gallery_dirs": [f"examples/{p}" for p in example_subdirs],
    "min_reported_time": 60,
    # Make the code snippet for own functions clickable
    "reference_url": {"cosmo-software-cookbook": None},
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "cosmo-software-cookbook"
copyright = "BSD 3-Clause License, Copyright (c) 2023, cosmo software cookbook team"

htmlhelp_basename = "cosmo-software-cookbook"

import os
os.system("patchelf --force-rpath --set-rpath '$ORIGIN/../../../../../lib:$ORIGIN/../../lib:$ORIGIN/../../src:$ORIGIN/../src:$ORIGIN/../../rascal.libs' /home/docs/checkouts/readthedocs.org/user_builds/software-cookbook/envs/13/lib/python3.10/site-packages/rascal/lib/_rascal.cpython-310-x86_64-linux-gnu.so")
