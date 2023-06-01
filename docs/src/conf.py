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
root_index_rst_content += ''.join([f"   {subdir}/index\n" for subdir in examples_subdirs])
print("Creating index.rst including all examples")
print(root_index_rst_content)

with open('index.rst', 'w') as f:
    f.write(root_index_rst_content)

# Temporary needed to update rpath correctly so we can include the prebuilt rascal version
import os
import rascal
rascal_shared_lib_path = '/'.join(rascal.__file__.split("/")[:-1]) + '/lib/_rascal.cpython-310-x86_64-linux-gnu.so'
os.system(f"patchelf --force-rpath --set-rpath '$ORIGIN/../../../../../lib:$ORIGIN/../../lib:$ORIGIN/../../src:$ORIGIN/../src:$ORIGIN/../../rascal.libs' {rascal_shared_lib_path}")
