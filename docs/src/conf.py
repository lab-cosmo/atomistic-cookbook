from datetime import datetime


# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx_sitemap",
    "sphinx_design",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.load_style",
    "chemiscope.sphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

project = "atomistic-cookbook"
copyright = (
    "BSD 3-Clause License, "
    f"Copyright (c) {datetime.now().date().year}, "
    "The atomistic cookbook team"
)

intersphinx_mapping = {
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "rascaline": ("https://luthaf.fr/rascaline/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

html_js_files = [
    (  # plausible.io tracking
        "https://plausible.io/js/script.file-downloads.hash.outbound-links.pageview-props.tagged-events.js",  # noqa: E501
        {"data-domain": "atomistic-cookbook.org", "defer": "defer"},
    ),
    "all-examples-data.js",  # data for the recipe-of-the-day
    "daily-recipe.js",  # loader for the recipe-of-the-day
]


htmlhelp_basename = "The Atomistic Cookbook"
html_theme = "furo"
html_static_path = ["_static"]
html_favicon = "_static/cookbook-icon.png"
html_logo = "_static/cookbook-icon.svg"
html_title = "The Atomistic Cookbook"

# sitemap/SEO settings
html_baseurl = "https://atomistic-cookbook.org/"
version = ""
relsease = ""
sitemap_url_scheme = "{link}"
html_extra_path = ["google4ae5e3529d19a84c.html", "robots.txt"]
