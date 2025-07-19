import os
import xml.etree.ElementTree as ET
from datetime import datetime


# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx_sitemap",
    "sphinx_design",
    "sphinx_copybutton",
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
    "featomic": ("https://metatensor.github.io/featomic/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torchpme": ("https://lab-cosmo.github.io/torch-pme/latest/", None),
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
html_theme_options = {
    "top_of_page_buttons": [],
}
html_use_index = False
html_static_path = ["_static"]
html_favicon = "_static/cookbook-icon.png"
html_logo = "_static/cookbook-icon.svg"
html_title = "The Atomistic Cookbook"

# sitemap/SEO settings
html_baseurl = "https://atomistic-cookbook.org/"
version = ""
release = ""
sitemap_url_scheme = "{link}"
html_extra_path = ["google4ae5e3529d19a84c.html", "robots.txt"]


# -- Custom post-build function ---------------------------------------------
def post_build_tweaks(app, exception):
    """
    Couple of small fixes (mostly SEO) after having
    built the docs.
    """

    conf_dir = os.path.abspath(os.path.dirname(__file__))
    build_folder = os.path.abspath(os.path.join(conf_dir, "..", "build", "html"))

    # adds custom urls to the sitemap
    custom_urls = [
        {"loc": "https://atomistic-cookbook.org", "priority": 1.0},
    ]

    sitemap_file = os.path.join(build_folder, "sitemap.xml")
    ET.register_namespace("", "http://www.sitemaps.org/schemas/sitemap/0.9")
    tree = ET.parse(sitemap_file)
    root = tree.getroot()

    for url in custom_urls:
        url_element = ET.SubElement(root, "url")
        loc_element = ET.SubElement(url_element, "loc")
        loc_element.text = url["loc"]
        if "priority" in url:
            priority_element = ET.SubElement(url_element, "priority")
            priority_element.text = str(url["priority"])

    tree.write(sitemap_file)

    # changes the canonical name of the main page
    file_path = os.path.join(build_folder, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the canonical URL (if it exists) with the desired URL
    new_content = content.replace(
        '<link rel="canonical" href="https://atomistic-cookbook.org/index.html" />',
        r"""
    <link rel="canonical" href="https://atomistic-cookbook.org" />
    <!-- Structured Data Markup -->
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "WebSite",
      "name": "The Atomistic Cookbook",
      "url": "https://atomistic-cookbook.org",
      "potentialAction": {
        "@type": "SearchAction",
        "target": "https://atomistic-cookbook.org/search.html?q={search_term_string}",
        "query-input": "required name=search_term_string"
      }
    }
    </script>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      "itemListElement": [
        {
          "@type": "ListItem",
          "position": 1,
          "name": "Home",
          "item": "https://atomistic-cookbook.org/"
        },
        {
          "@type": "ListItem",
          "position": 2,
          "name": "Recipes by Topic",
          "item": "https://atomistic-cookbook.org/topics/index.html"
        },
        {
          "@type": "ListItem",
          "position": 3,
          "name": "Recipes by Software Used",
          "item": "https://atomistic-cookbook.org/software/index.html"
        },
        {
          "@type": "ListItem",
          "position": 4,
          "name": "All recipes",
          "item": "https://atomistic-cookbook.org/all-examples.html"
        }
      ]
    }
    </script>
        """,
    )

    # Write the modified content back to index.html
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Canonical URL modified in {file_path}")


def setup(app):
    app.connect("build-finished", post_build_tweaks)
    app.add_css_file("cookbook.css")
