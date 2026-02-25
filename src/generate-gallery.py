import os
import shutil
import sys

import chemiscope  # noqa: F401
import sphinx_gallery.gen_gallery
import sphinx_gallery.gen_rst
from chemiscope.sphinx import ChemiscopeScraper


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))


class AttrDict(dict):
    def __init__(self):
        super().__init__()
        self.__dict__ = self


class PseudoSphinxApp:
    """
    Class pretending to be a sphinx App, used to configure and run sphinx-gallery
    from the command line, without having an actual sphinx project.
    """

    def __init__(self, example):
        gallery_dir = os.path.join(
            ROOT, "docs", "src", "examples", os.path.basename(example)
        )
        if os.path.exists(gallery_dir):
            shutil.rmtree(gallery_dir)

        # the options here are the minimal set of options to get sphinx-gallery to run
        # feel free to add more if sphinx-gallery uses more options in the future
        self.config = AttrDict()
        self.config.html_static_path = []
        self.config.templates_path = []
        self.config.source_suffix = [".rst"]
        self.config.default_role = ""
        self.config.sphinx_gallery_conf = {
            "filename_pattern": ".*",
            "examples_dirs": os.path.join(ROOT, example),
            "gallery_dirs": gallery_dir,
            "write_computation_times": False,
            "copyfile_regex": r".*\.(cp2k|jpg|jpeg|mdp|png|sh|xyz|yaml|yml|zip)",
            "matplotlib_animations": True,
            "within_subsection_order": "FileNameSortKey",
            "image_scrapers": ("matplotlib", ChemiscopeScraper()),
        }

        self.builder = AttrDict()
        self.builder.srcdir = os.path.join(ROOT, "docs", "src")
        self.builder.outdir = ""
        self.builder.name = os.path.basename(example)

        self.extensions = [
            "chemiscope.sphinx",
        ]

        self.builder.config = AttrDict()
        self.builder.config.plot_gallery = "True"
        self.builder.config.abort_on_example_error = True
        self.builder.config.highlight_language = None

    def add_css_file(self, path):
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <example/dir>")
        sys.exit(1)

    # To change the download text, we change the ZIP_DOWNLOAD variable in
    # sphinx_gallery.gen_rst. This is a bit of a hack, but arguably not
    # worse than postmodifying RST. We perform some checks here to make
    # sure that the hack is still valid and it does not fail silently.
    assert hasattr(sphinx_gallery.gen_rst, "ZIP_DOWNLOAD")
    assert isinstance(sphinx_gallery.gen_rst.ZIP_DOWNLOAD, str)

    sphinx_gallery.gen_rst.ZIP_DOWNLOAD = """
    .. container:: sphx-glr-download sphx-glr-download-zip

        :download:`Download recipe: {0} <{0}>`
    """

    app = PseudoSphinxApp(example=sys.argv[1])
    sphinx_gallery.gen_gallery.fill_gallery_conf_defaults(app, app.config)
    sphinx_gallery.gen_gallery.update_gallery_conf_builder_inited(app)
    sphinx_gallery.gen_gallery.generate_gallery_rst(app)
