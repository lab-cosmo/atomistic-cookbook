import os
import shutil
import subprocess
import sys
import traceback

import chemiscope  # noqa: F401
import sphinx_gallery.gen_gallery
import sphinx_gallery.gen_rst
from chemiscope.sphinx import ChemiscopeScraper


# Monkey-patch _LoggingTee.write to echo captured output to sys.__stderr__.
# sphinx-gallery's _LoggingTee replaces both sys.stdout and sys.stderr during
# recipe execution, so all print output vanishes from CI. sys.__stderr__ is
# the original fd preserved by Python at startup, immune to replacement.
_orig_tee_write = sphinx_gallery.gen_rst._LoggingTee.write


def _tee_write_with_ci(self, data):
    _orig_tee_write(self, data)
    sys.__stderr__.write(data)
    sys.__stderr__.flush()


sphinx_gallery.gen_rst._LoggingTee.write = _tee_write_with_ci


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
            "copyfile_regex": r".*\.(cp2k|jpg|jpeg|lmp|mdp|png|sh|xyz|yaml|yml|zip)",
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


def _gallery_has_output(example_dir: str) -> bool:
    """True if sphinx-gallery wrote recipe artifacts for this example."""
    gallery_dir = os.path.join(
        ROOT, "docs", "src", "examples", os.path.basename(example_dir)
    )
    if not os.path.isdir(gallery_dir):
        return False
    for name in os.listdir(gallery_dir):
        if name.endswith((".rst", ".ipynb", ".py")):
            return True
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <example/dir>")
        sys.exit(1)

    example_dir = sys.argv[1]

    # Run the gallery build in a child process. Some torch/native stacks abort
    # during interpreter teardown after a successful recipe (SIGABRT: -6 or
    # 134). If the gallery artifacts already exist, treat that as success;
    # otherwise retry once.
    if os.environ.get("_ATOMISTIC_COOKBOOK_GALLERY_INNER") != "1":
        env = {**os.environ, "_ATOMISTIC_COOKBOOK_GALLERY_INNER": "1"}
        abort_codes = {-6, 134}
        last_code = 1
        for attempt in range(2):
            proc = subprocess.run(
                [sys.executable, "-u", __file__, example_dir],
                env=env,
            )
            last_code = proc.returncode
            if last_code == 0:
                sys.exit(0)
            if last_code in abort_codes and _gallery_has_output(example_dir):
                print(
                    f"generate-gallery: child exited {last_code} after writing "
                    f"gallery for {example_dir}; treating as success",
                    file=sys.__stderr__,
                )
                sys.exit(0)
            if last_code in abort_codes and attempt == 0:
                print(
                    f"generate-gallery: child exited {last_code} "
                    "without usable gallery; retrying once",
                    file=sys.__stderr__,
                )
                continue
            break
        sys.exit(last_code if last_code > 0 else 1)

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

    app = PseudoSphinxApp(example=example_dir)
    sphinx_gallery.gen_gallery.fill_gallery_conf_defaults(app, app.config)
    sphinx_gallery.gen_gallery.update_gallery_conf_builder_inited(app)

    try:
        sphinx_gallery.gen_gallery.generate_gallery_rst(app)
    except Exception:
        traceback.print_exc(file=sys.__stderr__)
        sys.exit(1)
