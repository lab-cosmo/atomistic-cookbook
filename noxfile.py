import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import nox
import yaml
from docutils.core import publish_doctree, publish_parts
from docutils.nodes import comment, paragraph, title
from docutils.parsers.rst import Directive, directives


class DummyToctree(Directive):
    has_content = True

    def run(self):
        # collect non-blank body lines (sections)
        docs = [
            line.strip()
            for line in self.content
            if line.strip() and line.strip()[0] != ":"
        ]

        # stash them as "comments"
        marker = comment()
        marker["docnames"] = docs
        return [marker]


directives.register_directive("toctree", DummyToctree)

ROOT = os.path.realpath(os.path.dirname(__file__))

sys.path.insert(0, ROOT)
from src.get_examples import get_examples  # noqa: E402


# global nox options
nox.needs_version = ">=2024"
nox.options.reuse_venv = "yes"
nox.options.sessions = ["lint", "docs"]


EXAMPLES = get_examples()

# ==================================================================================== #
#                                 helper functions                                     #
# ==================================================================================== #


# Files that need to be linted & formatted
def get_lint_files():
    LINT_FILES = [
        "src/ipynb-to-gallery.py",
        "src/generate-gallery.py",
        "src/latest_docs_run.py",
        "noxfile.py",
        "docs/src/conf.py",
        "src/get_examples.py",
    ]
    return LINT_FILES + get_example_files()


def filter_files(tracked_files):
    """Filter tracked files of what we do not want to lint.
    The list of files tracked by `git ls-files` used in the `get_example_files`
    contains also input, trajectory and README files that we do not want
    to lint. This function filter these kind of files out of the main list."""
    returns = []
    for file in tracked_files.splitlines():
        tmp = file.split(".")[-1]
        if tmp in ["rst", "py"]:  # skips all files that are not rst or py
            if os.path.split(file)[-1] not in [
                ".gitignore",
                "README.rst",
                "INSTALLING.rst",
            ]:
                returns.append(file)

    return returns


# We want to mimic
# git ls-files examples
def get_example_files():
    folder = os.path.join(os.getcwd(), "examples")
    # Get the list of ignored files
    # Get the list of all tracked files
    tracked_files_command = ["git", "ls-files", folder]
    tracked_files_output = subprocess.check_output(
        tracked_files_command, cwd=folder, text=True
    )
    # Filter the tracked files to exclude ignored ones
    filtered_files = filter_files(tracked_files_output)

    return [os.path.join(folder, file) for file in filtered_files]


def get_example_other_files(fd):
    folder = os.path.join(os.getcwd(), fd)
    # Get the list of ignored files
    tracked_files_command = ["git", "ls-files", "--other", folder]
    tracked_files_output = subprocess.check_output(
        tracked_files_command, cwd=folder, text=True
    )

    return [os.path.join(folder, file) for file in tracked_files_output.splitlines()]


def should_reinstall_dependencies(session, **metadata):
    """
    Returns a bool indicating whether the dependencies should be re-installed in the
    venv.

    This works by hashing everything in metadata, and storing the hash in the session
    virtualenv. If the hash changes, we'll have to re-install!
    """

    to_hash = {}
    for key, value in metadata.items():
        if os.path.exists(value):
            with open(value) as fd:
                to_hash[key] = fd.read()
        else:
            to_hash[key] = str(value)

    to_hash = json.dumps(to_hash).encode("utf8")
    sha1 = hashlib.sha1(to_hash).hexdigest()
    sha1_path = os.path.join(session.virtualenv.location, "metadata.sha1")

    if session.virtualenv._reused:
        if os.path.exists(sha1_path):
            with open(sha1_path) as fd:
                should_reinstall = fd.read().strip() != sha1
        else:
            should_reinstall = True
    else:
        should_reinstall = True

    with open(sha1_path, "w") as fd:
        fd.write(sha1)

    if should_reinstall:
        session.debug("updating environment since the dependencies changed")

    return should_reinstall


def rst_to_html(rst_text):
    """
    Convert a reStructuredText (RST) string to HTML.

    Parameters:
        rst_text (str): The RST content to convert.

    Returns:
        str: The resulting HTML string.
    """
    settings_overrides = {
        "initial_header_level": 2,
        "report_level": 5,  # Suppress warnings
        "syntax_highlight": "short",
        "math_output": "mathjax",
    }

    parts = publish_parts(
        source=rst_text, writer_name="html5", settings_overrides=settings_overrides
    )

    return parts["fragment"]


def get_example_metadata(rst_file):
    metadata = {}
    # Path to the generated RST file (stripping docs/src/)
    gallery_dir, example_file = os.path.split(rst_file)
    gallery_dir = os.path.join(*(gallery_dir.split(os.sep)[2:]))
    example_name, _ = os.path.splitext(example_file)

    # Path to the thumbnail image
    thumbnail_file = os.path.join(
        gallery_dir, "images/thumb", f"sphx_glr_{example_name}_thumb.png"
    )

    # Parse the RST file
    with open(rst_file, "r") as file:
        rst_content = file.read()
    settings_overrides = {
        # Set the threshold for reporting messages to 'CRITICAL' (level 5)
        "report_level": 5,
        # Set the threshold for halting the processing to 'Severe' or higher
        "halt_level": 6,
        # Suppress warning output
        "warning_stream": None,
    }
    doctree = publish_doctree(rst_content, settings_overrides=settings_overrides)
    rst_title = None
    rst_description = None
    html_description = None

    # Traverse the document tree
    for node in doctree:
        if isinstance(node, title) and rst_title is None:
            rst_title = node.astext()
        if isinstance(node, paragraph) and rst_description is None:
            rst_description = node.astext().replace("\n", " ")
            html_description = rst_to_html(rst_description)
        if rst_title and rst_description:  # break when done
            break

    if rst_title is None:
        # Grab any text at start of line, followed by newline, followed by 3+ '=' signs
        match = re.search(r"^([^:\n].+)\n={3,}", rst_content, re.MULTILINE)
        if match:
            rst_title = match.group(1).strip()

    metadata["title"] = rst_title or ""
    metadata["description"] = rst_description or ""
    metadata["html"] = html_description or ""
    metadata["thumbnail"] = thumbnail_file
    metadata["ref"] = str(os.path.join(gallery_dir, example_name))

    return metadata


def build_gallery_section(template):
    """Builds the .rst for a section based on its template (.sec) file.

    Each .sec file contains an RST header, and is concluded by a :list:
    directive that contains the doc names of the examples that should be
    included in that section (relative to the root), e.g.

    - examples/lpr/lpr
    """

    rst_file, _ = os.path.splitext(template)
    rst_file += ".rst"
    rst_output = ""
    section_examples = []
    card_strings = []
    with open(template, "r") as fd:
        for line in fd:
            if line.strip()[:2] == "- ":
                section_examples.append(line.strip(" -\n"))
            else:
                rst_output += line

    with open(rst_file, "w") as fd:
        fd.write(rst_output)

        # gallery thumbnails
        fd.write(
            """

.. grid:: 1 2 2 3
    :gutter: 1 1 2 3
"""
        )
        # sort by title
        for example in section_examples:
            file = os.path.join("docs", "src", f"{example}.rst")
            if not os.path.exists(file):
                continue
            metadata = get_example_metadata(file)
            thumbnail = os.path.join("../", *metadata["thumbnail"].split(os.sep))

            # generates a thumbnail link
            card_string = f"""
        .. card:: {metadata["title"]}
            :link: ../{metadata["ref"]}
            :link-type: doc
            :text-align: center
            :shadow: md

            .. image:: {thumbnail}
                :alt: {metadata["description"]}
                :class: gallery-img

                """
            fd.write(f"""\n    .. grid-item::\n{card_string}""")
            card_strings.append(card_string)

    return card_strings


def post_process_gallery(name, example_dir, files_before):
    # Path of the generated gallery example.
    docs_example_dir = Path("docs/src/examples") / name

    # Get list of generated notebooks
    notebooks = [
        file
        for file in docs_example_dir.glob("*.ipynb")
        if not (example_dir / file.name).exists()
    ]
    # Get the source python files that generated the notebooks
    source_py_files = [notebook.with_suffix(".py").name for notebook in notebooks]
    # Remove them from the list of example files (we don't want to include
    # them in every zip file)
    example_files = [
        file
        for file in files_before
        if file.suffix != ".py" or file.name not in source_py_files
    ]

    # The src/generate-gallery.py script creates a zip file with just the
    # *.py and *.ipynb files. Downloading this zip file is not very useful
    # to reproduce the tutorial. Here we overwrite that zip file with one
    # that also contains the rest of the files present in the example directory
    # before running the example (including e.g. data and environment.yml).
    # We create a zip file for each notebook.
    for py_file, notebook in zip(source_py_files, notebooks):
        with zipfile.ZipFile(docs_example_dir / f"{notebook.stem}.zip", "w") as zipf:
            # Add files from the data dir (if present)
            for file in example_dir.rglob("data/*"):
                zipf.write(file, file.relative_to(example_dir))

            # Add the rest of files in the example dir (with an extra check
            # to make sure that they are still there)
            for file in example_files:
                if file.is_file() and os.path.split(file)[-1] not in [
                    "README.rst",
                    ".gitignore",
                ]:
                    zipf.write(file, file.relative_to(example_dir))

            # Add the .py and .ipynb files
            zipf.write(docs_example_dir / py_file, py_file)
            zipf.write(notebook, notebook.name)

            # Adds the installation instructions
            zipf.write("examples/INSTALLING.rst", "INSTALLING.rst")

    os.unlink(f"docs/src/examples/{name}/index.rst")


# For each of the dependencies as keys, add the value as an extra dependency
DEPENCENCIES_UPDATES = {
    "libtorch": "pytorch-cpu",
    "lammps-metatomic": "lammps-metatomic * cpu*nompi*",
    "plumed-metatomic": "plumed-metatomic * *nompi*",
}


def update_dependencies(environment_yml, session):
    output = session.run(
        "conda",
        "env",
        "create",
        f"--file={environment_yml}",
        "--name=atomistic-cookbook-tmp-env",
        "--solver=libmamba",
        "--json",
        "--quiet",
        "--dry-run",
        silent=True,
    )

    try:
        data = json.loads(output)
        dependencies = data["dependencies"]
    except json.JSONDecodeError:
        session.error(f"Conda did not return valid JSON. Output was: {output}")

    new_deps = set()
    for dep in dependencies:
        for to_update, new_dep in DEPENCENCIES_UPDATES.items():
            if f"::{to_update}==" in dep:
                new_deps.add(new_dep)

    if len(new_deps) != 0:
        with open(environment_yml) as fd:
            environment = yaml.safe_load(fd)

        for dep in new_deps:
            environment["dependencies"].append(dep)

        environment_yml = session.virtualenv.location + "/environment.yml"
        with open(environment_yml, "w") as fd:
            yaml.safe_dump(environment, fd)

    return environment_yml


# ==================================================================================== #
#                              nox sessions definitions                                #
# ==================================================================================== #


for name in EXAMPLES:

    @nox.session(name=name, venv_backend="conda")
    def example(session, name=name):
        example_dir = Path("examples") / name
        environment_yml = example_dir / "environment.yml"
        if should_reinstall_dependencies(session, environment_yml=environment_yml):
            environment_yml = update_dependencies(environment_yml, session)

            session.run(
                "conda",
                "env",
                "update",
                "--prune",
                f"--file={environment_yml}",
                f"--prefix={session.virtualenv.location}",
                "--solver=libmamba",
            )

            # install sphinx-gallery and its dependencies
            session.install(
                "sphinx-gallery",
                "sphinx",
                "pillow",
                "matplotlib",
                "chemiscope",
            )

        session.run(
            "conda",
            "list",
            f"--prefix={session.virtualenv.location}",
        )

        # Gather list of files before running the example
        files_before = list(example_dir.glob("*"))

        session.run("python", "src/generate-gallery.py", example_dir)

        post_process_gallery(name, example_dir, files_before)

        if "--no-website" not in session.posargs:
            session.notify("build_website")

        output = session.run(
            "git",
            "ls-files",
            "--exclude-standard",
            "--others",
            silent=True,
            external=True,
        )
        if output.strip() != "":
            session.warn(
                "WARNING: There are files untracked by git, you should add anything "
                "generated by an example to `.gitignore`, and commit the example files."
            )
            session.warn("The following files are not tracked:\n " + output.strip())
            session.warn("This will be an error when building the full website.")


@nox.session(venv_backend="none")
def docs(session):
    session.error("use nox -e website instead of nox -e docs")


@nox.session(venv_backend="none")
def website(session):
    """Run all examples and build the website"""

    for example in EXAMPLES:
        session.run("nox", "-e", example, "--", "--no-website", external=True)

    session.run("nox", "-e", "build_website", external=True)


@nox.session
def build_website(session):
    """
    Assemble the different examples into a website, assuming the examples files are
    already generated
    """

    # install build dependencies
    requirements = "docs/requirements.txt"
    if should_reinstall_dependencies(session, requirements=requirements):
        session.install("-r", requirements)

    # list all examples
    all_examples_rst = {}
    for file in glob.glob("docs/src/examples/*/*.rst"):
        if os.path.basename(file) != "sg_execution_times.rst":
            all_examples_rst[file] = get_example_metadata(file)

    # generate global list
    with open("docs/src/all-examples.rst", "w") as output:
        output.write(
            """
List of all recipes
===================

This section contains the list of all compiled recipes, including those
that are not part of any of the other sections.


.. grid:: 1 2 2 3
    :gutter: 1 1 2 3
"""
        )
        # sort by title
        for _, metadata in sorted(
            all_examples_rst.items(), key=(lambda kw: kw[1]["title"])
        ):
            # generates a thumbnail link
            output.write(
                f"""
    .. grid-item::
        .. card:: {metadata["title"]}
            :link: {metadata["ref"]}
            :link-type: doc
            :text-align: center
            :shadow: md

            .. image:: {metadata["thumbnail"]}
                :alt: {metadata["description"]}
                :class: gallery-img

                """
            )

        output.write(
            """
.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

"""
        )
        for _, metadata in sorted(
            all_examples_rst.items(), key=(lambda kw: kw[1]["title"])
        ):
            output.write(f"   {metadata['ref']}\n")

    # saves also example data as a json
    examples_data_js_path = os.path.join(
        "docs", "src", "_static", "all-examples-data.js"
    )
    # Prepare the examples data for JavaScript
    examples_data = []
    for _, metadata in sorted(all_examples_rst.items(), key=lambda kw: kw[1]["title"]):
        # use relative paths so it works both locally and when deployed
        metadata["thumbnail"] = "_images/" + os.path.split(metadata["thumbnail"])[-1]
        metadata["ref"] = metadata["ref"] + ".html"
        examples_data.append(metadata)

    # Write the data to examples_data.json in the _static directory
    os.makedirs(os.path.dirname(examples_data_js_path), exist_ok=True)
    with open(examples_data_js_path, "w") as fd:
        fd.write("var examplesData = ")
        json.dump(examples_data, fd)
        fd.write(";")

    # generates section files
    for section in glob.glob("docs/src/*/index.sec"):
        print("Processing section:", section)

        # parse to get the names of the subsections
        doctree = publish_doctree(Path(section).read_text(encoding="utf-8"))
        docnames = [
            name
            for node in doctree.traverse(comment)  # look for our marker nodes
            if "docnames" in node  # filter only the ones we made
            for name in node["docnames"]  # flatten the list
        ]

        # now builds the index.rst file as well as all the subsections
        rst_file, _ = os.path.splitext(section)
        rst_file += ".rst"
        rst_output = ""
        # first get the title and description from the index.sec file
        with open(section, "r") as fd:
            for line in fd:
                rst_output += line
        rst_output += "\n\n"

        # then add the sections from the toctree
        for template in docnames:
            cards = build_gallery_section(
                os.path.join(os.path.dirname(section), f"{template}.sec")
            )
            rst_output += f":doc:`{template}`\n" + ("~" * 256) + "\n\n"
            rst_output += ".. card-carousel:: 3\n\n"
            for card in cards:
                rst_output += card + "\n"

        with open(rst_file, "w") as fd:
            fd.write(rst_output)

    session.run("sphinx-build", "-b", "html", "docs/src", "docs/build/html")


@nox.session
def lint(session):
    """Run linters and type checks"""

    if not session.virtualenv._reused:
        session.install("black", "blackdoc")
        session.install("flake8", "flake8-bugbear", "flake8-sphinx-links")
        session.install("isort")
        session.install("sphinx-lint")

    # Get files
    LINT_FILES = get_lint_files()

    # Formatting
    session.run("black", "--check", "--diff", *LINT_FILES)
    session.run("blackdoc", "--check", "--diff", *LINT_FILES)
    session.run("isort", "--check-only", "--diff", *LINT_FILES)

    # Linting
    session.run(
        "flake8",
        "--max-line-length=88",
        "--exclude=docs/src/examples/",
        "--extend-ignore=E203",
        *LINT_FILES,
    )

    session.run(
        "sphinx-lint",
        "--enable=line-too-long",
        "--max-line-length=88",
        "--ignore=docs/src",
        "README.rst",
        "CONTRIBUTING.rst",
        *LINT_FILES,
    )


def remove_trailing_whitespace(file_path):
    with open(file_path, "r+") as file:
        lines = [line.rstrip() for line in file]
        file.seek(0)
        file.writelines(line + "\n" for line in lines)
        file.truncate()


@nox.session
def format(session):
    """Automatically format all files"""

    if not session.virtualenv._reused:
        # TODO: remove the pin after https://github.com/keewis/blackdoc/pull/256 lands
        session.install("black==25.1.0", "blackdoc==0.4.2")
        session.install("isort")
    # Get files
    LINT_FILES = get_lint_files()

    session.run("black", *LINT_FILES)
    session.run("blackdoc", *LINT_FILES)
    session.run("isort", *LINT_FILES)
    for file in LINT_FILES:
        remove_trailing_whitespace(file)


@nox.session
def clean_build(session):
    """Remove temporary files and building folders."""

    # remove build folders
    for i in ["docs/src/examples/", "docs/build"]:
        if os.path.isdir(i):
            shutil.rmtree(i)


@nox.session
def clean_examples(session):
    """Remove all untracked files from the example folders."""

    for ifile in get_example_other_files("examples"):
        os.remove(ifile)

    flist = glob.glob("examples/*")
    # Remove empty folders
    for path in flist:
        if len(glob.glob(os.path.join(path, "*"))) == 0:
            os.rmdir(path)
