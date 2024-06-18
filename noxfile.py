import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import nox


ROOT = os.path.realpath(os.path.dirname(__file__))

sys.path.append(ROOT)
from developer.get_examples import get_examples  # noqa: E402


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
        "ipynb-to-gallery.py",
        "generate-gallery.py",
        "noxfile.py",
        "docs/src/conf.py",
        "developer",
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
            if (
                file.split("/")[-1] != ".gitignore"
                and file.split("/")[-1] != "README.rst"
            ):
                returns.append(file)

    return returns


# We want to mimic
# git ls-files examples
def get_example_files():
    folder = os.getcwd() + "/examples"
    # Get the list of ignored files
    # Get the list of all tracked files
    tracked_files_command = ["git", "ls-files", folder]
    tracked_files_output = subprocess.check_output(
        tracked_files_command, cwd=folder, text=True
    )
    # Filter the tracked files to exclude ignored ones
    filtered_files = filter_files(tracked_files_output)

    return [folder + "/" + file for file in filtered_files]


# We want to mimic
# git ls-files --other examples
def get_example_other_files(fd):
    folder = os.getcwd() + "/" + fd
    # Get the list of ignored files
    # Get the list of all not tracked files
    tracked_files_command = ["git", "ls-files", "--other", folder]
    tracked_files_output = subprocess.check_output(
        tracked_files_command, cwd=folder, text=True
    )

    return [folder + "/" + file for file in tracked_files_output.splitlines()]


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
            to_hash[key] = value

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


# ==================================================================================== #
#                              nox sessions definitions                                #
# ==================================================================================== #


for name in EXAMPLES:

    @nox.session(name=name, venv_backend="conda")
    def example(session, name=name):
        environment_yml = f"examples/{name}/environment.yml"
        if should_reinstall_dependencies(session, environment_yml=environment_yml):
            session.run(
                "conda",
                "env",
                "update",
                "--prune",
                f"--file={environment_yml}",
                f"--prefix={session.virtualenv.location}",
            )

            # install sphinx-gallery and its dependencies
            session.install(
                "sphinx-gallery",
                "sphinx",
                "pillow",
                "matplotlib",
                "chemiscope",
            )

        session.run("python", "generate-gallery.py", f"examples/{name}")
        os.unlink(f"docs/src/examples/{name}/index.rst")

        if "--no-build-docs" not in session.posargs:
            session.notify("build_docs")


@nox.session(venv_backend="none")
def docs(session):
    """Run all examples and build the documentation"""

    for example in EXAMPLES:
        session.run("nox", "-e", example, "--", "--no-build-docs", external=True)

    session.run("nox", "-e", "build_docs", external=True)


@nox.session
def build_docs(session):
    """Assemble the documentation into a website, assuming pre-generated examples"""

    requirements = "docs/requirements.txt"
    if should_reinstall_dependencies(session, requirements=requirements):
        session.install("-r", requirements)

    with open("docs/src/index.rst", "w") as output:
        with open("docs/src/index.rst.in") as fd:
            output.write(fd.read())

        output.write("\n")
        for file in glob.glob("docs/src/examples/*/*.rst"):
            if os.path.basename(file) != "sg_execution_times.rst":
                path = file[9:-4]

                output.write(f"   {path}\n")

                # TODO: Explain
                with open(file) as fd:
                    content = fd.read()

                if "Download Conda environment file" in content:
                    # do not add the download link twice
                    pass
                else:
                    lines = content.split("\n")
                    with open(file, "w") as fd:
                        for line in lines:
                            if "sphx-glr-download-jupyter" in line:
                                # add the new download link before
                                fd.write(
                                    """
    .. container:: sphx-glr-download

      :download:`Download Conda environment file: environment.yml <environment.yml>`
"""
                                )

                            fd.write(line)
                            fd.write("\n")

    session.run("sphinx-build", "-W", "-b", "html", "docs/src", "docs/build/html")


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


@nox.session
def format(session):
    """Automatically format all files"""

    if not session.virtualenv._reused:
        session.install("black", "blackdoc")
        session.install("isort")
    # Get files
    LINT_FILES = get_lint_files()

    session.run("black", *LINT_FILES)
    session.run("blackdoc", *LINT_FILES)
    session.run("isort", *LINT_FILES)


@nox.session
def clean_build(session):
    """Remove temporary files and building folders"""

    # remove building folders
    for i in ["docs/src/examples/", "docs/build"]:
        if os.path.isdir(i):
            shutil.rmtree(i)
    # remove temp files if any
    for ifile in get_example_other_files("examples") + get_example_other_files("docs/"):
        os.remove(ifile)
    flist = glob.glob("examples/*")
    # Remove empty folders
    for fl in flist:
        if 0 == len(glob.glob(fl + "/*")):
            os.rmdir(fl)
