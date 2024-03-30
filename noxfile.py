import glob
import hashlib
import json
import os
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


# Files that need to be linted & formatted
LINT_FILES = [
    "ipynb-to-gallery.py",
    "generate-gallery.py",
    "noxfile.py",
    "docs/src/conf.py",
    "examples",
    "developer",
]

EXAMPLES = get_examples()

# ==================================================================================== #
#                                 helper functions                                     #
# ==================================================================================== #


def get_lint_files():
    LINT_FILES = [
        "ipynb-to-gallery.py",
        "generate-gallery.py",
        "noxfile.py",
        "docs/src/conf.py",
    ]
    return LINT_FILES


def get_command_output(command, cwd):
    """Execute a command and return its output."""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd
    )
    output, errors = process.communicate()
    if process.returncode == 0:
        return output
    else:
        raise RuntimeError(f"Command failed with errors: {errors}")


def filter_files(tracked_files, ignored_files):
    """Filter tracked files by removing any that appear in ignored_files."""
    returns = []
    for file in tracked_files.splitlines():
        tmp = file.split(".")[-1]
        if tmp != "xyz" and tmp != "sh" and tmp != "yml" and tmp != "cp2k":
            if (
                file.split("/")[-1] != ".gitignore"
                and file.split("/")[-1] != "README.rst"
            ):
                if file not in ignored_files.splitlines():
                    returns.append(file)

    return returns


# We want to mimic
# git ls-files examples |
# grep -v -x -f <(git ls-files --others --ignored --exclude-standard examples)
def get_list_of_ignored_files():
    folder = os.getcwd() + "/examples"
    # Get the list of ignored files
    ignored_files_command = [
        "git",
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        folder,
    ]
    ignored_files_output = get_command_output(ignored_files_command, cwd=folder)

    # Get the list of all tracked files
    tracked_files_command = ["git", "ls-files", folder]
    tracked_files_output = get_command_output(tracked_files_command, cwd=folder)

    # Filter the tracked files to exclude ignored ones
    filtered_files = filter_files(tracked_files_output, ignored_files_output)

    return [folder + "/" + file for file in filtered_files]


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
            session.install("sphinx-gallery", "sphinx", "pillow", "matplotlib")

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
    LINT_FILES = get_lint_files() + get_list_of_ignored_files()

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
    LINT_FILES = get_lint_files() + get_list_of_ignored_files()

    session.run("black", *LINT_FILES)
    session.run("blackdoc", *LINT_FILES)
    session.run("isort", *LINT_FILES)
