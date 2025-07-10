#!/usr/bin/env python3
"""Script to create JSON string suitable for a github action matrix."""

import glob
import json
import os
from pathlib import Path


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(ROOT, "examples")


# Changes in the following files will trigger all examples to be run.
GLOBAL_FILES = ["noxfile.py"]
GLOBAL_FILES += glob.glob("src/**")
GLOBAL_FILES += glob.glob(".github/workflows/*")


def get_examples():
    """The current list of examples, determined from the directories on disk"""
    return [
        os.path.basename(os.path.normpath(file))
        for file in glob.glob(f"{EXAMPLES}/*")
        if os.path.isdir(file)
    ]


def get_examples_from_modified_files(modified_files: list[Path]):
    """Get only examples for which files have been modified.

    Parameters
    ----------
    modified_files :
        List of modified files.

        This can be obtained from something like ``git diff --name-only``.
    """
    modified_files = [file.resolve().relative_to(ROOT) for file in modified_files]

    if any(str(file) in GLOBAL_FILES for file in modified_files):
        # This means that there is a modified file that could affect all examples.
        return get_examples()

    def is_inside_example(file: Path) -> bool:
        """Check if a file is inside an example."""
        # The path should start with "examples", and the second part should be
        # the example directory (not a file).
        return (
            file.parts[0] == "examples" and Path(file.parts[0], file.parts[1]).is_dir()
        )

    return list(
        set(file.parts[1] for file in modified_files if is_inside_example(file))
    )


def create_ci_jobs_json(modified_files: str):
    """A JSON string suitable for a github action matrix."""
    if modified_files:
        modified_files = map(Path, modified_files)
        examples = get_examples_from_modified_files(modified_files)
    else:
        examples = get_examples()

    if len(examples) == 0:
        # We return an empty string which will be easily detected
        # by the Github Workflow as having no examples to run.
        return ""

    return json.dumps({"example-name": list(sorted(examples))})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print a JSON string with the list of examples."
    )
    parser.add_argument(
        "--modified-files",
        type=str,
        nargs="*",
        help="""List of modified files.
        If provided, only examples with modified files will be included.""",
        default=None,
    )

    print(create_ci_jobs_json(parser.parse_args().modified_files))
