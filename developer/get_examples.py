#!/usr/bin/env python3
"""Script to create JSON string suitable for a github action matrix."""
import glob
import json
import os


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(ROOT, "examples")


def get_examples():
    """The current list of examples, determined from the directories on disk"""
    return [
        os.path.basename(os.path.normpath(file))
        for file in glob.glob(f"{EXAMPLES}/*")
        if os.path.isdir(file) and os.path.basename(file) != "_chemiscope_sphinx_data"
    ]


def create_json():
    """A JSON string suitable for a github action matrix."""
    return json.dumps({"example-name": get_examples()})


if __name__ == "__main__":
    print(create_json())
