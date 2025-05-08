#!/usr/bin/env python3
"""This script uses the Github API to get the ID of the latest
successful Documentation run on the main branch.

It then prints the ID"""

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import requests

from get_examples import get_examples


ROOT = Path(__file__).resolve().parent.parent

GITHUB_ACTIONS_API = "https://api.github.com/repos/lab-cosmo/atomistic-cookbook/actions"
NIGHTLY_LINK = "https://nightly.link/lab-cosmo/atomistic-cookbook/workflows/docs/main"


def get_latest_successful_docs_run():

    doc_runs_endpoint = GITHUB_ACTIONS_API + "/workflows/docs.yml/runs"

    response = requests.get(
        doc_runs_endpoint,
        params={
            "branch": "main",
            "per_page": 1,
            "status": "success",
            "exclude_pull_requests": True,
        },
    )

    if response.status_code != 200:
        print(f"Error [{response.status_code}]: {response.text}")
        return None

    latest_successful_run = response.json()["workflow_runs"][0]

    return latest_successful_run["id"]


def download_latest_examples(
    overwrite: bool = False,
    examples: Optional[list[str]] = None,
    exclude: list[str] = [],
):
    """Download built examples from latest Github's CI run on the main branch.

    It uses nightly.link instead of the Github API so that artifacts can be
    downloaded without logging in.
    """
    # If examples is not provided, get all examples
    if examples is None:
        examples = get_examples()

    # Remove examples that the user wants to exclude
    if len(exclude) > 0:
        examples = [example for example in examples if example not in exclude]

    # Iterate over examples and download each of them
    n_examples = len(examples)
    examples_dir = ROOT / "docs/src/examples"
    for i, example in enumerate(examples):

        # Skip examples that exist if overwrite is False
        if not overwrite and Path(examples_dir / example).exists():
            print(f"[{i + 1} / {n_examples}] Skipping {example} (already exists)")
            continue

        # Get the url to download the example artifact
        download_url = NIGHTLY_LINK + f"/example-{example}.zip"

        try:
            # Download the example artifact
            response = urllib.request.urlopen(download_url)

            print(f"[{i + 1} / {n_examples}] Downloaded {example}")

            # Extract the contents of the artifact into the examples directory
            myzip = zipfile.ZipFile(io.BytesIO(response.read()))
            myzip.extractall(examples_dir)
        except urllib.error.HTTPError as e:
            print(f"[{i + 1} / {n_examples}] Error downloading {example}: {e}")
            continue


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="CLI to interact with the latest successful docs run on Github Actions"
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "id",
        help="Get the ID of the latest successful docs run",
    )

    download_parser = subparsers.add_parser(
        "download-examples",
        help=download_latest_examples.__doc__,
    )

    download_parser.add_argument(
        "examples",
        default=None,
        nargs="*",
        help="List of examples to download. If None, all examples will be downloaded.",
    )

    download_parser.add_argument(
        "--exclude",
        default=[],
        nargs="*",
        help="Examples that should not be downloaded.",
    )

    download_parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing examples",
    )

    args = parser.parse_args()

    if args.command == "download-examples":
        download_latest_examples(
            overwrite=args.overwrite,
            examples=args.examples or None,
            exclude=args.exclude,
        )
    elif args.command == "id":
        print(get_latest_successful_docs_run())
