#!/usr/bin/env python3
"""Maintain the download cache shared by the recipes on CI.

All the matrix jobs of the documentation workflow restore the same cache
directory, which contains the files downloaded by the recipes themselves: the
HuggingFace and shiftml caches (under ``xdg/``, used as ``XDG_CACHE_HOME``) and
the files fetched by ``atomistic_cookbook_utils.download_with_retry``
(``downloads/``). Each job then archives the files it downloaded on top of the
restored cache, and a final job merges these archives into a new cache.

conda and pip packages are not cached: they are downloaded from fast and
reliable CDNs, while every job would have to restore the packages of all the
recipes. Files larger than a maximum size are not cached either: they are
typically used by a single recipe, and keeping them would slow down the
restoring of the cache in every job to save a few requests.

Subcommands:

- ``prune``: removes files older than the maximum age (the date of a file is
  the time it was downloaded) and any file that should not be cached, then the
  largest files until the cache fits in the size budget. Removing the largest
  files first keeps as many files as possible in the cache: rate limits apply
  to the number of requests rather than to their size. Optionally creates a
  marker file, used by ``collect`` to recognize new files.
- ``collect``: archives the cacheable files newer than the marker.
- ``merge``: extracts archives created by ``collect`` into the cache.
"""

import argparse
import math
import os
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


# Directories (relative to the cache root) whose content is cached
HF_HUB = ("xdg", "huggingface", "hub")
CACHED_TREES = [HF_HUB, ("xdg", "shiftml"), ("downloads",)]

# In the HuggingFace cache, the files in `snapshots/` are links to the downloaded
# files. Regular files in there are created by other tools (e.g. models exported
# by flashmd), and are not cached.
HF_SNAPSHOTS = "snapshots"

# Unfinished downloads and lock files
EXCLUDED_SUFFIXES = (".part", ".partial", ".incomplete", ".lock")
EXCLUDED_DIRS = {".locks"}


def is_cacheable(relpath: PurePosixPath, is_symlink: bool) -> bool:
    """Whether a file (given relative to the cache root) belongs in the cache"""
    parts = relpath.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if relpath.name.endswith(EXCLUDED_SUFFIXES):
        return False

    if parts[: len(HF_HUB)] == HF_HUB and HF_SNAPSHOTS in parts and not is_symlink:
        return False

    return any(
        len(parts) > len(tree) and parts[: len(tree)] == tree for tree in CACHED_TREES
    )


def max_file_size(args) -> float:
    if args.max_file_size_mb is None:
        return math.inf
    return args.max_file_size_mb * 1e6


def walk_files(root: Path):
    """Yield ``(path, relpath, lstat)`` for all files and symlinks under root"""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath, filename)
            yield path, PurePosixPath(path.relative_to(root).as_posix()), path.lstat()

        # symlinks to directories are listed in `dirnames` and not followed: the
        # cached tools do not create any


def set_github_output(name: str, value: str):
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as fd:
            fd.write(f"{name}={value}\n")


def human_size(size: int) -> str:
    for unit in ["B", "kB", "MB", "GB"]:
        if size < 1000 or unit == "GB":
            break
        size /= 1000
    return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"


def report(root: Path, summary: bool):
    sizes = {category: [0, 0] for category in CACHED_TREES}
    other = [0, 0]
    for _, relpath, stat in walk_files(root):
        for category in CACHED_TREES:
            if relpath.parts[: len(category)] == category:
                break
        else:
            category = None
        entry = sizes[category] if category is not None else other
        entry[0] += 1
        entry[1] += stat.st_size

    rows = [("/".join(category), *sizes[category]) for category in CACHED_TREES]
    rows.append(("other", *other))
    total = sum(row[2] for row in rows)
    rows.append(("total", sum(row[1] for row in rows), total))

    print(f"\ncontent of the download cache in {root}:")
    for name, count, size in rows:
        print(f"    {name:<18} {count:>7} files {human_size(size):>10}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and step_summary:
        with open(step_summary, "a") as fd:
            fd.write("### Download cache\n\n| | files | size |\n|---|---:|---:|\n")
            for name, count, size in rows:
                fd.write(f"| {name} | {count} | {human_size(size)} |\n")
            fd.write("\n")


def remove_empty_dirs(root: Path):
    for dirpath, _, _ in os.walk(root, topdown=False):
        if Path(dirpath) != root:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass  # not empty


def prune(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    now = time.time()
    max_age = args.max_age_days * 86400
    budget = None if args.max_size_gb is None else args.max_size_gb * 1e9
    max_size = max_file_size(args)

    # number of files and total size removed for each reason
    reasons = ["expired", "not cacheable", "too large", "over budget", "dangling"]
    removed = {reason: [0, 0] for reason in reasons}

    def remove(path, size, reason):
        path.unlink()
        removed[reason][0] += 1
        removed[reason][1] += size

    kept = []
    for path, relpath, stat in walk_files(root):
        if not is_cacheable(relpath, path.is_symlink()):
            remove(path, stat.st_size, "not cacheable")
        elif stat.st_size > max_size:
            remove(path, stat.st_size, "too large")
        elif now - stat.st_mtime > max_age:
            remove(path, stat.st_size, "expired")
        elif not path.is_symlink():
            kept.append((stat.st_size, stat.st_mtime, path))

    if budget is not None:
        total = sum(size for size, _, _ in kept)
        # largest first, and the oldest first among files of the same size
        for size, _, path in sorted(kept, key=lambda entry: (-entry[0], entry[1])):
            if total <= budget:
                break
            remove(path, size, "over budget")
            total -= size

    # dangling links (e.g. HuggingFace snapshots pointing to a pruned blob)
    for path, _, stat in walk_files(root):
        if path.is_symlink() and not path.exists():
            remove(path, stat.st_size, "dangling")

    remove_empty_dirs(root)

    print(f"removed from the download cache in {root}:")
    for reason, (count, size) in removed.items():
        print(f"    {reason:<18} {count:>7} files {human_size(size):>10}")

    report(root, args.summary)
    n_removed = sum(count for count, _ in removed.values())
    set_github_output("pruned", "true" if n_removed > 0 else "false")

    if args.marker is not None:
        marker = Path(args.marker)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        # make sure files written from now on are strictly newer than the marker
        time.sleep(0.01)


def collect(args):
    root = Path(args.root)
    output = Path(args.output)
    marker = Path(args.marker)
    if not marker.exists():
        # `prune` creates the marker, nothing to collect without it
        print(f"{marker} does not exist, not collecting anything")
        return
    marker_mtime = marker.stat().st_mtime_ns
    max_size = max_file_size(args)

    new_files = []
    size = 0
    if root.exists():
        for path, relpath, stat in walk_files(root):
            if (
                stat.st_mtime_ns > marker_mtime
                and is_cacheable(relpath, path.is_symlink())
                and stat.st_size <= max_size
            ):
                new_files.append(relpath)
                size += stat.st_size

    if not new_files:
        print("no new files in the download cache")
        output.unlink(missing_ok=True)
        return

    print(f"archiving {len(new_files)} new files ({human_size(size)}) to {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # no compression: most of the content is already compressed
    with tarfile.open(output, "w") as tar:
        for relpath in sorted(new_files):
            tar.add(root / relpath, arcname=str(relpath), recursive=False)


def find_archives(paths):
    archives = []
    for path in map(Path, paths):
        if path.is_dir():
            archives.extend(sorted(path.rglob("*.tar")))
        elif path.exists():
            archives.append(path)
    return archives


def merge(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    extract_filter = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    extracted = 0
    for archive in find_archives(args.archives):
        with tarfile.open(archive) as tar:
            members = [
                member
                for member in tar.getmembers()
                if (member.isfile() or member.issym())
                and is_cacheable(PurePosixPath(member.name), member.issym())
            ]

            print(f"extracting {len(members)} files from {archive}")
            tar.extractall(root, members=members, **extract_filter)
            extracted += len(members)

    set_github_output("merged", "true" if extracted > 0 else "false")


def add_max_file_size_argument(parser):
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        help="do not cache files larger than this",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prune_parser = subparsers.add_parser("prune", help="remove outdated files")
    prune_parser.add_argument("--root", required=True, help="cache directory")
    prune_parser.add_argument(
        "--max-age-days",
        type=float,
        required=True,
        help="remove files downloaded more than this number of days ago",
    )
    prune_parser.add_argument(
        "--max-size-gb",
        type=float,
        help="remove the largest files until the cache is smaller than this",
    )
    add_max_file_size_argument(prune_parser)
    prune_parser.add_argument(
        "--marker", help="create this file once the cache has been pruned"
    )
    prune_parser.add_argument(
        "--summary",
        action="store_true",
        help="add the size of the cache to the GitHub step summary",
    )
    prune_parser.set_defaults(function=prune)

    collect_parser = subparsers.add_parser(
        "collect", help="archive files added to the cache"
    )
    collect_parser.add_argument("--root", required=True, help="cache directory")
    collect_parser.add_argument(
        "--marker", required=True, help="marker file created by `prune`"
    )
    collect_parser.add_argument(
        "--output", required=True, help="archive to create (if there are new files)"
    )
    add_max_file_size_argument(collect_parser)
    collect_parser.set_defaults(function=collect)

    merge_parser = subparsers.add_parser("merge", help="merge archives in the cache")
    merge_parser.add_argument("--root", required=True, help="cache directory")
    merge_parser.add_argument(
        "archives",
        nargs="*",
        help="archives created by `collect`, or directories containing them",
    )
    merge_parser.set_defaults(function=merge)

    args = parser.parse_args()
    sys.exit(args.function(args))
