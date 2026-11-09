#!/usr/bin/env python3
"""Maintain the download cache shared by the recipes on CI.

All the matrix jobs of the documentation workflow restore the same cache
directory, which contains the conda package cache (``conda/``), the pip,
HuggingFace and shiftml caches (under ``xdg/``, used as ``XDG_CACHE_HOME``)
and the files fetched by ``atomistic_cookbook_utils.download_with_retry``
(``downloads/``). Each job then archives the files it downloaded on top of
the restored cache, and a final job merges these archives into a new cache.

Packages are always cached, while the files downloaded by the recipes
themselves (models, datasets) are only cached up to a maximum size: large files
are typically used by a single recipe, and keeping them would slow down the
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
import os
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath


# Directories (relative to the cache root) containing the files downloaded by the
# recipes, which are only cached up to a maximum size
HF_HUB = ("xdg", "huggingface", "hub")
RECIPE_DOWNLOADS = [HF_HUB, ("xdg", "shiftml"), ("downloads",)]

# Directories (relative to the cache root) whose whole content is cached
CACHED_TREES = [("xdg", "pip"), *RECIPE_DOWNLOADS]

# In the HuggingFace cache, the files in `snapshots/` are links to the downloaded
# files. Regular files in there are created by other tools (e.g. models exported
# by flashmd), and are not cached.
HF_SNAPSHOTS = "snapshots"

# conda keeps the downloaded archives and the extracted packages in the same
# directory: only the archives are cached, together with `urls.txt`, which conda
# needs to know which channel an archive comes from.
CONDA_DIR = "conda"
CONDA_URLS = PurePosixPath(CONDA_DIR, "urls.txt")
CONDA_ARCHIVES = (".conda", ".tar.bz2")

# pip stores each HTTP response in two files, `<key>` with the headers and
# `<key>.body` with the content, and only rewrites the headers when it revalidates
# a cached response. The two files are archived together or not at all, so that
# merging the archives of different jobs never pairs the headers of a response
# with the content of another version of it.
PIP_HTTP_CACHE = ("xdg", "pip", "http-v2")
PIP_BODY_SUFFIX = ".body"

# Unfinished downloads and lock files
EXCLUDED_SUFFIXES = (".part", ".partial", ".incomplete", ".lock")
EXCLUDED_DIRS = {".locks"}

# categories used when reporting the size of the cache
REPORT_CATEGORIES = [
    ("conda",),
    ("xdg", "pip"),
    ("xdg", "huggingface"),
    ("xdg", "shiftml"),
    ("downloads",),
]


def in_trees(relpath: PurePosixPath, trees) -> bool:
    """Whether a path is inside one of the given directories"""
    parts = relpath.parts
    return any(len(parts) > len(tree) and parts[: len(tree)] == tree for tree in trees)


def is_cacheable(relpath: PurePosixPath, is_symlink: bool) -> bool:
    """Whether a file (given relative to the cache root) belongs in the cache"""
    parts = relpath.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if relpath.name.endswith(EXCLUDED_SUFFIXES):
        return False

    if len(parts) == 2 and parts[0] == CONDA_DIR:
        return relpath == CONDA_URLS or relpath.name.endswith(CONDA_ARCHIVES)

    if parts[: len(HF_HUB)] == HF_HUB and HF_SNAPSHOTS in parts and not is_symlink:
        return False

    return in_trees(relpath, CACHED_TREES)


def is_too_large(relpath: PurePosixPath, size: int, max_size) -> bool:
    """Whether a file downloaded by a recipe is larger than the maximum size"""
    return (
        max_size is not None and size > max_size and in_trees(relpath, RECIPE_DOWNLOADS)
    )


def max_file_size(args):
    return None if args.max_file_size_mb is None else args.max_file_size_mb * 1e6


def pip_partner(relpath: PurePosixPath):
    """The other file of a pip HTTP cache entry, or None for other files"""
    if relpath.parts[: len(PIP_HTTP_CACHE)] != PIP_HTTP_CACHE:
        return None
    if relpath.name.endswith(PIP_BODY_SUFFIX):
        return relpath.with_name(relpath.name[: -len(PIP_BODY_SUFFIX)])
    return relpath.with_name(relpath.name + PIP_BODY_SUFFIX)


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
    sizes = {category: [0, 0] for category in REPORT_CATEGORIES}
    other = [0, 0]
    for _, relpath, stat in walk_files(root):
        for category in REPORT_CATEGORIES:
            if relpath.parts[: len(category)] == category:
                break
        else:
            category = None
        entry = sizes[category] if category is not None else other
        entry[0] += 1
        entry[1] += stat.st_size

    rows = [("/".join(category), *sizes[category]) for category in REPORT_CATEGORIES]
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
        elif is_too_large(relpath, stat.st_size, max_size):
            remove(path, stat.st_size, "too large")
        elif relpath != CONDA_URLS and now - stat.st_mtime > max_age:
            remove(path, stat.st_size, "expired")
        elif not path.is_symlink() and relpath != CONDA_URLS:
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

    # keep only the lines of urls.txt for archives present in the cache
    urls_txt = root / CONDA_URLS
    if urls_txt.exists():
        conda_dir = root / CONDA_DIR
        urls = urls_txt.read_text().splitlines()
        present = [url for url in urls if (conda_dir / url.split("/")[-1]).exists()]
        if len(present) != len(urls):
            urls_txt.write_text("".join(f"{url}\n" for url in present))

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

    # size of the new files, indexed by path relative to the root
    new_files = {}
    if root.exists():
        for path, relpath, stat in walk_files(root):
            if (
                stat.st_mtime_ns > marker_mtime
                and is_cacheable(relpath, path.is_symlink())
                and not is_too_large(relpath, stat.st_size, max_size)
            ):
                new_files[relpath] = stat.st_size

    # pip cache entries where only one of the files is new
    for relpath in list(new_files):
        partner = pip_partner(relpath)
        if partner is not None and partner not in new_files:
            del new_files[relpath]

    size = sum(new_files.values())
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

    urls_txt = root / CONDA_URLS
    urls = urls_txt.read_text().splitlines() if urls_txt.exists() else []
    known_urls = set(urls)

    extract_filter = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    extracted = 0
    for archive in find_archives(args.archives):
        with tarfile.open(archive) as tar:
            members = []
            for member in tar.getmembers():
                relpath = PurePosixPath(member.name)
                if relpath == CONDA_URLS:
                    # merge the list of URLs instead of overwriting it
                    content = tar.extractfile(member).read().decode("utf8")
                    for url in content.splitlines():
                        if url and url not in known_urls:
                            urls.append(url)
                            known_urls.add(url)
                elif (member.isfile() or member.issym()) and is_cacheable(
                    relpath, member.issym()
                ):
                    members.append(member)

            print(f"extracting {len(members)} files from {archive}")
            tar.extractall(root, members=members, **extract_filter)
            extracted += len(members)

    if urls:
        urls_txt.parent.mkdir(parents=True, exist_ok=True)
        urls_txt.write_text("".join(f"{url}\n" for url in urls))

    set_github_output("merged", "true" if extracted > 0 else "false")


def add_max_file_size_argument(parser):
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        help="do not cache files downloaded by the recipes that are larger than "
        "this (packages are always cached)",
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
