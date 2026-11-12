"""HTTP download with retries on transient errors."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import unquote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PathLike = Union[str, Path]

#: Environment variable pointing to a directory used as a shared download cache.
CACHE_ENV_VAR = "ATOMISTIC_COOKBOOK_CACHE"


def download_with_retry(
    url: str,
    destination: PathLike,
    *,
    retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    overwrite: bool = False,
    chunk_size: int = 1 << 16,
) -> Path:
    """Download ``url`` to ``destination`` with automatic retries.

    The destination's parent directories are created as needed. If the
    file already exists and ``overwrite`` is false (the default), the
    download is skipped and the existing path is returned.

    Retries follow urllib3's :class:`~urllib3.util.retry.Retry` with an
    exponential backoff. With the defaults (``retries=5``,
    ``backoff_factor=1``) the waits between attempts are roughly
    1, 2, 4, 8, 16 seconds. If the connection breaks while the file is
    being received, the download restarts from the beginning, with the
    same number of attempts and waits.

    If the ``ATOMISTIC_COOKBOOK_CACHE`` environment variable points to a
    directory, downloads go through a cache in that directory, keyed by
    URL: a file that was already downloaded (e.g. by another recipe) is
    copied from the cache instead of being fetched again. This is meant
    for CI, and is off when the variable is unset. Any problem with the
    cache directory falls back to a direct download, and
    ``overwrite=True`` always bypasses the cache.

    Parameters
    ----------
    url
        URL to fetch.
    destination
        Where to save the file.
    retries
        Maximum number of retry attempts on the status codes listed in
        ``status_forcelist`` (and on connect/read errors), and of restarts
        of a download interrupted while the file is being received.
    backoff_factor
        Exponential backoff factor passed to urllib3 ``Retry``.
    status_forcelist
        HTTP status codes that trigger a retry.
    overwrite
        If true, re-download even if the destination already exists.
    chunk_size
        Streaming chunk size in bytes.

    Returns
    -------
    pathlib.Path
        The resolved destination path.
    """
    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    def fetch(path: Path) -> None:
        _stream_to_file(
            url,
            path,
            retries=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            chunk_size=chunk_size,
        )

    cached = None if overwrite else _cache_path(url, destination)
    if cached is not None:
        try:
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                (cached.parent / "url.txt").write_text(url + "\n")
                fetch(cached)
            shutil.copyfile(cached, destination)
            return destination
        except requests.exceptions.RequestException:
            # requests exceptions derive from OSError: re-raise them here so
            # that only filesystem errors trigger the fallback below
            raise
        except OSError as error:
            warnings.warn(
                f"could not use the download cache in ${CACHE_ENV_VAR} "
                f"({error}), downloading {url} directly",
                stacklevel=2,
            )

    fetch(destination)
    return destination


def _cache_path(url: str, destination: Path) -> Optional[Path]:
    """Location of ``url`` in the download cache, or None if the cache is off."""
    cache_dir = os.environ.get(CACHE_ENV_VAR)
    if not cache_dir:
        return None

    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    name = Path(unquote(urlparse(url).path)).name or destination.name
    return Path(cache_dir) / key / name


def _stream_to_file(
    url: str,
    path: Path,
    *,
    retries: int,
    backoff_factor: float,
    status_forcelist: Tuple[int, ...],
    chunk_size: int,
) -> None:
    """Download ``url`` to ``path``.

    urllib3 retries the requests that fail before the content of the response
    is received. If the connection breaks while the content is received, the
    download is restarted from the beginning, up to ``retries`` times.
    """
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)

    with requests.Session() as session:
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        for attempt in range(retries + 1):
            with session.get(url, stream=True) as response:
                response.raise_for_status()
                try:
                    _save_content(response, path, chunk_size)
                    return
                except _BROKEN_TRANSFER as error:
                    if attempt == retries:
                        raise
                    reason = error

            delay = backoff_factor * 2**attempt
            warnings.warn(
                f"the download of {url} was interrupted ({reason}), "
                f"restarting it in {delay:g} s",
                stacklevel=4,
            )
            time.sleep(delay)


#: Errors raised when the connection breaks while the content of a response is
#: being received
_BROKEN_TRANSFER = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
)


def _save_content(response: requests.Response, path: Path, chunk_size: int) -> None:
    """Save the content of ``response`` to ``path`` through a ``.part`` file."""
    tmp_path = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
