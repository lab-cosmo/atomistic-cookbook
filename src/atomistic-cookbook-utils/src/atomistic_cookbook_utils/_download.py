"""HTTP download with retries on transient errors."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PathLike = Union[str, Path]


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
    1, 2, 4, 8, 16 seconds.

    Parameters
    ----------
    url
        URL to fetch.
    destination
        Where to save the file.
    retries
        Maximum number of retry attempts on the status codes listed in
        ``status_forcelist`` (and on connect/read errors).
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
        with session.get(url, stream=True) as response:
            response.raise_for_status()
            tmp_path = destination.with_suffix(destination.suffix + ".part")
            try:
                with open(tmp_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                tmp_path.replace(destination)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

    return destination
