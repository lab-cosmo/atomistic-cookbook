"""Tests for ``download_with_retry``."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from atomistic_cookbook_utils import download_with_retry


PAYLOAD = b"hello cookbook\n" * 1000


def _make_handler(fail_first_n: int, payload: bytes = PAYLOAD):
    state = {"calls": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # silence stderr noise
            pass

        def do_GET(self):
            state["calls"] += 1
            if state["calls"] <= fail_first_n:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler, state


@pytest.fixture
def http_server():
    servers = []

    def _start(fail_first_n: int):
        handler, state = _make_handler(fail_first_n)
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        host, port = server.server_address
        return f"http://{host}:{port}/file.bin", state

    yield _start

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_download_succeeds(http_server, tmp_path):
    url, state = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"

    result = download_with_retry(url, dest)

    assert result == dest
    assert dest.read_bytes() == PAYLOAD
    assert state["calls"] == 1


def test_download_retries_then_succeeds(http_server, tmp_path):
    url, state = http_server(fail_first_n=2)
    dest = tmp_path / "file.bin"

    download_with_retry(url, dest, retries=5, backoff_factor=0.0)

    assert dest.read_bytes() == PAYLOAD
    assert state["calls"] == 3  # two 503s then a 200


def test_download_skips_if_present(http_server, tmp_path):
    url, state = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"pre-existing")

    download_with_retry(url, dest)

    assert dest.read_bytes() == b"pre-existing"
    assert state["calls"] == 0


def test_download_overwrites_when_requested(http_server, tmp_path):
    url, _ = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"pre-existing")

    download_with_retry(url, dest, overwrite=True)

    assert dest.read_bytes() == PAYLOAD


def test_download_creates_parent_dirs(http_server, tmp_path):
    url, _ = http_server(fail_first_n=0)
    dest = tmp_path / "nested" / "deeper" / "file.bin"

    download_with_retry(url, dest)

    assert dest.read_bytes() == PAYLOAD


def test_download_exhausts_retries(http_server, tmp_path):
    url, _ = http_server(fail_first_n=10)
    dest = tmp_path / "file.bin"

    with pytest.raises(requests.exceptions.RetryError):
        download_with_retry(url, dest, retries=2, backoff_factor=0.0)

    assert not dest.exists()
    # The .part file should also have been cleaned up.
    assert not dest.with_suffix(dest.suffix + ".part").exists()


# ---------------------------------------------------------------------------- #
#                      shared cache ($ATOMISTIC_COOKBOOK_CACHE)                 #
# ---------------------------------------------------------------------------- #


def test_cache_is_off_by_default(http_server, tmp_path, monkeypatch):
    monkeypatch.delenv("ATOMISTIC_COOKBOOK_CACHE", raising=False)
    url, state = http_server(fail_first_n=0)

    download_with_retry(url, tmp_path / "a" / "file.bin")
    download_with_retry(url, tmp_path / "b" / "file.bin")

    assert state["calls"] == 2


def test_cache_shared_between_destinations(http_server, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("ATOMISTIC_COOKBOOK_CACHE", str(cache))
    url, state = http_server(fail_first_n=0)

    first = tmp_path / "recipe-a" / "file.bin"
    second = tmp_path / "recipe-b" / "renamed.dat"
    download_with_retry(url, first)
    download_with_retry(url, second)

    assert first.read_bytes() == PAYLOAD
    assert second.read_bytes() == PAYLOAD
    assert state["calls"] == 1

    # the cache holds one copy, named after the URL, plus the URL itself
    (entry,) = cache.iterdir()
    assert (entry / "file.bin").read_bytes() == PAYLOAD
    assert (entry / "url.txt").read_text().strip() == url
    assert not list(cache.rglob("*.part"))

    # the destination is a copy: modifying it leaves the cache intact
    second.write_bytes(b"modified")
    assert (entry / "file.bin").read_bytes() == PAYLOAD


def test_cache_bypassed_by_overwrite(http_server, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("ATOMISTIC_COOKBOOK_CACHE", str(cache))
    url, state = http_server(fail_first_n=0)

    download_with_retry(url, tmp_path / "a.bin")
    download_with_retry(url, tmp_path / "b.bin", overwrite=True)

    assert (tmp_path / "b.bin").read_bytes() == PAYLOAD
    assert state["calls"] == 2


def test_cache_falls_back_when_unusable(http_server, tmp_path, monkeypatch):
    # the "cache directory" is a regular file, so it cannot be created
    not_a_dir = tmp_path / "cache"
    not_a_dir.write_text("")
    monkeypatch.setenv("ATOMISTIC_COOKBOOK_CACHE", str(not_a_dir))
    url, state = http_server(fail_first_n=0)
    dest = tmp_path / "file.bin"

    with pytest.warns(UserWarning, match="download cache"):
        download_with_retry(url, dest)

    assert dest.read_bytes() == PAYLOAD
    assert state["calls"] == 1


def test_cache_does_not_mask_http_errors(http_server, tmp_path, monkeypatch):
    monkeypatch.setenv("ATOMISTIC_COOKBOOK_CACHE", str(tmp_path / "cache"))
    url, state = http_server(fail_first_n=10)
    dest = tmp_path / "file.bin"

    with pytest.raises(requests.exceptions.RetryError):
        download_with_retry(url, dest, retries=2, backoff_factor=0.0)

    # the HTTP error is raised after one round of retries, without falling
    # back to a direct download
    assert state["calls"] == 3
    assert not dest.exists()
    assert not list((tmp_path / "cache").rglob("*.bin*"))
