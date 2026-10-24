
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB cap per download


def safe_local_path(source: str) -> str:
    """Resolve a local path (following symlinks) and make sure it stays in READER_BASE_DIR."""
    path = os.path.realpath(source)
    base = os.environ.get("READER_BASE_DIR")
    if base:
        base_real = os.path.realpath(base)
        if os.path.commonpath([path, base_real]) != base_real:
            raise PermissionError("Access denied: file is outside the allowed folder.")
    return path


def _check_host(host: str) -> None:
    if os.environ.get("ALLOW_PRIVATE_URLS") == "1":
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ValueError(f"Could not resolve host: {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                or ip.is_multicast or ip.is_unspecified):
            raise PermissionError("Access denied: URL points to a private/internal address.")


def safe_get(url: str, headers=None, timeout: int = 20, max_redirects: int = 5,
             max_bytes: int = DEFAULT_MAX_BYTES) -> requests.Response:
    """GET with scheme check, private-IP block, per-hop redirect check and a size cap."""
    for _ in range(max_redirects + 1):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Only http/https URLs are allowed.")
        _check_host(parsed.hostname)

        resp = requests.get(url, headers=headers, timeout=timeout,
                            allow_redirects=False, stream=True)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            url = urljoin(url, resp.headers["Location"])
            resp.close()
            continue

        chunks, total = [], 0
        for chunk in resp.iter_content(65536):
            total += len(chunk)
            if total > max_bytes:
                resp.close()
                raise ValueError(f"Response larger than {max_bytes // (1024 * 1024)} MB.")
            chunks.append(chunk)
        resp._content = b"".join(chunks)
        return resp
    raise ValueError("Too many redirects.")
