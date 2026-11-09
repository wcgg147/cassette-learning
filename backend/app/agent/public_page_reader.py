"""Bounded, unauthenticated public HTTPS page excerpts for tool turns.

DNS results are validated and pinned to the TLS connection. No cookies,
environment proxies, model calls, material imports or database writes occur.
"""
from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
import socket
import threading
import time
import zlib
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

_SLOTS = threading.BoundedSemaphore(2)
MAX_BYTES = 256_000
MAX_CHARS = 12_000


def _target(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in value):
        raise ValueError("invalid_url")
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.port not in (None, 443):
        raise ValueError("public_https_required")
    host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith((".localhost", ".local", ".internal")) or "%" in host:
        raise ValueError("non_public_host")
    authority = f"[{host}]" if ":" in host else host
    return urlunsplit(("https", authority, parts.path or "/", parts.query, "")), host


def _addresses(host: str):
    rows = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return _validate_addresses(rows)


def _validate_addresses(rows):
    if not rows:
        raise ValueError("dns_empty")
    for row in rows:
        ip = ipaddress.ip_address(row[4][0])
        transition = isinstance(ip, ipaddress.IPv6Address) and (
            ip.ipv4_mapped is not None or ip.sixtofour is not None or ip.teredo is not None
            or ip.is_site_local or ip in ipaddress.ip_network("64:ff9b::/96")
            or ip in ipaddress.ip_network("64:ff9b:1::/48"))
        if transition or not ip.is_global or any((ip.is_loopback, ip.is_link_local, ip.is_multicast, ip.is_reserved, ip.is_unspecified)):
            raise ValueError("non_public_address")
    return rows


def _doh_addresses(host: str, deadline: float):
    # Explicit opt-in for environments whose system DNS uses synthetic IPs.
    # This one bootstrap endpoint is fixed, TLS verified and never redirected;
    # model-supplied URLs cannot control it. Only the public hostname is sent.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError()
    bootstrap = socket.getaddrinfo("cloudflare-dns.com", 443, type=socket.SOCK_STREAM)
    connection = _PinnedHTTPSConnection("cloudflare-dns.com", bootstrap, deadline)
    watchdog = _watch_deadline(connection, deadline)
    response = None
    try:
        connection.request("GET", "/dns-query?" + urlencode({"name": host, "type": "A"}),
                           headers={"Accept": "application/dns-json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("dns_failed")
        raw = response.read(65_537)
        if len(raw) > 65_536:
            raise ValueError("dns_failed")
        result = json.loads(raw)
        if result.get("Status") != 0:
            raise ValueError("dns_failed")
        rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (row["data"], 443))
                for row in result.get("Answer") or [] if row.get("type") == 1]
        return _validate_addresses(rows)
    finally:
        watchdog.cancel()
        if response:
            response.close()
        connection.close()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, addresses, deadline):
        self._addresses = addresses
        self._deadline = deadline
        super().__init__(host, timeout=max(0.01, deadline - time.monotonic()))

    def connect(self):
        # Never resolve the hostname again after its public-address check.
        family, kind, proto, _, address = self._addresses[0]
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        sock = socket.socket(family, kind, proto)
        self._active_socket = sock
        try:
            sock.settimeout(remaining)
            sock.connect(address)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host, do_handshake_on_connect=False)
            self._active_socket = self.sock
            if time.monotonic() >= self._deadline:
                raise TimeoutError()
            self.sock.do_handshake()
        except BaseException:
            self._active_socket.close()
            raise


def _watch_deadline(connection, deadline):
    def abort():
        sock = getattr(connection, "_active_socket", None)
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
    timer = threading.Timer(max(0, deadline-time.monotonic()), abort)
    timer.daemon = True
    timer.start()
    return timer


def _read(url: str, deadline: float, resolver: str = "system") -> dict:
    if not _SLOTS.acquire(blocking=False):
        return {"status": "error", "error_code": "reader_busy"}
    try:
        requested, _ = _target(url)
        current = requested
        https_upgrades = 0
        for hop in range(3):
            if time.monotonic() >= deadline:
                raise TimeoutError()
            current, host = _target(current)
            addresses = _addresses(host) if resolver == "system" else _doh_addresses(host, deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError()
            connection = _PinnedHTTPSConnection(host, addresses, deadline)
            watchdog = _watch_deadline(connection, deadline)
            response = None
            try:
                parts = urlsplit(current)
                path = parts.path + ("?" + parts.query if parts.query else "")
                connection.request("GET", path, headers={
                    "User-Agent": "LanguageTutorPublicReader/1.0", "Accept": "text/html,text/plain",
                    "Accept-Encoding": "identity"})
                response = connection.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    if not location or hop == 2:
                        raise ValueError("redirect_limit")
                    target = urljoin(current, location)
                    redirect = urlsplit(target)
                    if (redirect.scheme == "http" and redirect.hostname == urlsplit(current).hostname
                            and redirect.port in (None, 80) and not redirect.username and not redirect.password):
                        host_part = f"[{redirect.hostname}]" if ":" in redirect.hostname else redirect.hostname
                        target = urlunsplit(("https", host_part, redirect.path, redirect.query, ""))
                        https_upgrades += 1
                    current = target
                    continue
                content_type = response.headers.get_content_type()
                if response.status != 200:
                    return {"status": "error", "error_code": "http_error", "http_status": response.status}
                if content_type not in ("text/html", "text/plain"):
                    raise ValueError("unsupported_content_type")
                encoding = response.getheader("Content-Encoding", "identity").lower()
                if encoding not in ("", "identity", "gzip", "deflate"):
                    raise ValueError("unsupported_content_encoding")
                chunks, size = [], 0
                while size <= MAX_BYTES:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError()
                    if connection.sock:
                        connection.sock.settimeout(remaining)
                    chunk = response.read1(min(16_384, MAX_BYTES + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("page_too_large")
                if time.monotonic() >= deadline:
                    raise TimeoutError()
                raw = b"".join(chunks)
                body_bytes = raw
                if encoding in ("gzip", "deflate"):
                    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
                    body_bytes = decoder.decompress(raw, MAX_BYTES + 1)
                    if len(body_bytes) > MAX_BYTES or decoder.unconsumed_tail:
                        raise ValueError("page_too_large")
                    if not decoder.eof or decoder.unused_data:
                        raise ValueError("invalid_content_encoding")
                charset = response.headers.get_content_charset() or "utf-8"
                try:
                    decoded = body_bytes.decode(charset, errors="replace")
                except LookupError:
                    decoded = body_bytes.decode("utf-8", errors="replace")
                # Reuse deterministic extraction only, never the material
                # fetcher's video, comment, transcription or storage routes.
                from engine.link_fetcher import strip_html, longest_text_block
                title, body = strip_html(decoded) if content_type == "text/html" else ("", decoded)
                text = longest_text_block(body) or body
                if not text.strip():
                    raise ValueError("empty_page")
                return {"status": "ok", "evidence_kind": "fetched_page_excerpt",
                    "requested_url": requested, "url": current, "title": title[:180], "title_truncated": len(title) > 180,
                    "http_status": response.status, "content_type": content_type,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "text": text[:MAX_CHARS], "text_truncated": len(text) > MAX_CHARS,
                    "response_bytes": size, "response_sha256": hashlib.sha256(raw).hexdigest(),
                    "content_encoding": encoding or "identity", "decoded_bytes": len(body_bytes),
                    "redirect_count": hop, "redirect_https_upgrades": https_upgrades, "resolver": resolver}
            finally:
                watchdog.cancel()
                if response:
                    response.close()
                connection.close()
    except (ValueError, UnicodeError) as exc:
        # Only our bounded, predefined validation codes enter evidence.
        code = str(exc)
        allowed = {"invalid_url", "public_https_required", "non_public_host", "dns_empty",
                   "non_public_address", "dns_failed", "redirect_limit", "unsupported_content_type",
                   "unsupported_content_encoding", "invalid_content_encoding", "page_too_large", "empty_page"}
        return {"status": "error", "error_code": code if code in allowed else "invalid_url"}
    except Exception as exc:
        return {"status": "error", "error_code": type(exc).__name__}
    finally:
        _SLOTS.release()


async def read_public_page(url: str, *, timeout: float = 6, resolver: str = "system") -> dict:
    if resolver not in ("system", "cloudflare"):
        return {"status": "error", "error_code": "invalid_resolver"}
    budget = max(0.1, min(10.0, float(timeout)))
    deadline = time.monotonic() + budget
    try:
        return await asyncio.wait_for(asyncio.to_thread(_read, url, deadline, resolver), budget)
    except (TimeoutError, asyncio.TimeoutError):
        # A DNS lookup may outlive cancellation; it retains its slot and checks
        # the deadline before any connection. Late results have no session owner.
        return {"status": "error", "error_code": "reader_timeout"}
