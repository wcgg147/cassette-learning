import asyncio
from email.message import Message
import io
import gzip
import socket
import time
import threading
import unittest
from unittest.mock import patch, MagicMock

from agent import public_page_reader as reader


def addresses(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


class Response:
    def __init__(self, body=b"", status=200, location=None, content_type="text/html"):
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if location:
            self.headers["Location"] = location
        self._body = io.BytesIO(body)
    def getheader(self, name, default=None):
        return self.headers.get(name, default)
    def read1(self, size):
        return self._body.read(size)
    def close(self):
        self._body.close()


class PublicPageReaderTests(unittest.TestCase):
    def test_rejects_unsafe_urls_before_dns(self):
        for url in ("http://example.com", "https://user:secret@example.com/", "https://example.com:8443/",
                    "https://localhost/", "https://host.local/", "https://example.com/\nHeader: x"):
            with self.subTest(url=url), patch.object(reader.socket, "getaddrinfo") as dns:
                result = asyncio.run(reader.read_public_page(url))
                self.assertEqual(result["status"], "error")
                dns.assert_not_called()

    def test_rejects_private_and_mixed_dns_answers(self):
        for rows in (addresses("127.0.0.1"), addresses("10.0.0.1"), addresses("169.254.169.254"),
                     addresses("224.0.0.1"), addresses("8.8.8.8") + addresses("192.168.1.1")):
            with self.subTest(rows=rows), patch.object(reader.socket, "getaddrinfo", return_value=rows), \
                    patch.object(reader, "_PinnedHTTPSConnection") as connect:
                self.assertEqual(asyncio.run(reader.read_public_page("https://example.com"))["error_code"], "non_public_address")
                connect.assert_not_called()

    def test_connection_pins_resolved_ip_and_keeps_tls_hostname(self):
        sock, tls = MagicMock(), MagicMock()
        context = MagicMock()
        context.wrap_socket.return_value = tls
        connection = reader._PinnedHTTPSConnection("example.com", addresses("8.8.8.8"), time.monotonic()+1)
        connection._context = context
        with patch.object(reader.socket, "socket", return_value=sock), patch.object(reader.socket, "getaddrinfo") as dns:
            connection.connect()
        sock.connect.assert_called_once_with(("8.8.8.8", 443))
        context.wrap_socket.assert_called_once_with(sock, server_hostname="example.com", do_handshake_on_connect=False)
        tls.do_handshake.assert_called_once()
        dns.assert_not_called()
        connection.close()

    def test_redirect_is_revalidated_before_connection(self):
        with patch.object(reader.socket, "getaddrinfo", side_effect=[addresses("8.8.8.8"), addresses("127.0.0.1")]), \
                patch.object(reader, "_PinnedHTTPSConnection") as connect:
            connect.return_value.getresponse.return_value = Response(status=302, location="https://private.example/")
            result = asyncio.run(reader.read_public_page("https://example.com/"))
            self.assertEqual(result["error_code"], "non_public_address")
            self.assertEqual(connect.call_count, 1)
            connect.return_value.close.assert_called_once()

    def test_same_host_http_redirect_uses_secure_equivalent_only(self):
        with patch.object(reader.socket, "getaddrinfo", return_value=addresses("8.8.8.8")), \
                patch.object(reader, "_PinnedHTTPSConnection") as connect:
            connect.return_value.getresponse.side_effect = [
                Response(status=301, location="http://example.com/downloads/"),
                Response(b"<p>Actual public page content.</p>")]
            result = asyncio.run(reader.read_public_page("https://example.com/downloads"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["url"], "https://example.com/downloads/")
        self.assertEqual(result["redirect_https_upgrades"], 1)
        self.assertEqual(connect.call_count, 2)
        self.assertTrue(all(call.args[0] == "example.com" for call in connect.call_args_list))

    def test_page_metadata_and_excerpt_are_actual_extraction(self):
        body = b"<html><title>Public fixture</title><script>hidden()</script><p>" + b"Actual public content. "*40 + b"</p></html>"
        with patch.object(reader.socket, "getaddrinfo", return_value=addresses("8.8.8.8")), \
                patch.object(reader, "_PinnedHTTPSConnection") as connect:
            connect.return_value.getresponse.return_value = Response(body)
            result = asyncio.run(reader.read_public_page("https://example.com/"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evidence_kind"], "fetched_page_excerpt")
        self.assertEqual(result["url"], "https://example.com/")
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["content_type"], "text/html")
        self.assertNotIn("hidden()", result["text"])
        self.assertTrue(result["response_sha256"])

    def test_page_title_is_bounded_before_leaving_reader(self):
        body = b"<html><title>" + b"Long title "*1000 + b"</title><p>Public page text.</p></html>"
        with patch.object(reader.socket, "getaddrinfo", return_value=addresses("8.8.8.8")), \
                patch.object(reader, "_PinnedHTTPSConnection") as connect:
            connect.return_value.getresponse.return_value = Response(body)
            result = asyncio.run(reader.read_public_page("https://example.com/"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["title"]), 180)
        self.assertTrue(result["title_truncated"])

    def test_oversize_and_nontext_are_explicit_errors(self):
        for response, expected in ((Response(b"x"*(reader.MAX_BYTES+1)), "page_too_large"),
                                    (Response(content_type="application/pdf"), "unsupported_content_type")):
            with patch.object(reader.socket, "getaddrinfo", return_value=addresses("8.8.8.8")), \
                    patch.object(reader, "_PinnedHTTPSConnection") as connect:
                connect.return_value.getresponse.return_value = response
                result = asyncio.run(reader.read_public_page("https://example.com/"))
                self.assertEqual(result["error_code"], expected)

    def test_capacity_is_bounded(self):
        with patch.object(reader, "_SLOTS") as slots:
            slots.acquire.return_value = False
            self.assertEqual(asyncio.run(reader.read_public_page("https://example.com"))["error_code"], "reader_busy")
            slots.release.assert_not_called()

    def test_gzip_output_is_bounded_after_decompression(self):
        for body, expected in ((b"<p>Public page text.</p>", "ok"), (b"x"*(reader.MAX_BYTES+1), "error")):
            response = Response(gzip.compress(body))
            response.headers["Content-Encoding"] = "gzip"
            with patch.object(reader.socket, "getaddrinfo", return_value=addresses("8.8.8.8")), \
                    patch.object(reader, "_PinnedHTTPSConnection") as connect:
                connect.return_value.getresponse.return_value = response
                result = asyncio.run(reader.read_public_page("https://example.com/"))
            self.assertEqual(result["status"], expected)
            if expected == "error":
                self.assertEqual(result["error_code"], "page_too_large")

    def test_doh_is_explicit_and_still_rejects_private_answers(self):
        import json
        response = Response(json.dumps({"Status": 0, "Answer": [{"type": 1, "data": "127.0.0.1"}]}).encode())
        response.read = response.read1
        with patch.object(reader.socket, "getaddrinfo", return_value=addresses("198.18.0.1")), patch.object(reader, "_PinnedHTTPSConnection") as resolver:
            resolver.return_value.getresponse.return_value = response
            result = asyncio.run(reader.read_public_page("https://example.com/", resolver="cloudflare"))
        self.assertEqual(result["error_code"], "non_public_address")
        self.assertEqual(resolver.call_count, 1)  # fixed TLS bootstrap only
        self.assertEqual(resolver.call_args.args[0], "cloudflare-dns.com")

    def test_rejects_ipv6_transition_addresses(self):
        for ip in ("2002:7f00:1::", "::ffff:127.0.0.1", "64:ff9b::7f00:1", "fec0::1"):
            with self.subTest(ip=ip), self.assertRaises(ValueError):
                reader._validate_addresses([(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip,443,0,0))])

    def test_absolute_deadline_aborts_an_active_socket(self):
        connection = MagicMock()
        timer = reader._watch_deadline(connection, time.monotonic()+0.02)
        timer.join(timeout=1)
        self.assertFalse(timer.is_alive())
        connection._active_socket.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        connection._active_socket.close.assert_called_once()

    def test_stalled_response_is_aborted_and_releases_worker_capacity(self):
        async def run(resolver):
            stopped = threading.Event()
            connection = MagicMock()
            connection._active_socket.shutdown.side_effect = lambda *_: stopped.set()
            def stalled_headers():
                if not stopped.wait(1):
                    raise AssertionError("Watchdog did not abort stalled response")
                raise TimeoutError()
            connection.getresponse.side_effect = stalled_headers
            with patch.object(reader.socket, "getaddrinfo", return_value=addresses("8.8.8.8")), \
                    patch.object(reader, "_PinnedHTTPSConnection", return_value=connection):
                result = await reader.read_public_page("https://example.com/", timeout=0.1, resolver=resolver)
                self.assertEqual(result["status"], "error")
                await asyncio.sleep(0.05)
                self.assertTrue(stopped.is_set())
                acquired = [reader._SLOTS.acquire(blocking=False) for _ in range(2)]
                for success in acquired:
                    if success:
                        reader._SLOTS.release()
                self.assertEqual(acquired, [True, True])
        for resolver in ("system", "cloudflare"):
            asyncio.run(run(resolver))

    def test_dns_timeout_keeps_worker_slot_and_never_connects_late(self):
        def slow_dns(_):
            time.sleep(0.25)
            return addresses("8.8.8.8")
        async def run():
            with patch.object(reader, "_addresses", side_effect=slow_dns), patch.object(reader, "_PinnedHTTPSConnection") as connect:
                started = time.monotonic()
                result = await reader.read_public_page("https://example.com/", timeout=0.1)
                self.assertEqual(result["error_code"], "reader_timeout")
                self.assertLess(time.monotonic()-started, 0.2)
                # Late DNS must not connect after the caller's deadline.
                await asyncio.sleep(0.25)
                connect.assert_not_called()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
