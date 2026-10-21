"""Read-only idle TTL adapter derived from pinned public SEW1 source.

The private worker_control.py is NOT imported or executed. No setup/infer,
unload or shutdown message is sent. Call only after all engines are closed.
"""
import socket
import struct
import time

from resource_guard import CoverageError
from policy import Stamp, retention_deadline

MAGIC = 0x31574553


def decode(header, payload):
    if len(header) != 16 or struct.unpack("<IHHQ", header) != (MAGIC, 17, 0, 100) or \
            len(payload) != 100:
        raise CoverageError("cache-status-frame")
    version, state, session, arena, remaining = struct.unpack_from("<IIQQQ", payload)
    complete, = struct.unpack_from("<I", payload, 96)
    if version != 1 or state not in (0, 1, 2) or complete != 1 or \
            arena != 111012556800 or not 0 < remaining <= 1800000:
        raise CoverageError("verified-full-arena-idle-status-required")
    # Empty state/session after engine EOF is valid retained residency.
    return remaining


def read_exact(sock, count):
    parts = bytearray()
    while len(parts) < count:
        chunk = sock.recv(count - len(parts))
        if not chunk:
            raise CoverageError("short-cache-status")
        parts.extend(chunk)
    return bytes(parts)


def idle_deadline(port, owned_process, admission, completion, domain, *, engines_closed):
    if not engines_closed or type(port) is not int or not 0 < port < 65536:
        raise CoverageError("idle-only-owned-listener-required")
    owned_process.verify()
    start = Stamp(domain, time.monotonic(), time.time())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        sock.connect(("127.0.0.1", port))
        sock.sendall(struct.pack("<IHHQ", MAGIC, 16, 0, 0))
        header = read_exact(sock, 16)
        if struct.unpack("<IHHQ", header) != (MAGIC, 17, 0, 100):
            raise CoverageError("cache-status-header")
        remaining = decode(header, read_exact(sock, 100))
    owned_process.verify()
    if time.monotonic() - start.mono > 1:
        raise CoverageError("cache-status-total-timeout")
    return retention_deadline(admission, completion, start, remaining, domain)
