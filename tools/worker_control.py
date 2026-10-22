#!/usr/bin/env python3
"""Bounded loopback/owned-forward control for strata-expert-worker (no weights or prompts)."""
import argparse
import json
import socket
import struct
import sys
import time

MAGIC = 0x31574553


def exchange(sock, msg, expected):
    end = time.monotonic() + 5

    def recv(size):
        data = bytearray()
        while len(data) < size:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("worker control deadline")
            sock.settimeout(remaining)
            part = sock.recv(size - len(data))
            if not part:
                raise RuntimeError("worker closed the connection")
            data.extend(part)
        return data

    sock.settimeout(5)
    sock.sendall(struct.pack("<IHHQ", MAGIC, msg, 0, 0))
    magic, reply, status, size = struct.unpack("<IHHQ", recv(16))
    if magic != MAGIC or reply != expected or size > 128 or status:
        raise RuntimeError(f"worker refused control: reply={reply} status={status} size={size}")
    return recv(size)


def status(sock):
    payload = exchange(sock, 16, 17)
    if len(payload) != 100:
        raise RuntimeError("unsupported cache status length")
    version, state, *values = struct.unpack("<II11QI", payload)
    if version != 1 or state > 2 or values[-1] > 1:
        raise RuntimeError("unsupported cache status version/state")
    keys = ("session", "resident_bytes", "idle_remaining_ms", "cache_hits", "reused_bytes",
            "loaded_bytes", "verified_bytes", "peak_arena_bytes", "arena_allocations",
            "expirations", "unloads", "complete")
    out = dict(zip(keys, values))
    out["state"] = ("empty", "setup", "ready")[state]
    out["idle_expiry"] = "none" if out["idle_remaining_ms"] == (1 << 64) - 1 else "finite-or-empty"
    payload = exchange(sock, 9, 10)
    if len(payload) != 59:
        raise RuntimeError("unsupported legacy status length")
    out["network_weight_bytes"] = struct.unpack_from("<Q", payload, 8)[0]
    out["requests"] = struct.unpack_from("<Q", payload, 32)[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "unload", "stop"))
    parser.add_argument("--host", default="127.0.0.1", help="numeric IPv4 worker or owned SSH forward")
    parser.add_argument("--port", type=int, default=7431)
    args = parser.parse_args()
    socket.inet_pton(socket.AF_INET, args.host)  # no DNS wait
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        if args.action == "stop":
            if exchange(sock, 11, 10):
                raise RuntimeError("unexpected shutdown payload")
            print(json.dumps({"stop_acknowledged": True}))
        else:
            if args.action == "unload" and exchange(sock, 18, 19):
                raise RuntimeError("unexpected unload payload")
            print(json.dumps(status(sock), sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"worker control: {exc}", file=sys.stderr)
        sys.exit(1)
