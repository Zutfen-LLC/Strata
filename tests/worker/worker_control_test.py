#!/usr/bin/env python3
"""Fixture-free real-binary status/unload/stop and CLI validation. Run on inferswarm05 only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--restricted-listener", action="store_true", help="also exercise the real CLI peer allowlist")
    args = parser.parse_args()
    for value in ("0", "-1", "5", "garbage", "2x", "999999999999999999999999"):
        result = subprocess.run([args.worker, "--local-load-threads", value], capture_output=True, timeout=5)
        if result.returncode != 2 or b"listening on" in result.stderr:
            raise RuntimeError(f"invalid local loading thread count {value!r} was not refused")
    for value in ("-1", "86400001", "garbage", "999999999999999999999999"):
        result = subprocess.run([args.worker, "--residency-ms", value], capture_output=True, timeout=5)
        if result.returncode != 2:
            raise RuntimeError(f"invalid residency value {value!r} returned {result.returncode}")
    # Fail closed before listening. No physical interface is opened by these negative cases.
    for extra in (("--bind-ip", "0.0.0.0"), ("--bind-ip", "192.0.2.1"),
                  ("--bind-ip", "localhost"), ("--allow-peer-ip", "192.0.2.0/24"),
                  ("--bind-ip", "127.0.0.1", "--allow-peer-ip", "224.0.0.1"),
                  ("--allow-peer-ip", "127.0.0.1", "--deadline-ms", "0"),
                  ("--allow-peer-ip", "127.0.0.1", "--deadline-ms", "garbage")):
        result = subprocess.run([args.worker, *extra], capture_output=True, timeout=5)
        if result.returncode != 1 or b"listening on" in result.stderr:
            raise RuntimeError(f"unsafe listener policy was not refused: {extra!r}")
    with tempfile.TemporaryDirectory(prefix="worker-control-") as directory:
        log = Path(directory) / "worker.log"
        listener = (["--bind-ip", "127.0.0.1", "--allow-peer-ip", "127.0.0.1",
                     "--allow-peer-ip", "127.0.0.3", "--tcp-nodelay"] if args.restricted_listener else [])
        with log.open("w") as output:
            child = subprocess.Popen([args.worker, "--port", "0", "--workers", "2",
                                      "--memory-budget-mib", "1", "--residency-ms", "1000", *listener],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=output)
        try:
            deadline = time.monotonic() + 5
            port = None
            while time.monotonic() < deadline and child.poll() is None:
                for line in log.read_text().splitlines():
                    if "listening on 127.0.0.1:" in line:
                        port = line.split("listening on 127.0.0.1:", 1)[1].split()[0]
                        break
                if port:
                    break
                time.sleep(0.01)
            if not port:
                raise RuntimeError("worker did not listen: " + log.read_text())
            for action in ("status", "unload", "stop"):
                result = subprocess.run([sys.executable, args.control, action, "--port", port],
                                        capture_output=True, text=True, timeout=8, check=True)
                reply = json.loads(result.stdout)
                if action == "stop":
                    if reply != {"stop_acknowledged": True}:
                        raise RuntimeError("invalid stop acknowledgement")
                elif reply["session"] or reply["resident_bytes"] or reply["complete"] or reply["state"] != "empty":
                    raise RuntimeError("empty worker retained weights/session")
                elif action == "unload" and reply["unloads"] != 1:
                    raise RuntimeError("unload not counted")
                print(f"PASS real control {action}: {result.stdout.strip()}")
            if child.wait(timeout=2) != 0:
                raise RuntimeError("worker stopped unsuccessfully")
            if "shutdown: CPU pool joined; resident=0 B" not in log.read_text():
                raise RuntimeError("pool cleanup marker absent")
            print("PASS real binary control and cleanup")
        finally:
            # Popen retains ownership/status; never use a process-group or negative PID signal.
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=2)
            print(log.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
