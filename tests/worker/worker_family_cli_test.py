#!/usr/bin/env python3
"""Header-only family refusal for the CPU RPC timing flag; run remotely, no model or GPU work."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--native-experts", choices=("on", "off"), required=True)
    args = parser.parse_args()
    engine = str(Path(args.engine).resolve(strict=True))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    flags = ("--remote-cpu-rpc-timing", "--remote-cpu-worker")
    with tempfile.TemporaryDirectory(prefix="worker-family-cli-") as directory:
        def string(value):
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        family = Path(directory) / "qwen35-header.gguf"
        family.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) +
                           string("general.architecture") + struct.pack("<I", 8) + string("qwen35moe"))
        for flag in flags:
            for extra in ([flag], ["--dump-logits", flag]):
                command = [engine, "--native", str(family), "--serve", *extra]
                result = subprocess.run(command, env=env, cwd=directory, capture_output=True, text=True, timeout=5)
                expected = (f"{flag} requires STRATA_NATIVE_EXPERTS=ON" if args.native_experts == "off" else
                            f"strata-q36: {flag} is unsupported by this family")
                if result.returncode != 2 or expected not in result.stderr or result.stdout:
                    raise RuntimeError(f"family timing refusal failed: {command!r}, rc={result.returncode}, "
                                       f"stdout={result.stdout!r}, stderr={result.stderr!r}")
                if extra[0] == "--dump-logits" and "strata-q36:" not in result.stderr:
                    raise RuntimeError("option-shaped value did not reach the family refusal guard")
                print(f"PASS native-{args.native_experts} family timing refusal: {extra!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
