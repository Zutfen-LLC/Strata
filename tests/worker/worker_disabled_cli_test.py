#!/usr/bin/env python3
"""Real native-OFF engine CLI refusal checks. Execute remotely after building the correction."""
import argparse
import os
from pathlib import Path
import subprocess
import struct
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    engine = str(Path(args.engine).resolve(strict=True))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    with tempfile.TemporaryDirectory(prefix="worker-disabled-cli-") as directory:
        absent = str(Path(directory) / "absent-model.gguf")
        pack = str(Path(directory) / "absent-pack")
        cases = (("--remote-cpu-experts", "127.0.0.1:1"),
                 ("--remote-cpu-worker", "127.0.0.1:1"),
                 ("--remote-cpu-local-manifest", absent),
                 ("--remote-cpu-local-fallback",), ("--remote-cpu-resident",),
                 ("--remote-cpu-rpc-timing",),
                 ("--remote-cpu-regroup-diagnostic", "off"), ("--remote-cpu-regroup-diagnostic", "on"),
                 ("--remote-cpu-setup-ms", "600000"), ("--remote-cpu-deadline-ms", "10000"),
                 ("--remote-gpu-experts", "127.0.0.1:1"), ("--remote-gpu-plan", absent),
                 ("--remote-gpu-worker", "127.0.0.1:1", absent),
                 ("--remote-gpu-setup-ms", "600000"), ("--remote-gpu-deadline-ms", "10000"))
        for knobs in cases:
            # Both bare modifiers and real-looking invocations must stop at feature refusal,
            # before the earlier family/model-header path or the GPU architecture preflight.
            for command in (list(knobs), ["--native", absent, "--pack", pack, "--serve", *knobs]):
                result = subprocess.run([engine, *command], env=env, cwd=directory,
                                        capture_output=True, text=True, timeout=5)
                expected = f"{knobs[0]} requires STRATA_NATIVE_EXPERTS=ON"
                if result.returncode != 2 or expected not in result.stderr or result.stdout:
                    raise RuntimeError(f"unexpected refusal: {command!r}, rc={result.returncode}, "
                                       f"stdout={result.stdout!r}, stderr={result.stderr!r}")
                print(f"PASS unavailable worker option: {command!r}")
        # Option-shaped values remain values. --help stops before model/GPU setup; token files
        # contain only synthetic IDs and exercise the real parser's file-value consumption.
        for knobs in cases:
            alias = knobs[0]
            Path(directory, alias).write_text("1,2,3\n")
            for command in (["--pack", alias, "--help"], ["--tokens-file", alias, "--help"],
                            ["--dump-logits", alias, "--help"],
                            ["--control-vector-layer-range", "0", alias, "--help"],
                            ["--control-vector-layer-range", alias, "1", "--help"]):
                result = subprocess.run([engine, *command], env=env, cwd=directory,
                                        capture_output=True, text=True, timeout=5)
                if result.returncode != 0 or "requires STRATA_NATIVE_EXPERTS=ON" in result.stderr:
                    raise RuntimeError(f"option value mistaken for worker flag: {command!r}, "
                                       f"rc={result.returncode}, stderr={result.stderr!r}")
                print(f"PASS option-shaped value: {command!r}")
        # A header-only synthetic GGUF selects the other family parser. Its serve mode ignores
        # --dump-logits, whereas the main parser consumes a value; the actual worker flag must fail
        # there too, before this tensor-free fixture can reach model or GPU initialization.
        family = Path(directory) / "qwen35-header.gguf"
        def gguf_string(value):
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded
        family.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) +
                           gguf_string("general.architecture") + struct.pack("<I", 8) +
                           gguf_string("qwen35moe"))
        for knobs in cases:
            command = ["--native", str(family), "--serve", "--dump-logits", *knobs]
            result = subprocess.run([engine, *command], env=env, cwd=directory,
                                    capture_output=True, text=True, timeout=5)
            expected = f"strata-q36: {knobs[0]} requires STRATA_NATIVE_EXPERTS=ON"
            if result.returncode != 2 or expected not in result.stderr or result.stdout:
                raise RuntimeError(f"family parser did not refuse: {command!r}, "
                                   f"rc={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}")
            print(f"PASS family parser worker refusal: {command!r}")
        for flag in ("--help", "--version"):
            result = subprocess.run([engine, flag], env=env, cwd=directory,
                                    capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError(f"native-OFF {flag} failed: {result.stderr}")
            print(f"PASS native-OFF {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
