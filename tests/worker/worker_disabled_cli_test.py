#!/usr/bin/env python3
"""Real native-OFF engine CLI refusal checks. Execute remotely after building the correction."""
import argparse
import os
from pathlib import Path
import subprocess
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
                 ("--remote-cpu-local-manifest", absent),
                 ("--remote-cpu-local-fallback",), ("--remote-cpu-resident",),
                 ("--remote-cpu-setup-ms", "600000"), ("--remote-cpu-deadline-ms", "10000"))
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
        for flag in ("--help", "--version"):
            result = subprocess.run([engine, flag], env=env, cwd=directory,
                                    capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError(f"native-OFF {flag} failed: {result.stderr}")
            print(f"PASS native-OFF {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
