"""Fixed child entry points. Never use this file as an unsupervised launcher."""
import os
from pathlib import Path
import resource
import runpy
import sys


HERE = Path(__file__).resolve().parent
PLANNING = HERE.parent


def synthetic():
    # There is deliberately no live admission, proc namespace, cgroup or
    # subreaper operation on this path. Imported libraries are inert.
    print("RUNNER_FIXTURES_BEGIN_V1", flush=True)
    runpy.run_path(str(HERE / "test_runner.py"), run_name="__main__")
    print("INTEGRATION_44_BEGIN_V1", flush=True)
    runpy.run_path(str(PLANNING / "supervisor-integration-v1/test_integration.py"), run_name="__main__")


def live(config):
    sys.argv = [str(PLANNING / "supervisor-integration-v1/live_probe.py"), "--config", config]
    runpy.run_path(sys.argv[0], run_name="__main__")


def main():
    kind = sys.argv[1]
    sys.path[:0] = [str(HERE), str(PLANNING / "supervisor-integration-v1"), str(PLANNING)]
    # Exact isolated environment is installed by the parent before exec.
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1048576, 256 * 1048576))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1048576, 8 * 1048576))
    if kind == "synthetic" and len(sys.argv) == 2:
        synthetic()
    elif kind == "live" and len(sys.argv) == 3:
        live(sys.argv[2])
    else:
        raise ValueError("fixed-stage-arguments-required")


if __name__ == "__main__":
    main()
