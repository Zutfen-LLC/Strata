"""Future separately approved no-model probe; NOT EXECUTED by the author.

Only initial namespaces and a NEW evidence directory are accepted. No product
binary, model/cache path, endpoint, GPU tool or role config can be supplied.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from resource_guard import CoverageError, GuardStop, EVENT_KEYS, headroom, signature
from resource_telemetry import OwnedProcess, required, read_text
from linux_owned import identity
from policy import MIB
from runtime import Job, collect, packet


class ProbeGuard:
    def __init__(self, baseline):
        self.baseline = baseline

    def check(self, role, value, admission=False):
        if role != "ROLE_PROBE" or signature(value["cgroup"]) != signature(self.baseline["cgroup"]):
            raise CoverageError("probe-role-or-coverage-change")
        if required(value["host_bytes"]["MemAvailable"], "MemAvailable") < 256 * MIB:
            raise GuardStop("probe-host-headroom")
        room = headroom(value["cgroup"])
        if room is not None and room < 256 * MIB:
            raise GuardStop("probe-cgroup-headroom")
        if required(value["worker_memory_bytes"]["VmRSS"], "VmRSS") > 64 * MIB:
            raise GuardStop("probe-rss")
        initial = required(self.baseline["host_bytes"]["SwapFree"], "initial SwapFree")
        if required(value["host_bytes"]["SwapFree"], "SwapFree") < initial - 64 * MIB:
            raise GuardStop("probe-original-swap-boundary")
        for row, before in zip(value["cgroup"]["rows"], self.baseline["cgroup"]["rows"]):
            if row["coverage"] == "root-not-resource-controlled":
                continue
            for key in EVENT_KEYS:
                if required(row["events"][key], key) != required(before["events"][key], key):
                    raise GuardStop("probe-new-or-decreased-event")


class NoOuter:
    def sample(self):
        return {}

    def check(self, *args, **kwargs):
        return None


def run(config):
    if set(config) != {"namespaces", "evidence_root"}:
        raise CoverageError("probe-config-exact-schema")
    root = Path(config["evidence_root"])
    if not root.is_absolute():
        raise CoverageError("new-absolute-evidence-root-required")
    os.mkdir(root, 0o700)
    # Limits restrict these harmless processes; no pressure generation/allocation.
    resource.setrlimit(resource.RLIMIT_AS, (128 * MIB, 128 * MIB))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * MIB, 8 * MIB))
    ns = config["namespaces"]
    own = OwnedProcess.pin(os.getpid(), expected=identity(os.getpid(), ns), namespace_receipt=ns)
    try:
        from resource_telemetry import OwnedTree
        baseline = collect(OwnedTree(own), ns)
    finally:
        own.close()
    end = time.monotonic() + 45
    for name in ("natural_exit", "monitor_exit", "monitor_stall"):
        directory = root / name
        os.mkdir(directory, 0o700)
        key = ("probe-v1", "ROLE_PROBE", name)
        behavior = None
        if name == "monitor_exit":
            behavior = lambda channel: None
        elif name == "monitor_stall":
            def behavior(channel):
                packet(channel, {"kind": "COVERED", "key": key, "sequence": 0,
                                 "mono": time.monotonic()})
                time.sleep(30)
        seconds = 2 if name == "natural_exit" else 30
        argv = [sys.executable, "-B", "-I", "-c",
                "import time; print('HARMLESS_CHILD_STARTED_V1', flush=True); time.sleep(" + str(seconds) + ")"]
        input_fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
        job = None
        observed_failure = False
        try:
            job = Job(key, argv, {"PATH": "/usr/bin:/bin", "LANG": "C"}, input_fd, ns,
                      ProbeGuard(baseline), NoOuter(), min(end, time.monotonic() + 20),
                      str(directory), monitor_behavior=behavior)
            while True:
                try:
                    rc = job.poll()
                except CoverageError:
                    observed_failure = True
                    break
                if rc is not None:
                    if rc != 0:
                        raise CoverageError("probe-harmless-child-nonzero")
                    break
                time.sleep(.02)
        finally:
            os.close(input_fd)
            if job is not None:
                cleanup = job.stop("probe-completion", grace=1)
                (directory / "cleanup.json").write_text(json.dumps(cleanup, sort_keys=True) + "\n")
                if not cleanup["complete"] or not all(pid in cleanup["reaped"] for pid in job.children):
                    raise CoverageError("probe-incomplete-true-parent-reaping")
        if observed_failure != (name != "natural_exit"):
            raise CoverageError("probe-expected-failure-mismatch:" + name)
        if time.monotonic() >= end:
            raise CoverageError("probe-overall-wall-bound")
        print("SUPERVISOR_OWNED_PROBE_CASE_PASS_V1 case=" + name, flush=True)
    print("SUPERVISOR_OWNED_PROBE_PASS_V1 cases=3 models=0 gpus=0 services=0", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(json.loads(required(read_text(args.config), "private-probe-config")))


if __name__ == "__main__":
    main()
