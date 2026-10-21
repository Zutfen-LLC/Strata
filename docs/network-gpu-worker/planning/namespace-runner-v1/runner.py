"""One deterministic no-model runner. Publication is NOT runtime approval."""
import time
ENTRY_MONO, ENTRY_WALL = time.monotonic(), time.time()

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import select
import signal
import stat
import sys

HERE = Path(__file__).resolve().parent
PLANNING = HERE.parent
sys.path[:0] = [str(HERE), str(PLANNING / "supervisor-integration-v1"), str(PLANNING)]
from admission import Blocked, live_admission, observe, regular_bytes, hex_digest
from owned_child import Child
from resource_telemetry import parse_stat, read_text

RUNNER_TESTS = 25
RUNNER_MARKER = "NAMESPACE_RUNNER_FIXTURES_PASS_V1 tests=25 live_signals=0 workloads=0"
INTEGRATION_MARKER = "SUPERVISOR_INTEGRATION_FIXTURES_PASS_V1 tests=44 live_signals=0 workloads=0"
LIVE_MARKER = "SUPERVISOR_OWNED_PROBE_PASS_V1 cases=3 models=0 gpus=0 services=0"
REQUIRED = {"resource_guard.py", "resource_telemetry.py"}
REQUIRED.update("supervisor-integration-v1/" + n + ".py" for n in
                ("agent", "cache_status", "commands", "controller", "linux_owned", "live_probe", "outer", "policy", "runtime", "test_integration"))
REQUIRED.update("namespace-runner-v1/" + n + ".py" for n in
                ("runner", "admission", "owned_child", "stage_entry", "test_runner"))


class Deadline(RuntimeError):
    pass


def stamp():
    return {"mono": time.monotonic(), "wall": time.time(), "domain": "target-runner-local"}


class Budget:
    def __init__(self, start):
        self.start, self.deadline, self.work_end = start, start + 210, start + 150

    def check(self, end=None):
        if time.monotonic() >= (self.deadline if end is None else end):
            raise Deadline("fixed-monotonic-deadline")

    def admit_stage(self, seconds=60):
        self.check()
        if time.monotonic() + seconds + 5 > self.work_end:
            raise Blocked("budget", "full-stage-and-cleanup-cannot-fit-before-60s-reserve")

    @contextmanager
    def limited(self, seconds):
        if time.monotonic() + seconds > self.work_end:
            raise Blocked("budget", "full-admission-cannot-fit-before-60s-reserve")
        end = min(self.work_end, time.monotonic() + seconds)
        signal.setitimer(signal.ITIMER_REAL, max(.001, end - time.monotonic()))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, max(0, self.deadline - time.monotonic()))


class Evidence:
    def __init__(self, root, budget):
        self.root, self.budget = Path(root), budget
        if not self.root.is_absolute() or str(self.root.parent.resolve(strict=True)) != str(self.root.parent):
            raise Blocked("path", "absolute-canonical-new-evidence-root-required")
        os.mkdir(self.root, 0o700)
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        st = os.fstat(self.fd)
        if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
            raise Blocked("path", "private-owned-evidence-directory-required")

    def write(self, name, raw):
        self.budget.check()
        if "/" in name or len(raw) > 65536:
            raise Blocked("evidence", "bounded-root-file-required")
        fd = os.open(name + ".new", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=self.fd)
        try:
            if os.write(fd, raw) != len(raw):
                raise OSError("short-evidence-write")
        finally:
            os.close(fd)
        os.replace(name + ".new", name, src_dir_fd=self.fd, dst_dir_fd=self.fd)
        self.budget.check()

    def json(self, name, value):
        self.write(name, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("ascii"))

    def log(self, name):
        return os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                       0o600, dir_fd=self.fd)

    def seal(self, report):
        """Bounded own-tree inventory. No archives/model scans or late copying."""
        report["seal_started"] = stamp()
        rows, total, count = [], 0, 0
        pending = [self.root]
        while pending:
            self.budget.check()
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    self.budget.check()
                    count += 1
                    if count > 64 or entry.is_symlink():
                        raise Blocked("seal", "evidence-entry-cap-or-symlink")
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        if path.parent == self.root and path.name in ("REPORT.json", "SEALED.json", "SHA256SUMS"):
                            continue
                        data = regular_bytes(path, 8 * 1048576, self.budget.check, allow_empty=True)
                        total += len(data)
                        if total > 32 * 1048576:
                            raise Blocked("seal", "aggregate-evidence-byte-cap")
                        rows.append((str(path.relative_to(self.root)), hashlib.sha256(data).hexdigest()))
                    else:
                        raise Blocked("seal", "nonregular-evidence")
        report["payload_sealed_at"] = stamp()
        report["payload_bytes"] = total
        self.json("REPORT.json", report)
        data = regular_bytes(self.root / "REPORT.json", 65536, self.budget.check)
        rows.append(("REPORT.json", hashlib.sha256(data).hexdigest()))
        manifest = "".join(digest + "  " + name + "\n" for name, digest in sorted(rows)).encode("ascii")
        self.write("SHA256SUMS", manifest)
        self.json("SEALED.json", {"manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                                  "observed_after_manifest_write": stamp(), "files": len(rows)})
        self.budget.check()


def artifact_admission(manifest_digest, interpreter_digest, budget):
    if not hex_digest(manifest_digest) or not hex_digest(interpreter_digest):
        raise Blocked("provenance", "approved-artifact-and-interpreter-pins-required")
    raw = regular_bytes(HERE / "SHA256SUMS.runner-v1", 32768, budget.check)
    if hashlib.sha256(raw).hexdigest() != manifest_digest:
        raise Blocked("digest-mismatch", "artifact-manifest")
    found = set()
    for line in raw.decode("ascii").splitlines():
        digest, sep, name = line.partition("  ")
        path = Path(name)
        if not sep or not hex_digest(digest) or path.is_absolute() or \
                any(p in (".", "..") for p in path.parts) or name in found:
            raise Blocked("malformed", "artifact-manifest-row")
        found.add(name)
        if hashlib.sha256(regular_bytes(PLANNING / name, 131072, budget.check)).hexdigest() != digest:
            raise Blocked("digest-mismatch", "artifact:" + name)
    if not REQUIRED <= found or len(found) > 32:
        raise Blocked("malformed", "artifact-manifest-required-set")
    executable = Path(sys.executable).resolve(strict=True)
    data = regular_bytes(executable, 32 * 1048576, budget.check)
    if hashlib.sha256(data).hexdigest() != interpreter_digest:
        raise Blocked("digest-mismatch", "interpreter")
    st, current = os.stat(executable), os.stat("/proc/self/exe")
    if (st.st_dev, st.st_ino) != (current.st_dev, current.st_ino):
        raise Blocked("mismatch", "running-interpreter")
    actual, _, _ = parse_stat(read_text("/proc/self/stat"))
    if actual != os.getpid() or not callable(getattr(signal, "pidfd_send_signal", None)):
        raise Blocked("ownership", "local-pid-view-or-pidfd-signal")
    fd = os.pidfd_open(os.getpid(), 0)
    os.close(fd)
    return {"executable": str(executable), "executable_identity": (st.st_dev, st.st_ino),
            "interpreter_sha256": interpreter_digest, "manifest_sha256": manifest_digest,
            "python": sys.version, "uid": os.getuid(), "files": len(found)}


def classify(kind, rc, stdout, cleanup):
    lines = stdout.decode("ascii", errors="replace").splitlines()
    def marker(value):
        return lines.count(value) == 1
    result = {"rc": rc, "status": "FAIL", "cleanup": cleanup}
    if kind == "synthetic":
        result["runner_fixtures"] = "PASS" if marker(RUNNER_MARKER) else "FAIL"
        result["integration_44"] = "PASS" if marker(INTEGRATION_MARKER) else \
            "FAIL" if "INTEGRATION_44_BEGIN_V1" in lines else "NOT_RUN"
        sequence = ["RUNNER_FIXTURES_BEGIN_V1", RUNNER_MARKER, "INTEGRATION_44_BEGIN_V1", INTEGRATION_MARKER]
    else:
        sequence = ["SUPERVISOR_OWNED_PROBE_CASE_PASS_V1 case=" + case
                    for case in ("natural_exit", "monitor_exit", "monitor_stall")] + [LIVE_MARKER]
    if rc == 0 and cleanup["complete"] and all(marker(m) for m in sequence) and \
            [lines.index(m) for m in sequence] == sorted(lines.index(m) for m in sequence):
        result["status"] = "PASS"
    return result


def stage(kind, argv, identity, evidence, budget, report):
    budget.admit_stage()
    record = {"kind": kind, "status": "RUNNING", "start": stamp()}
    report["stages"].append(record)
    evidence.json("REPORT.json", report)
    logs, child, output, size = {}, None, bytearray(), 0
    failure = None
    end = record["start"]["mono"] + 60
    try:
        for name in ("stdout", "stderr"):
            logs[name] = evidence.log(kind + "." + name)
        child = Child(argv, tuple(identity))
        record.update(pid=child.pid, process_identity=child.identity, spawn_start_mono=child.started)
        while True:
            budget.check(min(end, budget.work_end))
            for fd in select.select(list(child.streams), [], [], .02)[0]:
                data = os.read(fd, 65536)
                if not data:
                    del child.streams[fd]
                    continue
                size += len(data)
                if size > 8 * 1048576:
                    raise Blocked("output-cap", kind)
                if os.write(logs[child.streams[fd]], data) != len(data):
                    raise OSError("short-stage-log-write")
                if child.streams[fd] == "stdout":
                    output.extend(data)
            rc = child.poll()
            if rc is not None and not child.streams:
                record["exit_observed"] = stamp()
                break
    except (OSError, ValueError, Deadline) as exc:
        failure = type(exc).__name__ + ":" + str(exc)
        if getattr(exc, "cleanup", None) is not None:
            record["cleanup"] = exc.cleanup
    finally:
        try:
            if child is not None:
                try:
                    record["cleanup"] = child.stop(min(budget.work_end, end + 5, time.monotonic() + 5))
                except (OSError, ValueError, Deadline) as exc:
                    record["cleanup"] = {"complete": False, "error": type(exc).__name__ + ":" + str(exc)}
                    failure = failure or "cleanup-failed"
        finally:
            close_errors = []
            for fd in logs.values():
                try:
                    os.close(fd)
                except OSError as exc:
                    close_errors.append(type(exc).__name__)
            if close_errors:
                record["log_close_errors"] = close_errors
                failure = failure or "log-close-failed"
    record.update(classify(kind, None if child is None else child.rc, output,
                           record.get("cleanup", {"complete": False})))
    record["end"] = stamp()
    if failure:
        record.update(status="FAIL", reason=failure)
    evidence.json("REPORT.json", report)
    return record


def execute(args, evidence, budget, report):
    report["live_result"] = {"status": "NOT_RUN"}
    admission = {"start": stamp(), "kind": "in-process", "status": "BLOCKED", "observations": []}
    report["synthetic_admission"] = admission
    try:
        with budget.limited(10):
            facts = observe("synthetic-protections-and-provenance", lambda:
                            artifact_admission(args.manifest_sha256, args.interpreter_sha256, budget))
            admission["observations"].append(facts)
            if facts["status"] != "ok":
                return "BLOCKED_SYNTHETIC"
            admission["status"] = "PASS"
    finally:
        admission["end"] = stamp()
        evidence.json("REPORT.json", report)
    facts = facts["value"]
    argv = [facts["executable"], "-B", "-I", str(HERE / "stage_entry.py"), "synthetic"]
    result = stage("synthetic", argv, facts["executable_identity"], evidence, budget, report)
    if result["status"] != "PASS":
        return "FAIL_SYNTHETIC"
    if args.mode == "synthetic-only":
        report["live"] = {"status": "NOT_REQUESTED"}
        report["live_result"] = {"status": "NOT_REQUESTED"}
        return "PASS_SYNTHETIC_ONLY"
    receipt = {"status": "BLOCKED", "start": stamp()}
    report["live"] = receipt
    try:
        with budget.limited(5):
            receipt.update(live_admission(args.receipt, {"digest": args.approved_receipt_sha256,
                           "source": args.approved_source_id, "host": args.expected_host_id,
                           "machine": args.expected_machine_id_sha256}, time.time(), budget.check))
    finally:
        receipt["end"] = stamp()
        evidence.json("REPORT.json", report)
    if receipt["status"] != "PASS":
        report["live_result"] = {"status": "BLOCKED", "reason": receipt["reason"]}
        return "BLOCKED_LIVE"
    evidence.json("probe-config.json", {"namespaces": receipt["namespaces"],
                                       "evidence_root": str(evidence.root / "live")})
    argv[-1:] = ["live", str(evidence.root / "probe-config.json")]
    result = stage("live", argv, facts["executable_identity"], evidence, budget, report)
    report["live_result"] = {"status": result["status"]}
    return "PASS_SYNTHETIC_AND_LIVE" if result["status"] == "PASS" else "FAIL_LIVE"


def main():
    budget = Budget(ENTRY_MONO)
    def expired(*_):
        raise Deadline("alarm-deadline")
    signal.signal(signal.SIGALRM, expired)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, budget.deadline - time.monotonic()))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("synthetic-only", "synthetic-then-live"))
    for flag in ("evidence-root", "manifest-sha256", "interpreter-sha256"):
        ap.add_argument("--" + flag, required=True)
    for flag in ("receipt", "approved-receipt-sha256", "approved-source-id", "expected-host-id", "expected-machine-id-sha256"):
        ap.add_argument("--" + flag)
    args = ap.parse_args()
    evidence = None
    report = {"runner_entry": {"mono": ENTRY_MONO, "wall": ENTRY_WALL}, "deadline": budget.deadline,
              "work_end": budget.work_end, "closeout_reserve_seconds": 60, "mode": args.mode,
              "status": "RUNNING", "stages": []}
    try:
        budget.check()
        # Only this runner's soft limit changes; child stages have their own caps.
        old = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (min(128 * 1048576, old[0]) if old[0] >= 0 else 128 * 1048576, old[1]))
        evidence = Evidence(args.evidence_root, budget)
        evidence.json("REPORT.json", report)
        report["status"] = execute(args, evidence, budget, report)
    except (OSError, ValueError, Deadline) as exc:
        report["status"] = "BLOCKED" if isinstance(exc, (Blocked, Deadline)) else "FAIL"
        report["reason"] = type(exc).__name__ + ":" + str(exc)
    finally:
        if evidence is not None:
            try:
                budget.check()
                evidence.seal(report)
                budget.check()
                print("NAMESPACE_RUNNER_FINAL_V1 status=" + report["status"], flush=True)
            except (OSError, ValueError, Deadline) as exc:
                report["status"] = "INCOMPLETE"
                # No post-deadline bookkeeping. Earlier REPORT/stage results remain.
                if time.monotonic() < budget.deadline:
                    report["seal_error"] = type(exc).__name__ + ":" + str(exc)
                    evidence.json("INCOMPLETE.json", report)
            finally:
                os.close(evidence.fd)
        elif time.monotonic() < budget.deadline:
            print(json.dumps({"marker": "NAMESPACE_RUNNER_NO_EVIDENCE_V1", "report": report},
                             sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["status"].startswith("PASS_") else 3 if report["status"].startswith("BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
