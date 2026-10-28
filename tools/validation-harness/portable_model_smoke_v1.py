#!/usr/bin/env python3
"""Parent-dispatched, single-run local portable-model proof. No network or retries."""
import argparse
import csv
import json
import os
import pathlib
import signal
import subprocess
import time

PROMPT = [248045, 846, 198, 7734, 264, 12654, 436, 34810, 6297, 523, 1608,
          51425, 11, 440, 8390, 10206, 11, 615, 321, 2113, 5245, 11, 321,
          75555, 13, 3301, 1970, 1132, 13, 248046, 198, 248045, 74455, 198,
          248068, 271, 248069, 271]
REFERENCE_PREFIX = [71093, 12305, 198]
PCI = ["0000:07:00.0", "0000:0b:00.0"]
MODEL_ROOT = pathlib.Path("/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66")
MANIFEST = "/home/hermes/strata-pr10-9ed478e-20261025-05-peer/assets/manifest"
HOST_ANONYMOUS_CAP = 32 << 30
HOST_RESERVE = 8 << 30


def model_working_set():
    files, identities = [], set()
    for number in range(1, 7):
        path = MODEL_ROOT / f"Qwen3.8-Flash-Next-Q8_0-0000{number}-of-00006.gguf"
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError("model shard paths do not identify six unique files")
        identities.add(identity)
        files.append({"path": str(path), "bytes": stat.st_size})
    # Worst-case file working set, not a measurement of resident file pages.
    return {"files": files, "working_set_bound_bytes": sum(item["bytes"] for item in files)}


def cgroup_limits(leaf, root):
    limits = []
    while True:
        maximum = leaf / "memory.max"
        if maximum.exists():
            value = maximum.read_text().strip()
            if value != "max":
                limit = int(value)
                current = int((leaf / "memory.current").read_text())
                limits.append({"path": str(leaf), "limit": limit, "current": current,
                               "available": max(0, limit-current)})
        if leaf == root:
            return limits
        leaf = leaf.parent


def cgroup_memory():
    memberships = [line.split(":", 2) for line in pathlib.Path("/proc/self/cgroup").read_text().splitlines()]
    unified = next((path for group, controllers, path in memberships if group == "0" and not controllers), None)
    if unified is None:
        if any("memory" in controllers.split(",") for _, controllers, _ in memberships):
            raise ValueError("bounded host guard needs readable cgroup-v2 memory accounting; v1 memory controller found")
        return {"limits": [], "available": None}
    for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines():
        mount, filesystem = line.split(" - ", 1)
        if filesystem.split()[0] != "cgroup2":
            continue
        fields = mount.split()
        root = pathlib.Path(fields[4].replace(chr(92) + "040", " ").replace(chr(92) + "134", chr(92))).resolve(strict=True)
        mount_root = pathlib.PurePosixPath(fields[3])
        membership = pathlib.PurePosixPath(unified)
        # cgroup namespaces can expose a membership relative to their own
        # root rather than the mount's global root. Never escape the mount.
        relative = membership.relative_to(mount_root) if membership.is_relative_to(mount_root) else membership.relative_to("/")
        leaf = (root / relative).resolve(strict=True)
        if not leaf.is_relative_to(root):
            raise ValueError("cannot resolve effective memory cgroup beneath its mount")
        limits = cgroup_limits(leaf, root)
        return {"limits": limits, "available": min((item["available"] for item in limits), default=None)}
    raise ValueError("unified cgroup membership has no readable cgroup2 mount")


def host_memory():
    memory = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            memory[key] = int(value.split()[0]) * 1024
    for line in pathlib.Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in {"pswpin", "pswpout"}:
            memory[key] = int(value)  # kernel counters in pages
    for key in {"MemAvailable", "SwapTotal", "SwapFree", "pswpin", "pswpout"}:
        if key not in memory:
            raise ValueError(f"host memory observation missing {key}")
    memory["SwapUsed"] = memory["SwapTotal"] - memory["SwapFree"]
    memory["cgroup"] = cgroup_memory()
    available = memory["cgroup"]["available"]
    memory["effective_available"] = min(memory["MemAvailable"], available) if available is not None else memory["MemAvailable"]
    return memory


def host_admission_terms(working_set):
    return {"working_set_bound_bytes": working_set, "anonymous_cap_bytes": HOST_ANONYMOUS_CAP,
            "reserve_bytes": HOST_RESERVE,
            "required_available_bytes": working_set + HOST_ANONYMOUS_CAP + HOST_RESERVE}


def admit_host(memory, working_set):
    terms = host_admission_terms(working_set)
    required = terms["required_available_bytes"]
    if memory["effective_available"] < required:
        raise ValueError(f"insufficient host/cgroup headroom: available={memory['effective_available']} required={required}")
    return terms


def guard_host(memory, baseline, process):
    if memory["effective_available"] < HOST_RESERVE:
        raise ValueError("host/cgroup available memory breached 8 GiB reserve")
    if memory["pswpout"] > baseline["pswpout"]:
        raise ValueError("system swap-out grew from pre-launch baseline")
    if process.get("VmSwap", 0) > 0:
        raise ValueError("owned child VmSwap grew from zero baseline")


def vram(pci):
    root = pathlib.Path("/sys/bus/pci/devices") / pci
    return {"pci": pci, "used": int((root / "mem_info_vram_used").read_text()),
            "total": int((root / "mem_info_vram_total").read_text())}


def rss(pid):
    try:
        rows = pathlib.Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return {}
    return {line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in rows if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:"))}


def placements(path, devices, split):
    with path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    seen = set()
    load_roles = set()
    tensor_counts = {str(device): 0 for device in devices}
    for row in rows:
        layer = int(row["layer"])
        if row["phase"] == "load":
            roles = {f"blk.{layer}.indexer.q_proj.weight", f"blk.{layer}.indexer.k_proj.weight"}
            if (layer not in range(3, 48, 4) or row["op"] != "BF16_TO_F32_EXACT" or
                    row["tensor"] not in roles or row["backend"] != "CPU" or row["fallback"]):
                raise ValueError(f"unexpected host load conversion: {row}")
            load_roles.add((layer, row["tensor"]))
            continue
        if row["phase"] == "host":
            if row["op"] != "ROUTE_AND_MERGE" or row["backend"] != "CPU":
                raise ValueError(f"unexpected host work: {row}")
            continue
        if row["metadata"] == "1":
            continue
        if not 0 <= layer <= 48:
            raise ValueError(f"missing/invalid layer attribution: {row}")
        die = 0 if layer < split else 1
        expected = f"Vulkan{devices[die]}"
        if row["backend"] != expected or row["fallback"]:
            raise ValueError(f"unexpected tensor backend/fallback: {row}")
        tensor_counts[str(devices[die])] += 1
        seen.add(layer)
    if seen != set(range(49)):
        raise ValueError(f"layers/head missing actual device assignments: {sorted(set(range(49)) - seen)}")
    expected_loads = {(layer, f"blk.{layer}.indexer.{role}_proj.weight")
                      for layer in range(3, 48, 4) for role in ("q", "k")}
    if load_roles != expected_loads:
        raise ValueError("actual BF16 indexer load conversions do not cover all24 artifact roles")
    for layer in range(48):
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        if layer % 4 == 3:
            if not any(row["phase"] == "qsa-attend" and row["op"] == "MUL_MAT" for row in layer_rows):
                raise ValueError(f"no executed QSA attention graph for layer {layer}")
        else:
            operations = {row["op"] for row in layer_rows}
            if not {"GATED_DELTA_NET", "SSM_CONV"} <= operations:
                raise ValueError(f"missing GDN graph operations for layer {layer}")
    if not any(row["layer"] == "48" and row["phase"] == "global-head" for row in rows):
        raise ValueError("missing global head")
    return tensor_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--evidence", type=pathlib.Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--split", type=int, default=32)
    parser.add_argument("--device-cap-mib", type=int, default=7680)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--expert-dispatch-oracle", action="store_true",
                        help="comparison only: copy and verify experts on every dispatch")
    args = parser.parse_args()
    devices = [int(value) for value in args.devices.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error("two distinct nonnegative Vulkan indices are required")
    if not 0 < args.split < 48 or not 0 < args.device_cap_mib <= 8192 or not 0 < args.timeout_seconds <= 1800:
        parser.error("invalid split/cap or timeout outside [1,1800]")
    # An existing evidence path is never overwritten. A failed run is retained.
    args.evidence.mkdir(parents=False, exist_ok=False)
    evidence = args.evidence
    (evidence / "logits").mkdir()
    binary = args.binary.resolve(strict=True)
    command = [str(binary), "--model", str(MODEL_ROOT / "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"),
               "--manifest", MANIFEST, "--tokens", ",".join(map(str, PROMPT)),
               "--context", "64", "--max-new", "3", "--threads", "4", "--expert-workers", "12",
               "--host-cap-mib", str(HOST_ANONYMOUS_CAP >> 20),
               "--vulkan", args.devices, "--device-cap-mib", ",".join([str(args.device_cap_mib)] * 2),
               "--split", str(args.split), "--dump-logits", str(evidence / "logits"),
               "--placements", str(evidence / "placements.tsv")]
    if args.expert_dispatch_oracle:
        command.append("--expert-dispatch-oracle")
    (evidence / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
    cap = args.device_cap_mib << 20
    summary = {"launched": False, "status": "STOP", "prompt_tokens": len(PROMPT),
               "first_generation_logits": f"logits/pos-{len(PROMPT)-1}.f32",
               "numeric_reference_comparison": "pending; host-local retained02 row required",
               "memory_scope": "sampled aggregate per-PCI VRAM, including other users; not an exact transient peak"}
    process = None
    try:
        working_set = model_working_set()
        summary["model_file_working_set"] = working_set
        baseline_host = host_memory()
        summary["baseline_host"] = baseline_host
        summary["host_admission"] = host_admission_terms(working_set["working_set_bound_bytes"])
        admit_host(baseline_host, working_set["working_set_bound_bytes"])
        summary["minimum_effective_available"] = baseline_host["effective_available"]
        initial = [vram(pci) for pci in PCI]
        summary["baseline_vram"] = initial
        for item in initial:
            if item["total"] - item["used"] < cap:
                raise ValueError(f"per-die free memory cannot admit cap with headroom: {item}")
        peak = [item["used"] for item in initial]
        start = time.monotonic()
        with (evidence / "stdout.jsonl").open("w") as stdout, (evidence / "stderr.log").open("w") as stderr, \
                (evidence / "memory.jsonl").open("w") as memory:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            summary["launched"] = True
            summary["baseline_process"] = rss(process.pid)
            guard_host(baseline_host, baseline_host, summary["baseline_process"])
            while True:
                now = time.monotonic()
                sample = [vram(pci) for pci in PCI]
                host = host_memory()
                child = rss(process.pid)
                memory.write(json.dumps({"seconds": now-start, "vram": sample, "host": host, "process": child}) + "\n")
                memory.flush()
                summary["last_host"] = host
                summary["last_process"] = child
                summary["minimum_effective_available"] = min(summary["minimum_effective_available"], host["effective_available"])
                guard_host(host, baseline_host, child)
                for index, item in enumerate(sample):
                    peak[index] = max(peak[index], item["used"])
                    if item["used"] - initial[index]["used"] > cap:
                        raise ValueError(f"aggregate observed VRAM growth exceeds die cap: {item}")
                summary["sampled_peak_vram"] = peak[:]
                summary["sampled_peak_growth"] = [peak[i]-initial[i]["used"] for i in range(2)]
                status = process.poll()
                if status is not None:
                    break
                if now - start >= args.timeout_seconds:
                    raise TimeoutError("single-run time budget exhausted")
                time.sleep(0.2)
        summary["exit_status"] = status
        summary["seconds"] = time.monotonic()-start
        summary["sampled_peak_vram"] = peak
        summary["sampled_peak_growth"] = [peak[i]-initial[i]["used"] for i in range(2)]
        if status:
            raise RuntimeError(f"portable_model exited {status}; preserve stderr and do not retry")
        summary["tensor_operations_by_device"] = placements(evidence / "placements.tsv", devices, args.split)
        summary["host_bf16_load_conversions"] = 24
        records = [json.loads(line) for line in (evidence / "stdout.jsonl").read_text().splitlines()]
        tokens = [row for row in records if "token" in row]
        if [row["position"] for row in tokens] != list(range(len(tokens))):
            raise ValueError("positions are not contiguous from zero")
        if [row["token"] for row in tokens if not row["generated"]] != PROMPT:
            raise ValueError("consumed prompt differs from retained exact IDs")
        generated = [row["token"] for row in tokens if row["generated"]]
        summary["generated_ids"] = generated
        summary["retained_prefix"] = REFERENCE_PREFIX
        summary["prefix_equal"] = generated == REFERENCE_PREFIX
        summary["first_token_divergence"] = next((i for i, (a, b) in enumerate(zip(generated, REFERENCE_PREFIX)) if a != b),
                                                min(len(generated), len(REFERENCE_PREFIX)) if len(generated) != len(REFERENCE_PREFIX) else None)
        summary["stop"] = records[-1]
        for row in tokens:
            if (evidence / "logits" / f"pos-{row['position']}.f32").stat().st_size != 993280:
                raise ValueError("full-vocabulary logit row size differs from248320 F32")
        summary["status"] = "mechanism_checks_passed; numerical/quality verdict pending"
    except BaseException as error:
        summary["error"] = str(error)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if process is not None:
            summary["exit_status"] = process.returncode
        raise
    finally:
        (evidence / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
