#!/usr/bin/env python3
"""Parent-dispatched, single-run local portable-model proof. No network or retries."""
import argparse
import csv
import json
import os
import pathlib
import re
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


def model_working_set(root=None):
    root = MODEL_ROOT if root is None else root
    files, identities = [], set()
    for number in range(1, 7):
        path = root / f"Qwen3.8-Flash-Next-Q8_0-0000{number}-of-00006.gguf"
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


def pci_id(text):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", text)
    if not match or int(match[1], 16) > 65535:
        raise ValueError(f"invalid PCI identity: {text}")
    return f"{int(match[1],16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}"


def observation(pci, used, total):
    if total <= 0 or used < 0 or used > total:
        raise ValueError(f"invalid VRAM telemetry for {pci}: used={used} total={total}")
    return dict(pci=pci, used=used, total=total)


def vram(pci):
    root = pathlib.Path("/sys/bus/pci/devices") / pci
    vendor = int((root / "vendor").read_text().strip(), 16)
    if vendor == 0x1002:
        return observation(pci, int((root / "mem_info_vram_used").read_text()),
                           int((root / "mem_info_vram_total").read_text()))
    if vendor == 0x10de:
        # Query by physical bus identity, never by NVIDIA's unrelated index.
        result = subprocess.run(["nvidia-smi", "--id=" + pci,
                                 "--query-gpu=pci.bus_id,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                text=True, timeout=5)
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) != 3 or pci_id(rows[0][0].strip()) != pci:
            raise ValueError("NVIDIA telemetry PCI identity mismatch")
        return observation(pci, int(rows[0][1]) << 20, int(rows[0][2]) << 20)
    raise ValueError(f"unsupported guarded VRAM telemetry vendor {vendor:#x} at {pci}; adapter required")


def device_arguments(parser):
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--pci", default=",".join(PCI), help="ordered physical PCI IDs for selected logical Vulkan indices")
    parser.add_argument("--split", type=int, default=32)
    parser.add_argument("--dense-topology", choices=('split', 'primary'), default='split')
    parser.add_argument("--device-cap-mib", type=cap_value, default=7680,
                        help="policy ceiling in MiB: scalar or two caps in selected device order")
    parser.add_argument("--model-root", type=pathlib.Path, default=None)
    parser.add_argument("--manifest", default=MANIFEST)


def dense_topology(args):
    value = getattr(args, 'dense_topology', 'split')
    if value not in ('split', 'primary'):
        raise ValueError('invalid dense topology')
    return value


def topology_argv(args):
    # Keep historical/default engine argv unchanged.
    return ['--dense-topology', 'primary'] if dense_topology(args) == 'primary' else []


def argv_topology(command):
    if '--dense-topology' not in command:
        return 'split'
    at = command.index('--dense-topology')
    if command.count('--dense-topology') != 1 or at+1 >= len(command) or command[at+1] not in ('split', 'primary'):
        raise ValueError('invalid dense topology argv')
    return command[at+1]


def device_policy(args, physical):
    return dict(devices=args.devices, pci=physical, split=args.split,
                device_cap_mib=args.device_cap_mib, dense_topology=dense_topology(args))


def device_selection(parser, args):
    try:
        indices = [int(value) for value in args.devices.split(",")]
        physical = [pci_id(value) for value in args.pci.split(",")]
    except ValueError as error:
        parser.error(str(error))
    if (len(indices) != 2 or len(set(indices)) != 2 or min(indices) < 0 or
            len(physical) != 2 or len(set(physical)) != 2):
        parser.error("two distinct Vulkan indices and two distinct ordered PCI identities required")
    try:
        device_caps(args)
    except ValueError as error:
        parser.error(str(error))
    if dense_topology(args) == 'split' and not 0 < args.split < 48:
        parser.error("invalid split")
    return indices, physical


def verified_devices(binary, indices, physical):
    result = subprocess.run([str(binary), "--list-vulkan"], check=True, capture_output=True,
                            text=True, timeout=30)
    found = {}
    for line in result.stdout.splitlines():
        index, pci, free, total, description = line.split("\t", 4)
        index = int(index)
        if index in found:
            raise ValueError("duplicate logical Vulkan identity")
        found[index] = dict(pci=pci, free=int(free), total=int(total), description=description)
    selected = []
    for index, pci in zip(indices, physical):
        item = found.get(index)
        if (item is None or item['pci'] == '-' or pci_id(item['pci']) != pci or
                any(word in item['description'].lower() for word in ('llvmpipe', 'lavapipe', 'software'))):
            raise ValueError(f"Vulkan{index} physical PCI mapping missing/mismatched or software driver: {pci}")
        root = pathlib.Path('/sys/bus/pci/devices') / pci
        if int((root / 'class').read_text().strip(), 16) >> 16 != 3:
            raise ValueError(f"selected PCI device is not a physical display controller: {pci}")
        if not 0 <= item['free'] <= item['total'] or item['total'] <= 0:
            raise ValueError("invalid Vulkan physical memory capacity")
        selected.append(dict(index=index, **item))
    return selected


# Keep the scalar spelling/default compatible; normalize only at the boundary.
MAX_MIB = ((1 << 63) - 1) >> 20
VRAM_RESERVE = 512 << 20


def cap_value(text):
    values = str(text).split(',')
    if len(values) not in (1, 2) or any(not v.isdecimal() or not 0 < int(v) <= MAX_MIB for v in values):
        raise ValueError('expected positive scalar or two overflow-safe MiB caps')
    return int(values[0]) if len(values) == 1 else ','.join(str(int(v)) for v in values)


def device_caps(args):
    value = cap_value(args.device_cap_mib)
    return [value << 20] * 2 if isinstance(value, int) else [int(v) << 20 for v in value.split(',')]


def cap_argv(args):
    return ','.join(str(v >> 20) for v in device_caps(args))


def device_reserves(args):
    value = getattr(args, 'expert_free_reserve_mib', getattr(args, 'free_reserve_mib', '512,512'))
    values = str(value).split(',')
    if len(values) != 2 or any(not v.isdecimal() or not 0 < int(v) <= MAX_MIB for v in values):
        raise ValueError('two positive overflow-safe reserve values required')
    return [max(VRAM_RESERVE, int(v) << 20) for v in values]


def admit_vram(initial, mapping, caps, reserves=None):
    caps = [caps] * 2 if type(caps) is int else caps
    reserves = [VRAM_RESERVE] * 2 if reserves is None else reserves
    if not len(initial) == len(mapping) == len(caps) == len(reserves) == 2:
        raise ValueError('two complete ordered memory observations required')
    budgets = []
    for item, device, cap, reserve in zip(initial, mapping, caps, reserves):
        # Backend effective budgets and physical telemetry can have different totals.
        # Both must admit the policy ceiling; use the smaller free observation.
        if item.get('pci') != device.get('pci') or not item.get('pci'):
            raise ValueError('memory observation PCI identity mismatch')
        observation(item['pci'], item['used'], item['total'])
        free = device['free']
        if (type(free) is not int or not 0 <= free <= device['total'] or
                cap <= 0 or cap > device['total'] or cap > item['total']):
            raise ValueError('selected physical capacity/free memory cannot admit cap')
        available = min(free, item['total'] - item['used'])
        if reserve <= 0 or available <= reserve:
            raise ValueError('selected physical free memory cannot preserve reserve')
        budgets.append(dict(policy_cap_bytes=cap, effective_limit_bytes=min(cap, available-reserve),
                            available_bytes=available, reserve_bytes=reserve,
                            baseline_used_bytes=item['used'], physical_total_bytes=item['total'],
                            backend_total_bytes=device['total'], pci=item['pci']))
    return budgets


def budget_argv(budgets):
    # This observation can only tighten the engine's own live backend admission.
    return ['--device-free-bytes', ','.join(str(b['available_bytes']) for b in budgets)]


def guard_vram(sample, initial, budgets):
    if not len(sample) == len(initial) == len(budgets) == 2:
        raise ValueError('incomplete runtime memory evidence')
    for value, baseline, budget in zip(sample, initial, budgets):
        observation(value['pci'], value['used'], value['total'])
        if (value['pci'] != budget['pci'] or value['total'] != budget['physical_total_bytes'] or
                value['used'] - baseline['used'] > budget['effective_limit_bytes'] or
                value['total'] - value['used'] < budget['reserve_bytes']):
            raise ValueError('per-device growth cap/free reserve or identity drift')


def sample_vram(physical, initial=None):
    sample = [vram(pci) for pci in physical]
    if initial is not None and any(a['total'] != b['total'] for a, b in zip(sample, initial)):
        raise ValueError("physical VRAM capacity changed during run")
    return sample


def rss(pid):
    try:
        rows = pathlib.Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return {}
    return {line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in rows if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:"))}


def placements(path, devices, split, dense_topology='split'):
    if dense_topology not in ('split', 'primary'):
        raise ValueError('invalid dense topology')
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
        die = 0 if dense_topology == 'primary' or layer < split else 1
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
    device_arguments(parser)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--expert-dispatch-oracle", action="store_true",
                        help="comparison only: copy and verify experts on every dispatch")
    args = parser.parse_args()
    devices, physical = device_selection(parser, args)
    if not 0 < args.timeout_seconds <= 1800:
        parser.error("timeout outside [1,1800]")
    # An existing evidence path is never overwritten. A failed run is retained.
    args.evidence.mkdir(parents=False, exist_ok=False)
    evidence = args.evidence
    (evidence / "logits").mkdir()
    binary = args.binary.resolve(strict=True)
    command = [str(binary), "--model", str((args.model_root or MODEL_ROOT) / "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"),
               "--manifest", args.manifest, "--tokens", ",".join(map(str, PROMPT)),
               "--context", "64", "--max-new", "3", "--threads", "4", "--expert-workers", "12",
               "--host-cap-mib", str(HOST_ANONYMOUS_CAP >> 20),
               "--vulkan", args.devices, "--device-cap-mib", cap_argv(args),
               "--split", str(args.split), "--dump-logits", str(evidence / "logits"),
               "--placements", str(evidence / "placements.tsv")]
    command += topology_argv(args)
    if args.expert_dispatch_oracle:
        command.append("--expert-dispatch-oracle")
    (evidence / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
    caps = device_caps(args)
    summary = {"launched": False, "status": "STOP", "prompt_tokens": len(PROMPT),
               "first_generation_logits": f"logits/pos-{len(PROMPT)-1}.f32",
               "numeric_reference_comparison": "pending; host-local retained02 row required",
               "memory_scope": "sampled aggregate per-PCI VRAM, including other users; not an exact transient peak"}
    process = None
    summary['device_policy'] = device_policy(args, physical)
    try:
        working_set = model_working_set() if args.model_root is None else model_working_set(args.model_root)
        summary["model_file_working_set"] = working_set
        baseline_host = host_memory()
        summary["baseline_host"] = baseline_host
        summary["host_admission"] = host_admission_terms(working_set["working_set_bound_bytes"])
        admit_host(baseline_host, working_set["working_set_bound_bytes"])
        summary["minimum_effective_available"] = baseline_host["effective_available"]
        mapping = verified_devices(binary, devices, physical)
        summary["selected_devices"] = mapping
        initial = sample_vram(physical)
        summary["baseline_vram"] = initial
        budgets = admit_vram(initial, mapping, caps, device_reserves(args))
        summary["device_budgets"] = budgets
        command += budget_argv(budgets)
        (evidence / "argv.json").write_text(json.dumps(command, indent=2) + "\n")
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
                sample = sample_vram(physical, initial)
                host = host_memory()
                child = rss(process.pid)
                memory.write(json.dumps({"seconds": now-start, "vram": sample, "host": host, "process": child}) + "\n")
                memory.flush()
                summary["last_host"] = host
                summary["last_process"] = child
                summary["minimum_effective_available"] = min(summary["minimum_effective_available"], host["effective_available"])
                guard_host(host, baseline_host, child)
                guard_vram(sample, initial, budgets)
                for index, item in enumerate(sample):
                    peak[index] = max(peak[index], item["used"])
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
        summary["tensor_operations_by_device"] = placements(evidence / "placements.tsv", devices, args.split, dense_topology(args))
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
    def terminate(_sig, _frame): raise SystemExit("termination requested; owned cleanup")
    signal.signal(signal.SIGTERM, terminate)
    main()
