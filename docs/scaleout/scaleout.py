"""Offline placement/request/render/report helpers and bounded output clock.

No subprocess, socket, model read or PID signalling. Import has no side effects.
Run only through Hermes on the approved controller/03, never on user desktops.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import time

PRODUCT = "9dd56a946b025bd56e4bd0174997d6b9e51f4e02"
ENGINE = "92df6d5ccc859d6e2f2606517ca6c9ff6d13547cf54b19094a6c14a4fb057753"
WORKER = "4ae79b97ef4761088f87167a239a271f80025d8c2a507567aee1985378cb1576"
EVALUATOR = "8d2cc2cf6aff51c70157a6e1d22374fe14956acc5fd9be19625db6ca4cbb2587"
PLAN = "1c81fa79b9a26f66b0a4f7b2e6eae92858c82f47de827f08751b312948a21792"
PLAN04 = "a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578"
PROFILE = "8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf"
MIB, GIB, BLOB, WORKSPACE = 1 << 20, 1 << 30, 5222400, 1598720
ORDER = ("C0", "S0", "S1", "C1")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def bounded(path, cap=1048576):
    with open(path, "rb") as f:
        raw = f.read(cap + 1)
    need(0 < len(raw) <= cap, "bounded input: " + str(path))
    return raw


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(bounded(path))


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, allow_nan=False, indent=2)
        f.write("\n")


def task_plan(path):
    raw = bounded(path)
    need(sha(raw) == PLAN, "exact retained task plan; no retokenization")
    p = json.loads(raw)
    need(len(p["tasks"]) == 5, "five tasks")
    for row in p["tasks"]:
        need(row["max_new"] == 32 and 2 <= len(row["ids"]) <= 64 and all(
            type(t) is int and 0 <= t < 248320 for t in row["ids"]), "task/token bound")
    return p


def place(a):
    admission = read_json(a.admission)
    raw = bounded(a.profile)
    need(sha(raw) == PROFILE, "retained ranking SHA")
    magic, ver, nl, ne, slots, count = struct.unpack_from("<4s5I", raw)
    need((magic, ver, nl, ne) == (b"STRP", 1, 48, 512) and 0 < count <= slots <= 24576 and
         len(raw) == 24 + 4 * count + 4 * 24576, "STRP bounds")
    ranked = list(struct.iter_unpack("<HH", raw[24:24 + 4 * count]))
    need(len(set(ranked)) == count and all(l < 48 and e < 512 for l, e in ranked), "ranked pairs")
    primary, quota = set(), [0] * 48
    for p in ranked:
        if quota[p[0]] < 20:
            primary.add(p)
            quota[p[0]] += 1
    helper = set([p for p in ranked if p not in primary][:2247])
    cpu = {(l, e) for l in range(48) for e in range(512)} - primary - helper
    need(quota == [20] * 48 and len(helper) == 2247 and len(cpu) == 21369, "frozen local ownership")
    bitmap = bounded(a.cpu_bitmap, 3072)
    expected = bytearray(3072)
    for l, e in cpu:
        expected[l * 64 + e // 8] |= 1 << (e % 8)
    need(bitmap == expected and sha(bitmap) == admission["cpu_bitmap_sha256"], "retained CPU bitmap mismatch")
    plan04 = bounded(a.plan04)
    need(sha(plan04) == PLAN04 and len(plan04) == 92156, "unchanged 04 plan")
    head = struct.unpack_from("<8sQ6H2QI", plan04)
    need(head[0] == b"SGPUPLN1" and head[2:8] == (48, 512, 2560, 640, 8, 8) and head[-1] == 3289,
         "04 plan geometry")
    remote04 = set()
    for i in range(3289):
        l, e, size = struct.unpack_from("<HHQ", plan04, 48 + i * 28)
        need(size == BLOB and (l, e) in cpu, "04 assigned asset")
        remote04.add((l, e))
    need(len(remote04) == 3289, "04 duplicate assignment")
    available = [p for p in ranked if p in cpu and p not in remote04]
    results, offset = [], 0
    for index, limit in ((0, 6 * GIB), (1, 10 * GIB)):
        d = admission["devices"][index]
        need(d["index"] == index and isinstance(d["uuid"], str) and d["uuid"], "physical device binding")
        free, total = d["free_bytes"], d["total_bytes"]
        need(type(free) is int and type(total) is int and 0 < free <= total, "fresh VRAM admission")
        reserve_mib = max(1536, math.ceil(total / (10 * MIB)))
        capacity = min(limit, free - reserve_mib * MIB - WORKSPACE - 256 * MIB)
        n = capacity // BLOB
        need(n > 0 and offset + n < len(available), "positive bounded assignment leaving CPU work")
        need(d["cold_payload_bytes"] > 0 and d["cold_setup_seconds"] > 0 and
             math.isfinite(d["cold_setup_seconds"]),
             "retained same-route cold setup goodput required")
        rate = d["cold_payload_bytes"] / d["cold_setup_seconds"]
        # This effective rate includes context/cache overhead; it is not raw link bandwidth.
        setup_s = math.ceil(1.5 * n * BLOB / rate + 300)
        need(0 < setup_s <= 86400, "supported bounded worker setup interval")
        pairs = available[offset:offset + n]
        offset += n
        results.append({"device": index, "uuid": d["uuid"], "count": n, "assigned_bytes": n * BLOB,
                        "reserve_mib": reserve_mib, "workspace_bytes": WORKSPACE,
                        "driver_margin_bytes": 256 * MIB, "cache_budget_mib": math.ceil(n * BLOB / MIB) + 64,
                        "effective_setup_goodput_Bps": rate, "setup_s": setup_s, "pairs": pairs})
    need(admission["cache_filesystem_free_bytes"] >= sum(r["cache_budget_mib"] for r in results) * MIB + 4 * GIB,
         "combined 02 disk budget plus 4 GiB headroom")
    setup_ms = max(300, *(r["setup_s"] for r in results)) * 1000
    cold_setup_s = 300 + sum(r["setup_s"] for r in results)
    # C0/C1 startup 300 each; S1 startup 600; four 240s task batches;
    # per-phase cleanup 45 each, global cleanup reserve 45, admission/relays 300.
    window = cold_setup_s + 300 + 300 + 600 + 4 * 240 + 4 * 45 + 45 + 300
    need(window <= 14400, "derived window exceeds four hours; report estimate before launching, do not shrink silently")
    out = Path(a.out)
    need(out.is_absolute(), "new absolute private output")
    os.mkdir(out, 0o700)
    for r in results:
        with open(out / ("assigned02-" + str(r["device"]) + ".tsv"), "x", encoding="ascii") as f:
            f.write("".join(f"{l} {e}\n" for l, e in r.pop("pairs")))
    write_json(out / "placement.json", {"product": PRODUCT, "engine_sha256": ENGINE, "worker_sha256": WORKER,
               "profile_sha256": PROFILE, "cpu_bitmap_sha256": sha(bitmap), "plan04_sha256": PLAN04,
               "primary": 960, "helper": 2247, "cpu_full": 21369, "remote04": 3289,
               "cpu_uncovered_scaled": 18080 - offset, "devices": results, "setup_ms": setup_ms,
               "cold_setup_s": cold_setup_s, "window_s": window, "cleanup_reserve_s": 45,
               "order": ORDER, "request_cap": 20, "generated_token_cap": 640,
               "selection": "existing profile order, CPU complement minus 04; device 0 then device 1"})


def render(a):
    b, placement = read_json(a.bindings), read_json(a.placement)
    need(a.phase in ORDER and placement["product"] == PRODUCT, "phase/product binding")
    need(b["engine_sha256"] == ENGINE and b["worker_sha256"] == WORKER, "validated ELF pins")
    need(sha(bounded(b["PROFILE_A"])) == PROFILE and b["endpoints"][0]["plan_sha256"] == PLAN04,
         "unchanged actual engine profile and 04 assignment")
    p = task_plan(a.plan)
    out = Path(a.out)
    need(out.is_absolute(), "new absolute private output")
    os.mkdir(out, 0o700)
    with open(out / "requests.txt", "xb") as f:
        f.write(("".join("GEN 32 " + ",".join(map(str, r["ids"])) + "\n" for r in p["tasks"]) + "QUIT\n").encode("ascii"))
    env = dict(OMP_NUM_THREADS="2", PATH="/usr/bin:/bin", LANG="C", LC_ALL="C", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=b["GPU_SELECT_A"], STRATA_IO_PF_THREADS="2", STRATA_ROUTE_TAIL_SKIP="0",
               STRATA_LOOKAHEAD="0", STRATA_LOOKAHEAD_STATS="0", STRATA_FS_SLOTS="0")
    argv = [b["BIN_ENGINE_A"], "--serve", "--native", b["NATIVE_SHARD_A"], "--pack", b["NATIVE_PACK_A"],
            "--gpu", b["GPU_SELECT_A"], "--mmap-experts", "--greedy", "--pool-workers", "2",
            "--remote-cpu-experts", b["ENDPOINT_CPU_C_ON_A"], "--remote-cpu-deadline-ms", "10000",
            "--remote-cpu-local-manifest", b["MANIFEST_A"], "--remote-cpu-resident", "--remote-cpu-setup-ms", "600000",
            "--expert-cache", "960", "--expert-cache-per-layer", "--expert-profile", b["PROFILE_A"],
            "--expert-cache-device1", "2247", "--remote-expert-opt", "--vram-reserve-mib", "1152",
            "--adapt-every", "0", "--adapt-swaps", "0", "--pcie-frac", "0", "--short-read", "64",
            "--prefill", "64", "--prompt-cache", "0", "--spec", "2", "--suffix-draft", "0", "--lookup-chain", "0",
            "--pipeline-windows", "0", "--max-context", "1024"]
    endpoints = b["endpoints"][:1] if a.phase.startswith("C") else b["endpoints"]
    need(len(b["endpoints"]) == 3 and [r["label"] for r in b["endpoints"]] == ["04", "02-0", "02-1"], "fixed endpoint order")
    for r in endpoints:
        need(isinstance(r["endpoint"], str) and r["endpoint"] and Path(r["plan"]).is_absolute(), "private endpoint/plan binding")
        need(sha(bounded(r["plan"], 4 << 20)) == r["plan_sha256"], "frozen per-endpoint plan bytes")
        argv += ["--remote-gpu-worker", r["endpoint"], r["plan"]]
    argv += ["--remote-gpu-setup-ms", str(placement["setup_ms"]), "--remote-gpu-deadline-ms", "10000"]
    need(all(isinstance(x, str) and x for x in argv), "nonempty exact argv")
    write_json(out / "launch.json", {"phase": a.phase, "argv": argv, "env": env, "stdin": str(out / "requests.txt"),
               "engine_sha256": ENGINE, "plan_sha256": PLAN, "requests": 5, "generated_tokens_max": 160})


def clock(a):
    end = time.monotonic() + a.seconds
    events, offset, tail = [], 0, b""
    need(0 < a.seconds <= 14400 and not Path(a.receipt).exists(), "bounded new clock receipt")
    while time.monotonic() < end:
        with open(a.output, "rb") as f:
            f.seek(offset)
            raw = f.read(1048577 - offset)
        offset += len(raw)
        need(offset <= 1048576, "bounded engine stdout")
        lines = (tail + raw).split(b"\n")
        tail = lines.pop()
        for line in lines:
            need(not line.startswith(b"ERR"), "engine ERR")
            if line.startswith((b"READY", b"DONE ")):
                events.append({"observed_mono": time.monotonic(), "line": line.decode("ascii")})
                need(len(events) <= 6 and events[0]["line"].startswith("READY"), "clock event order")
                temp = a.receipt + ".tmp"
                with open(temp, "w", encoding="ascii") as f:
                    json.dump({"events": events, "scope": "03 file observation, 50 ms polls; includes polling/guard jitter"}, f)
                os.replace(temp, a.receipt)
        if len(events) == 6:
            return
        time.sleep(.05)
    raise TimeoutError("bounded output clock incomplete; retain partial receipt and real process exit")


def output_rows(path, rows):
    """Same DONE/task association; zero CPU share is legal after scale-out."""
    result, tokens, ready = [], [], 0
    for line in bounded(path).decode("utf-8").splitlines():
        if line.startswith("READY"):
            ready += 1
        elif line.startswith("ERR"):
            raise ValueError(line)
        elif line.startswith("T "):
            token = int(line[2:])
            need(len(result) < len(rows) and len(tokens) < rows[len(result)]["max_new"] and 0 <= token < 248320,
                 "task output token bound")
            tokens.append(token)
        elif line.startswith("DONE "):
            f = line.split()[1:]
            need(len(result) < len(rows) and len(f) >= 15, "DONE schema/count")
            row = rows[len(result)]
            count, prompt = int(f[0]), int(f[1])
            pp, dec = float(f[2]), float(f[3])
            extra = dict(v.split("=", 1) for v in f[15:] if "=" in v)
            entries, exchange = int(extra["worker_entries"]), float(extra["worker_req_wall"])
            need(count == len(tokens) and 0 < count <= 32 and prompt == len(row["ids"]) and
                 f[4] in ("stop", "length") and (f[4] != "length" or count == 32) and int(f[7]) == 0 and
                 entries >= 0 and all(math.isfinite(x) and x >= 0 for x in (pp, dec, exchange)), "complete finite DONE")
            result.append({"name": row["name"], "tokens": tokens, "finish": f[4], "prompt_ms": pp, "decode_ms": dec,
                           "prompt_tokens": prompt, "worker_entries": entries, "worker_exchange_ms": exchange,
                           "decode_tokens_per_s": 1000 * count / dec if dec else None, "raw_done": line})
            tokens = []
    need(ready == 1 and not tokens and len(result) == 5, "complete five-task stream")
    return result


def phase_report(tasks, tok, plan, root, phase):
    exit_receipt = read_json(root / "exit.json")
    need(exit_receipt.get("rc") == 0 and exit_receipt.get("phase") == phase and
         exit_receipt.get("engine_sha256") == ENGINE and exit_receipt.get("plan_sha256") == PLAN,
         "actual exit/phase/engine/plan receipt")
    rows = output_rows(root / "workload.out", plan["tasks"])
    for row in rows:
        text = tok.decode(row["tokens"])
        for terminal in ("<|im_end|>", "<|endoftext|>"):
            if text.endswith(terminal):
                text = text[:-len(terminal)]
        passed = tasks.check_answer(row["name"], text)
        row.update(text=text, strict_pass=passed and row["finish"] == "stop", output_cap_reached=row["finish"] == "length")
    return {"complete": True, "tasks": rows, "wall": tasks.output_clock(root / "output-clock.json", rows),
            "prompt_ms": sum(r["prompt_ms"] for r in rows), "decode_ms": sum(r["decode_ms"] for r in rows),
            "tokens": sum(len(r["tokens"]) for r in rows)}


def report(a):
    need(sha(bounded(a.evaluator)) == EVALUATOR, "unchanged retained answer evaluator")
    spec = importlib.util.spec_from_file_location("paired_tasks", a.evaluator)
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    plan = task_plan(a.plan)
    tok, _ = tasks.tokenizer(a.pack)
    need({name: sha(bounded(Path(a.pack) / name, 32 << 20)) for name in tasks.METADATA} == plan["tokenizer_metadata_sha256"],
         "retained decoder metadata")
    phases = {}
    for phase in ORDER:
        root = Path(a.root) / phase
        try:
            phases[phase] = phase_report(tasks, tok, plan, root, phase)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            phases[phase] = {"complete": False, "reason": type(exc).__name__ + ":" + str(exc)}
    print(json.dumps({"product": PRODUCT, "engine_sha256": ENGINE, "task_plan_sha256": PLAN, "order": ORDER,
          "phases": phases, "historical_numerical_verdict": "NUMERICAL_FAIL unchanged",
          "limits": "two samples per mode, ABBA; no general scaling/quality claim; compare actual work and setup separately"}, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("place")
    for n in ("admission", "profile", "cpu-bitmap", "plan04", "out"):
        p.add_argument("--" + n, required=True)
    p = sub.add_parser("render")
    for n in ("bindings", "placement", "plan", "phase", "out"):
        p.add_argument("--" + n, required=True)
    p = sub.add_parser("clock")
    for n in ("output", "receipt"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--seconds", required=True, type=float)
    p = sub.add_parser("report")
    for n in ("evaluator", "plan", "pack", "root"):
        p.add_argument("--" + n, required=True)
    a = parser.parse_args()
    {"place": place, "render": render, "clock": clock, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
