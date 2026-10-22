"""Bounded retained-B check and unchanged evaluator for a separate G-only run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_tasks as tasks

TAG = "GPU-PAIRED-G-CONTINUATION-V1-20261022"
PLAN_SHA = "1c81fa79b9a26f66b0a4f7b2e6eae92858c82f47de827f08751b312948a21792"
BASELINE_REVISION = "e0d11fc765db3f8b07ea70d3535738a630fc129e"
PHASES = ("G0", "G1")
REQUEST_CAP, TOKEN_CAP = 7, 162


def retained_plan(path):
    with open(path, "rb") as source:
        raw = source.read(1048577)
    if not 0 < len(raw) <= 1048576 or hashlib.sha256(raw).hexdigest() != PLAN_SHA:
        raise ValueError("exact retained paired plan required; no retokenization")
    plan = json.loads(raw)
    tasks.validate_plan(plan)
    return plan


def check_baseline(plan, root, engine_config):
    """Original config/plan association plus complete bounded B outputs."""
    cfg = tasks.read_json(engine_config)
    if cfg.get("diagnostic") != tasks.TAG or cfg.get("diagnostic_revision") != BASELINE_REVISION or \
            cfg.get("role") != "ROLE_ENGINE_A" or cfg.get("plan_sha256") != PLAN_SHA:
        raise ValueError("retained B engine config/plan binding")
    root = Path(root)
    answers = tasks.protocol(root / "B0/workload.out", plan["tasks"])
    probes = tasks.protocol(root / "B1/workload.out", plan["probes"])
    tasks.output_clock(root / "B0/output-clock.json", answers)
    tasks.likelihood(root / "B1.logpos", plan["probes"])
    for i, row in enumerate(probes):
        path = root / ("B1.first-logits." + str(i))
        if tasks.compare_logits(path, path)["top1_b"] != row["tokens"][0]:
            raise ValueError("retained B probe/logit association")
    tokens = sum(len(row["tokens"]) for row in answers + probes)
    if tokens != 53:
        raise ValueError("retained B receipt expected 51 task and 2 probe tokens")
    names = ("B0/workload.out", "B0/output-clock.json", "B1/workload.out", "B1.logpos",
             "B1.first-logits.0", "B1.first-logits.1")
    hashes = {}
    for name in names:
        with open(root / name, "rb") as source:
            raw = source.read(1048577)
        if not 0 < len(raw) <= 1048576:
            raise ValueError("bounded retained B artifact")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {"complete": True, "revision": BASELINE_REVISION, "plan_sha256": PLAN_SHA,
            "started_requests": 7, "generated_tokens": tokens, "artifact_sha256": hashes,
            "engine_binary_sha256": cfg["binary_sha256"], "separate_run_comparison": True}


def continuation_report(plan, pack, b_root, g_root, engine_config):
    baseline = check_baseline(plan, b_root, engine_config)
    hashes = {name: hashlib.sha256((Path(pack) / name).read_bytes()).hexdigest() for name in tasks.METADATA}
    if hashes != plan["tokenizer_metadata_sha256"]:
        raise ValueError("retained decoder metadata mismatch")
    result = tasks.report(plan, pack, b_root, g_root)  # Same checks/metrics; decode only, no make_plan().
    result.update(diagnostic=TAG, request_cap=REQUEST_CAP, generated_token_cap=TOKEN_CAP,
                  retained_baseline=baseline, separate_run_comparison=True,
                  historical_requests=24, historical_generated_tokens=888,
                  cumulative_started_request_ceiling=31, cumulative_generated_token_ceiling=1050,
                  baseline_format_note="Supplied B code was substantively correct but fenced; strict checker remains unchanged.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--b-root", required=True)
    parser.add_argument("--b-engine-config", required=True)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check")
    finish = modes.add_parser("report")
    finish.add_argument("--pack", required=True)
    finish.add_argument("--g-root", required=True)
    args = parser.parse_args()
    plan = retained_plan(args.plan)
    result = check_baseline(plan, args.b_root, args.b_engine_config) if args.mode == "check" else \
        continuation_report(plan, args.pack, args.b_root, args.g_root, args.b_engine_config)
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
