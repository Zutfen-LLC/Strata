"""Candidate identity and offline comparison for the CPU-first scheduling change."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired_g_continuation as gpu
import paired_tasks as tasks

TAG = "CPU-FIRST-REMOTE-OVERLAP-V1-20261022"
BASE = "e122215a28791062259f391a810d232e10c542d7"


def candidate_identity(config):
    candidate = config.get("candidate_engine_sha256", "")
    accepted = config.get("retained_engine_sha256", "")
    revision = config.get("diagnostic_revision", "")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision) or \
            config.get("candidate_base_revision") != BASE or config.get("candidate_source_revision") != revision or \
            len(candidate) != 64 or len(accepted) != 64 or candidate == accepted or any(
                c not in "0123456789abcdef" for c in candidate + accepted):
        raise ValueError("reviewed scheduling candidate and accepted engine identities required")
    if config.get("role") == "ROLE_ENGINE_A" and config.get("binary_sha256") != candidate:
        raise ValueError("actual engine must match the pinned candidate ELF")


def prior_g(plan, root, engine_config, accepted_sha):
    cfg = tasks.read_json(engine_config)
    if cfg.get("diagnostic") != gpu.TAG or cfg.get("diagnostic_revision") != BASE or \
            cfg.get("role") != "ROLE_ENGINE_A" or cfg.get("plan_sha256") != gpu.PLAN_SHA or cfg.get("binary_sha256") != accepted_sha:
        raise ValueError("accepted G config/plan/engine binding")
    root = Path(root)
    answers = tasks.protocol(root / "G0/workload.out", plan["tasks"])
    probes = tasks.protocol(root / "G1/workload.out", plan["probes"])
    clock = tasks.output_clock(root / "G0/output-clock.json", answers)
    likelihood = tasks.likelihood(root / "G1.logpos", plan["probes"])
    for i, row in enumerate(probes):
        path = root / ("G1.first-logits." + str(i))
        if tasks.compare_logits(path, path)["top1_g"] != row["tokens"][0]:
            raise ValueError("accepted G probe/logit association")
    if sum(len(r["tokens"]) for r in answers + probes) != 53:
        raise ValueError("accepted G receipt expected 51 task and 2 probe tokens")
    return {"tasks": answers, "probes": probes, "performance_clock": clock, "likelihood": likelihood,
            "accepted_engine_sha256": accepted_sha, "source_revision": BASE}


def report(plan, pack, b_root, b_config, old_root, old_config, new_root, candidate_config):
    cfg = tasks.read_json(candidate_config)
    if cfg.get("role") != "ROLE_ENGINE_A":
        raise ValueError("candidate engine config required for report")
    candidate_identity(cfg)
    if cfg.get("diagnostic") != TAG or cfg.get("plan_sha256") != gpu.PLAN_SHA:
        raise ValueError("candidate comparison config")
    old = prior_g(plan, old_root, old_config, cfg["retained_engine_sha256"])
    result = gpu.continuation_report(plan, pack, b_root, new_root, b_config)
    if result["retained_baseline"]["engine_binary_sha256"] != cfg["retained_engine_sha256"]:
        raise ValueError("B/G accepted engine identity mismatch")
    new = result["modes"]["G"]
    tokens_equal = all(a["tokens"] == b["tokens"] for name in ("tasks", "probes") for a, b in zip(old[name], new[name]))
    logits_equal = []
    for i in range(2):
        name = "G1.first-logits." + str(i)
        logits_equal.append((Path(old_root) / name).read_bytes() == (Path(new_root) / name).read_bytes())
    likelihood_equal = old["likelihood"] == new["likelihood"]
    result.update(diagnostic=TAG, historical_requests=31, historical_generated_tokens=941,
                  cumulative_started_request_ceiling=38, cumulative_generated_token_ceiling=1103,
                  accepted_g=old, candidate_engine_sha256=cfg["candidate_engine_sha256"],
                  candidate_source_revision=cfg["candidate_source_revision"],
                  same_mode_tokens_bitwise=tokens_equal, same_mode_probe_logits_bitwise=logits_equal,
                  same_mode_reference_likelihood_equal=likelihood_equal,
                  scheduling_verdict="NO_OBSERVED_DRIFT" if tokens_equal and all(logits_equal) and likelihood_equal else "SCHEDULING_REGRESSION")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "pack", "b-root", "b-engine-config", "prior-g-root", "prior-g-engine-config", "g-root", "candidate-engine-config"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(report(gpu.retained_plan(args.plan), args.pack, args.b_root, args.b_engine_config,
                            args.prior_g_root, args.prior_g_engine_config, args.g_root, args.candidate_engine_config),
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
