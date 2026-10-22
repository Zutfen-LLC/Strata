"""Fixed small task set and offline report. Import never starts a workload."""
import argparse
from array import array
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

TAG = "GPU-PAIRED-PRACTICAL-V1-20261022"
PHASES = ("B0", "B1", "G0", "G1")
WINDOW = 1500
REQUEST_CAP = 14
TOKEN_CAP = 324
METADATA = ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja")
TASKS = (
    ("arithmetic", "18 pens minus 7 plus 5: how many? Reply only with the integer.", "16"),
    ("ordering", "Sort 9,2,11,4 ascending. Reply only with a JSON array.", "[2,4,9,11]"),
    ("extraction", "Mira:7; Jon:12. Return only JSON {\"name\":...,\"score\":...} for the higher score.", '{"name":"Jon","score":12}'),
    ("logic", "All zogs are blue. No blue thing is red. Can a zog be red? Reply only yes or no.", "no"),
    ("code", "Write Python def is_even(n) returning n%2==0. Output only the function, no explanation.", "def is_even(n):\n    return n % 2 == 0"),
)
PROBES = (0, 1)  # Teacher-force the declared answer to these two tasks; GEN 1 follows it.


def read_json(path, cap=1048576):
    with open(path, "rb") as source:
        raw = source.read(cap + 1)
    if not 0 < len(raw) <= cap:
        raise ValueError("bounded JSON input required")
    return json.loads(raw)


def tokenizer(pack):
    """Use the existing tokenizer and template; no GGUF/model tensor reads."""
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root), str(root / "tools")]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    pack = Path(pack)
    vocab = read_json(pack / "vocab.json", 32 * 1048576)
    tokens = [None] * len(vocab)
    for text, index in vocab.items():
        if type(index) is not int or not 0 <= index < len(tokens) or tokens[index] is not None:
            raise ValueError("vocabulary IDs")
        tokens[index] = text
    meta = read_json(pack / "tokenizer.json")
    merges = (pack / "merges.txt").read_text(encoding="utf-8").splitlines()
    tok = Tokenizer(tokens, merges, read_json(pack / "token_type.json", 4 * 1048576),
                    pre=meta.get("pre", "qwen35"), special_ids=meta.get("special_ids"))
    return tok, ChatTemplate(pack / "chat_template.jinja")


def make_plan(tok, template):
    tasks, probes = [], []
    for index, (name, prompt, answer) in enumerate(TASKS):
        prefix = template.render([{"role": "user", "content": prompt}], enable_thinking=False)
        ids = tok.encode(prefix, parse_special=True)
        tasks.append({"name": name, "ids": ids, "max_new": 32})
        if index in PROBES:
            full = tok.encode(prefix + answer, parse_special=True)
            boundary = 0
            while boundary < min(len(ids), len(full)) and ids[boundary] == full[boundary]:
                boundary += 1
            if not 0 < boundary < len(full):
                raise ValueError("answer boundary must contain scored tokens")
            probes.append({"name": name, "ids": full, "max_new": 1, "answer_start": boundary})
    plan = {"diagnostic": TAG, "tasks": tasks, "probes": probes}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if plan.get("diagnostic") != TAG or len(plan.get("tasks", ())) != 5 or len(plan.get("probes", ())) != 2:
        raise ValueError("fixed paired plan")
    for items, names, cap in ((plan["tasks"], [t[0] for t in TASKS], 32),
                             (plan["probes"], [TASKS[i][0] for i in PROBES], 1)):
        if [r["name"] for r in items] != names:
            raise ValueError("fixed task order")
        for row in items:
            if row["max_new"] != cap or not 2 <= len(row["ids"]) <= 64 or any(
                    type(t) is not int or not 0 <= t < 248320 for t in row["ids"]):
                raise ValueError("native remote-CPU prompt bound is 64 tokens; never truncate")
            if cap == 1 and not 0 < row["answer_start"] < len(row["ids"]):
                raise ValueError("teacher-forced answer range")
    for phase in PHASES:
        if len(requests(plan, phase, validate=False)) > 4096:
            raise ValueError("prelaunch stdin pipe bound")


def phase_rows(plan, phase):
    if phase not in PHASES:
        raise ValueError("paired phase")
    return plan["tasks" if phase.endswith("0") else "probes"]


def requests(plan, phase, *, validate=True):
    if validate:
        validate_plan(plan)
    return ("".join("GEN " + str(r["max_new"]) + " " + ",".join(map(str, r["ids"])) + "\n"
                    for r in phase_rows(plan, phase)) + "QUIT\n").encode("ascii")


def check_answer(name, text):
    """Transparent strict format checks; never execute generated Python."""
    value = text.strip()
    if name == "arithmetic":
        return value == "16"
    if name == "logic":
        return value.lower() == "no"
    if name in ("ordering", "extraction"):
        try:
            actual = json.loads(value)
        except (ValueError, TypeError):
            return False
        expected = [2, 4, 9, 11] if name == "ordering" else {"name": "Jon", "score": 12}
        return actual == expected and (name != "extraction" or type(actual.get("score")) is int)
    if name == "code":
        try:
            actual = ast.dump(ast.parse(value), include_attributes=False)
            expected = ast.dump(ast.parse(TASKS[-1][2]), include_attributes=False)
            return actual == expected
        except (SyntaxError, ValueError):
            return False
    raise ValueError("unknown task")


def protocol(path, rows):
    raw = Path(path).read_bytes()
    if len(raw) > 1048576:
        raise ValueError("bounded protocol output")
    result, tokens, ready = [], [], 0
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if line.startswith("READY"):
            ready += 1
        elif line.startswith("ERR"):
            raise ValueError("product ERR: " + line)
        elif line.startswith("T "):
            token = int(line[2:])
            if not 0 <= token < 248320 or len(result) >= len(rows) or len(tokens) >= rows[len(result)]["max_new"]:
                raise ValueError("output token cap")
            tokens.append(token)
        elif line.startswith("DONE "):
            f = line.split()[1:]
            if len(result) >= len(rows) or len(f) < 15:
                raise ValueError("DONE schema/count")
            row = rows[len(result)]
            generated, prompt_n = int(f[0]), int(f[1])
            prompt_ms, decode_ms = float(f[2]), float(f[3])
            if generated != len(tokens) or not 0 < generated <= row["max_new"] or prompt_n != len(row["ids"]) or \
                    f[4] not in ("stop", "length") or (f[4] == "length" and generated != row["max_new"]) or \
                    not all(math.isfinite(x) and x >= 0 for x in (prompt_ms, decode_ms)) or int(f[7]) != 0:
                raise ValueError("incomplete/nonfinite/cancelled/mismatched DONE or unexpected prefix reuse")
            extra = dict(field.split("=", 1) for field in f[15:] if "=" in field)
            entries, worker_ms = int(extra["worker_entries"]), float(extra["worker_req_wall"])
            if entries <= 0 or not math.isfinite(worker_ms) or worker_ms < 0:
                raise ValueError("required CPU participation")
            result.append({"name": row["name"], "tokens": tokens, "finish": f[4],
                           "prompt_ms": prompt_ms, "decode_ms": decode_ms,
                           "engine_request_ms": prompt_ms + decode_ms,
                           "decode_tokens_per_s": 1000 * generated / decode_ms if decode_ms else None,
                           "prompt_tokens": prompt_n, "read_tokens": int(f[13]),
                           "worker_entries": entries, "worker_exchange_ms": worker_ms,
                           "raw_done": line})
            tokens = []
    if ready != 1 or tokens or len(result) != len(rows):
        raise ValueError("missing READY/DONE or partial output")
    return result


def likelihood(path, probes):
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if len(lines) > 128:
        raise ValueError("logpos row cap")
    groups, current = [], {}
    for line in lines:
        f = line.split()
        if len(f) != 8:
            raise ValueError("logpos schema")
        pos, target, lp = int(f[0]), int(f[1]), float(f[2])
        if pos == 0 and current:
            groups.append(current)
            current = {}
        if pos in current or not math.isfinite(lp) or lp > 0:
            raise ValueError("logpos position/probability")
        current[pos] = (target, lp)
    if current:
        groups.append(current)
    if len(groups) != len(probes):
        raise ValueError("logpos request association")
    result = []
    for group, row in zip(groups, probes):
        if set(group) != set(range(len(row["ids"]) - 1)) or any(
                target != row["ids"][pos + 1] for pos, (target, _) in group.items()):
            raise ValueError("logpos must cover the actual identical supplied sequence")
        values = [group[p - 1][1] for p in range(row["answer_start"], len(row["ids"]))]
        result.append({"name": row["name"], "answer_tokens": len(values), "answer_logprobs": values,
                       "answer_mean_nll": -sum(values) / len(values)})
    return result


def compare_logits(a_path, b_path):
    def load(path):
        raw = Path(path).read_bytes()
        if len(raw) != 248320 * 4 or sys.byteorder != "little":
            raise ValueError("finite 248320 little-endian float32 logits required")
        values = array("f")
        values.frombytes(raw)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite logits")
        return values
    a, b = load(a_path), load(b_path)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    denom = sum(abs(x - mean_a) for x in a)
    rel = sum(abs((x - mean_a) - (y - mean_b)) for x, y in zip(a, b)) / max(denom, 1e-12)
    ma, mb = max(a), max(b)
    za, zb = sum(math.exp(x - ma) for x in a), sum(math.exp(x - mb) for x in b)
    la, lb = ma + math.log(za), mb + math.log(zb)
    kl = sum(math.exp(x - la) * ((x - la) - (y - lb)) for x, y in zip(a, b))
    return {"centered_rel_l1": rel, "kl_b_to_g_nats": kl,
            "top1_b": max(range(len(a)), key=a.__getitem__), "top1_g": max(range(len(b)), key=b.__getitem__),
            "within_unchanged_0.03_limits": rel <= .03 and kl <= .03}


def output_clock(path, rows):
    clock = read_json(path)
    events = clock["events"]
    if len(events) != len(rows) + 1 or not events[0]["line"].startswith("READY") or any(
            event["line"] != row["raw_done"] for event, row in zip(events[1:], rows)):
        raise ValueError("clock/output association")
    stamps = [event["observed_mono"] for event in events]
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in stamps) or any(
            b < a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("monotonic clock")
    return {"scope": clock["scope"], "ready_to_last_done_s": stamps[-1] - stamps[0],
            "completion_intervals_s": [b - a for a, b in zip(stamps, stamps[1:])]}


def report(plan, pack, b_root, g_root):
    validate_plan(plan)
    tok, _ = tokenizer(pack)
    modes = {}
    for mode, root in (("B", Path(b_root)), ("G", Path(g_root))):
        answers = protocol(root / (mode + "0/workload.out"), plan["tasks"])
        for row in answers:
            # Decode first, then remove only terminal control tokens, never reasoning/prose/fences.
            text = tok.decode(row["tokens"])
            for control in ("<|im_end|>", "<|endoftext|>"):
                if text.endswith(control):
                    text = text[:-len(control)]
            checked = check_answer(row["name"], text)
            row.update(text=text, check_pass=checked, task_pass=checked and row["finish"] == "stop",
                       outcome="OUTPUT_CAP_REACHED" if row["finish"] == "length" else
                               "TASK_CHECK_PASS" if checked else "TASK_CHECK_FAIL")
        probe_output = protocol(root / (mode + "1/workload.out"), plan["probes"])
        modes[mode] = {"tasks": answers, "probes": probe_output,
                       "performance_clock": output_clock(root / (mode + "0/output-clock.json"), answers),
                       "decode_tokens_per_s_aggregate": 1000 * sum(len(r["tokens"]) for r in answers) /
                           max(sum(r["decode_ms"] for r in answers), 1e-12),
                       "likelihood": likelihood(root / (mode + "1.logpos"), plan["probes"])}
    numerical = [compare_logits(Path(b_root) / ("B1.first-logits." + str(i)),
                                Path(g_root) / ("G1.first-logits." + str(i))) for i in range(2)]
    return {"diagnostic": TAG, "historical_numerical_verdict": "NUMERICAL_FAIL (unchanged)",
            "request_cap": REQUEST_CAP, "generated_token_cap": TOKEN_CAP, "modes": modes,
            "common_input_logits": numerical, "task_pass_counts": {m: sum(r["task_pass"] for r in v["tasks"]) for m, v in modes.items()},
            "output_cap_counts": {m: sum(r["outcome"] == "OUTPUT_CAP_REACHED" for r in v["tasks"]) for m, v in modes.items()},
            "limits": "Five simple tasks; strict format/code-form checks; one fixed B-then-G order; no general quality claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    finish = sub.add_parser("report")
    finish.add_argument("--plan", required=True)
    finish.add_argument("--b-root", required=True)
    finish.add_argument("--g-root", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        tok, template = tokenizer(args.pack)
        plan = make_plan(tok, template)
        hashes = {name: hashlib.sha256((Path(args.pack) / name).read_bytes()).hexdigest() for name in METADATA}
        plan["tokenizer_metadata_sha256"] = hashes
        with open(args.output, "x", encoding="utf-8") as output:
            json.dump(plan, output, allow_nan=False)
        print("PAIRED_PLAN_READY requests=14 generated_tokens_max=324 prompt_tokens_max=64")
    else:
        plan = read_json(args.plan)
        regenerated = make_plan(*tokenizer(args.pack))
        current_hashes = {name: hashlib.sha256((Path(args.pack) / name).read_bytes()).hexdigest() for name in METADATA}
        if current_hashes != plan["tokenizer_metadata_sha256"] or any(
                plan[key] != regenerated[key] for key in ("diagnostic", "tasks", "probes")):
            raise ValueError("plan must match fixed prompts and current tokenizer metadata")
        print(json.dumps(report(plan, args.pack, args.b_root, args.g_root), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
