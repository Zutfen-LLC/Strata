"""Frozen, opt-in backend answer screen; no model execution, network or generated-code execution."""
import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'harness'))
import bundle
import compare_logits

TEMPLATE = '<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
# Frozen before any CPU or Vulkan outputs. Each case is worth one point.
CASES = [
    ('arithmetic', 'Return only the integer: 17 * 23.', 'exact', '391'),
    ('arithmetic', 'Return only the integer: (84 - 19) / 5.', 'exact', '13'),
    ('extraction', 'Record: name=Mira; ticket=BX-204; city=Lima. Return only the ticket value.', 'exact', 'BX-204'),
    ('extraction', 'Text: red=7, blue=12, green=4. Return only the value of blue.', 'exact', '12'),
    ('structured', 'Return a JSON object with exactly two keys: city with string value Oslo, and count with integer value 3. No prose.', 'json', {'city': 'Oslo', 'count': 3}),
    ('structured', 'Return only a JSON array containing the integers 9, 2, 5 sorted ascending.', 'json', [2, 5, 9]),
    ('instruction', 'Return only the word READY in uppercase. No punctuation.', 'exact', 'READY'),
    ('instruction', 'Return exactly two lines: first line alpha, second line beta. No other text.', 'exact', 'alpha\nbeta'),
    ('reasoning', 'All dax are blue. No blue object is round. Can a dax be round? Return only yes or no.', 'exact', 'no'),
    ('reasoning', 'Ada is taller than Ben. Ben is taller than Cy. Who is shortest? Return only the name.', 'exact', 'Cy'),
    ('code', 'Python code: xs = [2, 4, 6]; print(sum(x * 2 for x in xs)). Return only the printed integer.', 'exact', '24'),
    ('code', "Python code: s = 'strata'; print(s[1:4]). Return only the printed string, without quotes.", 'exact', 'tra'),
]
MAX_NEW = 32
MAX_PROMPT = 224


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with pathlib.Path(path).open('x') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)


def tokenizer(gguf):
    from strata_tokenizer import Tokenizer
    return Tokenizer.from_gguf(gguf)  # existing metadata reader; no model download or fallback


def prepare(tok, output):
    output = pathlib.Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    lines = []
    for _, prompt, _, _ in CASES:
        text = TEMPLATE.format(p=prompt)
        ids = tok.encode(text, parse_special=True)
        if not ids or len(ids) > MAX_PROMPT or tok.decode(ids, errors='strict') != text:
            raise ValueError('prompt bound/roundtrip failed')
        lines.append('GEN ' + str(MAX_NEW) + ' ' + ' '.join(map(str, ids)))
    requests = ('\n'.join(lines) + '\n').encode()
    (output / 'requests').write_bytes(requests)
    # Seal the vocabulary as well as prompts; rechecked when decoding retained outputs.
    vocab = json.dumps(tok.tokens, ensure_ascii=False).encode()
    write(output / 'quality_manifest', dict(version=1, cases=CASES, template=TEMPLATE,
          max_new=MAX_NEW, max_prompt=MAX_PROMPT, requests_sha256=digest(requests),
          vocabulary_sha256=digest(vocab), scorer_sha256=bundle.sha(__file__)))


def correct(text, rule, expected):
    # Only surrounding whitespace and ONE terminal protocol EOS are removed. No reasoning/fence extraction.
    for eos in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(eos):
            text = text[:-len(eos)]
            break
    text = text.strip()
    if rule == 'exact':
        return text == expected
    try:
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result: raise ValueError('duplicate JSON key')
                result[key] = value
            return result
        actual = json.loads(text, object_pairs_hook=pairs,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except (ValueError, TypeError):
        return False


def participation(text, count):
    # One endpoint only. Existing startup report, then one cumulative report per completed GEN.
    reports = [line for line in text.splitlines() if 'Vulkan/native-Q8-host-boundaries approximate' in line]
    if len(reports) != count + 1:
        raise ValueError('missing/extra Vulkan endpoint reports')
    previous = 0
    entries = []
    identities = set()
    for i, line in enumerate(reports):
        fields = dict(re.findall(r'([a-z_]+)=([^\s;]+)', line))
        if ' READY transport=mTLS ' not in line or int(fields['failures']) != 0:
            raise ValueError('Vulkan transport/runtime failure')
        identities.add(tuple(fields[k] for k in ('model', 'layout', 'placement', 'session')))
        current = int(fields['entries'])
        if current < previous or (i == 0 and current != 0):
            raise ValueError('invalid Vulkan entry counters')
        if i: entries.append(current - previous)
        previous = current
    if len(identities) != 1: raise ValueError('Vulkan identity changed')
    return entries


def comparable(a, b):
    def argv(cfg):
        args = cfg['argv'][1:]
        cleaned = []
        i = 0
        while i < len(args):
            flag = args[i]
            if flag == '--remote-gpu-worker':
                i += 3
            elif flag.startswith('--remote-gpu-'):
                # Only transport/setup controls may differ; numerical/placement changes are refused.
                if flag not in ('--remote-gpu-tls-ca', '--remote-gpu-tls-cert', '--remote-gpu-tls-key',
                                '--remote-gpu-tls-peer', '--remote-gpu-deadline-ms', '--remote-gpu-setup-ms'):
                    raise ValueError('unapproved GPU option difference: ' + flag)
                i += 2
            else:
                cleaned.append('@manifest' if flag == cfg['assets']['manifest'] else flag)
                i += 1
        return cleaned
    if '--remote-gpu-worker' in a['argv'] or b['argv'].count('--remote-gpu-worker') != 1:
        raise ValueError('screen requires CPU-backed baseline and one Vulkan endpoint')
    j = b['argv'].index('--remote-gpu-worker')
    if not b['argv'][j+1].startswith('vulkan@'): raise ValueError('candidate must use Vulkan')
    for asset in ('engine', 'worker', 'manifest', 'requests', 'quality_manifest'):
        if bundle.sha(a['assets'][asset]) != bundle.sha(b['assets'][asset]):
            raise ValueError('arm asset differs: ' + asset)
    if argv(a) != argv(b): raise ValueError('coordinator settings differ beyond offload transport')
    for key in ('env', 'worker_env', 'monitor_env', 'model_paths', 'worker_bytes', 'request_seconds',
                'attach_root', 'cpu_route', 'gpu_uuids', 'worker_peers'):
        if a.get(key) != b.get(key): raise ValueError('arm setting differs: ' + key)
    def worker_args(cfg):
        bindings = {cfg['assets'][name]: '@' + name for name in ('worker', 'manifest')}
        return [bindings.get(arg, arg) for arg in cfg['worker_argv']]
    if worker_args(a) != worker_args(b): raise ValueError('worker settings differ')
    for key in ('workers', 'loader_threads', 'caller_mask', 'pool_masks', 'engine_main_mask', 'requests'):
        if a['arms'][0][key] != b['arms'][0][key]: raise ValueError('arm policy differs: ' + key)


def score(tok, baseline, candidate, output):
    roots = [pathlib.Path(baseline), pathlib.Path(candidate)]
    configs = [bundle.verify(root) for root in roots]
    comparable(*configs)
    for helper in bundle.HELPERS:
        if bundle.sha(roots[0] / helper) != bundle.sha(roots[1] / helper):
            raise ValueError('runtime helper differs: ' + helper)
    manifest = json.loads(pathlib.Path(configs[0]['assets']['quality_manifest']).read_text())
    if manifest['cases'] != [list(c) for c in CASES] or manifest['scorer_sha256'] != bundle.sha(__file__):
        raise ValueError('frozen suite/scorer differs')
    if manifest['vocabulary_sha256'] != digest(json.dumps(tok.tokens, ensure_ascii=False).encode()):
        raise ValueError('decode vocabulary differs')
    arms = []
    for cfg, root in zip(configs, roots):
        if (root / 'failure.json').exists(): raise ValueError('failed runtime arm')
        summary = json.loads((root / 'summary.json').read_text())
        arm = summary['arms'][0]
        rows = arm['requests']
        if [r['index'] for r in rows] != list(range(1, 13)): raise ValueError('incomplete/reordered suite')
        for row in rows:
            if row['finish'] not in ('stop', 'length') or not 0 < len(row['tokens']) <= MAX_NEW or \
               row['output_count'] != len(row['tokens']) or \
               any(type(t) is not int or not 0 <= t < len(tok.tokens) for t in row['tokens']):
                raise ValueError('invalid completion/token IDs')
        arms.append(arm)
    entries = participation((roots[1] / arms[1]['arm'] / 'stderr.log').read_text(), 12)
    rows = []
    for i, (category, prompt, rule, expected) in enumerate(CASES):
        pair = [arm['requests'][i] for arm in arms]
        answers = [tok.decode(r['tokens'], errors='strict') for r in pair]
        points = [correct(answer, rule, expected) for answer in answers]
        logits = [compare_logits.read(root / (arm['arm'] + '-first-logits.' + str(i)))
                  for root, arm in zip(roots, arms)]
        rows.append(dict(index=i+1, category=category, baseline_correct=points[0], candidate_correct=points[1],
             correctness_disagreement=points[0] != points[1], baseline_answer=answers[0], candidate_answer=answers[1],
             token_ids_equal=pair[0]['tokens'] == pair[1]['tokens'],
             token_positions_different=sum(x != y for x, y in zip(pair[0]['tokens'], pair[1]['tokens'])) +
                 abs(len(pair[0]['tokens']) - len(pair[1]['tokens'])),
             baseline_tokens=pair[0]['tokens'], candidate_tokens=pair[1]['tokens'], vulkan_entries=entries[i],
             exact_first_logits_pass=logits[0].tobytes() == logits[1].tobytes(),
             fixed_numeric_diagnostic=compare_logits.compare(*logits),
             runtime=[{k: r[k] for k in ('prompt_s', 'decode_s', 'finish')} for r in pair]))
    totals = [sum(row[key] for row in rows) for key in ('baseline_correct', 'candidate_correct')]
    result = ('inconclusive-baseline' if totals[0] != 12 else
              'inconclusive-participation' if not all(entries) else
              'screen-pass' if totals[1] == 12 else 'screen-regression')
    write(output, dict(mode='exploratory-backend-quality', result=result, baseline_score=totals[0],
          candidate_score=totals[1], total=12, cases=rows, numeric_limits=compare_logits.LIMITS,
          scope='One real Q8 expert on one V340 die; full CPU backing; one run per arm. '
                'No broad quality equivalence, reproducibility, full-GPU or readiness claim.'))
    return 0 if result == 'screen-pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True, help='existing identical model GGUF; metadata only')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True)
    p = sub.add_parser('score')
    p.add_argument('--baseline', required=True); p.add_argument('--candidate', required=True)
    p.add_argument('--out', required=True)
    args = parser.parse_args()
    tok = tokenizer(args.gguf)
    if args.command == 'prepare': prepare(tok, args.out); return 0
    return score(tok, args.baseline, args.candidate, args.out)


if __name__ == '__main__': raise SystemExit(main())
