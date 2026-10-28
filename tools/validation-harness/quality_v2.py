"""Independent frozen 100-case backend regression screen. Original quality.py stays immutable."""
import argparse
import collections
import fractions
import json
import math
import pathlib
import re

import quality as Q

HERE = pathlib.Path(__file__).resolve().parent
SUITE = 'v2-100'
COUNT = 100
MAX_NEW = 32
MAX_PROMPT = 224


def inventory():
    cases = json.loads((HERE / 'quality_v2_cases.json').read_text())
    counts = collections.Counter(c['category'] for c in cases)
    if counts != dict(arithmetic=17, extraction=17, structured=17, instruction=17, reasoning=16, code=16):
        raise ValueError('invalid category inventory')
    if len({c['id'] for c in cases}) != COUNT or len({c['prompt'] for c in cases}) != COUNT:
        raise ValueError('duplicate case ID/prompt')
    for case in cases:
        if case['semantic_rule'] not in ('exact', 'yes_no', 'json', 'number') or not case['rationale']:
            raise ValueError('invalid case rule/rationale')
        if case['semantic_rule'] == 'yes_no' and case['expected'] not in ('yes', 'no'):
            raise ValueError('invalid yes/no gold')
        if case['format_rule'] not in ('json-value', 'reduced-fraction'): re.compile(case['format_rule'])
    return cases


def sources():
    paths = [HERE / name for name in ('quality_v2.py', 'quality_v2_cases.json', 'quality.py', 'compare_logits.py')]
    paths += [Q.ROOT / 'tools' / name for name in ('strata_tokenizer.py', 'gguf_reader.py')]
    return {p.name: Q.bundle.sha(p) for p in paths}


def prepare(tok, output, placement_plan):
    cases = inventory()
    plan_sha256 = Q.bundle.sha(placement_plan)
    output = pathlib.Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    lines = []
    for case in cases:
        prompt = Q.TEMPLATE.format(p=case['prompt'])
        ids = tok.encode(prompt, parse_special=True)
        if not ids or len(ids) > MAX_PROMPT or tok.decode(ids, errors='strict') != prompt:
            raise ValueError('prompt bound/roundtrip failed: ' + case['id'])
        lines.append('GEN 32 ' + ' '.join(map(str, ids)))
    requests = ('\n'.join(lines) + '\n').encode()
    (output / 'requests').write_bytes(requests)
    Q.write(output / 'quality_manifest', dict(version=2, suite=SUITE, cases=cases, template=Q.TEMPLATE,
          max_new=MAX_NEW, max_prompt=MAX_PROMPT, requests_sha256=Q.digest(requests), plan_sha256=plan_sha256,
          vocabulary_sha256=Q.digest(json.dumps(tok.tokens, ensure_ascii=False).encode()), sources=sources()))


def clean(text):
    # Protocol EOS is not answer content. Ignore surrounding whitespace, never internal whitespace.
    text = text.strip()
    for eos in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(eos): return text[:-len(eos)].strip()
    return text


def json_value(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError('nonfinite JSON number')
        return value
    def constant(text): raise ValueError('invalid JSON constant: ' + text)
    return json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def assess(case, answer):
    text = clean(answer)
    rule = case['semantic_rule']
    if rule == 'json':
        try:
            actual = json_value(text)
            semantic = json.dumps(actual, sort_keys=True) == json.dumps(case['expected'], sort_keys=True)
        except (ValueError, TypeError):
            semantic = False
    elif rule == 'number':
        # Arithmetic only: exact rational equality, never a tolerance or prose extraction.
        numeric = re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/[0-9]+)', text)
        try: semantic = bool(numeric) and fractions.Fraction(text) == fractions.Fraction(case['expected'])
        except (ValueError, ZeroDivisionError): semantic = False
    elif rule == 'yes_no':
        semantic = text.casefold() == case['expected']  # only explicitly binary yes/no tasks
    else:
        semantic = text == case['expected']
    if case['format_rule'] == 'json-value':
        try: json_value(text); formatting = True
        except (ValueError, TypeError): formatting = False
    elif case['format_rule'] == 'reduced-fraction':
        match = re.fullmatch(r'(-?(?:0|[1-9][0-9]*))/([1-9][0-9]*)', text)
        formatting = bool(match) and math.gcd(int(match[1]), int(match[2])) == 1
    else:
        formatting = re.fullmatch(case['format_rule'], text, flags=re.ASCII) is not None
    return dict(semantic_correct=semantic, format_compliant=formatting)


def paired(rows, metric):
    result = dict(total=len(rows), both_correct=0, regressions=0, improvements=0, shared_failures=0,
                  baseline_correct=0, candidate_correct=0)
    result['case_ids'] = {key: [] for key in ('both_correct', 'regressions', 'improvements', 'shared_failures')}
    for row in rows:
        a, b = (row[side][metric] for side in ('baseline', 'candidate'))
        result['baseline_correct'] += a
        result['candidate_correct'] += b
        key = 'both_correct' if a and b else 'regressions' if a else 'improvements' if b else 'shared_failures'
        result[key] += 1
        result['case_ids'][key].append(row['id'])
    return result


def summarize(rows):
    exercised = [row for row in rows if row['vulkan_entries'] > 0]
    excluded = [row['id'] for row in rows if not row['vulkan_entries']]
    semantic = paired(exercised, 'semantic_correct')
    formatting = paired(exercised, 'format_compliant')
    if semantic['regressions']: result = 'screen-semantic-regression'
    elif formatting['regressions']: result = 'screen-format-regression'
    elif not exercised: result = 'screen-unexercised'
    elif excluded: result = 'screen-coverage-limited'
    else: result = 'screen-no-observed-regression'
    categories = {}
    for category in sorted({row['category'] for row in rows}):
        all_rows = [row for row in rows if row['category'] == category]
        active = [row for row in all_rows if row['vulkan_entries'] > 0]
        categories[category] = dict(all_cases=dict(semantic=paired(all_rows, 'semantic_correct'),
            formatting=paired(all_rows, 'format_compliant')), exercised=dict(semantic=paired(active, 'semantic_correct'),
            formatting=paired(active, 'format_compliant')))
    return dict(result=result, all_cases=dict(semantic=paired(rows, 'semantic_correct'),
                formatting=paired(rows, 'format_compliant')),
                offload_evidence=dict(semantic=semantic, formatting=formatting),
                nonparticipating_case_ids=excluded, categories=categories)


def score(tok, baseline, candidate, output):
    cases = inventory()
    roots = [pathlib.Path(baseline), pathlib.Path(candidate)]
    configs = [Q.bundle.verify(root) for root in roots]
    if any(cfg.get('evaluation_mode') != 'quality' or cfg.get('quality_suite') != SUITE for cfg in configs):
        raise ValueError('v2 requires explicit quality mode and v2-100 suite')
    Q.comparable(*configs)
    for helper in Q.bundle.HELPERS:
        if Q.bundle.sha(roots[0] / helper) != Q.bundle.sha(roots[1] / helper):
            raise ValueError('runtime helper differs: ' + helper)
    frozen = json.loads(pathlib.Path(configs[0]['assets']['quality_manifest']).read_text())
    gpu = configs[1]['argv']
    plan_sha256 = Q.bundle.sha(gpu[gpu.index('--remote-gpu-worker') + 2])
    if frozen != dict(version=2, suite=SUITE, cases=cases, template=Q.TEMPLATE, max_new=MAX_NEW,
            max_prompt=MAX_PROMPT, requests_sha256=Q.bundle.sha(configs[0]['assets']['requests']), plan_sha256=plan_sha256,
            vocabulary_sha256=Q.digest(json.dumps(tok.tokens, ensure_ascii=False).encode()), sources=sources()):
        raise ValueError('frozen v2 inventory, vocabulary, settings or sources differ')
    arms = []
    for cfg, root in zip(configs, roots):
        if (root / 'failure.json').exists(): raise ValueError('failed runtime arm')
        summary = json.loads((root / 'summary.json').read_text())
        if len(summary['arms']) != 1: raise ValueError('expected one completed arm')
        arm = summary['arms'][0]
        if arm['arm'] != cfg['arms'][0]['name'] or [r['index'] for r in arm['requests']] != list(range(1, COUNT+1)):
            raise ValueError('incomplete/reordered suite')
        for row in arm['requests']:
            if row['finish'] not in ('stop', 'length') or not 0 < len(row['tokens']) <= MAX_NEW or \
               row['output_count'] != len(row['tokens']) or \
               any(type(t) is not int or not 0 <= t < len(tok.tokens) for t in row['tokens']):
                raise ValueError('invalid completion/token IDs')
        arms.append(arm)
    entries = Q.participation((roots[1] / arms[1]['arm'] / 'stderr.log').read_text(), COUNT)
    rows = []
    for i, case in enumerate(cases):
        pair = [arm['requests'][i] for arm in arms]
        answers = [tok.decode(r['tokens'], errors='strict') for r in pair]
        logits = [Q.compare_logits.read(root / (arm['arm'] + '-first-logits.' + str(i)))
                  for root, arm in zip(roots, arms)]
        if any(len(row) != len(tok.tokens) for row in logits): raise ValueError('logit vocabulary differs')
        rows.append(dict(index=i+1, id=case['id'], category=case['category'], vulkan_entries=entries[i],
             baseline=dict(answer=answers[0], tokens=pair[0]['tokens'], **assess(case, answers[0])),
             candidate=dict(answer=answers[1], tokens=pair[1]['tokens'], **assess(case, answers[1])),
             token_ids_equal=pair[0]['tokens'] == pair[1]['tokens'],
             token_positions_different=sum(x != y for x, y in zip(pair[0]['tokens'], pair[1]['tokens'])) +
                 abs(len(pair[0]['tokens']) - len(pair[1]['tokens'])),
             exact_first_logits_pass=logits[0].tobytes() == logits[1].tobytes(),
             fixed_numeric_diagnostic=Q.compare_logits.compare(*logits),
             runtime=[{k: r[k] for k in ('prompt_s', 'decode_s', 'finish')} for r in pair]))
    report = summarize(rows)
    Q.write(output, dict(mode='exploratory-backend-quality', suite=SUITE, sources=sources(),
            **report, cases=rows, numeric_limits=Q.compare_logits.LIMITS,
            scope='One fixed expert302 plan, one V340 die, full CPU backing; one run per arm. '
                  'Descriptive paired screen, not statistical noninferiority or broad quality equivalence.'))
    return 0 if report['result'] == 'screen-no-observed-regression' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True, help='existing same model GGUF, metadata only')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--out', required=True)
    p.add_argument('--placement-plan', required=True, help='existing admitted expert302 plan; freezes its bytes')
    p = sub.add_parser('score'); p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args()
    tok = Q.tokenizer(args.gguf)
    if args.command == 'prepare': prepare(tok, args.out, args.placement_plan); return 0
    return score(tok, args.baseline, args.candidate, args.out)


if __name__ == '__main__': raise SystemExit(main())
