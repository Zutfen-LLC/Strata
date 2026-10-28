#!/usr/bin/env python3
"""Frozen quality.py adapter for the existing portable CLI; no model builds or services."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import quality as Q
import portable_model_smoke_v1 as S


def frozen(tok, root):
    root = Path(root)
    manifest = json.loads((root / 'quality_manifest').read_text())
    lines = (root / 'requests').read_text().splitlines()
    expected = []
    for _, prompt, _, _ in Q.CASES:
        text = Q.TEMPLATE.format(p=prompt)
        ids = tok.encode(text, parse_special=True)
        if not 0 < len(ids) <= Q.MAX_PROMPT or tok.decode(ids, errors='strict') != text:
            raise ValueError('frozen prompt bound/roundtrip')
        expected.append('GEN ' + str(Q.MAX_NEW) + ' ' + ' '.join(map(str, ids)))
    binding = dict(version=1, cases=[list(c) for c in Q.CASES], template=Q.TEMPLATE,
                   max_new=Q.MAX_NEW, max_prompt=Q.MAX_PROMPT,
                   requests_sha256=Q.bundle.sha(root / 'requests'),
                   vocabulary_sha256=Q.digest(json.dumps(tok.tokens, ensure_ascii=False).encode()),
                   scorer_sha256=Q.bundle.sha(Q.__file__))
    if manifest != binding or lines != expected or len(lines) != 12:
        raise ValueError('frozen suite/request/scorer/vocabulary mismatch')
    return [[int(t) for t in line.split()[2:]] for line in lines], binding


def completion(records, prompt, vocabulary):
    if not records or set(records[-1]) != {'stop', 'position'}:
        raise ValueError('missing final stop receipt')
    rows = records[:-1]
    if any(set(r) != {'position', 'token', 'generated', 'greedy_next'} for r in rows):
        raise ValueError('unknown/malformed token receipt')
    if [r['position'] for r in rows] != list(range(len(rows))):
        raise ValueError('noncontiguous positions')
    if any(type(r['generated']) is not bool or type(r['token']) is not int or
           not 0 <= r['token'] < vocabulary or type(r['greedy_next']) is not int or
           not 0 <= r['greedy_next'] < vocabulary for r in rows):
        raise ValueError('invalid token or greedy ID')
    if len(rows) <= len(prompt) or [r['token'] for r in rows[:len(prompt)]] != prompt or \
       any(r['generated'] != (i >= len(prompt)) for i, r in enumerate(rows)):
        raise ValueError('prompt/generated boundary mismatch')
    generated = [r['token'] for r in rows[len(prompt):]]
    if len(generated) > Q.MAX_NEW or len(rows) > 256 or records[-1]['position'] != len(rows):
        raise ValueError('generation/context bound')
    if any(rows[i]['token'] != rows[i-1]['greedy_next'] for i in range(len(prompt), len(rows))):
        raise ValueError('output is not serial greedy')
    stop = records[-1]['stop']
    if stop not in ('eos', 'max_new') or (stop == 'max_new' and len(generated) != Q.MAX_NEW):
        raise ValueError('abnormal or incomplete stop')
    if (stop == 'eos') != (generated[-1] == 248046) or 248046 in generated[:-1]:
        raise ValueError('generated EOS mismatch')
    return generated, stop


def run_portable(args, tok):
    prompts, binding = frozen(tok, args.suite)
    binary = args.binary.resolve(strict=True)
    if Q.bundle.sha(binary) != args.binary_sha256:
        raise ValueError('qualified PR43 binary digest mismatch')
    args.out.mkdir(mode=0o700, exist_ok=False)
    summary = dict(status='STOP', source='ac85ff3fdf0834c13437b69b1568d8704e9b0717',
                   binary_sha256=args.binary_sha256, suite=binding, requests=[])
    baseline = S.host_memory()
    work = S.model_working_set()
    summary.update(baseline_host=baseline, host_admission=S.host_admission_terms(work['working_set_bound_bytes']))
    start = time.monotonic()
    try:
        for index, prompt in enumerate(prompts, 1):
            S.admit_host(S.host_memory(), work['working_set_bound_bytes'])
            case = args.out / f'case-{index:02d}'
            case.mkdir()
            initial = [S.vram(pci) for pci in S.PCI]
            if any(v['total'] - v['used'] < 7680 << 20 for v in initial):
                raise ValueError('insufficient per-die cap headroom')
            command = [str(binary), '--model', str(S.MODEL_ROOT / 'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'),
                       '--manifest', S.MANIFEST, '--tokens', ','.join(map(str, prompt)),
                       '--context', '256', '--max-new', '32', '--threads', '4', '--expert-workers', '12',
                       '--host-cap-mib', '32768', '--vulkan', '0,1', '--device-cap-mib', '7680,7680',
                       '--split', '32', '--placements', str(case / 'placements.tsv')]
            Q.write(case / 'argv.json', command)
            process = None
            peak = [v['used'] for v in initial]
            minimum = baseline['effective_available']
            began = time.monotonic()
            try:
                with (case / 'stdout.jsonl').open('x') as stdout, (case / 'stderr.log').open('x') as stderr:
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
                    while True:
                        host, child = S.host_memory(), S.rss(process.pid)
                        S.guard_host(host, baseline, child)
                        minimum = min(minimum, host['effective_available'])
                        for die, pci in enumerate(S.PCI):
                            used = S.vram(pci)['used']
                            peak[die] = max(peak[die], used)
                            if used - initial[die]['used'] > 7680 << 20:
                                raise ValueError('aggregate per-die VRAM growth cap')
                        if any((case / name).stat().st_size > 8 << 20 for name in ('stdout.jsonl', 'stderr.log')):
                            raise ValueError('bounded output log exceeded8MiB')
                        if time.monotonic() - began > 600 or time.monotonic() - start > 3600:
                            raise TimeoutError('case600s/suite3600s budget exhausted; no retry')
                        status = process.poll()
                        if status is not None:
                            break
                        time.sleep(.2)
                if status:
                    raise RuntimeError(f'portable_model exited{status}')
                participation = S.placements(case / 'placements.tsv', [0, 1], 32)
                records = [json.loads(line) for line in (case / 'stdout.jsonl').read_text().splitlines()]
                ids, stop = completion(records, prompt, len(tok.tokens))
                summary['requests'].append(dict(index=index, tokens=ids, finish=stop, seconds=time.monotonic()-began,
                    tensor_operations_by_device=participation, sampled_peak_vram=peak,
                    minimum_effective_available=minimum))
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        summary['status'] = 'complete'
    except BaseException as error:
        summary['error'] = str(error)
        raise
    finally:
        Q.write(args.out / 'summary.json', summary)



def native_configuration(original, suite, output, expected_sha):
    # Reuse the accepted lifecycle. Never start/restart/stop a resident worker.
    import copy
    cfg = copy.deepcopy(original)
    if not cfg.get('attach_root') or not cfg.get('attached_worker_idle') or \
       cfg.get('final_worker') != 'retain' or cfg.get('failure_worker') != 'retain':
        raise ValueError('native template must retain an explicitly attached idle resident')
    if len(cfg['arms']) != 1 or cfg['arms'][0].get('engine', 'engine') != 'engine' or cfg['arms'][0]['restart'] or cfg.get('worker_peers') or \
       any(a.startswith('--remote-gpu-') for a in cfg['argv']):
        raise ValueError('need one native arm without worker restart/offload peers')
    if Q.bundle.sha(cfg['assets']['engine']) != expected_sha:
        raise ValueError('native template executable is not selected reference')
    if 'STRATA_FIRST_REQUEST_TRACE' in cfg['env']:
        raise ValueError('trace-stop native template cannot run quality suite')
    assets = {name: cfg['assets'][name] for name in ('engine', 'worker', 'manifest')}
    assets.update(requests=str(Path(suite).resolve() / 'requests'),
                  quality_manifest=str(Path(suite).resolve() / 'quality_manifest'))
    old_paths = {cfg['assets'][name]: '@'+name for name in ('engine', 'worker', 'manifest')}
    cfg['argv'] = [old_paths.get(arg, arg) for arg in cfg['argv']]
    cfg['worker_argv'] = [old_paths.get(arg, arg) for arg in cfg['worker_argv']]
    cfg.update(assets=assets, run_root=str(Path(output).resolve()), evaluation_mode='quality', capture_first_logits=True)
    cfg['arms'][0].update(name='NATIVE12', requests=list(range(12)))
    for key in ('reference', 'first_logits_reference'):
        cfg['arms'][0].pop(key, None)
    # Native existing harness must still validate all identities/limits/masks.
    return cfg


def native_rows(root, binding, prompts, expected_sha):
    root = Path(root)
    if (root / 'failure.json').exists():
        raise ValueError('failed native run')
    cfg = Q.bundle.verify(root)
    if Q.bundle.sha(cfg['assets']['engine']) != expected_sha:
        raise ValueError('native baseline binary differs from selected reference')
    if json.loads(Path(cfg['assets']['quality_manifest']).read_text()) != binding or \
       Q.bundle.sha(cfg['assets']['requests']) != binding['requests_sha256']:
        raise ValueError('native frozen suite mismatch')
    if cfg['request_lines'] != ['GEN 32 ' + ' '.join(map(str, p)) for p in prompts]:
        raise ValueError('native prompt IDs differ')
    if cfg.get('worker_peers') or any(a.startswith('--remote-gpu-') for a in cfg['argv']):
        raise ValueError('baseline must be native local-GPU/CPU, no offload peers')
    if '--native' not in cfg['argv'] or cfg['argv'][cfg['argv'].index('--native')+1].split('/')[-1] != \
       'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf':
        raise ValueError('native model family/path differs')
    if Q.bundle.sha(cfg['assets']['manifest']) != '8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318':
        raise ValueError('expert manifest identity differs')
    arms = json.loads((root / 'summary.json').read_text())['arms']
    if len(arms) != 1 or cfg['arms'][0]['requests'] != list(range(12)):
        raise ValueError('native baseline needs one complete ordered12-request arm')
    if cfg['arms'][0].get('engine', 'engine') != 'engine':
        raise ValueError('native arm selects a different executable')
    for index in range(12):
        row = Q.compare_logits.read(root / (arms[0]['arm'] + '-first-logits.' + str(index)))
        if len(row) != 248320:
            raise ValueError('native first-logit vocabulary differs')
    return arms[0]['requests']


def score(tok, args):
    prompts, binding = frozen(tok, args.suite)
    portable = json.loads((args.portable_root / 'summary.json').read_text())
    if portable['status'] != 'complete' or portable['suite'] != binding:
        raise ValueError('incomplete/foreign portable run')
    native = native_rows(args.native_root, binding, prompts, args.native_sha256)
    rows = []
    for arm in (native, portable['requests']):
        if [r['index'] for r in arm] != list(range(1, 13)):
            raise ValueError('incomplete/reordered/duplicated requests')
        for row in arm:
            if row['finish'] not in ('stop', 'length', 'eos', 'max_new') or not 0 < len(row['tokens']) <= 32 or \
               any(type(t) is not int or not 0 <= t < len(tok.tokens) for t in row['tokens']):
                raise ValueError('invalid/abnormal completion')
    for index, case in enumerate(Q.CASES):
        pair = [tok.decode(arm[index]['tokens'], errors='strict') for arm in (native, portable['requests'])]
        points = [Q.correct(text, case[2], case[3]) for text in pair]
        rows.append(dict(index=index+1, category=case[0], native_correct=points[0], portable_correct=points[1],
                         native_answer=pair[0], portable_answer=pair[1],
                         tokens_equal=native[index]['tokens'] == portable['requests'][index]['tokens']))
    totals = [sum(row[key] for row in rows) for key in ('native_correct', 'portable_correct')]
    result = 'inconclusive-baseline' if totals[0] != 12 else 'screen-pass' if totals[1] == 12 else 'screen-regression'
    Q.write(args.out, dict(result=result, native_score=totals[0], portable_score=totals[1], total=12,
                          cases=rows, numerical_parity='prior whole-model numerical gate FAILED; unchanged',
                          scope='Frozen12-case functional screen only; no broad quality equivalence.'))
    return int(result != 'screen-pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('portable')
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('stage-native')
    p.add_argument('--template-root', type=Path, required=True)
    p.add_argument('--native-sha256', required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('score')
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--portable-root', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--native-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tok = Q.tokenizer(args.gguf)
    if args.command == 'stage-native':
        frozen(tok, args.suite)
        cfg = native_configuration(Q.bundle.verify(args.template_root), args.suite, args.out, args.native_sha256)
        configuration = args.out.with_name(args.out.name + '.input.json')
        Q.write(configuration, cfg)
        print(Q.bundle.stage(configuration))
        return 0
    if args.command == 'portable':
        run_portable(args, tok)
        return 0
    return score(tok, args)


if __name__ == '__main__':
    raise SystemExit(main())
