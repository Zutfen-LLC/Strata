#!/usr/bin/env python3
"""Frozen quality.py adapter for the existing portable CLI; no model builds or services."""
import argparse
import json
import math
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


def benchmark_prompts(prompts, binding, enabled):
    # Freeze/verify all twelve as before, then predeclare only case 1 repeated.
    if not enabled:
        return prompts, binding
    return [prompts[0]] * 4, dict(binding, benchmark_case=1, excluded_warmup=1, measured_trials=3)


def timing_fields(row, prompt, ids):
    for key in ('setup_ms', 'prompt_ms', 'decode_ms'):
        if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError('invalid timing field: ' + key)
    if row.get('prompt_count') != len(prompt) or row.get('output_count') != len(ids):
        raise ValueError('timing token count mismatch')
    return row


def benchmark_receipt(args, summary):
    if not getattr(args, 'benchmark', False):
        return
    rows = summary['requests']
    for i, row in enumerate(rows):
        row['excluded_warmup'] = i == 0
    summary['numerical_parity'] = 'FAILED; performance does not waive logit gates'
    summary['cache_policy'] = 'fresh engine per trial; one excluded warmup; OS/file caches not flushed; no KV reuse'
    # Raw individual trials are authoritative; deliberately no pooled TPS.
    Q.write(args.out / 'benchmark.json', summary)


def run_portable(args, tok):
    prompts, binding = frozen(tok, args.suite)
    benchmark = getattr(args, 'benchmark', False)
    prompts, binding = benchmark_prompts(prompts, binding, benchmark)
    binary = args.binary.resolve(strict=True)
    if Q.bundle.sha(binary) != args.binary_sha256:
        raise ValueError('portable binary digest mismatch')
    args.out.mkdir(mode=0o700, exist_ok=False)
    summary = dict(status='STOP', source='ac85ff3fdf0834c13437b69b1568d8704e9b0717',
                   binary_sha256=args.binary_sha256, suite=binding, requests=[])
    if getattr(args,'expert_plan',[]):
        summary['expert_plans']=args.expert_plan
        summary['source']='local resident expert candidate; see exact source export/binary SHA'
    elif benchmark:
        summary['source'] = 'PR43 c76d03c7 plus opt-in CLI phase timers; see source export SHA'
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
            for plan in getattr(args, 'expert_plan', []):
                command += ['--expert-plan', plan]
            if getattr(args, 'expert_plan', []):
                for name in ('expert_tensor_mib','expert_workspace_mib','expert_free_reserve_mib'):
                    command += ['--'+name.replace('_','-'), getattr(args,name)]
                if getattr(args,'expert_check_rows',False): command += ['--expert-check-rows']
            if benchmark:
                command += ['--timings', str(case / 'timings.json')]
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
                        if time.monotonic() - began > (300 if benchmark else 600) or time.monotonic() - start > (1200 if benchmark else 3600):
                            raise TimeoutError('case/suite budget exhausted; no retry')
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
                if benchmark:
                    timing = json.loads((case / 'timings.json').read_text())
                    timing_fields(timing, prompt, ids)
                    for key in ('selection_prompt_ms', 'selection_decode_ms', 'terminal_feed_ms'):
                        value = timing.get(key)
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError('invalid selection timing: ' + key)
                    if timing.get('selection_prompt_steps') != len(prompt)-1 or timing.get('selection_decode_steps') != len(ids):
                        raise ValueError('selection step count mismatch')
                    summary['requests'][-1]['timing'] = timing
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
        benchmark_receipt(args, summary)



DRIVER_SHA256 = 'b5f1a5c10a0337d6254776ad01a19dd787076f41fd114538ea01f28708a1ae2f'
NATIVE_SHA256 = '931569c089a4f04114395a428c85f85227142c733f4fa0511243d05f97b46b89'
EXPERT_MANIFEST_SHA256 = '8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318'


def retained_driver(path):
    import importlib.util
    if Q.bundle.sha(path) != DRIVER_SHA256:
        raise ValueError('retained driver5 digest mismatch')
    spec = importlib.util.spec_from_file_location('retained_native_driver5', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)  # imports definitions; never calls its main/distributed/TLS campaign
    return driver


def owned_native(driver, ndc, loadc, prompts, summary, output, benchmark=False):
    manifest, load, local, _unused_distributed = driver.prepare_argv(ndc, loadc)
    worker_argv = [a.replace('@worker', ndc['assets']['worker']).replace('@manifest', manifest)
                   for a in ndc['worker_argv']]
    worker_argv[0] = ndc['assets']['worker']
    driver.set_single_value(worker_argv, '--local-manifest', manifest)
    driver.require_once(worker_argv, '--residency-forever')
    driver.require_once(worker_argv, '--port', '17431')
    if any(a in worker_argv for a in ('--bind', '--bind-ip', '--allow-peer')):
        raise ValueError('only retained default loopback worker binding is allowed')
    if driver.port_open(17431):
        raise ValueError('127.0.0.1:17431 already occupied; do not touch existing listener')
    driver.R, driver.DRIVER_LOG = output, output / 'driver.log'
    worker, log = None, (output / 'cpu-worker.log').open('xb')
    try:
        worker = driver.subprocess.Popen(worker_argv,
            env={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','OMP_NUM_THREADS':'2'},
            cwd=str(output), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        summary['owned_worker'] = dict(pid=worker.pid, argv=worker_argv)
        started = time.monotonic()
        while not driver.port_open(17431):
            if worker.poll() is not None:
                raise RuntimeError('owned CPU worker exited during startup')
            if time.monotonic()-started > driver.WORKER_START_SECONDS:
                raise TimeoutError('owned CPU worker startup budget')
            time.sleep(2)
        if worker.poll() is not None:
            raise RuntimeError('owned CPU worker exited before LOAD')
        if benchmark:
            summary['worker_startup_s'] = time.monotonic() - started
        summary['load'] = driver.engine_stage('load', loadc, load, '', False, 0, False)
        if not driver.stage_ok(summary['load'], measured=False):
            raise RuntimeError('owned resident LOAD failed; no requests started')
        # Unchanged driver5 engine_stage owns READY/GEN/DONE/QUIT and reaps its engine.
        # One fresh engine per case matches the portable CLI's sequence isolation.
        for index, prompt in enumerate(prompts, 1):
            case = output / f'case-{index:02d}'
            case.mkdir()
            driver.R = case
            (case / 'request.txt').write_text('GEN 32 ' + ' '.join(map(str, prompt)) + '\n')
            result = driver.engine_stage('local', ndc, local,
                'GEN 32 ' + ' '.join(map(str, prompt)), True, 0, not benchmark)
            Q.write(case / 'result.json', result)
            if not driver.stage_ok(result, measured=True):
                raise RuntimeError(f'native case{index} failed; no retry')
            done = result['done']
            ids = result['tokens']
            if done['finish'] not in ('stop','length') or not 0 < len(ids) <= 32 or \
               done['output_count'] != len(ids) or any(type(t) is not int or not 0 <= t < 248320 for t in ids):
                raise ValueError('native abnormal/nonfinite/invalid completion')
            logits = case / 'local' / 'local-first-logits.0'
            if not benchmark and len(Q.compare_logits.read(logits)) != 248320:
                raise ValueError('native finite first-logit vocabulary mismatch')
            summary['requests'].append(dict(index=index, tokens=ids, finish=done['finish']))
            if benchmark:
                prompt_count = int(done['raw'].split()[2])
                timing = dict(setup_ms=result['setup_s']*1000, prompt_ms=done['prompt_ms'],
                              decode_ms=done['decode_ms'], prompt_count=prompt_count, output_count=len(ids))
                summary['requests'][-1]['timing'] = timing_fields(timing, prompt, ids)
    finally:
        driver.R = output
        if worker is not None:
            summary['cpu_worker_rc'] = driver.stop_process_group(worker, 'owned CPU worker')
        log.close()
    if summary.get('cpu_worker_rc') != 0 or driver.port_open(17431):
        raise RuntimeError('owned CPU worker cleanup did not complete cleanly')


def run_native(args, tok):
    prompts, binding = frozen(tok, args.suite)
    benchmark = getattr(args, 'benchmark', False)
    prompts, binding = benchmark_prompts(prompts, binding, benchmark)
    driver = retained_driver(args.driver)
    ndc = json.loads(driver.NDC_PATH.read_text())
    loadc = json.loads(driver.LOADC_PATH.read_text())
    worker = ndc['assets']['worker']
    manifest = ndc['assets']['manifest']
    if driver.sha256(driver.ENGINE) != NATIVE_SHA256 or not driver.sha256(worker).startswith(driver.CPU_WORKER_SHA_PREFIX):
        raise ValueError('retained native/worker executable identity mismatch')
    if driver.sha256(manifest) != EXPERT_MANIFEST_SHA256:
        raise ValueError('retained expert manifest mismatch')
    args.out.mkdir(mode=0o700, exist_ok=False)
    summary = dict(status='STOP', suite=binding, binary_sha256=NATIVE_SHA256, driver_sha256=DRIVER_SHA256,
                   worker_sha256=driver.sha256(worker), manifest_sha256=driver.sha256(manifest), requests=[])
    Q.write(args.out / 'input-pins.json', dict(ndon_config_sha256=driver.sha256(driver.NDC_PATH),
            load_config_sha256=driver.sha256(driver.LOADC_PATH), driver_sha256=DRIVER_SHA256, suite=binding))
    try:
        owned_native(driver, ndc, loadc, prompts, summary, args.out, benchmark=benchmark)
        summary['status'] = 'complete'
    except BaseException as error:
        summary['error'] = str(error)
        raise
    finally:
        Q.write(args.out / 'summary.json', summary)
        benchmark_receipt(args, summary)


def native_rows(root, binding, prompts, expected_sha):
    root = Path(root)
    result = json.loads((root / 'summary.json').read_text())
    if result['status'] != 'complete' or result['suite'] != binding or \
       result['binary_sha256'] != expected_sha or result['driver_sha256'] != DRIVER_SHA256 or \
       result['manifest_sha256'] != EXPERT_MANIFEST_SHA256 or result.get('cpu_worker_rc') != 0:
        raise ValueError('incomplete/foreign/unclean native run')
    for index in range(12):
        if (root / f'case-{index+1:02d}' / 'request.txt').read_text() != 'GEN 32 ' + ' '.join(map(str, prompts[index])) + '\n':
            raise ValueError('native consumed prompt receipt differs')
        argv = json.loads((root / f'case-{index+1:02d}' / 'local' / 'engine-argv.json').read_text())
        if any(a.startswith('--remote-gpu-') for a in argv) or '--native' not in argv:
            raise ValueError('native arm is not retained local policy')
        if len(Q.compare_logits.read(root / f'case-{index+1:02d}' / 'local' / 'local-first-logits.0')) != 248320:
            raise ValueError('native first-logit vocabulary differs')
    return result['requests']


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
    p.add_argument('--expert-plan', action='append', default=[], metavar='SLOT:FILE')
    p.add_argument('--expert-tensor-mib', default='256,256')
    p.add_argument('--expert-workspace-mib', default='128,128')
    p.add_argument('--expert-free-reserve-mib', default='512,512')
    p.add_argument('--expert-check-rows', action='store_true')
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = commands.add_parser('native')
    p.add_argument('--driver', type=Path, required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    for name in ('portable', 'native'):
        commands.choices[name].add_argument('--benchmark', action='store_true',
            help='only frozen case 1: one excluded warmup, three measured fresh engines; no scoring')
    p = commands.add_parser('score')
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--portable-root', type=Path, required=True)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--native-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    tok = Q.tokenizer(args.gguf)
    if args.command == 'native':
        run_native(args, tok)
        return 0
    if args.command == 'portable':
        run_portable(args, tok)
        return 0
    return score(tok, args)


if __name__ == '__main__':
    def terminate(_signum, _frame):
        raise SystemExit('termination requested; cleaning owned processes')
    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())
