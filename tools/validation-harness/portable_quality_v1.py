#!/usr/bin/env python3
"""Frozen quality.py adapter for the existing portable CLI; no model builds or services."""
import argparse
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import time

import executed_logits_v1 as E
import quality as Q
import portable_model_smoke_v1 as S
import portable_local_expert_smoke_v1 as L


GDN_CAUSAL_FLAG = '--gdn-causal-chunk'
GDN_PROJECTION_FLAG = '--gdn-batch-input-projections'
RESIDENCY_FLAGS = ('--prompt-head-elision', '--qsa-production', '--qsa-reuse-attention', '--gdn-production', '--gdn-resident-weights', '--gdn-resident-state', '--dense-resident-weights')


def residency_arguments(parser):
    parser.add_argument(GDN_CAUSAL_FLAG,action='store_true',help='experimental production GDN causal chunk; requires batch projections')
    parser.add_argument(GDN_PROJECTION_FLAG,action='store_true',help='batch independent GDN projections; requires prompt chunks')
    parser.add_argument('--prompt-chunk-size',type=int,choices=range(2,9),default=0,help='bounded layer-major prompt chunks; implies head elision')
    for flag in RESIDENCY_FLAGS:
        parser.add_argument(flag, action='store_true', help='forward opt-in model residency policy')


def residency_flags(args):
    size=chunk_size(args)
    projections=projection_policy(args)
    causal=causal_policy(args)
    flags=[flag for flag in RESIDENCY_FLAGS if getattr(args,flag[2:].replace('-','_'),False) or
           (flag=='--prompt-head-elision' and size)]
    if projections:flags += [GDN_PROJECTION_FLAG]
    if causal:flags += [GDN_CAUSAL_FLAG]
    if size: flags += ['--prompt-chunk-size',str(size)]
    return flags


CHUNK_COUNTERS={'logical_tokens_delta','prompt_chunks_delta','prompt_chunk_tokens_delta','chunk_layer_calls_delta',
    'chunk_ffn_pre_calls_delta','chunk_provider_calls_delta','chunk_ffn_post_calls_delta',
    'model_forward_calls_delta','model_layer_calls_delta','ffn_pre_calls_delta','provider_calls_delta','ffn_post_calls_delta'}
def chunk_size(args):
    value=getattr(args,'prompt_chunk_size',0)
    if type(value) is not int or value not in (0,2,3,4,5,6,7,8): raise ValueError('invalid prompt chunk size')
    return value

def chunk_argv(command):
    flag='--prompt-chunk-size'
    if flag not in command:return 0
    if command.count(flag)!=1 or command.index(flag)+1>=len(command): raise ValueError('duplicate/missing chunk size')
    value=command[command.index(flag)+1]
    if value not in tuple(map(str,range(2,9))):raise ValueError('invalid chunk argv size')
    return int(value)

def projection_policy(args):
    enabled=getattr(args,'gdn_batch_input_projections',False)
    if type(enabled) is not bool or (enabled and not chunk_size(args)):
        raise ValueError('GDN batch input projections require prompt chunks and a boolean policy')
    return enabled

def projection_argv(command):
    count=command.count('--gdn-batch-input-projections')
    if count>1 or (count and not chunk_argv(command)):
        raise ValueError('duplicate GDN batch input projections or missing prompt chunks')
    return bool(count)

def projection_binding(record,enabled):
    value=record.get('gdn_batch_input_projections',False)
    if type(enabled) is not bool or type(value) is not bool or value is not enabled:
        raise ValueError('GDN batch input projection policy mismatch')


def causal_policy(args):
    enabled=getattr(args,'gdn_causal_chunk',False)
    if type(enabled) is not bool or (enabled and (not projection_policy(args) or not getattr(args,'gdn_production',False))):
        raise ValueError('GDN causal chunk requires batched projections and production')
    return enabled

def causal_argv(command):
    count=command.count(GDN_CAUSAL_FLAG)
    if count>1 or (count and (not projection_argv(command) or command.count('--gdn-production')!=1)):
        raise ValueError('duplicate GDN causal chunk or missing projection/production policy')
    return bool(count)

def causal_binding(record,enabled):
    value=record.get('gdn_causal_chunk',False)
    if type(enabled) is not bool or type(value) is not bool or value is not enabled:
        raise ValueError('GDN causal chunk policy mismatch')


def chunk_binding(record,size):
    if type(size) is not int or size not in (0,2,3,4,5,6,7,8):raise ValueError('invalid expected chunk size')
    if size:
        if type(record.get('prompt_chunk_size')) is not int or record['prompt_chunk_size']!=size or record.get('chunk_policy')!='layer_major_v1':
            raise ValueError('prompt chunk policy mismatch')
    elif 'prompt_chunk_size' in record or 'chunk_policy' in record: raise ValueError('foreign prompt chunk policy')

def chunk_fields(metrics,prompt_count,consumed,size):
    chunk_binding(metrics,size)
    if not size:
        if any(key in metrics for key in CHUNK_COUNTERS) or 'gdn_subphases_available' in metrics:raise ValueError('foreign chunk counters')
        return consumed
    chunks=(prompt_count-1+size-1)//size
    forwards=chunks+consumed-(prompt_count-1)
    expected={'logical_tokens_delta':consumed,'prompt_chunks_delta':chunks,'prompt_chunk_tokens_delta':prompt_count-1,
        'chunk_layer_calls_delta':48*chunks,'chunk_ffn_pre_calls_delta':48*chunks,
        'chunk_provider_calls_delta':48*chunks,'chunk_ffn_post_calls_delta':48*chunks,
        'model_forward_calls_delta':forwards,'model_layer_calls_delta':48*forwards,
        'ffn_pre_calls_delta':48*forwards,'provider_calls_delta':48*forwards,'ffn_post_calls_delta':48*forwards}
    if metrics.get('gdn_subphases_available') is not False:raise ValueError('chunk GDN detailed phases unavailable')
    for key,value in expected.items():
        if type(metrics.get(key)) is not int or metrics[key]!=value:raise ValueError('chunk execution count mismatch: '+key)
    return forwards


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


def token_receipt(row, position, prompt_count, vocabulary, head_policy='eager_diagnostic', prompt_chunk_size=0):
    if head_policy not in ('eager_diagnostic', 'prompt_head_elision'):
        raise ValueError('unknown head policy')
    chunk_binding(row,prompt_chunk_size)
    if prompt_chunk_size and head_policy!='prompt_head_elision':raise ValueError('chunking requires head elision')
    fields = {'position', 'token', 'generated', 'greedy_next'}
    if prompt_chunk_size:fields|={'chunk_policy','prompt_chunk_size'}
    elision = head_policy == 'prompt_head_elision'
    available = not elision or position + 1 >= prompt_count
    if elision:
        fields |= {'schema', 'head_policy', 'logits_available'}
        if (type(row.get('schema')) is not int or row['schema'] != 2 or
                row.get('head_policy') != head_policy or
                row.get('logits_available') is not available):
            raise ValueError('head receipt policy/availability mismatch')
    if set(row) != fields:
        raise ValueError('unknown/malformed token receipt')
    if (type(row['position']) is not int or row['position'] != position or
            type(row['generated']) is not bool or row['generated'] != (position >= prompt_count) or
            type(row['token']) is not int or not 0 <= row['token'] < vocabulary or
            (available and (type(row['greedy_next']) is not int or not 0 <= row['greedy_next'] < vocabulary)) or
            (not available and row['greedy_next'] is not None)):
        raise ValueError('invalid token or greedy ID')


# The native Q8 component prints these two diagnostics to stdout at creation.
# Keep the retained stdio supervisor's two-name allowlist, with source-shaped
# fields. Raw stdout remains untouched; diagnostics never count as token work.
NATIVE_STDOUT_DIAGNOSTICS = (
    re.compile(r'native_q8_integer_dot requested=[01] queried_support=[01] logical_enabled=[01] '
               r'ggml_accelerated=[01] device=[^\x00-\x1f]+'),
    re.compile(r'native_q8_strict_fma requested=[01] queried_support=[01] logical_enabled=[01] qualified=false'),
)


def stdout_records(path):
    records, diagnostics = [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if any(pattern.fullmatch(line) for pattern in NATIVE_STDOUT_DIAGNOSTICS):
            diagnostics.append(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'malformed JSON or unexpected stdout at line {number}') from exc
        if not isinstance(record, dict):
            raise ValueError(f'stdout record is not an object at line {number}')
        if 'error' in record:
            raise ValueError(f'child error record at line {number}')
        records.append(record)
    return records, diagnostics


def completion(records, prompt, vocabulary, head_policy='eager_diagnostic', prompt_chunk_size=0, native=None):
    if not prompt:
        raise ValueError('empty prompt')
    terminal = records[-1] if records else {}
    if native is not None:
        native_work(terminal, native, len(records)-1)
        terminal = {key: value for key, value in terminal.items() if key not in NATIVE_WORK_FIELDS}
    if not records or set(terminal) != {'stop', 'position'}:
        raise ValueError('missing final stop receipt')
    rows = records[:-1]
    for i, row in enumerate(rows):
        token_receipt(row,i,len(prompt),vocabulary,head_policy,prompt_chunk_size)
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


def benchmark_prompts(prompts, binding, enabled, single=False):
    # Freeze/verify all twelve as before, then predeclare only case 1 repeated.
    if single:
        return [prompts[0]], dict(binding, benchmark_case=1, excluded_warmup=0, measured_trials=1)
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
    if not (getattr(args, 'benchmark', False) or getattr(args, 'single_case', False)):
        return
    rows = summary['requests']
    for i, row in enumerate(rows):
        row['excluded_warmup'] = i == 0 and not getattr(args, 'single_case', False)
    summary['numerical_parity'] = 'FAILED; performance does not waive logit gates'
    summary['cache_policy'] = ('fresh engine; no excluded warmup; OS/file caches not flushed; no KV reuse'
        if getattr(args, 'single_case', False) else
        'fresh engine per trial; one excluded warmup; OS/file caches not flushed; no KV reuse')
    # Raw individual trials are authoritative; deliberately no pooled TPS.
    Q.write(args.out / 'benchmark.json', summary)


# Explicit adapter switches use booleans; product receipt prefixes are reserved
# per case. Omission preserves the historical adapter path and receipts.
NATIVE_SWITCHES = ('native_router', 'native_shared', 'native_moe_combine')
NATIVE_WORK_FIELDS = {
    'native_router_method', 'router_numerical_parity', 'native_router_calls_delta', 'native_router_rows_delta',
    'native_shared_method', 'shared_numerical_parity', 'native_shared_calls_delta', 'native_shared_rows_delta',
}


def native_arguments(parser):
    for name in NATIVE_SWITCHES:
        parser.add_argument('--'+name.replace('_', '-'), action=argparse.BooleanOptionalAction, default=None,
                            help='explicit native method selection; declare all three switches together')
    parser.add_argument('--expert-native-q8-observation', action='store_true',
                        help='native grouped expert observation in both declared method arms; fresh case receipts')


def native_policy(args):
    values = [getattr(args, name, None) for name in NATIVE_SWITCHES]
    expert = getattr(args, 'expert_native_q8_observation', False)
    if all(value is None for value in values) and expert is False:
        return None
    if any(type(value) is not bool for value in values) or expert is not True:
        raise ValueError('declare all three native method switches and --expert-native-q8-observation')
    if not getattr(args, 'expert_plan', []):
        raise ValueError('native method screen requires explicit frozen expert plans')
    if (getattr(args, 'benchmark', False) or getattr(args, 'single_case', False) or chunk_size(args) or
            getattr(args, 'prompt_head_elision', False) or getattr(args, 'dump_executed_logits', False)):
        raise ValueError('native method screen requires full twelve-case serial eager suite')
    return dict(zip(NATIVE_SWITCHES, values), expert_native_q8_observation=True)


def native_plans(args):
    # Existing bounded plan decoder, including the native placement digest.
    import quality_v2 as V
    result = []
    for record in args.expert_plan:
        slot, path = record.split(':', 1)
        if slot not in ('0', '1') or any(row['slot'] == int(slot) for row in result):
            raise ValueError('duplicate/invalid native expert plan slot')
        target = Path(path).resolve(strict=True)
        binding, _ = V.read_plan(target)
        result.append(dict(slot=int(slot), path=str(target), **binding))
    return result


def native_flags(policy, case):
    if policy is None:
        return []
    receipt = case/'native-receipts'
    receipt.mkdir()  # Refuse collisions before launching a child.
    flags = ['--expert-native-q8-observation', str(receipt/'expert')]
    for name in ('native_router', 'native_shared'):
        if policy[name]:
            flags += ['--'+name.replace('_', '-'), str(receipt/name.removeprefix('native_'))]
    if policy['native_moe_combine']:
        flags += ['--native-moe-combine']
    return flags


def native_work(row, policy, consumed):
    expected = {}
    for name, label in (('native_router', 'router'), ('native_shared', 'shared')):
        if policy[name]:
            expected.update({name+'_method': True, label+'_numerical_parity': False,
                             name+'_calls_delta': 48*consumed, name+'_rows_delta': 48*consumed})
    actual = {key: row[key] for key in NATIVE_WORK_FIELDS if key in row}
    if actual.keys() != expected.keys() or any(type(actual[key]) is not type(value) or actual[key] != value
                                              for key, value in expected.items()):
        raise ValueError('native completed-method counters/policy mismatch')
    return actual


def native_case_receipts(case, policy, plans, selected, consumed):
    # Check actual completed provider entries; the frozen one-plan screen does
    # not require exercising an unassigned helper device or pretend CPU misses vanished.
    log = (case/'stderr.log').read_text()
    reports = L.devices(log, 'token-complete', selected)
    if set(reports) != {row['slot'] for row in plans}:
        raise ValueError('native expert completion reports differ from frozen plan slots')
    entries = 0
    for plan in plans:
        report = reports[plan['slot']]
        count = int(report['entries'])
        if (int(report['position']) != consumed-1 or int(report['assigned']) != plan['assigned_experts'] or
                int(report['weight_bytes']) != plan['assigned_bytes'] or count < 0 or
                int(report['row_checks']) != count):
            raise ValueError('native expert assignment/entry/row-check receipt mismatch')
        for key in ('row_max_abs', 'row_relative_l2'):
            if not math.isfinite(float(report[key])) or float(report[key]) < 0:
                raise ValueError('nonfinite native expert row observation')
        entries += count
    if entries <= 0:
        raise ValueError('native screen has no actual Vulkan expert participation')
    dispatches = [dict(part.split('=', 1) for part in line.split()[1:]) for line in log.splitlines()
                  if line.startswith('expert_dispatch ')]
    if not dispatches or any(int(dispatches[-1].get(key, -1)) != value for key, value in
            (('position', consumed-1), ('gpu_entries', entries), ('cpu_entries', 48*10*consumed-entries))):
        raise ValueError('native GPU/CPU completion entry totals mismatch')
    root = case/'native-receipts'
    expected = {f'expert.slot-{i}.{stage}.spv' for i in range(len(plans)) for stage in ('activation', 'gu', 'down')}
    for layer in range(48):
        if policy['native_router']:
            expected.add(f'router.layer-{layer}.t-1.owner-*/effective.spv')
        if policy['native_shared']:
            expected.update(f'shared.layer-{layer}.t-1.owner-*/{name}-effective.spv'
                            for name in ('dense-scalar', 'quant'))
        if policy['native_moe_combine']:
            expected.add(f'expert.combine.layer-{layer}.t-1.owner-*/effective.spv')
    # Owner counters are process-local allocation identities, not layer IDs.
    # Bind each layer and filename once, allowing retained failed-plan owners.
    found = sorted(root.rglob('*.spv'))
    for pattern in expected:
        if not list(root.glob(pattern)):
            raise ValueError('missing native effective module: '+pattern)
    if len(found) > 1024:
        raise ValueError('native effective module count exceeded')
    for path in found:
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or
                not 20 <= path.stat().st_size <= 1 << 20 or
                not any(path.relative_to(root).match(pattern) for pattern in expected)):
            raise ValueError('invalid/unrequested native effective module receipt')
        with path.open('rb') as module:
            if module.read(4) != b'\x03\x02\x23\x07':
                raise ValueError('invalid native effective module header')
    return dict(expert_entries=entries, expert_reports=reports,
                effective_modules={str(path.relative_to(root)): Q.bundle.sha(path) for path in found})


def run_portable(args, tok):
    native = native_policy(args)
    plans = native_plans(args) if native is not None else None
    selected, physical = S.device_selection(argparse.ArgumentParser(), args)
    caps = S.device_caps(args)
    prompts, binding = frozen(tok, args.suite)
    single = getattr(args, 'single_case', False)
    benchmark = getattr(args, 'benchmark', False) or single
    prompts, binding = benchmark_prompts(prompts, binding, benchmark, single)
    binary = args.binary.resolve(strict=True)
    if Q.bundle.sha(binary) != args.binary_sha256:
        raise ValueError('portable binary digest mismatch')
    projections=projection_policy(args)
    causal=causal_policy(args)
    args.out.mkdir(mode=0o700, exist_ok=False)
    summary = dict(status='STOP', source='ac85ff3fdf0834c13437b69b1568d8704e9b0717',
                   binary_sha256=args.binary_sha256, suite=binding, requests=[],
                   head_policy='prompt_head_elision' if getattr(args,'prompt_head_elision',False) or chunk_size(args) else 'eager_diagnostic',
                   qsa_policy='device_production' if getattr(args, 'qsa_production', False) else 'eager_diagnostic',
                   qsa_reuse_attention=bool(getattr(args, 'qsa_reuse_attention', False)),
                   gdn_policy='device_production' if getattr(args, 'gdn_production', False) else 'eager_diagnostic')
    if native is not None:
        summary.update(native_methods=native, native_expert_plans=plans)
    if projections:summary['gdn_batch_input_projections']=True
    if causal:summary['gdn_causal_chunk']=True
    size=chunk_size(args)
    capture=getattr(args,'dump_executed_logits',False)
    if capture and summary['head_policy']!='prompt_head_elision':raise ValueError('executed-head capture requires head elision')
    if size:summary.update(prompt_chunk_size=size,chunk_policy='layer_major_v1')
    if getattr(args,'expert_plan',[]):
        summary['expert_plans']=args.expert_plan
        summary['source']='local resident expert candidate; see exact source export/binary SHA'
    elif benchmark:
        summary['source'] = 'PR43 c76d03c7 plus opt-in CLI phase timers; see source export SHA'
    if size: summary['source']='bounded layer-major prompt chunks; see exact source export/binary SHA'
    baseline = S.host_memory()
    work = S.model_working_set() if not getattr(args, 'model_root', None) else S.model_working_set(args.model_root)
    summary.update(baseline_host=baseline, host_admission=S.host_admission_terms(work['working_set_bound_bytes']))
    start = time.monotonic()
    try:
        mapping = S.verified_devices(binary, selected, physical)
        summary['selected_devices'] = mapping
        summary['device_policy'] = S.device_policy(args, physical)
        if S.dense_topology(args) == 'primary':
            if [record.split(':', 1)[0] for record in getattr(args, 'expert_plan', [])] != ['0', '1']:
                raise ValueError('primary topology requires ordered slot 0/1 expert plans')
            primary_paths = [Path(record.split(':', 1)[1]) for record in args.expert_plan]
            summary['primary_expert_plans'] = L.primary_plans(primary_paths)
        for index, prompt in enumerate(prompts, 1):
            S.admit_host(S.host_memory(), work['working_set_bound_bytes'])
            case = args.out / f'case-{index:02d}'
            case.mkdir()
            initial = S.sample_vram(physical)
            budgets = S.admit_vram(initial, mapping, caps, S.device_reserves(args))
            summary['last_baseline_vram'] = initial
            summary['last_device_budgets'] = budgets
            command = [str(binary), '--model', str((getattr(args, 'model_root', None) or S.MODEL_ROOT) / 'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'),
                       '--manifest', getattr(args, 'manifest', S.MANIFEST), '--tokens', ','.join(map(str, prompt)),
                       '--context', '256', '--max-new', '32', '--threads', '4', '--expert-workers', '12',
                       '--host-cap-mib', '32768', '--vulkan', args.devices,
                       '--device-cap-mib', S.cap_argv(args),
                       '--split', str(args.split), '--placements', str(case / 'placements.tsv')]
            command += S.topology_argv(args)
            for plan in getattr(args, 'expert_plan', []):
                command += ['--expert-plan', plan]
            if getattr(args, 'expert_plan', []):
                for name in ('expert_tensor_mib','expert_workspace_mib','expert_free_reserve_mib'):
                    command += ['--'+name.replace('_','-'), getattr(args,name)]
                if getattr(args,'expert_check_rows',False): command += ['--expert-check-rows']
            if benchmark or size:
                command += ['--timings', str(case / 'timings.json')]
            command += residency_flags(args)
            if native is not None:
                if native_plans(args) != plans:
                    raise ValueError('frozen native expert plans changed')
                command += ['--expert-check-rows'] if '--expert-check-rows' not in command else []
                command += native_flags(native, case)
            if capture:
                requested=case/'executed-logits';requested.mkdir()
                command += ['--dump-executed-logits',str(requested)]
            Q.write(case / 'device-budgets.json', budgets)
            command += S.budget_argv(budgets)
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
                        samples = S.sample_vram(physical, initial)
                        summary['last_vram'] = samples
                        S.guard_vram(samples, initial, budgets)
                        for die, sample in enumerate(samples):
                            used = sample['used']
                            peak[die] = max(peak[die], used)
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
                participation = S.placements(case / 'placements.tsv', selected, args.split, S.dense_topology(args))
                if S.dense_topology(args) == 'primary':
                    L.primary_plan_reports((case/'stderr.log').read_text(), primary_paths,
                                           summary['primary_expert_plans'], selected)
                records, diagnostics = stdout_records(case / 'stdout.jsonl')
                ids, stop = completion(records,prompt,len(tok.tokens),summary['head_policy'],size,native)
                native_receipt = None
                if native is not None:
                    if native_plans(args) != plans:
                        raise ValueError('frozen native expert plans changed')
                    native_receipt = native_case_receipts(case, native, plans, selected, len(records)-1)
                summary['requests'].append(dict(index=index, tokens=ids, finish=stop, seconds=time.monotonic()-began,
                    tensor_operations_by_device=participation, baseline_vram=initial, device_budgets=budgets, sampled_peak_vram=peak,
                    minimum_effective_available=minimum))
                if diagnostics:
                    summary['requests'][-1]['stdout_diagnostics'] = diagnostics
                if native is not None:
                    summary['requests'][-1]['native_work'] = native_work(records[-1], native, len(records)-1)
                    summary['requests'][-1]['native_receipts'] = native_receipt
                if capture:
                    diagnostic,_=E.load(requested,size)
                    if diagnostic['prompt_tokens']!=prompt or diagnostic['output_count']!=len(ids) or diagnostic['stop']!=stop or \
                       [f['token'] for f in diagnostic['frames']]!=[prompt[-1]]+ids:
                        raise ValueError('executed-head manifest differs from request receipt')
                    summary['requests'][-1]['executed_logits_manifest']=diagnostic
                if size:
                    timing=json.loads((case/'timings.json').read_text())
                    chunk_fields(timing,len(prompt),len(records)-1,size)
                    projection_binding(timing,projections)
                    causal_binding(timing,causal)
                    summary['requests'][-1]['chunk_work']=timing
                if benchmark:
                    timing = json.loads((case / 'timings.json').read_text())
                    projection_binding(timing,projections)
                    causal_binding(timing,causal)
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
    S.device_arguments(p)
    residency_arguments(p)
    native_arguments(p)
    p.add_argument('--dump-executed-logits',action='store_true',help='capture executed heads in each case/executed-logits; requires head elision')
    p.add_argument('--single-case', action='store_true', help='only frozen case 1 once, timed; no warmup or scoring')
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
    if getattr(args,'single_case',False) and getattr(args,'benchmark',False):
        parser.error('--single-case excludes --benchmark')
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
