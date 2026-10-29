#!/usr/bin/env python3
"""One-token local expert discovery or strict two-die smoke. No network, retries or clocks."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import portable_model_smoke_v1 as S
import quality as Q


def devices(log, stage, selected=(0, 1)):
    out = {}
    for line in log.splitlines():
        if not line.startswith('expert_device stage='+stage+' '):
            continue
        fields = dict(part.split('=', 1) for part in line.split()[1:])
        slot = int(fields['slot'])
        if slot not in (0, 1) or fields['backend'] != f'Vulkan{selected[slot]}' or fields['numerics'] != 'VULKAN_HYBRID_Q8':
            raise ValueError('unexpected expert device/backend contract')
        out[slot] = fields
    return out


def primary_plans(paths):
    # Reuse the existing bounded decoder and native placement identity check.
    # SHA binding retains exact expert IDs without inventing a selection policy.
    import quality_v2 as V
    if len(paths) != 2:
        raise ValueError('primary topology requires two explicit expert plans')
    return V.plan_binding(paths)


def primary_plan_reports(log, paths, binding, selected):
    if primary_plans(paths) != binding:
        raise ValueError('primary expert plans changed')
    if sum(line.startswith('expert_device stage=loaded ') for line in log.splitlines()) != 2:
        raise ValueError('primary expert load reports missing or duplicated')
    reports = devices(log, 'loaded', selected)
    if set(reports) != {0, 1}:
        raise ValueError('primary expert load reports missing')
    for slot, plan in enumerate(binding['plans']):
        if (int(reports[slot]['assigned']) != plan['assigned_experts'] or
                int(reports[slot]['weight_bytes']) != plan['assigned_bytes']):
            raise ValueError('primary expert plan/load ownership mismatch')
    return reports


def assignments(log, split=32, dense_topology='split'):
    if dense_topology != 'split':
        raise ValueError('route discovery is split-only; supply exact native expert plans')
    pools = [set(), set()]
    for line in log.splitlines():
        match = re.fullmatch(r'expert_route position=0 layer=(\d+) ids=([\d,]+)', line)
        if match:
            layer = int(match[1])
            if layer not in range(48): raise ValueError('invalid layer')
            ids = list(map(int, match[2].split(',')))
            if len(ids) != 10 or len(set(ids)) != 10 or any(i not in range(512) for i in ids):
                raise ValueError('invalid route IDs')
            pools[int(layer >= split)].update((layer, expert) for expert in ids)
    if any(len(pool) < 16 for pool in pools): raise ValueError('incomplete route discovery')
    return [sorted(pool)[:16] for pool in pools]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=('discover', 'smoke'), required=True)
    p.add_argument('--plan', type=Path, action='append', default=[])
    p.add_argument('--tensor-mib', default='256,256')
    p.add_argument('--workspace-mib', default='128,128')
    p.add_argument('--free-reserve-mib', default='512,512')
    S.device_arguments(p)
    args = p.parse_args()
    selected, physical = S.device_selection(p, args)
    caps = S.device_caps(args)
    if len(args.plan) != (2 if args.phase == 'smoke' else 0): p.error('smoke requires exactly two plans; discover has none')
    if S.dense_topology(args) == 'primary' and args.phase == 'discover':
        p.error('primary topology requires exact plans; split route discovery is incompatible')
    binary = args.binary.resolve(strict=True)
    if Q.bundle.sha(binary) != args.binary_sha256: raise ValueError('binary hash mismatch')
    args.out.mkdir(mode=0o700, exist_ok=False)
    summary = dict(status='STOP', phase=args.phase, binary_sha256=args.binary_sha256, prompt=[S.PROMPT[0]], max_new=0)
    summary['device_policy'] = S.device_policy(args, physical)
    if S.dense_topology(args) == 'primary':
        summary['primary_expert_plans'] = primary_plans(args.plan)
    command = [str(binary), '--model', str((args.model_root or S.MODEL_ROOT)/'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'),
               '--manifest', args.manifest, '--tokens', str(S.PROMPT[0]), '--context', '256', '--max-new', '0',
               '--threads', '4', '--expert-workers', '12', '--host-cap-mib', '32768',
               '--vulkan', args.devices, '--device-cap-mib', S.cap_argv(args), '--split', str(args.split),
               '--placements', str(args.out/'placements.tsv')]
    command += S.topology_argv(args)
    if args.phase == 'discover': command += ['--expert-record-routes']
    else:
        for slot, plan in enumerate(args.plan): command += ['--expert-plan', f'{slot}:{plan.resolve(strict=True)}']
        command += ['--expert-tensor-mib', args.tensor_mib, '--expert-workspace-mib', args.workspace_mib,
                    '--expert-free-reserve-mib', args.free_reserve_mib, '--expert-check-rows']
    Q.write(args.out/'planned-argv.json', command)
    child = None
    try:
        baseline = S.host_memory()
        work = S.model_working_set() if args.model_root is None else S.model_working_set(args.model_root)
        summary['baseline_host'] = baseline
        summary['minimum_effective_available'] = baseline['effective_available']
        summary['host_admission'] = S.admit_host(baseline, work['working_set_bound_bytes'])
        mapping = S.verified_devices(binary, selected, physical)
        summary['selected_devices'] = mapping
        initial = S.sample_vram(physical)
        summary['baseline_vram'] = initial
        summary['sampled_peak_vram'] = [v['used'] for v in initial]
        summary['minimum_vram_free'] = [v['total']-v['used'] for v in initial]
        summary['sample_count'] = 0
        summary['peak_process'] = dict(VmRSS=0, VmHWM=0, VmSwap=0)
        summary['peak_swap_used'] = baseline.get('SwapUsed', 0)
        summary['maximum_swap_out_pages_delta'] = 0
        budgets = S.admit_vram(initial, mapping, caps, S.device_reserves(args))
        summary['device_budgets'] = budgets
        command += S.budget_argv(budgets)
        Q.write(args.out/'argv.json', command)
        start = time.monotonic()
        with (args.out/'stdout.jsonl').open('x') as stdout, (args.out/'stderr.log').open('x') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            while True:
                host, process = S.host_memory(), S.rss(child.pid)
                sample = S.sample_vram(physical, initial)
                # Save the failing sample too; compact extrema stay bounded in size.
                summary['sample_count'] += 1
                summary['last_host'], summary['last_process'], summary['last_vram'] = host, process, sample
                summary['minimum_effective_available'] = min(summary['minimum_effective_available'], host['effective_available'])
                summary['peak_swap_used'] = max(summary['peak_swap_used'], host.get('SwapUsed', 0))
                summary['maximum_swap_out_pages_delta'] = max(summary['maximum_swap_out_pages_delta'], host.get('pswpout', 0)-baseline.get('pswpout', 0))
                for key in summary['peak_process']:
                    summary['peak_process'][key] = max(summary['peak_process'][key], process.get(key, 0))
                for die, value in enumerate(sample):
                    summary['sampled_peak_vram'][die] = max(summary['sampled_peak_vram'][die], value['used'])
                    summary['minimum_vram_free'][die] = min(summary['minimum_vram_free'][die], value['total']-value['used'])
                summary['sampled_peak_growth'] = [summary['sampled_peak_vram'][i]-initial[i]['used'] for i in range(2)]
                S.guard_host(host, baseline, process)
                S.guard_vram(sample, initial, budgets)
                if time.monotonic()-start > 300: raise TimeoutError('one-token case budget 300s; no retry')
                if any((args.out/name).stat().st_size > 16 << 20 for name in ('stdout.jsonl','stderr.log')):
                    raise ValueError('output log cap')
                rc = child.poll()
                if rc is not None: break
                time.sleep(.2)
        summary['exit_code'] = rc
        if rc: raise RuntimeError(f'portable_model exited {rc}')
        rows = [json.loads(line) for line in (args.out/'stdout.jsonl').read_text().splitlines()]
        if len(rows) != 2 or rows[0]['token'] != S.PROMPT[0] or rows[0]['generated'] or rows[0]['position'] != 0 or rows[-1] != dict(stop='max_new', position=1):
            raise ValueError('one-token output contract')
        S.placements(args.out/'placements.tsv', selected, args.split, S.dense_topology(args))
        log = (args.out/'stderr.log').read_text()
        if S.dense_topology(args) == 'primary':
            summary['expert_devices_loaded'] = primary_plan_reports(log, args.plan, summary['primary_expert_plans'], selected)
        if args.phase == 'discover':
            for slot, pairs in enumerate(assignments(log, args.split)):
                (args.out/f'assignments-{slot}.txt').write_text(''.join(f'{l} {e}\n' for l,e in pairs))
        else:
            report = devices(log, 'token-complete', selected)
            if set(report) != {0,1} or any(int(d['entries']) <= 0 or int(d['row_checks']) != int(d['entries']) for d in report.values()):
                raise ValueError('actual expert hits and row diagnostics required on both dies')
            summary['expert_devices'] = report
        summary['status'] = 'complete'
    except BaseException as error:
        summary['error'] = str(error)
        raise
    finally:
        if child is not None:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: summary['exit_code'] = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    summary['exit_code'] = child.wait()
            else:
                summary['exit_code'] = child.wait()
            summary['child_reaped'] = True
        Q.write(args.out/'summary.json', summary)


if __name__ == '__main__':
    def terminate(_sig, _frame): raise SystemExit('termination requested; owned cleanup')
    signal.signal(signal.SIGTERM, terminate)
    main()
