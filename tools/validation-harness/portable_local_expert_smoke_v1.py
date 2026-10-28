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


def devices(log, stage):
    out = {}
    for line in log.splitlines():
        if not line.startswith('expert_device stage='+stage+' '):
            continue
        fields = dict(part.split('=', 1) for part in line.split()[1:])
        slot = int(fields['slot'])
        if slot not in (0, 1) or fields['backend'] != f'Vulkan{slot}' or fields['numerics'] != 'VULKAN_HYBRID_Q8':
            raise ValueError('unexpected expert device/backend contract')
        out[slot] = fields
    return out


def assignments(log):
    pools = [set(), set()]
    for line in log.splitlines():
        match = re.fullmatch(r'expert_route position=0 layer=(\d+) ids=([\d,]+)', line)
        if match:
            layer = int(match[1])
            if layer not in range(48): raise ValueError('invalid layer')
            ids = list(map(int, match[2].split(',')))
            if len(ids) != 10 or len(set(ids)) != 10 or any(i not in range(512) for i in ids):
                raise ValueError('invalid route IDs')
            pools[int(layer >= 32)].update((layer, expert) for expert in ids)
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
    args = p.parse_args()
    if len(args.plan) != (2 if args.phase == 'smoke' else 0): p.error('smoke requires exactly two plans; discover has none')
    binary = args.binary.resolve(strict=True)
    if Q.bundle.sha(binary) != args.binary_sha256: raise ValueError('binary hash mismatch')
    args.out.mkdir(mode=0o700, exist_ok=False)
    summary = dict(status='STOP', phase=args.phase, binary_sha256=args.binary_sha256, prompt=[S.PROMPT[0]], max_new=0)
    command = [str(binary), '--model', str(S.MODEL_ROOT/'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'),
               '--manifest', S.MANIFEST, '--tokens', str(S.PROMPT[0]), '--context', '256', '--max-new', '0',
               '--threads', '4', '--expert-workers', '12', '--host-cap-mib', '32768',
               '--vulkan', '0,1', '--device-cap-mib', '7680,7680', '--split', '32',
               '--placements', str(args.out/'placements.tsv')]
    if args.phase == 'discover': command += ['--expert-record-routes']
    else:
        for slot, plan in enumerate(args.plan): command += ['--expert-plan', f'{slot}:{plan.resolve(strict=True)}']
        command += ['--expert-tensor-mib', args.tensor_mib, '--expert-workspace-mib', args.workspace_mib,
                    '--expert-free-reserve-mib', args.free_reserve_mib, '--expert-check-rows']
    Q.write(args.out/'argv.json', command)
    child = None
    try:
        baseline = S.host_memory(); work = S.model_working_set()
        summary['baseline_host'] = baseline
        summary['minimum_effective_available'] = baseline['effective_available']
        summary['host_admission'] = S.admit_host(baseline, work['working_set_bound_bytes'])
        initial = [S.vram(pci) for pci in S.PCI]
        summary['baseline_vram'] = initial
        summary['sampled_peak_vram'] = [v['used'] for v in initial]
        summary['minimum_vram_free'] = [v['total']-v['used'] for v in initial]
        summary['sample_count'] = 0
        summary['peak_process'] = dict(VmRSS=0, VmHWM=0, VmSwap=0)
        summary['peak_swap_used'] = baseline.get('SwapUsed', 0)
        summary['maximum_swap_out_pages_delta'] = 0
        if any(v['total']-v['used'] < 7680 << 20 for v in initial): raise ValueError('per-die free admission')
        start = time.monotonic()
        with (args.out/'stdout.jsonl').open('x') as stdout, (args.out/'stderr.log').open('x') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            while True:
                host, process = S.host_memory(), S.rss(child.pid)
                sample = [S.vram(pci) for pci in S.PCI]
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
                for die, value in enumerate(sample):
                    if value['used']-initial[die]['used'] > 7680 << 20: raise ValueError('per-die growth cap')
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
        S.placements(args.out/'placements.tsv', [0,1], 32)
        log = (args.out/'stderr.log').read_text()
        if args.phase == 'discover':
            for slot, pairs in enumerate(assignments(log)):
                (args.out/f'assignments-{slot}.txt').write_text(''.join(f'{l} {e}\n' for l,e in pairs))
        else:
            report = devices(log, 'token-complete')
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
