#!/usr/bin/env python3
"""Four identical sequential requests, one excluded warmup. No network or retries."""
import argparse
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import portable_model_smoke_v1 as S
import portable_local_expert_smoke_v1 as L
import quality as Q


class Resident:
    """Owned finite child. tick runs throughout load, requests and cleanup waits."""
    def __init__(self, command, out, tick=lambda _pid: None, load_seconds=180, request_seconds=180, total_seconds=960):
        self.out, self.tick = out, tick
        self.request_seconds, self.load_seconds = request_seconds, load_seconds
        self.start = time.monotonic()
        self.end = self.start + total_seconds
        self.buffer = b''
        self.log = (out/'stderr.log').open('xb')
        self.transcript = (out/'stdout.jsonl').open('xb')
        self.child = None
        self.selector = selectors.DefaultSelector()
        try:
            self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=self.log, start_new_session=True)
            self.selector.register(self.child.stdout, selectors.EVENT_READ)
        except BaseException:
            self.close()
            raise

    def row(self, deadline):
        while True:
            self.tick(self.child.pid)
            if time.monotonic() > min(deadline, self.end):
                raise TimeoutError('resident process/request deadline; no retry')
            if (self.out/'stderr.log').stat().st_size > 16 << 20:
                raise ValueError('stderr cap')
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                self.transcript.write(line+b'\n'); self.transcript.flush()
                if self.transcript.tell() > 1 << 20: raise ValueError('stdout cap')
                return json.loads(line)
            if not self.selector.select(.2): continue
            data = os.read(self.child.stdout.fileno(), 65536)
            if not data: raise RuntimeError('resident EOF before response completed')
            self.buffer += data
            if len(self.buffer) > 65536: raise ValueError('response framing cap')

    def ready(self):
        row = self.row(self.start+self.load_seconds)
        if row.get('ready') is not True or row.get('schema') != 1:
            raise ValueError('missing readiness')
        return row

    def request(self, tokens, max_new, index):
        if not tokens or len(tokens)>256 or max_new not in range(257): raise ValueError('request bound')
        self.child.stdin.write(('GEN '+str(max_new)+' '+','.join(map(str,tokens))+'\n').encode())
        self.child.stdin.flush()
        deadline = time.monotonic()+self.request_seconds
        rows, stop = [], None
        for _ in range(len(tokens)+max_new+2):
            row = self.row(deadline)
            if 'done' in row:
                if row['done'] != index or stop is None: raise ValueError('request framing')
                if row['prompt_count'] != len(tokens) or row['output_count'] != len(rows)-len(tokens):
                    raise ValueError('actual token counts')
                if stop.get('position')!=len(rows) or len(rows)<len(tokens): raise ValueError('stop position')
                count=len(rows)-len(tokens)
                reason=stop.get('stop')
                if reason not in ('eos','context','max_new'): raise ValueError('unknown stop')
                if reason=='eos' and (not count or rows[-1]['token']!=248046): raise ValueError('EOS stop')
                if reason=='max_new' and count!=max_new: raise ValueError('incomplete max-new')
                if reason=='context' and len(rows)!=256: raise ValueError('unexpected context stop')
                if any(rows[i]['token']!=rows[i-1]['greedy_next'] for i in range(len(tokens),len(rows))):
                    raise ValueError('non-greedy completion')
                for key, count in (('gpu_infer_rows_ms_delta', False), ('gpu_infer_rows_calls_delta', True)):
                    if key in row:
                        if not isinstance(row[key], list) or any(not isinstance(v, (int,float)) or
                                not math.isfinite(v) or v<0 or (count and type(v)!=int) for v in row[key]):
                            raise ValueError('invalid GPU timing/call delta')
                for key, value in row.items():
                    if (key.endswith('_ms') or key.endswith('_ms_delta')) and not isinstance(value,list) and (not math.isfinite(value) or value<0): raise ValueError('invalid timer')
                    if key.endswith('_delta') and not key.endswith('_ms_delta') and key not in ('gpu_entries_per_device_delta','gpu_infer_rows_ms_delta','gpu_infer_rows_calls_delta') and (not isinstance(value,int) or value<0): raise ValueError('invalid delta')
                return dict(rows=rows, stop=stop, metrics=row)
            if 'stop' in row:
                if stop is not None: raise ValueError('duplicate stop')
                stop=row
                continue
            pos=len(rows)
            if stop is not None or row.get('position')!=pos or not isinstance(row.get('token'),int):
                raise ValueError('invalid token row')
            if row.get('generated') != (pos>=len(tokens)): raise ValueError('token kind')
            if pos<len(tokens) and row['token']!=tokens[pos]: raise ValueError('prompt changed')
            rows.append(row)
        raise ValueError('too many response rows')

    def finish(self):
        self.child.stdin.write(b'QUIT\n'); self.child.stdin.flush(); self.child.stdin.close()
        deadline=min(self.end,time.monotonic()+15)
        while self.child.poll() is None:
            self.tick(self.child.pid)
            if time.monotonic()>deadline: raise TimeoutError('resident exit deadline')
            time.sleep(.1)
        if self.child.wait()!=0: raise RuntimeError('resident failed on exit')
        if self.buffer or self.child.stdout.read(1): raise ValueError('unexpected extra response')

    def close(self):
        if self.child is not None:
            if self.child.poll() is None:
                try: os.killpg(self.child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(self.child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    self.child.wait(timeout=5)
            else: self.child.wait()
            if self.child.stdin and not self.child.stdin.closed:
                try: self.child.stdin.close()
                except BrokenPipeError: pass
            self.child.stdout.close()
        self.selector.close(); self.log.close(); self.transcript.close()


def model_phase_fields(metrics, consumed):
    for key, multiplier in (('model_step_calls_delta',1),('gdn_layer_calls_delta',36),
                            ('qsa_layer_calls_delta',12),('output_head_calls_delta',1),
                            ('gdn_attention_exchange_calls_delta',36),('gdn_attention_compute_calls_delta',36),
                            ('gdn_ffn_calls_delta',36)):
        if metrics[key] != consumed*multiplier: raise ValueError('model phase call capture incomplete')
    step=metrics['model_step_ms_delta']; gdn=metrics['gdn_layer_ms_delta']
    qsa=metrics['qsa_layer_ms_delta']; head=metrics['output_head_ms_delta']
    gp=metrics['gdn_provider_ms_delta']; qp=metrics['qsa_provider_ms_delta']
    residual=metrics['model_residual_ms_delta']; provider=metrics['expert_provider_ms_delta']
    epsilon=max(1e-6,step*1e-9)
    if min(step,gdn,qsa,head,gp,qp,residual,provider)<0 or not all(math.isfinite(v) for v in (step,gdn,qsa,head,gp,qp,residual,provider)) or \
       gdn+qsa+head>step+epsilon or gp>gdn+epsilon or qp>qsa+epsilon or \
       abs(gp+qp-provider)>epsilon or \
       abs(step-gdn-qsa-head-residual)>epsilon:
        raise ValueError('model phase containment/reconciliation failed')
    attention=metrics['gdn_attention_exchange_ms_delta']; compute=metrics['gdn_attention_compute_ms_delta']
    ffn=metrics['gdn_ffn_ms_delta']; outer=metrics['gdn_outer_residual_ms_delta']
    surrounding=metrics['gdn_attention_surrounding_ms_delta']; nonprovider=metrics['gdn_ffn_non_provider_ms_delta']
    values=(attention,compute,ffn,outer,surrounding,nonprovider)
    if not all(math.isfinite(v) and v>=0 for v in values) or compute>attention+epsilon or gp>ffn+epsilon or \
       attention+ffn>gdn+epsilon or abs(gdn-attention-ffn-outer)>epsilon or \
       abs(attention-compute-surrounding)>epsilon or abs(ffn-gp-nonprovider)>epsilon:
        raise ValueError('GDN phase containment/reconciliation failed')
    # Additive diagnostics: old qualified binaries may omit the whole group.
    keys = ['gdn_'+scope+'_'+unit+'_delta' for scope in ('candidate_readback','candidate_finite','state_commit')
            for unit in ('ms','calls')]
    if any(key in metrics for key in keys):
        if not all(key in metrics for key in keys): raise ValueError('incomplete GDN candidate diagnostics')
        for key in keys:
            value = metrics[key]
            if type(value) not in ((int,float) if '_ms_' in key else (int,)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid GDN candidate diagnostic: '+key)
        readback, scan, commit = [metrics['gdn_'+scope+'_ms_delta']
                                  for scope in ('candidate_readback','candidate_finite','state_commit')]
        commits = metrics['gdn_state_commit_calls_delta']
        if metrics['gdn_candidate_readback_calls_delta'] != 109*consumed or \
           metrics['gdn_candidate_finite_calls_delta'] != 109*consumed or commits not in (0,36*consumed) or \
           readback+scan > surrounding+epsilon or commit > outer+epsilon or (commits == 0 and commit != 0):
            raise ValueError('GDN candidate diagnostic containment/counts failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens-file', type=Path, required=True, help='frozen JSON array of token IDs')
    p.add_argument('--max-new', type=int, required=True)
    p.add_argument('--expert-tensor-mib', default='2404,2404')
    p.add_argument('--expert-timings', action='store_true', help='coarse provider and per-device executor call timing')
    p.add_argument('--gdn-resident-weights', action='store_true', help='forward the opt-in GDN attention weight placement trial')
    p.add_argument('--gdn-resident-state', action='store_true', help='forward session-lifetime GDN state residency')
    p.add_argument('--dense-resident-weights', action='store_true', help='forward model-lifetime dense weight placement')
    p.add_argument('--plan', type=Path, action='append', required=True)
    S.device_arguments(p)
    args=p.parse_args()
    selected,physical=S.device_selection(p,args)
    if len(args.plan)!=2 or args.max_new not in range(257): p.error('two plans and bounded max-new required')
    caps=args.expert_tensor_mib.split(',')
    if len(caps)!=2 or any(not c.isdecimal() or not 0<int(c)<=8192 for c in caps):
        p.error('two positive bounded expert tensor caps required')
    tokens=json.loads(args.tokens_file.read_text())
    if not isinstance(tokens,list) or not 1<=len(tokens)<=256 or any(type(t)!=int or t<0 or t>2147483647 for t in tokens):
        p.error('bounded nonempty token IDs required')
    binary=args.binary.resolve(strict=True)
    if Q.bundle.sha(binary)!=args.binary_sha256: raise ValueError('binary hash mismatch')
    args.out.mkdir(mode=0o700,exist_ok=False)
    receipt=dict(status='STOP',binary_sha256=args.binary_sha256,prompt=tokens,max_new=args.max_new,
                 excluded_warmups=1,measured_requests=3)
    runner=None
    try:
        baseline=S.host_memory()
        receipt['baseline_host']=baseline
        work=S.model_working_set() if args.model_root is None else S.model_working_set(args.model_root)
        receipt['host_admission']=S.admit_host(baseline,work['working_set_bound_bytes'])
        mapping=S.verified_devices(binary,selected,physical)
        receipt['selected_devices']=mapping
        initial=S.sample_vram(physical)
        receipt['baseline_vram']=initial
        cap=args.device_cap_mib<<20
        S.admit_vram(initial,mapping,cap)
        receipt.update(sample_count=0,minimum_effective_available=baseline['effective_available'],
                       peak_process=dict(VmRSS=0,VmHWM=0,VmSwap=0),peak_swap_used=baseline.get('SwapUsed',0),
                       maximum_swap_out_pages_delta=0,sampled_peak_vram=[v['used'] for v in initial],
                       minimum_vram_free=[v['total']-v['used'] for v in initial])
        last_sample=0.0
        def tick(pid):
            nonlocal last_sample
            now=time.monotonic()
            if now-last_sample<.2: return
            last_sample=now
            host,process=S.host_memory(),S.rss(pid)
            sample=S.sample_vram(physical,initial)
            receipt['sample_count']+=1
            receipt.update(last_host=host,last_process=process,last_vram=sample)
            receipt['minimum_effective_available']=min(receipt['minimum_effective_available'],host['effective_available'])
            receipt['peak_swap_used']=max(receipt['peak_swap_used'],host.get('SwapUsed',0))
            receipt['maximum_swap_out_pages_delta']=max(receipt['maximum_swap_out_pages_delta'],host.get('pswpout',0)-baseline.get('pswpout',0))
            for key in receipt['peak_process']: receipt['peak_process'][key]=max(receipt['peak_process'][key],process.get(key,0))
            for i,v in enumerate(sample):
                receipt['sampled_peak_vram'][i]=max(receipt['sampled_peak_vram'][i],v['used'])
                receipt['minimum_vram_free'][i]=min(receipt['minimum_vram_free'][i],v['total']-v['used'])
            S.guard_host(host,baseline,process)
            if any(v['used']-initial[i]['used']>cap or v['total']-v['used']<512<<20 for i,v in enumerate(sample)):
                raise ValueError('per-die cap/free reserve')
        command=[str(binary),'--model',str((args.model_root or S.MODEL_ROOT)/'Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf'),
                 '--manifest',args.manifest,'--stdio','--max-requests','5','--context','256','--threads','4',
                 '--expert-workers','12','--host-cap-mib','32768','--vulkan',args.devices,
                 '--device-cap-mib',','.join([str(args.device_cap_mib)]*2),'--split',str(args.split),
                 '--expert-tensor-mib',args.expert_tensor_mib,'--expert-workspace-mib','128,128',
                 '--expert-free-reserve-mib','512,512','--placements',str(args.out/'placements.tsv')]
        for i,plan in enumerate(args.plan): command+=['--expert-plan',f'{i}:{plan.resolve(strict=True)}']
        if args.expert_timings: command += ['--expert-timings']
        if args.gdn_resident_weights: command += ['--gdn-resident-weights']
        if args.gdn_resident_state: command += ['--gdn-resident-state']
        if args.dense_resident_weights: command += ['--dense-resident-weights']
        receipt['plan_sha256']=[Q.bundle.sha(plan) for plan in args.plan]
        Q.write(args.out/'argv.json',command)
        runner=Resident(command,args.out,tick)
        receipt['ready']=runner.ready()
        results=[]
        for index in range(1,5):
            result=runner.request(tokens,args.max_new,index)
            if args.expert_timings:
                m=result['metrics']
                model_phase_fields(m,len(result['rows']))
                if m['expert_provider_calls_delta']!=len(result['rows'])*48 or m['expert_provider_ms_delta']<=0:
                    raise ValueError('provider timing/call capture incomplete')
                if len(m['gpu_infer_rows_ms_delta'])!=2 or len(m['gpu_infer_rows_calls_delta'])!=2 or \
                   min(m['gpu_infer_rows_ms_delta'])<=0 or min(m['gpu_infer_rows_calls_delta'])<=0:
                    raise ValueError('GPU timing/call capture incomplete')
            results.append(result)
            Q.write(args.out/f'request-{index}.json',result)
            if index>1:
                if result['rows']!=results[0]['rows'] or result['stop']!=results[0]['stop']:
                    raise ValueError('same-backend repeat changed tokens/rows')
                if any(result['metrics'][key] for key in ('cpu_verifications_delta','cpu_copied_bytes_delta','cpu_digest_bytes_delta','source_invalidations_delta')):
                    raise ValueError('warm request unexpectedly reverified/copied experts')
        runner.finish()
        if any(len(r['metrics']['gpu_entries_per_device_delta'])!=2 or min(r['metrics']['gpu_entries_per_device_delta'])<=0 for r in results):
            raise ValueError('each request must participate on both dies')
        S.placements(args.out/'placements.tsv',selected,args.split)
        log=(args.out/'stderr.log').read_text()
        report=L.devices(log,'token-complete',selected)
        if set(report)!={0,1} or any(int(d['entries'])<=0 for d in report.values()): raise ValueError('GPU participation required on both dies')
        receipt['expert_devices_cumulative']=report
        receipt['requests']=[r['metrics'] for r in results]
        receipt['status']='complete'
    except BaseException as error:
        receipt['error']=str(error)
        raise
    finally:
        if runner:
            try: runner.close()
            except BaseException as error: receipt['cleanup_error']=str(error)
            receipt.update(exit_code=runner.child.returncode,child_reaped=runner.child.returncode is not None)
            if not receipt['child_reaped'] or 'cleanup_error' in receipt: receipt['status']='STOP'
        Q.write(args.out/'summary.json',receipt)
        if runner and (not receipt['child_reaped'] or 'cleanup_error' in receipt):
            raise RuntimeError('owned cleanup uncertain; see summary.json')


if __name__=='__main__':
    def terminate(_sig,_frame): raise SystemExit('termination requested; owned cleanup')
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    main()
