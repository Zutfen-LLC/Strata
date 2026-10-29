"""Real sparse-model CLI lifecycle checks; driven by the C++ fixture owner."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from portable_resident_v1 import Resident, model_phase_fields
import portable_quality_v1 as P
import executed_logits_v1 as E

binary,model,manifest=sys.argv[1:]
command=[binary,'--model',model,'--manifest',manifest,'--context','4','--stdio','--max-requests','8']
with tempfile.TemporaryDirectory() as d:
    root=Path(d)
    def start(name):
        out=root/name;out.mkdir()
        r=Resident(command,out,load_seconds=90,request_seconds=90,total_seconds=240)
        r.ready()
        return r
    baseline=subprocess.run([binary,'--model',model,'--manifest',manifest,'--tokens','5,7','--max-new','2','--context','4'],
                                capture_output=True,text=True,check=True,timeout=90)
    baseline_rows=[json.loads(line) for line in baseline.stdout.splitlines()]
    command+=['--expert-timings']
    r=start('reuse')
    try:
        a=r.request([5,7],2,1)
        assert a['rows']==baseline_rows[:-1] and a['stop']==baseline_rows[-1]
        assert a['stop']['stop']=='eos' and a['metrics']['output_count']==1
        b=r.request([6],1,2)
        again=r.request([5,7],2,3)
        assert a['rows']==again['rows'] and a['stop']==again['stop']
        assert again['metrics']['cpu_verifications_delta']==0
        assert again['metrics']['cpu_hits_delta']==480*3
        assert a['metrics']['cpu_verifications_delta']==480
        assert a['metrics']['cpu_hits_delta']==480*2
        for result in (a,b,again):
            values=[result['metrics']['cpu_pool_'+key+'_ms_delta'] for key in ('gu','quant','down','elapsed')]
            assert all(math.isfinite(value) and value>=0 for value in values)
            assert values[3]>0 and sum(values[:3])<=values[3]+1e-6
        assert all('cpu_pool_' not in line for line in baseline.stdout.splitlines())
        for result in (a,b,again):
            m=result['metrics']
            assert m['expert_provider_calls_delta']==len(result['rows'])*48
            assert math.isfinite(m['expert_provider_ms_delta']) and m['expert_provider_ms_delta']>=m['cpu_pool_elapsed_ms_delta']
            assert m['model_step_calls_delta']==len(result['rows']) and m['output_head_calls_delta']==len(result['rows'])
            assert m['gdn_layer_calls_delta']==36*len(result['rows']) and m['qsa_layer_calls_delta']==12*len(result['rows'])
            model_phase_fields(m,len(result['rows']))
            assert m['gdn_candidate_readback_calls_delta'] == 109*len(result['rows'])
            assert m['gdn_candidate_finite_calls_delta'] == 109*len(result['rows'])
            assert m['gdn_state_commit_calls_delta'] == 0 and m['gdn_state_commit_ms_delta'] == 0
            assert m['model_residual_ms_delta']>=0
            assert abs(m['gdn_provider_ms_delta']+m['qsa_provider_ms_delta']-m['expert_provider_ms_delta'])<1e-6
            assert m['gdn_layer_ms_delta']>=m['gdn_provider_ms_delta'] and m['qsa_layer_ms_delta']>=m['qsa_provider_ms_delta']
            assert m['gpu_infer_rows_ms_delta']==[] and m['gpu_infer_rows_calls_delta']==[]
        r.finish()
    finally: r.close()
    # Same sparse backend and final/generated rows; skipped rows expose no
    # prediction. A/B/A includes different prompt lengths and max-new zero.
    command+=['--prompt-head-elision']
    r=start('head-elision')
    try:
        a=r.request([5,7],2,1)
        assert a['rows'][0]['greedy_next'] is None and not a['rows'][0]['logits_available']
        for actual,expected in zip(a['rows'][1:],baseline_rows[1:-1]):
            assert {k:actual[k] for k in expected}==expected
        assert a['stop']==baseline_rows[-1]
        fresh=subprocess.run([binary,'--model',model,'--manifest',manifest,'--tokens','5,7',
                              '--max-new','2','--context','4','--prompt-head-elision'],
                             capture_output=True,text=True,check=True,timeout=90)
        assert [json.loads(line) for line in fresh.stdout.splitlines()]==a['rows']+[a['stop']]
        assert 'prompt_head_work executed=2 skipped=1' in fresh.stderr
        b=r.request([6],0,2)
        again=r.request([5,7],2,3)
        assert a['rows']==again['rows'] and a['stop']==again['stop']
        for result,prompt_count in ((a,2),(b,1),(again,2)):
            m=result['metrics']
            assert m['heads_skipped_delta']==prompt_count-1
            assert m['heads_executed_delta']==len(result['rows'])-prompt_count+1
            model_phase_fields(m,len(result['rows']),'eager_diagnostic','prompt_head_elision')
        r.finish()
    finally: r.close()
    command.remove('--prompt-head-elision')
    serial_dump=root/'serial-executed';serial_dump.mkdir()
    subprocess.run([binary,'--model',model,'--manifest',manifest,'--tokens','5,6,7','--max-new','1',
                    '--context','4','--prompt-head-elision','--dump-executed-logits',str(serial_dump)],
                   capture_output=True,text=True,check=True,timeout=90)
    serial_manifest,_=E.load(serial_dump,0)
    assert [f['position'] for f in serial_manifest['frames']]==[2,3]
    assert [f['role'] for f in serial_manifest['frames']]==['final_prompt','terminal']
    # Actual fresh/resident chunk policy, not a fake child: implicit head
    # elision, T1 tails, terminal feed, A/B/A, and actual execution counts.
    for size in (2,8):
        projection_flags=['--gdn-batch-input-projections'] if size==8 else []
        command+=projection_flags+['--prompt-chunk-size',str(size)]
        r=start('chunks-'+str(size))
        try:
            a=r.request([5,6,7],1,1)
            b=r.request([6],0,2)
            again=r.request([5,6,7],1,3)
            assert a['rows']==again['rows'] and a['stop']==again['stop']
            assert a['stop']['stop']=='eos'
            for result,prompt in ((a,[5,6,7]),(b,[6]),(again,[5,6,7])):
                m=result['metrics'];n=len(result['rows'])
                assert m.get('gdn_batch_input_projections',False)==bool(projection_flags)
                forwards=P.chunk_fields(m,len(prompt),n,size)
                model_phase_fields(m,n,'eager_diagnostic','prompt_head_elision',size)
                assert m['model_step_calls_delta']==forwards
                assert m['expert_provider_calls_delta']==48*forwards
                assert not any(k.startswith('gdn_attention_') or k.startswith('gdn_candidate_') for k in m)
            assert a['metrics']['logical_tokens_delta']==4 and a['metrics']['model_forward_calls_delta']==3
            assert a['metrics']['chunk_ffn_pre_calls_delta']==48
            assert a['rows'][0]['greedy_next'] is None and a['rows'][1]['greedy_next'] is None
            assert a['rows'][2]['logits_available'] and a['rows'][3]['generated']
            assert again['metrics']['cpu_verifications_delta']==0
            assert again['metrics']['cpu_hits_delta']==480*3
            timing=root/f'chunk-{size}-timings.json'
            requested=root/f'chunk-{size}-executed';requested.mkdir()
            fresh=subprocess.run([binary,'--model',model,'--manifest',manifest,'--tokens','5,6,7',
                                  '--max-new','1','--context','4','--prompt-chunk-size',str(size),'--timings',str(timing),'--dump-executed-logits',str(requested),*projection_flags],
                                 capture_output=True,text=True,check=True,timeout=90)
            assert [json.loads(line) for line in fresh.stdout.splitlines()]==a['rows']+[a['stop']]
            assert E.compare(serial_dump,requested,0,size)['numeric_pass']
            assert 'prompt_chunk_admission shape_envelope=1' in fresh.stderr
            P.chunk_fields(json.loads(timing.read_text()),3,4,size)
            r.finish()
        finally:r.close()
        command=command[:-(2+len(projection_flags))]
    final_only=root/'final-only';final_only.mkdir()
    subprocess.run([binary,'--model',model,'--manifest',manifest,'--tokens','6','--max-new','0',
                    '--context','4','--prompt-head-elision','--dump-executed-logits',str(final_only)],
                   capture_output=True,text=True,check=True,timeout=90)
    only,_=E.load(final_only,0)
    assert only['output_count']==0 and only['frames'][0]['position']==0 and only['frames'][0]['role']=='final_prompt'
    for flags in ([],['--stdio','--prompt-head-elision'],['--prompt-head-elision','--dump-logits',str(root)]):
        rejected=subprocess.run([binary,*flags,'--dump-executed-logits',str(root)],capture_output=True,text=True,timeout=5)
        assert rejected.returncode!=0 and 'requires fresh head elision' in rejected.stderr
    rejected=subprocess.run([binary,'--prompt-chunk-size','2','--dump-logits',str(root)],
                            capture_output=True,text=True,timeout=5)
    assert rejected.returncode!=0 and 'cannot dump skipped logits' in rejected.stderr
    rejected=subprocess.run([binary,'--gdn-batch-input-projections'],capture_output=True,text=True,timeout=5)
    assert rejected.returncode!=0 and 'requires --prompt-chunk-size' in rejected.stderr
    for invalid in ('0','1','9'):
        rejected=subprocess.run([binary,'--prompt-chunk-size',invalid],capture_output=True,text=True,timeout=5)
        assert rejected.returncode!=0 and 'prompt chunk size' in rejected.stderr
    # Explicit exhaustive dumps remain eager; incompatible policies fail before
    # model load rather than inventing intermediate output files.
    rejected=subprocess.run([binary,'--prompt-head-elision','--dump-logits',str(root)],
                            capture_output=True,text=True,timeout=5)
    assert rejected.returncode!=0 and 'cannot dump skipped logits' in rejected.stderr
    for mode in ('eof','quit','signal','malformed','mutation'):
        r=start(mode)
        try:
            if mode=='eof': r.child.stdin.close()
            elif mode=='quit': r.finish()
            elif mode=='signal': r.child.send_signal(signal.SIGTERM)
            elif mode=='malformed':
                r.child.stdin.write(b'GEN 1 5,999999\nGEN 0 5\n');r.child.stdin.flush()
            else:
                stamp=os.stat(model).st_mtime_ns
                os.utime(model,ns=(os.stat(model).st_atime_ns,stamp+2000000000))
                r.child.stdin.write(b'GEN 0 5\n');r.child.stdin.flush()
            rc=r.child.wait(timeout=30)
            assert (rc==0)==(mode in ('eof','quit')), (mode,rc)
            assert not r.child.stdout.read(), 'rejected request produced token rows'
        finally: r.close()
print('real fresh/resident chunk CLI: policy/counters, unavailable logits, A/B/A and terminal T1 passed; resident lifecycle: A/B/A, EOS, verification deltas, EOF/QUIT/signal, preflight and source mutation passed')
