"""Real sparse-model CLI lifecycle checks; driven by the C++ fixture owner."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from portable_resident_v1 import Resident

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
        r.finish()
    finally: r.close()
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
print('real resident CLI: A/B/A, EOS, verification deltas, EOF/QUIT/signal, preflight and source mutation passed')
