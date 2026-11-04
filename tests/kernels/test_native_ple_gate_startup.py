#!/usr/bin/env python3
"""Exercise the actual component contract from an unrelated working directory."""
import json,struct,subprocess,sys,tempfile
from pathlib import Path

if len(sys.argv)!=3:raise SystemExit('usage: test_native_ple_gate_startup.py COMPONENT SCRATCH')
exe=Path(sys.argv[1]).resolve(strict=True);scratch=Path(sys.argv[2]).resolve(strict=True)
with tempfile.TemporaryDirectory(dir=scratch) as t:
    root=Path(t);fixture=root/'fixture';fixture.mkdir();unrelated=root/'unrelated';unrelated.mkdir()
    key=[1.0]*(4*2560);query=[0.0]*(4*2560)
    for i in range(2560):query[i]=1.0;query[2*2560+i]=-1.0
    for tid in range(512):
        for j,x in enumerate((16777216.0,1.0,-16777216.0,1.0)):
            query[2560+tid+j*512]=x
    query[3*2560:3*2560+4]=[16777216.0,1.0,-16777216.0,1.0]
    (fixture/'key.f32le').write_bytes(struct.pack('<10240f',*key));(fixture/'query.f32le').write_bytes(struct.pack('<10240f',*query))
    contract=root/'contract.json';subprocess.run([str(exe),'--contract',str(fixture),str(contract)],cwd=unrelated,check=True,capture_output=True)
    assert json.loads(contract.read_text())=={'key_bytes':40960,'query_bytes':40960,'gate_bytes':16,'diagnostic_bytes':64,'qualified':False}
    for data in (b'\x00'*40959,struct.pack('<I',0x7f800000)+bytes(40956),struct.pack('<I',1)+bytes(40956)):
        (fixture/'key.f32le').write_bytes(data);refused=root/'refused.json'
        result=subprocess.run([str(exe),'--contract',str(fixture),str(refused)],cwd=unrelated,capture_output=True)
        assert result.returncode!=0 and not refused.exists(),'malformed/nonfinite/subnormal fixture accepted'
    print('Actual PLE gate startup contract passes and malformed/unsupported fixtures refuse without GPU initialization.')
