"""Exercise role framing, all-layer accounting and actual block-byte scanning."""
from pathlib import Path
import importlib.util,json,struct,subprocess,sys,tempfile
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
spec=importlib.util.spec_from_file_location('wo_roles',root/'tools/native_qsa_wo_role_preflight.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

with tempfile.TemporaryDirectory(prefix='wo-role-test-') as tmp:
    d=Path(tmp);paths=[];names=sorted(m.NAMES);starts=[]
    for index in range(6):
        p=d/(str(index)+'.gguf');paths.append(p)
        text=lambda s:struct.pack('<Q',len(s))+s.encode()
        header=bytearray(struct.pack('<IIQQ',0x46554747,3,2,4 if index==0 else 3))
        if index==0:header+=text('general.architecture')+struct.pack('<I',8)+text('qwen4exp')
        for key,typ,value in [('split.count',2,6),('split.no',2,index),('split.tensors.count',5,12)]:
            header+=text(key)+struct.pack('<I',typ)+struct.pack('<H' if typ==2 else '<i',value)
        for j in range(2):header+=text(names[2*index+j])+struct.pack('<IQQIQ',2,6144,2560,8,j*m.BYTES)
        header+=bytes((-len(header))%32);starts.append(len(header))
        with p.open('wb') as f:f.write(header);f.truncate(len(header)+2*m.BYTES)
    report=m.scan(paths);assert report['supported'] and len(report['roles'])==12
    assert all(x['scale_classes']['zero']==m.BLOCKS for x in report['roles'].values())
    if len(sys.argv)>1:
        control=Path(sys.argv[1]).resolve();packed=d/'packed.bin';out=d/'out.f32le';packed.write_bytes(bytes(6912));out.write_bytes(bytes(10240))
        command=[str(control),*(str(p) for p in paths),str(packed),str(out)]
        r=subprocess.run(command,capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr
        assert json.loads(r.stdout)['exact'] is True
        bad=bytearray(out.read_bytes());bad[0]=1;out.write_bytes(bad)
        r=subprocess.run(command,capture_output=True,text=True,timeout=60);assert r.returncode and 'projected byte mismatch' in r.stderr
    with paths[0].open('r+b') as f:f.seek(starts[0]);f.write(b'\x01\x00')
    report=m.scan(paths);assert not report['supported'] and report['roles'][names[0]]['scale_classes']['subnormal']==1
    # Type field in the first directory entry, while keeping a valid GGUF header.
    bad=m.GGUFFile(paths[0]);assert bad.tensors[0].type_id==8
    with paths[0].open('rb') as f:blob=f.read(starts[0])
    needle=names[0].encode()+struct.pack('<IQQI',2,6144,2560,8)
    at=blob.index(needle)+len(names[0])+4+8+8
    with paths[0].open('r+b') as f:f.seek(at);f.write(struct.pack('<I',2))
    try:m.scan(paths)
    except ValueError as e:assert 'type or shape' in str(e)
    else:raise AssertionError('wrong role shape accepted')
print('all-role framing, byte hashes, scale classes and malformed-role refusal passed')
