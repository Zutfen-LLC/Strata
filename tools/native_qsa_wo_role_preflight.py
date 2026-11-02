"""Read the selected GGUF WO roles locally; emit identities and scale classes only."""
from pathlib import Path
import hashlib, json, sys
from gguf_reader import GGUFFile

ROWS=2560
BLOCKS=ROWS*192
BYTES=BLOCKS*34
NAMES={f'blk.{layer}.attn_output.weight' for layer in range(3,48,4)}

def inspect_role(path,offset,blocks):
    counts={'zero':0,'normal':0,'subnormal':0,'nonfinite':0};digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        f.seek(offset)
        while blocks:
            n=min(blocks,16384);data=f.read(n*34)
            if len(data)!=n*34: raise ValueError('truncated Q8_0 role')
            digest.update(data)
            for i in range(0,len(data),34):
                h=data[i]|data[i+1]<<8;e=(h>>10)&31;m=h&1023
                key='nonfinite' if e==31 else 'normal' if e else 'subnormal' if m else 'zero'
                counts[key]+=1
            blocks-=n
    return counts,digest.hexdigest()

def scan(paths):
    if len(paths)!=6: raise ValueError('exactly six GGUF shards are required')
    paths=[Path(p).resolve(strict=True) for p in paths]
    if len(set(paths))!=6: raise ValueError('duplicate GGUF shard path')
    files=[GGUFFile(p) for p in paths];split=[];expected_total=None
    for f in files:
        count=f.metadata.get('split.count');number=f.metadata.get('split.no');total=f.metadata.get('split.tensors.count')
        if count!=6 or not isinstance(number,int) or not 0<=number<6 or not isinstance(total,int): raise ValueError('invalid GGUF split metadata')
        if expected_total is None:expected_total=total
        if total!=expected_total:raise ValueError('GGUF split tensor totals differ')
        split.append(number)
    if set(split)!=set(range(6)) or sum(len(f.tensors) for f in files)!=expected_total:raise ValueError('incomplete GGUF split')
    roles={}
    for f in files:
        for t in f.tensors:
            if t.name not in NAMES:continue
            if t.name in roles:raise ValueError('duplicate role '+t.name)
            if t.type_id!=8 or t.shape!=[6144,2560] or t.expected_bytes()!=BYTES:raise ValueError('unsupported role type or shape: '+t.name)
            start=f.data_start+t.offset
            if t.offset<0 or start+BYTES>f.path.stat().st_size:raise ValueError('role outside GGUF: '+t.name)
            counts,sha=inspect_role(f.path,start,BLOCKS)
            roles[t.name]={'type':'Q8_0','shape':t.shape,'bytes':BYTES,'sha256':sha,'scale_classes':counts}
    if set(roles)!=NAMES:raise ValueError('missing WO roles: '+', '.join(sorted(NAMES-set(roles))))
    return {'roles':dict(sorted(roles.items())),'supported':all(not r['scale_classes']['subnormal'] and not r['scale_classes']['nonfinite'] for r in roles.values()),'qualified':False}

if __name__=='__main__':
    try:
        report=scan(sys.argv[1:]);print(json.dumps(report,indent=2));raise SystemExit(0 if report['supported'] else 1)
    except (OSError,ValueError,KeyError) as e:
        print('QSA WO role preflight: '+str(e),file=sys.stderr);raise SystemExit(1)
