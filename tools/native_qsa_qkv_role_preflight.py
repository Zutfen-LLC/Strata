"""Inspect all ordinary-QSA Q/K/V GGUF roles without extracting weights."""
from pathlib import Path
import json,sys
from gguf_reader import GGUFFile
from native_qsa_wo_role_preflight import inspect_role

ROWS={'attn_q.weight':12288,'attn_k.weight':512,'attn_v.weight':512}
NAMES={f'blk.{layer}.{name}':rows for layer in range(3,48,4) for name,rows in ROWS.items()}
def scan(paths):
    if len(paths)!=6:raise ValueError('exactly six GGUF shards are required')
    paths=[Path(p).resolve(strict=True) for p in paths]
    if len(set(paths))!=6:raise ValueError('duplicate GGUF shard path')
    files=[GGUFFile(p) for p in paths];split=[];expected_total=None
    for f in files:
        count=f.metadata.get('split.count');number=f.metadata.get('split.no');total=f.metadata.get('split.tensors.count')
        if count!=6 or not isinstance(number,int) or not 0<=number<6 or not isinstance(total,int):raise ValueError('invalid GGUF split metadata')
        if expected_total is None:expected_total=total
        if total!=expected_total:raise ValueError('GGUF split tensor totals differ')
        split.append(number)
    if split!=list(range(6)) or sum(len(f.tensors) for f in files)!=expected_total:raise ValueError('incomplete or unordered GGUF split')
    if files[0].metadata.get('general.architecture')!='qwen4exp':raise ValueError('unsupported GGUF architecture')
    roles={}
    for f in files:
        for t in f.tensors:
            if t.name not in NAMES:continue
            if t.name in roles:raise ValueError('duplicate role '+t.name)
            rows=NAMES[t.name];blocks=rows*80;size=blocks*34
            if t.type_id!=8 or t.shape!=[2560,rows] or t.expected_bytes()!=size:raise ValueError('unsupported role type or shape: '+t.name)
            start=f.data_start+t.offset
            if t.offset<0 or start+size>f.path.stat().st_size:raise ValueError('role outside GGUF: '+t.name)
            counts,digest=inspect_role(f.path,start,blocks)
            roles[t.name]={'type':'Q8_0','shape':t.shape,'bytes':size,'sha256':digest,'scale_classes':counts}
    if set(roles)!=set(NAMES):raise ValueError('missing QKV roles: '+', '.join(sorted(set(NAMES)-set(roles))))
    return {'roles':dict(sorted(roles.items())),'supported':all(not r['scale_classes']['nonfinite'] for r in roles.values()),'qualified':False}
if __name__=='__main__':
    try:
        report=scan(sys.argv[1:]);print(json.dumps(report,indent=2));raise SystemExit(0 if report['supported'] else 1)
    except (OSError,ValueError,KeyError) as e:
        print('QSA QKV role preflight: '+str(e),file=sys.stderr);raise SystemExit(1)
