#!/usr/bin/env python3
"""Exact bounded first-divergence diagnosis, not a numerical tolerance or parity verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

COUNTS=(10240,786432,30720,10240,2560,512,10,10,2560,25600,2560,4,786432,30720,512,1,10240,512)
NAMES=('embedding','state_before','history_before','pending_residual','moe_input','router_scores',
       'router_ids','router_weights','shared_output','expert_rows','combined_output','inject',
       'state_after','history_after','primary_ownership','token','materialized_post_residual','helper_ownership')
OPTIONAL={12,13,16}
# Edges describe semantic inputs, not an asserted serial schedule for parallel branches.
DEPENDENCIES={0:(),1:(),2:(),3:(0,1,2),4:(3,),5:(4,),6:(5,),7:(5,6),8:(4,),9:(4,6,14,17),10:(6,7,8,9),11:(3,),12:(0,1,2),13:(0,1,2)}
ORDER=(0,1,2,3,11,4,5,6,7,8,9,10,12,13)
LIMIT=8<<20

def read(path):
    path=Path(path)
    if not 72<=path.stat().st_size<=LIMIT:raise ValueError('trace size outside bound')
    with path.open("rb") as f:raw=f.read(LIMIT+1)
    if not 72<=len(raw)<=LIMIT:raise ValueError("changed trace size outside bound")
    if raw[:8]!=b'STRL001\0':raise ValueError('unsupported magic')
    binding=raw[8:72].decode('ascii')
    if len(binding)!=64 or any(c not in '0123456789abcdef' for c in binding):raise ValueError('invalid binding')
    fields={};at=72
    while at+8<=len(raw):
        key,count=struct.unpack_from('<II',raw,at);at+=8
        if key==0xffffffff:
            if count or at!=len(raw) or set(fields)!=set(range(18)):raise ValueError('invalid seal or missing fields')
            break
        if key>=18 or key in fields or count not in (0,COUNTS[key]) or at+count*4>len(raw):raise ValueError('invalid field/shape')
        data=raw[at:at+count*4];at+=count*4;fields[key]=data if count else None
        if count and key not in (6,14,15,17):
            if any((v[0]&0x7f800000)==0x7f800000 for v in struct.iter_unpack('<I',data)):raise ValueError('nonfinite '+NAMES[key])
    else:raise ValueError('unsealed trace')
    if any(fields[k] is None for k in range(18) if k not in OPTIONAL):raise ValueError('required boundary unavailable')
    if any(v[0] not in (0,1) for k in (14,17) for v in struct.iter_unpack('<i',fields[k])):raise ValueError('invalid ownership')
    if any(a[0] and b[0] for a,b in zip(struct.iter_unpack('<i',fields[14]),struct.iter_unpack('<i',fields[17]))):raise ValueError('overlapping ownership')
    ids=[v[0] for v in struct.iter_unpack('<i',fields[6])]
    if len(set(ids))!=10 or any(not 0<=v<512 for v in ids):raise ValueError('invalid route IDs')
    if struct.unpack('<i',fields[15])[0]<0:raise ValueError('invalid token')
    return binding,fields,hashlib.sha256(raw).hexdigest()

def compare(native,portable):
    a,x,ha=read(native);b,y,hb=read(portable)
    if a!=b or x[15]!=y[15] or x[14]!=y[14] or x[17]!=y[17]:raise ValueError('input/token/primary ownership binding differs')
    equal={i:x[i] is not None and y[i] is not None and x[i]==y[i] for i in range(18)}
    stages=[]
    for i in ORDER:
        available=x[i] is not None and y[i] is not None
        comparable=available and (i not in (7,9) or equal[6])
        different=None;first=None
        if comparable:
            words=[j for j,(u,v) in enumerate(zip(struct.iter_unpack('<I',x[i]),struct.iter_unpack('<I',y[i]))) if u!=v]
            different=len(words);first=words[0] if words else None
        prerequisites=DEPENDENCIES[i]
        stages.append(dict(boundary=NAMES[i],available=available,comparable=comparable,exact=equal[i] if comparable else None,
            different_words=different,first_word=first,matching_inputs_and_state=all(equal[j] for j in prerequisites),
            attribution=('unavailable' if not available else 'route_association_differs' if not comparable else 'equal' if equal[i] else
                         'local_boundary_candidate' if all(equal[j] for j in prerequisites) else 'upstream_difference_present')))
    differences=[s for s in stages if s['comparable'] and not s['exact']]
    return dict(scope='position0/layer0 diagnostic through combine; no numerical/quality/performance qualification',
        qualified=False,native_sha256=ha,portable_sha256=hb,binding=a,
        exact_observed_boundaries=not differences,first_difference=differences[0]['boundary'] if differences else None,
        materialized_post_residual='unavailable on native; never reconstructed or compared',stages=stages)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('native',type=Path);p.add_argument('portable',type=Path);p.add_argument('report',type=Path)
    args=p.parse_args()
    try:
        result=compare(args.native,args.portable)
        with args.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    except (ValueError,OSError,UnicodeError) as error:
        print('STOP: '+str(error),file=sys.stderr);return 2
    return 0 if result['exact_observed_boundaries'] else 1
if __name__=='__main__':raise SystemExit(main())
