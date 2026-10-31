#!/usr/bin/env python3
"""Exact bounded first-divergence diagnosis, not a numerical tolerance or parity verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

COUNTS=(10240,786432,30720,10240,2560,512,10,10,2560,25600,2560,4,786432,30720,512,1,10240,512,1,2560,4)
NAMES=('embedding','state_before','history_before','pending_residual','moe_input','router_scores',
       'router_ids','router_weights','shared_output','expert_rows','combined_output','inject',
       'state_after','history_after','primary_ownership','token','materialized_post_residual','helper_ownership','helper_reduction','attention_mixed','attention_inject')
OPTIONAL={12,13,16}
# Edges describe semantic inputs, not an asserted serial schedule for parallel branches.
DEPENDENCIES={0:(),1:(),2:(),3:(0,1,2),4:(3,),5:(4,),6:(5,),7:(5,6),8:(4,),9:(4,6,14,17),10:(6,7,8,9),11:(3,),12:(0,1,2),13:(0,1,2),19:(0,),20:(0,)}
ORDER=(0,1,2,19,20,3,11,4,5,6,7,8,9,10,12,13)
LIMIT=8<<20

def read(path):
    path=Path(path)
    if not 72<=path.stat().st_size<=LIMIT:raise ValueError('trace size outside bound')
    with path.open("rb") as f:raw=f.read(LIMIT+1)
    if not 72<=len(raw)<=LIMIT:raise ValueError("changed trace size outside bound")
    if raw[:8] not in (b'STRL001\0',b'STRL002\0',b'STRL003\0'):raise ValueError('unsupported magic')
    field_count={b'STRL001\0':18,b'STRL002\0':19,b'STRL003\0':21}[raw[:8]]
    binding=raw[8:72].decode('ascii')
    if len(binding)!=64 or any(c not in '0123456789abcdef' for c in binding):raise ValueError('invalid binding')
    fields={};at=72
    while at+8<=len(raw):
        key,count=struct.unpack_from('<II',raw,at);at+=8
        if key==0xffffffff:
            if count or at!=len(raw) or set(fields)!=set(range(field_count)):raise ValueError('invalid seal or missing fields')
            break
        if key>=field_count or key in fields or count not in (0,COUNTS[key]) or at+count*4>len(raw):raise ValueError('invalid field/shape')
        data=raw[at:at+count*4];at+=count*4;fields[key]=data if count else None
        if count and key not in (6,14,15,17,18):
            if any((v[0]&0x7f800000)==0x7f800000 for v in struct.iter_unpack('<I',data)):raise ValueError('nonfinite '+NAMES[key])
    else:raise ValueError('unsealed trace')
    if field_count==18:fields[18]=struct.pack('<i',0) # Legacy format only supported unreduced rows.
    if fields[18] is None or struct.unpack('<i',fields[18])[0] not in (0,1):raise ValueError('invalid helper reduction mode')
    if field_count<21:fields.update({19:None,20:None}) # Older captures retain their original scope.
    reduced=struct.unpack('<i',fields[18])[0]==1
    optional=OPTIONAL|({9} if reduced else set())|({19,20} if field_count<21 else set())
    if any(fields[k] is None for k in range(21) if k not in optional):raise ValueError('required boundary unavailable')
    if reduced and fields[9] is not None:raise ValueError('reduced helper cannot claim complete unweighted rows')
    if any(v[0] not in (0,1) for k in (14,17) for v in struct.iter_unpack('<i',fields[k])):raise ValueError('invalid ownership')
    if any(a[0] and b[0] for a,b in zip(struct.iter_unpack('<i',fields[14]),struct.iter_unpack('<i',fields[17]))):raise ValueError('overlapping ownership')
    ids=[v[0] for v in struct.iter_unpack('<i',fields[6])]
    if len(set(ids))!=10 or any(not 0<=v<512 for v in ids):raise ValueError('invalid route IDs')
    if struct.unpack('<i',fields[15])[0]<0:raise ValueError('invalid token')
    return binding,fields,hashlib.sha256(raw).hexdigest()

def compare(native,portable):
    a,x,ha=read(native);b,y,hb=read(portable)
    if a!=b or x[15]!=y[15] or x[14]!=y[14] or x[17]!=y[17]:raise ValueError('input/token/primary ownership binding differs')
    equal={i:x[i] is not None and y[i] is not None and x[i]==y[i] for i in range(21)}
    stages=[]
    for i in ORDER:
        available=x[i] is not None and y[i] is not None
        comparable=available and (i not in (7,9) or equal[6])
        different=None;first=None
        if comparable:
            words=[j for j,(u,v) in enumerate(zip(struct.iter_unpack('<I',x[i]),struct.iter_unpack('<I',y[i]))) if u!=v]
            different=len(words);first=words[0] if words else None
        # Legacy captures cannot assert the new HC boundary. Preserve their
        # original composite attention attribution rather than invent values.
        hc_observed=all(x[j] is not None and y[j] is not None for j in (19,20))
        prerequisites=DEPENDENCIES[i]+((19,20) if i==3 and hc_observed else ())
        missing_inputs=[NAMES[j] for j in prerequisites if x[j] is None or y[j] is None]
        matched_state=i in (0,1,2) or all(equal[j] for j in (0,1,2))
        matched_inputs=matched_state and all(equal[j] for j in prerequisites)
        stages.append(dict(boundary=NAMES[i],available=available,comparable=comparable,exact=equal[i] if comparable else None,
            different_words=different,first_word=first,matching_inputs_and_state=matched_inputs,unavailable_inputs=missing_inputs,
            attribution=('unavailable' if not available else 'route_association_differs' if not comparable else 'equal' if equal[i] else
                         'input_boundary_unavailable' if missing_inputs else 'local_boundary_candidate' if matched_inputs else 'upstream_difference_present')))
    differences=[s for s in stages if s['comparable'] and not s['exact']]
    return dict(scope='position0/layer0 diagnostic through combine; no numerical/quality/performance qualification',
        qualified=False,native_sha256=ha,portable_sha256=hb,binding=a,
        attention_hc_inputs_observed=all(x[j] is not None and y[j] is not None for j in (19,20)),
        native_helper_reduction=bool(struct.unpack('<i',x[18])[0]),portable_helper_reduction=bool(struct.unpack('<i',y[18])[0]),
        complete_unweighted_rows_available=x[9] is not None and y[9] is not None and equal[6],
        combine_inputs_observed_and_equal=all(equal[j] for j in DEPENDENCIES[10]) and all(equal[j] for j in (0,1,2)),
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
