#!/usr/bin/env python3
"""Score selected expert assignments against one bounded grouped-route count file."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from score_remote_prompt_placement import assignments

HEADER=re.compile(r'STRATA_REMOTE_ROUTE_COUNTS_V1 complete=1 calls=(\d+) generated=(\d+) finish=(stop|length) patterns=(\d+) layers=(\d+) experts=(\d+)')


def read(path):
    p=Path(path)
    if p.stat().st_size>64*1024*1024:raise ValueError('count file exceeds bound')
    with p.open() as f:
        match=HEADER.fullmatch(f.readline().rstrip('\n'))
        if not match:raise ValueError('unsealed, incomplete or unsuccessful count file')
        calls,generated,_,patterns,layers,experts=match.groups()
        calls,generated,patterns,layers,experts=map(int,(calls,generated,patterns,layers,experts))
        if not (0<calls and generated>0 and 0<patterns<=65536 and 0<layers<=1024 and 0<experts<=65536):
            raise ValueError('invalid count totals')
        rows=[];seen=set();total=0
        for line in f:
            a=line.split()
            if len(a)<5 or any(not x.isascii() or not x.isdecimal() for x in a):raise ValueError('malformed count record')
            layer,T,K,count=map(int,a[:4]);ids=tuple(map(int,a[4:]))
            key=(layer,T,K,ids)
            if layer>=layers or not 1<=T<=8 or not 1<=K<=10 or count<1 or len(ids)!=T*K or any(x>=experts for x in ids) or key in seen:
                raise ValueError('invalid or duplicate count record')
            seen.add(key);total+=count;rows.append((layer,ids,count))
            if len(rows)>patterns:raise ValueError('excess count records')
        if len(rows)!=patterns or total!=calls:raise ValueError('count total mismatch')
    return layers,experts,generated,rows


def score(rows,selected):
    counts=Counter();layers=Counter();slices=0
    for layer,ids,n in rows:
        hits=sum((layer,e) in selected for e in ids)
        counts[hits]+=n;layers[layer]+=hits*n;slices+=n
    calls=sum(n for k,n in counts.items() if k);entries=sum(k*n for k,n in counts.items())
    return {'assigned':len(selected),'slices':slices,'selected_entries':entries,'nonempty_calls':calls,
            'entries_per_call':entries/calls if calls else None,'slice_histogram':dict(sorted(counts.items())),
            'entries_by_layer':dict(sorted((k,v) for k,v in layers.items() if v)),
            'scope':'one request; includes prompt and executed speculative verification; not a latency prediction'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('counts',type=Path)
    p.add_argument('--cpu-assignments',type=Path,required=True)
    p.add_argument('plans',type=Path,nargs='+');a=p.parse_args()
    layers,experts,generated,rows=read(a.counts);cpu=assignments(a.cpu_assignments,layers,experts)
    frequency=Counter()
    for layer,ids,n in rows:
        for expert in ids:
            if (layer,expert) in cpu:frequency[(layer,expert)]+=n
    out={}
    for path in a.plans:
        selected=assignments(path,layers,experts)
        if not selected<=cpu:raise ValueError('assignment outside supplied CPU partition')
        out[str(path)]=score(rows,selected)
    print(json.dumps({'generated':generated,
        'cpu_frequency':[[l,e,n] for (l,e),n in sorted(frequency.items(),key=lambda x:(-x[1],x[0]))],
        'placements':out},indent=2))

if __name__=='__main__':main()
