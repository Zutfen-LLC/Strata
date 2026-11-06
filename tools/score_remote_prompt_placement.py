#!/usr/bin/env python3
"""Score fixed remote expert assignments on an existing local first-request trace.

Use the baseline with a CPU worker and no network GPU: owner 3 is CPU-worker
execution; host-local CPU is -1. For one destination, nonempty slices count
potential RPCs; score each peer independently. The retained trace covers prompt
and possible speculative first-verification rows, not accepted decode history.
Unobserved assignments require separate frozen-partition validation. It does not
predict decode, latency, traffic, or validate the complete frozen placement.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import compare_first_request_trace as trace


def assignments(path, layers, experts):
    result=set()
    for line in Path(path).read_text().splitlines():
        fields=line.partition('#')[0].split()
        if not fields:
            continue
        if len(fields)!=2 or not all(s.isascii() and s.isdecimal() for s in fields):
            raise ValueError('assignment must contain LAYER EXPERT per line')
        pair=tuple(map(int,fields))
        if pair[0]>=layers or pair[1]>=experts or pair in result:
            raise ValueError('duplicate or out-of-range assignment')
        result.add(pair)
    if not result:
        raise ValueError('empty assignment')
    return result


def score(records, selected):
    ids={k[:3]:v for k,v in records.items() if k[3]==4}
    hist=Counter();by_layer=Counter();observed=set();eligible=0
    for k,owners in sorted(records.items()):
        if k[3]!=9:
            continue
        w,pos,layer,_,begin,rows,topk=k
        routed=ids.get((w,pos,layer))
        if routed is None or begin<0 or begin+rows>routed.shape[0] or topk!=routed.shape[1] or owners.shape!=(rows,topk):
            raise ValueError('ownership slice does not match router IDs')
        hits=0
        for t in range(rows):
            for j in range(topk):
                pair=(layer,int(routed[begin+t,j]));owner=int(owners[t,j])
                eligible+=owner==3
                if pair in selected:
                    if owner!=3:
                        raise ValueError('selected pair is not CPU-owned in retained observation')
                    observed.add(pair);hits+=1
        hist[hits]+=1;by_layer[layer]+=hits
    if not hist:
        raise ValueError('no ownership slices')
    calls=sum(n for hits,n in hist.items() if hits)
    entries=sum(hits*n for hits,n in hist.items())
    return {'assigned':len(selected),'assigned_observed':len(observed),
            'cpu_entries_observed':eligible,'selected_entries':entries,'nonempty_calls':calls,
            'entries_per_call':entries/calls if calls else None,
            'entries_per_cpu_entry':entries/eligible if eligible else None,
            'slice_histogram':dict(sorted(hist.items())),
            'entries_by_layer':dict(sorted((k,v) for k,v in by_layer.items() if v)),
            'scope':'retained prompt and first verification only; not full decode, a timing prediction or a placement admission'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path);p.add_argument('assignments',nargs='+',type=Path)
    a=p.parse_args();records=trace.read(a.trace)
    binding=[v.reshape(-1) for k,v in records.items() if k[3]==10][0]
    layers,experts=int(binding[4]),int(binding[6])
    reports={str(path):score(records,assignments(path,layers,experts)) for path in a.assignments}
    print(json.dumps(reports,indent=2))

if __name__=='__main__':
    main()
