"""Compare the existing STRATA_DUMP_FIRST_LOGITS raw F32 rows. Fixed screening limits, no tuning CLI."""
import argparse
import array
import json
import math
import pathlib
import sys

MAX_BYTES = 4 << 20
LIMITS = dict(logit_abs_floor=1e-2, logit_abs_scale=1e-4, logit_relative_l2=1e-4,
              probability_abs=1e-3, kl_a_b=1e-5)


def read(path):
    path=pathlib.Path(path);size=path.stat().st_size
    if not size or size%4 or size>MAX_BYTES:raise ValueError('invalid/bounded raw F32 row size')
    row=array.array('f');row.frombytes(path.read_bytes())
    if sys.byteorder!='little':row.byteswap()
    if any(not math.isfinite(value) for value in row):raise ValueError('nonfinite logit')
    return row


def compare(a,b):
    if not a or len(a)!=len(b) or any(not math.isfinite(v) for row in (a,b) for v in row):
        raise ValueError('logit rows must have matching nonempty finite vocabulary')
    delta=[float(y)-float(x) for x,y in zip(a,b)]
    maximum=max(map(abs,delta));scale=max(map(abs,a))
    l2=math.sqrt(math.fsum(d*d for d in delta))/max(1.0,math.sqrt(math.fsum(float(v)*v for v in a)))
    am=max(a);bm=max(b)
    az=math.log(math.fsum(math.exp(v-am) for v in a));bz=math.log(math.fsum(math.exp(v-bm) for v in b))
    logpa=[(v-am)-az for v in a];logpb=[(v-bm)-bz for v in b]
    pa=[math.exp(v) for v in logpa];pb=[math.exp(v) for v in logpb]
    probability=max(abs(x-y) for x,y in zip(pa,pb))
    kl=max(0.0,math.fsum(p*(x-y) for p,x,y in zip(pa,logpa,logpb)))
    metrics=dict(vocabulary=len(a),max_abs_logit=maximum,
        max_rel_logit=max(abs(d)/max(1e-12,abs(float(x))) for x,d in zip(a,delta)),
        logit_relative_l2=l2,max_abs_probability=probability,kl_a_b=kl,
        top1_equal=max(range(len(a)),key=lambda i:a[i])==max(range(len(b)),key=lambda i:b[i]))
    metrics['numeric_pass']=(maximum<=LIMITS['logit_abs_floor']+LIMITS['logit_abs_scale']*scale and
        l2<=LIMITS['logit_relative_l2'] and probability<=LIMITS['probability_abs'] and kl<=LIMITS['kl_a_b'])
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',nargs=3,required=True)
    parser.add_argument('--candidate',nargs=3,required=True)
    args=parser.parse_args()
    results=[compare(read(a),read(b)) for a,b in zip(args.baseline,args.candidate)]
    print(json.dumps(dict(limits=LIMITS,requests=results),indent=2,allow_nan=False))
    return 0 if all(row['numeric_pass'] for row in results) else 1


if __name__=='__main__':raise SystemExit(main())
