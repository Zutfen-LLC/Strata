"""Exact WO comparison with independently specified binary16 literal outputs."""
from pathlib import Path
import argparse,hashlib,json,struct
import native_q8_wo_compare as base

GOLDEN_SHA256='698d1878ec7b4b36e26c4ed2e43cb9ff34c1d4fb4a511032f5996ddcfdc2cb80'


def expected():
    values=[0.,1.,32510.,1.5,0.,0.,16.,20.]
    values.extend(m/2**24 for m in range(1,1024))
    values.extend(-m/2**24 for m in range(1,1024))
    values.extend([1023/2**24,1024/2**24,1/2**25,-1023/2**23,-128/2**24,0.,
                   1025/2**24,-1025/2**24,1023/2**23,-1/2**25])
    values.extend([0.]*(2560-len(values)))
    return struct.pack('<2560f',*values)


def compare(weights,weight_sha,packed,packed_sha,native,vulkan,control,control_sha):
    result=base.compare(weights,weight_sha,packed,packed_sha,2560,native,vulkan,control,control_sha)
    golden=expected();result['literal_sha256']=hashlib.sha256(golden).hexdigest()
    if result['literal_sha256']!=GOLDEN_SHA256:raise ValueError('frozen subnormal literal image differs')
    for kind,root in (('native',Path(native)),('vulkan',Path(vulkan))):
        actual=base.read(root/'out.f32le');diff=sum(a!=b for a,b in zip(actual,golden))
        result[kind+'_literal_different_bytes']=diff
    result['all_literals_exact']=result['native_literal_different_bytes']==0 and result['vulkan_literal_different_bytes']==0
    result['exact']=result['exact'] and result['all_literals_exact']
    result['signed_subnormal_encodings']=2046
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('weights','weight-sha','packed','packed-sha','native','vulkan','control','control-sha'):p.add_argument('--'+name,required=True)
    args=p.parse_args();result=compare(args.weights,args.weight_sha,args.packed,args.packed_sha,args.native,args.vulkan,args.control,args.control_sha)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['exact'] else 1)
