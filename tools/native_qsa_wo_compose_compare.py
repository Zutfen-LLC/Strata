"""Exact packed-image and WO-output comparison; lifecycle gates are external."""
from pathlib import Path
import argparse, hashlib, json, subprocess

def read(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file(): raise ValueError('regular file required')
    return p.read_bytes()
def digest(data): return hashlib.sha256(data).hexdigest()

def compare(weights,weight_sha,fixture,fixture_sha,native,vulkan,control,control_sha):
    for path,expected in ((weights,weight_sha),(fixture,fixture_sha),(control,control_sha)):
        if len(expected)!=64 or digest(read(path))!=expected: raise ValueError('bound input or control differs')
    outputs=[]
    for kind,root in (('native',Path(native)),('vulkan',Path(vulkan))):
        report=json.loads(read(root/'report.json'))
        required=['producer_buffer_consumed','stale_changes_output','invalid_noncommit_retry','immutable_canary_repeat']
        required+=['actual_native_quantizer_and_mmvq','same_stream'] if kind=='native' else ['actual_vulkan_quantizer_and_wo']
        if report.get('qualified') is not False or any(report.get(k) is not True for k in required): raise ValueError(kind+' report differs')
        if read(root/'fixture.bin')!=read(fixture): raise ValueError(kind+' input fixture differs')
        packed=read(root/'qx.bin');output=read(root/'out.f32le')
        if len(packed)!=6912 or len(output)!=10240 or read(root/'qx-repeat.bin')!=packed or read(root/'out-repeat.f32le')!=output:
            raise ValueError(kind+' output size or repeat differs')
        check=subprocess.run([str(control),'--validate-packed',str(weights),str(fixture),str(root/'qx.bin')],capture_output=True,timeout=60)
        if check.returncode or check.stdout or check.stderr: raise ValueError(kind+' compiled domain validator refused')
        outputs.append((packed,output))
    packed_diff=sum(a!=b for a,b in zip(outputs[0][0],outputs[1][0]));output_diff=sum(a!=b for a,b in zip(outputs[0][1],outputs[1][1]))
    return {'packed_bytes':6912,'packed_different_bytes':packed_diff,'native_packed_sha256':digest(outputs[0][0]),'vulkan_packed_sha256':digest(outputs[1][0]),
            'output_bytes':10240,'output_different_bytes':output_diff,'native_output_sha256':digest(outputs[0][1]),'vulkan_output_sha256':digest(outputs[1][1]),
            'exact':packed_diff==0 and output_diff==0,'qualified':False,'model_integration':False}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('weights','weight-sha','fixture','fixture-sha','native','vulkan','control','control-sha'): p.add_argument('--'+name,required=True)
    a=p.parse_args();result=compare(a.weights,a.weight_sha,a.fixture,a.fixture_sha,a.native,a.vulkan,a.control,a.control_sha)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['exact'] else 1)
