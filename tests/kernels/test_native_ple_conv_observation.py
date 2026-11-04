#!/usr/bin/env python3
"""Static shader and independent literal history/layout controls; no GPU execution."""
import argparse,re,struct,subprocess,tempfile
from pathlib import Path

def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def add(a,b): return f32(f32(a)+f32(b))
def mul(a,b): return f32(f32(a)*f32(b))
def preactivation(history,channel,normalized,weight):
    # Source layout is history[channel * 9 + row], weight[channel * 4 + tap].
    terms=[mul(history[channel*9+i],weight[channel*4+j]) for j,i in enumerate((0,3,6))]
    terms.append(mul(normalized[channel],weight[channel*4+3]))
    return add(add(add(terms[0],terms[1]),terms[2]),terms[3])

def test_controls():
    history=[1.,91.,92.,2.,94.,95.,3.,97.,98., 16777216.,81.,82.,1.,84.,85.,-16777216.,87.,88.]
    normalized=[4.,1.];weight=[1.,2.,4.,8., 1.,1.,1.,1.];before=history[:]
    assert preactivation(history,0,normalized,weight)==49.
    assert preactivation(history,1,normalized,weight)==1. # sequential FP32, not reassociation
    candidate=[history[c*9+i+1] for c in range(2) for i in range(8)]
    candidate=candidate[:8]+[normalized[0]]+candidate[8:]+[normalized[1]]
    assert candidate==[91.,92.,2.,94.,95.,3.,97.,98.,4.,81.,82.,1.,84.,85.,-16777216.,87.,88.,1.]
    assert history==before # candidate production is not a commit
    assert struct.unpack('<e',struct.pack('<H',0x0001))[0]==2.0**-24
    assert struct.unpack('<e',struct.pack('<H',0x8001))[0]==-2.0**-24

def check_module(text):
    # Four separately rounded products, three serial adds, denominator add,
    # division and two residual adds; no fused multiply/add may appear.
    assert not re.search(r'\b(?:OpFmaKHR|Fma|FClamp)\b',text)
    expected={'OpFMul':4,'OpFAdd':6,'OpFDiv':1}
    for op,count in expected.items():
        results=re.findall(r'(%\w+)\s*=\s*'+op+r'\b',text)
        assert len(results)==count,(op,len(results))
        for result in results:assert re.search(r'OpDecorate\s+'+re.escape(result)+r'\s+NoContraction\b',text),(op,result)
    assert len(re.findall(r'\bUnpackHalf2x16\b',text))==2
    assert len(re.findall(r'\bExp\b',text))==1

def main():
    p=argparse.ArgumentParser()
    for n in ('shader','glslc','validator','disassembler','scratch'):p.add_argument('--'+n,required=True)
    a=p.parse_args();test_controls()
    with tempfile.TemporaryDirectory(dir=a.scratch) as d:
        spv=Path(d)/'observation.spv';asm=Path(d)/'observation.spvasm'
        subprocess.run([a.glslc,'-O','--target-env=vulkan1.2','-fshader-stage=compute',a.shader,'-o',str(spv)],check=True)
        subprocess.run([a.validator,'--target-env','vulkan1.2',str(spv)],check=True)
        subprocess.run([a.disassembler,str(spv),'-o',str(asm)],check=True)
        text=asm.read_text();check_module(text)
        try:check_module(text.replace('OpFMul','OpFmaKHR',1))
        except AssertionError:pass
        else:raise AssertionError('FMA mutation accepted')
    print('literal history/tap controls and optimized SPIR-V ordering passed; no GPU execution')
if __name__=='__main__':main()
