#!/usr/bin/env python3
"""New head HC per-backend contract; the frozen exact gate is unchanged.

This checks numerical captures only. Source, shader, device, canary and lifecycle
bindings are separate required physical gates. Native does not export gates.
"""
import argparse,hashlib,json,struct,tempfile
from pathlib import Path
import native_hc_numeric as n
import native_hc_mix_numeric as m
import native_head_hc_numeric as exact

VERSION='native-head-hc-own-primitives-rne-overflow-v2'

def rne_div(top,den):
 if den==0x7f800000 and m.head_silu_overflow_candidate(top):return 0x80000000
 m.quotient_guard(top,den)
 if not top&0x7fffffff:return top
 q=n.rational(top)/n.rational(den)
 return (0x80000000 if q<0 else 0)|n.nearest(abs(q))

def evaluate(case,reference,candidate,retained=False):
 raw,outputs=exact.load(case,reference,candidate)
 x=struct.unpack_from('<10240I',raw,8);w=struct.unpack_from('<10240I',raw,8+exact.D*4)
 totals,weighted=n.source_inputs((1,x,w,[0]*(4*exact.D)));controls={};rne_bad=[]
 with tempfile.TemporaryDirectory(prefix='head-contract-') as d:
  root=Path(d)
  for backend in ('native','vulkan'):
   values=outputs[backend]
   n.need(all(n.rs_allowed(a,b,backend) for a,b in zip(values['rs'],totals)),backend+' rsqrt bound')
   expected=tuple(n.mul(v,values['rs'][i//2560]) for i,v in enumerate(weighted))
   n.need(expected==values['normalized'],backend+' normalized operation order')
   fixture=root/(backend+'.bin');fixture.write_bytes(b'HCMIX01\0'+struct.pack('<10240I',*values['normalized'])+raw[8+2*exact.D*4:])
   source=Path(reference) if backend=='native' else Path(candidate)/'candidate';dest=root/backend;dest.mkdir()
   for name in (('low','mixed') if backend=='native' else m.NAMES):
    path=source/(('' if backend=='native' else 'first-')+name+'.f32le')
    n.need(not path.is_symlink(),'symlink output');(dest/(name+'.f32le')).write_bytes(path.read_bytes())
   controls[backend]=m.check(fixture,dest,backend,head_silu_overflow=True)
   if backend=='vulkan':
    pre=exact.words(source/'first-low-pre.f32le',320);exp=exact.words(source/'first-low-exp.f32le',320)
    for i,(top,e,actual) in enumerate(zip(pre,exp,values['low'])):
     if actual!=rne_div(top,0x7f800000 if e==0x7f800000 and m.head_silu_overflow_candidate(top) else n.add(0x3f800000,e)):rne_bad.append(i)
 differences={name:sum(a!=b for a,b in zip(outputs['native'][name],outputs['vulkan'][name])) for name in ('rs','normalized','low','mixed')}
 # A historical capture is never accepted as the new shader qualification.
 return {'contract':VERSION,'mode':'retained_observation' if retained else 'head_rne_candidate',
         'fixture_sha256':hashlib.sha256(raw).hexdigest(),'per_backend_source_and_primitive_controls':True,
         'native_sum_scope':'not_exported_reference_sum_and_rsqrt_bound',
         'native_dot_scope':'not_exported_reference_dots_and_activation_bounds',
         'native_gate_scope':'not_exported_existing_propagation_bound',
         'vulkan_captured_dots_exact':True,'vulkan_own_gate_mixed_exact':True,
         'vulkan_rne_silu':{'exact':not rne_bad,'different_words':len(rne_bad),'first':rne_bad[0] if rne_bad else None},
         'exact_diagnostic':{'different_words':differences,'all_equal':not any(differences.values())},
         'numerical_acceptance':not retained and not rne_bad,'physical_gates_required':True,
         'original_exact_gate':'separate_not_reclassified','qualified':False,'full_model_parity':False}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=sorted(exact.PINS));p.add_argument('native');p.add_argument('vulkan');p.add_argument('output');p.add_argument('--retained-observation',action='store_true');a=p.parse_args()
 try:
  output=Path(a.output);n.need(not output.exists() and not output.is_symlink(),'fresh output required')
  result=evaluate(a.case,a.native,a.vulkan,a.retained_observation)
  with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  raise SystemExit(0 if result['numerical_acceptance'] or a.retained_observation else 1)
 except (OSError,ValueError,KeyError,struct.error) as e:print('STOP: '+str(e),file=__import__('sys').stderr);raise SystemExit(2)
