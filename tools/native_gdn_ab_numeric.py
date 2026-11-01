"""Native GDN a/b component: original primitive bounds, no gate/beta epsilon."""
from pathlib import Path
import struct,json,argparse
import native_hc_numeric as N
import native_hc_mix_numeric as M
import native_gdn_log1p_numeric as L
FIELDS=('alpha','beta-logit','v','softplus-exp','softplus','beta-exp','beta-denominator','gate','beta')
def words(p,n):
 raw=Path(p).read_bytes();N.need(len(raw)==n*4,'GDN ab field shape');return list(struct.unpack('<'+'I'*n,raw))
def softplus_controls(v,e,sp,backend):
 N.normal(v);N.normal(sp)
 if N.value(v)>20:
  N.need(e==0 and sp==v,'GDN ab strict >20 original bypass');return
 L.domain(e)
 if M.in_activation_domain(v):N.need(M.inside(e,M.exp_bounds(v,backend)),'GDN ab actual softplus exp bound')
 else:N.need(N.value(v)<0 and M.sigmoid_exp_allowed(v^0x80000000,e,backend),'GDN ab actual exp underflow bound')
 reference=L.reference(e);distance=abs(sp-reference) if sp>>31==reference>>31 else 0xffffffff
 N.need(distance<=1 and (e!=0 or sp==0),'GDN ab unchanged log1pf one-step gate')
def check(fixture,result,backend):
 N.need(backend in ('native','vulkan'),'GDN ab backend');raw=Path(fixture).read_bytes();N.need(len(raw)==502176 and raw[:32]==struct.pack('<8I',0x31424147,1,2560,48,1,30,0,0),'GDN ab geometry/type fixture')
 root=Path(result);N.need((root/'fixture.bin').read_bytes()==raw,'GDN ab actual fixture binding');v={k:words(root/(k+'.f32le'),48) for k in FIELDS}
 dt=struct.unpack_from('<48I',raw,32+2560*4+48*2560*4);ssa=struct.unpack_from('<48I',raw,len(raw)-192)
 for i in range(48):
  N.need(v['v'][i]==N.add(v['alpha'][i],dt[i]),'GDN ab dt exact local association');softplus_controls(v['v'][i],v['softplus-exp'][i],v['softplus'][i],backend);M.sigmoid_controls(v['beta-logit'][i],v['beta-exp'][i],v['beta'][i],backend)
  if M.sigmoid_class(v['beta-logit'][i])=='normal':N.need(v['beta-denominator'][i]==N.add(0x3f800000,v['beta-exp'][i]),'GDN ab sigmoid exact denominator')
  else:N.need(v['beta-denominator'][i]==(0x3f800000 if N.value(v['beta-logit'][i])>0 else 0x7f800000),'GDN ab original sigmoid tail denominator')
  N.need(v['gate'][i]==N.mul(v['softplus'][i],ssa[i]),'GDN ab gate exact multiplication');N.normal(v['gate'][i]);N.normal(v['beta'][i])
 return {'contract':'native-gdn-ab-T1-v1','backend':backend,'native_outer_order':True,'primitive_contracts':True,'exact_dot_reduction_control_required':True,'qualified':False,'model_integration':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fixture',required=True);p.add_argument('--result',required=True);p.add_argument('--backend',required=True,choices=['native','vulkan']);a=p.parse_args();print(json.dumps(check(a.fixture,a.result,a.backend),indent=2))
