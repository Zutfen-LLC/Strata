#!/usr/bin/env python3
"""Compare raw Q8_1 blocks; file roles and source are operator assertions."""
import argparse
import hashlib
import json
from pathlib import Path
import re

NAMES = ('control', 'signed_subnormal', 'normal_boundary', 'rounding')
def read(path):
    data = path.read_bytes()
    if len(data) != 144:
        raise ValueError(f'{path}: expected exactly four 36-byte Q8_1 blocks')
    return data, [{'scale_bits_le': data[i:i+2].hex(), 'sum_bits_le': data[i+2:i+4].hex(),
                   'quantized_bytes': data[i+4:i+36].hex()} for i in range(0, 144, 36)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', type=Path, required=True)
    p.add_argument('--vulkan', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--source-head', required=True, help='operator-declared 40-digit source commit')
    p.add_argument('--producer', choices=('expert','gdn-projection'), required=True)
    p.add_argument('--gdn-domain', type=Path)
    a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',a.source_head):p.error('source-head must be 40 lowercase hex digits')
    input_bytes=a.input.read_bytes()
    if len(input_bytes)!=512:raise ValueError('expected 128 float32 inputs')
    cb,c=read(a.cuda);vb,v=read(a.vulkan)
    domain=None
    if a.producer=='gdn-projection':
        if a.gdn_domain is None:p.error('gdn-projection requires the Vulkan domain report')
        domain=json.loads(a.gdn_domain.read_text())
        valid={'unsupported_input','unsupported_packed_scale','block_checks_passed'}
        if domain.get('model_execution') is not False or len(domain.get('blocks',[]))!=4 or any(b.get('state') not in valid for b in domain['blocks']):
            raise ValueError('invalid GDN domain report')
    elif a.gdn_domain is not None:p.error('domain report applies to gdn-projection only')
    rows=[]
    for i,(n,x,y) in enumerate(zip(NAMES,c,v)):
        row={'block':n,'cuda':x,'vulkan':y,'isolated_bytes_equal':x==y}
        if domain is not None:row['gdn_block_domain']=domain['blocks'][i]
        rows.append(row)
    equal=all(r['isolated_bytes_equal'] for r in rows)
    supported=domain is None or all(b['state']=='block_checks_passed' for b in domain['blocks'])
    digest=lambda b:hashlib.sha256(b).hexdigest()
    print(json.dumps({'declared_source_head':a.source_head, 'producer':a.producer,
                      'cuda_entrypoint':'quantize_q8_1_rows' if a.producer=='expert' else 'native_quantize_q8_1',
                      'vulkan_variant':'quant_rn_observation' if a.producer=='expert' else 'quant_rcp_observation',
                      'role_and_source_are_operator_assertions':True, 'inputs_bound_to_execution':False,
                      'input_sha256':digest(input_bytes),'cuda_output_sha256':digest(cb),'vulkan_output_sha256':digest(vb),
                      'qualified':False,'model_execution':False,'fp32_denorm':'unspecified',
                      'isolated_all_equal':equal,'gdn_block_domain_supported':supported if domain else None,'blocks':rows},indent=2))
    return 0 if equal and supported else 1
if __name__=='__main__': raise SystemExit(main())
