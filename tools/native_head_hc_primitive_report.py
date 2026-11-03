"""Read the fixed CUDA primitive observation; make no parity or accuracy claim."""
from pathlib import Path
import json,struct,sys
INPUTS=['bf5544aa','33c013a7','3e800022','3fffffff']
NAMES=('up_input','up_exp','up_den','up_gate','low_input','low_exp','low_den','low_silu','rsqrt','captured_low_div')
def check(root):
 root=Path(root).resolve(strict=True)
 if {p.name for p in root.iterdir()}!={'report.json','words.u32le'} or any(p.is_symlink() or not p.is_file() for p in root.iterdir()):raise ValueError('probe output closure differs')
 report=json.loads((root/'report.json').read_text());data=(root/'words.u32le').read_bytes()
 if len(data)!=40:raise ValueError('probe word span differs')
 values=struct.unpack('<10I',data)
 if set(report)!={'probe','inputs','words','repeat','canaries','compiled_instruction_review_required','qualified'} or report['probe']!='native_head_hc_primitive_v1' or report['inputs']!=INPUTS or report['words']!=[f'{x:08x}' for x in values] or report['repeat'] is not True or report['canaries'] is not True or report['compiled_instruction_review_required'] is not True or report['qualified'] is not False or values[0]!=0xbf5544aa or values[4]!=0x33c013a7:raise ValueError('probe report/input differs')
 return {'observation':dict(zip(NAMES,(f'{x:08x}' for x in values))),
         'comparisons':{'exp_matches_retained_vulkan':values[1]==0x4013399a,'denominator_matches_retained_vulkan':values[2]==0x4053399a,'gate_matches_inferred_native':values[3]==0x3e9b2215,'gate_matches_retained_vulkan':values[3]==0x3e9b2216,'rsqrt_matches_retained_native':values[8]==0x3fffffdd,'fixed_division_matches_rne':values[9]==0x334013a8},
         'compiled_instruction_review_required':True,'qualified':False,'full_model_parity':False}
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('usage: native_head_hc_primitive_report.py PROBE_OUTPUT_DIR')
 try:print(json.dumps(check(sys.argv[1]),indent=2,sort_keys=True))
 except (OSError,ValueError,KeyError,struct.error) as e:print('STOP: '+str(e),file=sys.stderr);raise SystemExit(1)
