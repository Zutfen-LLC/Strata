"""Real product parser/shared-precondition tests, before files/backends/admission."""
import json,os,subprocess,sys
binary=sys.argv[1]
base=['--model','/absent/model.gguf','--manifest','/absent/manifest','--stdio','--max-requests','2','--context','256','--threads','4','--expert-workers','12','--host-cap-mib','32768','--vulkan','0,1','--device-cap-mib','7680,7680','--dense-topology','primary','--expert-native-q8-observation','/absent/expert','--native-moe-combine','--native-router','/absent/router','--native-shared','/absent/shared','--prompt-head-elision','--expert-native-profile','/absent/profile','--expert-native-primary-slots','864','--expert-native-helper-slots','1346','--expert-native-plan-prefix','/absent/plan','--validate-options']
hc=['--native-hc-norm-inject-layer0','/absent/hc']
count=0
def run(extra,ok,error=None,env=None):
 global count
 p=subprocess.run([binary,*base,*extra],capture_output=True,text=True,timeout=5,env=env);count+=1
 assert 'Found 2 Vulkan devices' not in p.stderr and 'device_memory' not in p.stderr,'preflight initialized backend'
 if ok:
  assert p.returncode==0,p.stderr
  r=json.loads(p.stdout);assert r['validated_options'] is True and all(r[k] is False for k in ['backend_initialized','model_metadata_checked','memory_admission_checked'])
  return r
 assert p.returncode!=0 and (error is None or error in p.stderr),p.stderr
# Both nonresident controls remain valid. The historically attempted on config
# is rejected by the actual same predicate used by session creation.
assert run(["--head-shape-admission"],True)["head_shape_admission"]
assert not run([],True)["head_shape_admission"]
run(['--dump-executed-logits','/absent/heads'],True)
run(hc,False,'native HC layer0 requires')
a=run(['--gdn-resident-weights'],True);b=run([*hc,'--gdn-resident-weights'],True)
assert a['gdn_resident_weights'] and b['gdn_resident_weights'] and not a['dense_resident_weights'] and not b['dense_resident_weights']
assert not a['native_hc_norm_inject_layer0'] and b['native_hc_norm_inject_layer0']
run([*hc,'--dense-resident-weights'],True) # valid syntax, deliberately no memory-fit claim
for flags in [['--allow-cpu-fallback'],['--gdn-production','--gdn-resident-state'],['--prompt-chunk-size','2']]:run([*hc,'--gdn-resident-weights',*flags],False)
run(['--native-hc-norm-inject-layer0','','--gdn-resident-weights'],False,'receipt prefix')
run([*hc,'--gdn-resident-weights','--qsa-production'],False,'QSA production requires')
run([*hc,'--gdn-resident-weights'],False,'recorded-unset HC mode',dict(os.environ,STRATA_HC_Q8='1'))
read=['--native-hc-read-layer0','/absent/hc-read']
r=run([*read,'--gdn-resident-weights'],True);assert r['native_hc_norm_inject_layer0'] and r['native_hc_mix_layer0']
assert not a['native_hc_mix_layer0'] and not b['native_hc_mix_layer0']
run(read,False,'native HC layer0 requires')
for flags in [['--allow-cpu-fallback'],['--gdn-production','--gdn-resident-state'],['--prompt-chunk-size','2']]:run([*read,'--gdn-resident-weights',*flags],False)
run(['--native-hc-read-layer0','','--gdn-resident-weights'],False,'receipt prefix')
ffn=['--native-hc-ffn-read-layer0','/absent/hc-ffn']
assert not a['layer0_resident_weights'] and not a['native_hc_ffn_read_layer0']
f=run([*ffn,'--layer0-resident-weights'],True)
assert f['native_hc_ffn_read_layer0'] and f['native_hc_mix_layer0'] and f['native_hc_norm_inject_layer0'] and f['layer0_resident_weights']
assert not f['dense_resident_weights'] and not f['gdn_resident_weights']
run([*ffn,'--gdn-resident-weights'],False,'explicit layer0/dense resident weights')
run(ffn,False,'explicit layer0/dense resident weights')
run([*ffn,'--dense-resident-weights'],True)
run(['--native-hc-ffn-read-layer0','','--layer0-resident-weights'],False,'receipt prefix')
for flags in [['--allow-cpu-fallback'],['--gdn-production','--gdn-resident-state'],['--prompt-chunk-size','2']]:run([*ffn,'--layer0-resident-weights',*flags],False)
run([*ffn,'--layer0-resident-weights'],False,'legacy observer',dict(os.environ,STRATA_LAYER0_OBSERVER='/absent/legacy.bin'))
run([*ffn,'--layer0-resident-weights'],False,'recorded-unset HC mode',dict(os.environ,STRATA_HC_Q8='1'))
print(str(count)+' actual product mode/parser preflights passed; no model/backend/memory admission performed')

write=['--native-hc-ffn-write-layer0','/absent/hc-write']
w=run([*write,'--layer0-resident-weights'],True)
assert w['native_hc_ffn_write_layer0'] and w['native_hc_ffn_read_layer0'] and w['native_hc_mix_layer0'] and not a['native_hc_ffn_write_layer0']
run(write,False,'explicit layer0/dense resident weights')
for flags in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--gdn-production','--gdn-resident-state']]:run([*write,'--layer0-resident-weights',*flags],False)
print('final-write actual CLI preconditions/defaults passed')

carry=['--native-hc-pending-layer1-2','/absent/hc-boundary']
b=run([*carry,'--gdn-resident-weights'],True)
assert b['native_hc_pending_layer1_2'] and not a['native_hc_pending_layer1_2'] and not b['native_hc_ffn_write_layer0'] and not b['native_hc_norm_inject_layer0']
run(carry,False,'resident attention')
for flags in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--gdn-production','--gdn-resident-state']]:run([*carry,'--gdn-resident-weights',*flags],False)
run([*carry,'--gdn-resident-weights'],False,'legacy observer',dict(os.environ,STRATA_LAYER0_OBSERVER='/absent/observer.bin'))
print('ordinary layer1-2 pending boundary actual CLI preconditions/defaults passed')

qsa=['--native-hc-pending-layer2-3','/absent/qsa-hc']
q=run([*qsa,'--gdn-resident-weights'],True)
assert q['native_hc_pending_layer2_3'] and not a['native_hc_pending_layer2_3'] and not q['native_hc_pending_layer1_2']
run(qsa,False,'resident GDN')
for flags in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--qsa-production','--qsa-reuse-attention'],['--gdn-production','--gdn-resident-state']]:run([*qsa,'--gdn-resident-weights',*flags],False)
run([*qsa,'--gdn-resident-weights'],False,'legacy observer',dict(os.environ,STRATA_LAYER0_OBSERVER='/absent/observer.bin'))
print('ordinary QSA layer2-3 pending actual CLI defaults/preconditions passed')

cycle=['--native-hc-serial-cycle','/absent/cycle','--hc-cycle-resident-weights','--gdn-resident-weights']
c=run(cycle,True);assert c['native_hc_serial_cycle'] and c['hc_cycle_resident_weights'] and c['native_hc_pending_layer1_2'] and c['native_hc_pending_layer2_3'] and not c['dense_resident_weights'];assert not a['native_hc_serial_cycle'] and not a['hc_cycle_resident_weights']
run(['--native-hc-serial-cycle','/absent/cycle','--gdn-resident-weights'],False,'matched FFN residency')
for f in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--qsa-production','--qsa-reuse-attention'],['--gdn-production','--gdn-resident-state']]:run([*cycle,*f],False)
run([*cycle],False,'legacy observer',dict(os.environ,STRATA_LAYER0_OBSERVER='/absent/observer.bin'))
run(['--native-hc-serial-cycle','','--hc-cycle-resident-weights','--gdn-resident-weights'],False,'receipt prefix')
res=run(['--hc-cycle-resident-weights'],True);assert res['hc_cycle_resident_weights'] and not res['native_hc_serial_cycle'] and not res['native_hc_pending_layer1_2']
print('serial HC cycle actual CLI/defaults/residency/refusal contracts passed')
# Prefix order must preserve explicitly supplied entry-owner paths.
run(['--native-hc-pending-layer1-2','/absent/entry','--native-hc-pending-layer2-3','/absent/qsa',*cycle],True)
