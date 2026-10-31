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
print(str(count)+' actual product mode/parser preflights passed; no model/backend/memory admission performed')
