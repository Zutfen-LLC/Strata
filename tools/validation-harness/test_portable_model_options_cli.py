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

ordinary=['--native-hc-serial','/absent/ordinary','--hc-serial-resident-weights','--gdn-resident-weights','--native-hc-ffn-write-layer0','/absent/zero','--layer0-resident-weights']
r=run(ordinary,True);assert r['native_hc_serial'] and r['hc_serial_resident_weights'] and not r['native_hc_serial_cycle']
assert not run([],True)['native_hc_serial']
run(['--native-hc-serial','/absent/no-residency'],False,'ordinary native HC requires')
for flags in [['--native-hc-serial-cycle','/absent/cycle'],['--dense-topology','split'],['--allow-cpu-fallback'],['--prompt-chunk-size','2']]:run([*ordinary,*flags],False)
run(ordinary,False,'fused final head',dict(os.environ,STRATA_FUSE_HEAD_GR='1'))
run(ordinary,True,env=dict(os.environ,STRATA_FUSE_HEAD_GR='0'))
assert not run([],True).get('native_head_hc_t1',False)
run(['--native-head-hc-t1','/absent/head'],False,'native head HC requires')
run([*ordinary,'--native-head-hc-t1','/absent/head'],False,'native head HC requires')
head=[*ordinary,'--dense-resident-weights','--native-head-hc-t1','/absent/head']
assert run(head,True)['native_head_hc_t1']
for flags in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--dense-topology','split']]:run([*head,*flags],False)
run([*ordinary,'--dense-resident-weights','--native-head-hc-t1',''],False,'receipt')
run(head,False,'fused final head',dict(os.environ,STRATA_FUSE_HEAD_GR='1'))
print('native head HC actual CLI default, materialized T1 and residency refusals pass')

# The GDN family is explicit and default-disabled; complete residency is validated before model load.
assert not a.get('native_gdn_t1',False)
gdn=['--gdn-resident-weights','--gdn-resident-state','--dense-topology','primary','--native-gdn-t1','/tmp/native-gdn-options-only']
r=run(gdn,True);assert r['native_gdn_t1'] and r['gdn_resident_state']
for missing in ('--gdn-resident-weights','--gdn-resident-state'):
 bad=[x for x in gdn if x!=missing];run(bad,False)
run(gdn+['--allow-cpu-fallback'],False)

prep_weights=[*gdn,'--gdn-native-prep-bf16']
prep=[*prep_weights,'--native-gdn-prep-t1','/absent/prep']
r=run(gdn,True);assert not r['native_gdn_preparation_t1'] and not r['native_gdn_preparation_weights']
r=run(prep_weights,True);assert not r['native_gdn_preparation_t1'] and r['native_gdn_preparation_weights']
r=run(prep,True);assert r['native_gdn_preparation_t1'] and r['native_gdn_preparation_weights']
run(['--gdn-native-prep-bf16'],False,'native recurrence')
run([*gdn,'--native-gdn-prep-t1','/absent/no-bank'],False,'explicit BF16 residency')
run([*prep_weights,'--native-gdn-prep-t1',''],False,'receipt')
for flags in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--gdn-production']]:run([*prep,*flags],False)
print('native GDN preparation actual CLI defaults/matched BF16/preconditions passed')

# Actual QSA residency/dispatch options, before model access/backend initialization.
nr=['--qsa-norm-rope-resident-weights','/absent/qsa-native']
assert not run([],True)['native_qsa_norm_rope_t1']
assert run(nr,True)['qsa_norm_rope_resident_weights']
assert not run(nr,True)['native_qsa_norm_rope_t1']
assert run([*nr,'--native-qsa-norm-rope-t1'],True)['native_qsa_norm_rope_t1']
run(['--native-qsa-norm-rope-t1'],False,'matched resident roles')
for flags in [['--allow-cpu-fallback'],['--prompt-chunk-size','2'],['--qsa-production','--qsa-reuse-attention']]:run([*nr,*flags],False)
for key in ['STRATA_ROPE_TABLE','STRATA_NO_NORM_ROPE']:run(nr,False,'analytic fused text method',dict(os.environ,**{key:'1'}))
print('native QSA actual CLI residency/dispatch/default/unsupported-mode preconditions passed')

indexer=['--qsa-indexer-resident-weights','/absent/indexer']
i=run(indexer,True);assert i['qsa_indexer_resident_weights'] and not i['native_qsa_indexer_t1']
i=run([*indexer,'--native-qsa-indexer-t1'],True);assert i['native_qsa_indexer_t1']
assert not run([],True)['native_qsa_indexer_t1'] and not run([],True)['qsa_indexer_resident_weights']
run(['--native-qsa-indexer-t1'],False,'matched resident')
for bad in [['--allow-cpu-fallback'],['--qsa-production','--qsa-reuse-attention'],['--prompt-chunk-size','2']]:run([*indexer,'--native-qsa-indexer-t1',*bad],False)
run(['--qsa-indexer-resident-weights',''],False,'receipt')
run([*indexer,'--native-qsa-indexer-t1'],False,'analytic fused text',dict(os.environ,STRATA_ROPE_TABLE='1'))
print('native indexer actual CLI/default/residency/mode preconditions pass')

score=[*indexer,'--native-qsa-indexer-t1','--qsa-score-select-banks','/absent/scores']
r=run(score,True);assert r['qsa_score_select_banks'] and not r['native_qsa_score_select_t1']
r=run([*score,'--native-qsa-score-select-t1'],True);assert r['native_qsa_score_select_t1']
assert not run([],True)['qsa_score_select_banks'] and not run([],True)['native_qsa_score_select_t1']
run(['--native-qsa-score-select-t1'],False,'matched candidate banks')
run(['--qsa-score-select-banks','/absent/scores'],False,'actual native indexer')
for bad in [['--allow-cpu-fallback'],['--qsa-production','--qsa-reuse-attention'],['--prompt-chunk-size','2']]:run([*score,'--native-qsa-score-select-t1',*bad],False)
run([*indexer,'--native-qsa-indexer-t1','--qsa-score-select-banks',''],False,'receipt')
print('native score/selection actual CLI/default/banks/native producer/mode preconditions pass')

run(score,False,'alternative native dispatch',dict(os.environ,STRATA_TOPK_OLD='1'))

attention=[*score,'--native-qsa-score-select-t1','--qsa-norm-rope-resident-weights','/absent/norm','--native-qsa-norm-rope-t1','--qsa-attention-banks','/absent/attention']
a=run(attention,True);assert a['qsa_attention_banks'] and not a['native_qsa_attention_t1']
assert run([*attention,'--native-qsa-attention-t1'],True)['native_qsa_attention_t1']
assert not run([],True)['qsa_attention_banks'] and not run([],True)['native_qsa_attention_t1']
run(['--native-qsa-attention-t1'],False,'matched candidate banks')
run(['--qsa-attention-banks','/absent/attention'],False,'native query/IDs')
for bad in [['--allow-cpu-fallback'],['--qsa-production','--qsa-reuse-attention'],['--prompt-chunk-size','2']]:run([*attention,'--native-qsa-attention-t1',*bad],False)
run(attention,False,'alternative native attention',dict(os.environ,STRATA_ATTN_MERGE_V2='1'))
print('native FP16 attention/gate actual CLI/default/matched-bank/native producer/mode preconditions pass')

assert not run([],True)['native_qsa_wo_t1']
run(['--native-qsa-wo-t1'],False,'native QSA WO requires')
run([*attention,'--native-qsa-attention-t1','--native-qsa-wo-t1'],False,'native QSA WO requires')
wo=[*ordinary,*attention,'--native-qsa-attention-t1','--dense-resident-weights','--native-qsa-wo-t1']
assert run(wo,True)['native_qsa_wo_t1']
for bad in [['--allow-cpu-fallback'],['--qsa-production','--qsa-reuse-attention'],['--prompt-chunk-size','2']]:run([*wo,*bad],False)
print('native QSA WO CLI default and ordinary eager native-attention/residency preconditions pass')

assert not run([],True).get('native_qsa_qkv_t1',False)
run(['--native-qsa-qkv-t1'],False,'native QSA QKV requires')
assert run([*wo,'--native-qsa-qkv-t1'],True)['native_qsa_qkv_t1']
for bad in [['--allow-cpu-fallback'],['--qsa-production','--qsa-reuse-attention'],['--prompt-chunk-size','2']]:
    run([*wo,'--native-qsa-qkv-t1',*bad],False)
print('native QSA QKV CLI default and matched native WO preconditions pass')

assert not run([],True).get('native_qsa_indexer_projection_t1',False)
run(['--native-qsa-indexer-projection-t1'],False,'native QSA indexer projections require')
idx_projection=[*ordinary,*indexer,'--native-qsa-indexer-t1',*nr,'--native-qsa-norm-rope-t1','--dense-resident-weights','--native-qsa-indexer-projection-t1']
assert run(idx_projection,True)['native_qsa_indexer_projection_t1']
for bad in [['--allow-cpu-fallback'],['--qsa-production','--qsa-reuse-attention'],['--prompt-chunk-size','2']]:run([*idx_projection,*bad],False)
print('native BF16 indexer projection CLI default and strict producer/consumer preconditions pass')
