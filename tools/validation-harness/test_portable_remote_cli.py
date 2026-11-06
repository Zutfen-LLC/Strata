"""Real CLI preflight checks; no model, network, or credential access."""
import json
import subprocess
import sys

binary = sys.argv[1]
tls_enabled = len(sys.argv) > 2 and sys.argv[2] == '1'
native_enabled = len(sys.argv) > 3 and sys.argv[3] == '1'
base = ['--model', '/absent/model', '--manifest', '/absent/manifest', '--tokens', '1']
remote = ['--remote-gpu-worker', 'vulkan-native-q8@127.0.0.1:4567', '--remote-gpu-plan', '/absent/plan']
count = 0

def run(extra, success, validate=True):
    global count
    args = [binary, *base, *extra]
    if validate:
        args.append('--validate-options')
    result = subprocess.run(args, capture_output=True, text=True, timeout=5)
    count += 1
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)
    if success:
        doc = json.loads(result.stdout)
        assert doc['validated_options'] and not doc['backend_initialized']
        assert not doc['model_metadata_checked'] and not doc['memory_admission_checked']
        return doc
    return result.stderr

assert 'remote_experts' not in run([], True)
doc = run(remote, True)
assert doc['remote_experts'] and not doc['remote_plan_checked']
run(remote + ['--remote-gpu-deadline-ms', '123', '--remote-gpu-setup-ms', '456'], True)
local = ['--expert-plan', '0:/absent/local-plan', '--vulkan', '0', '--device-cap-mib', '1024']
run(remote + local, False)
run(remote + local + ['--expert-native-q8-observation', '/absent/receipt'], True)
# Native serial PLE/GDN/FFN and three-way fixed ownership must be admitted together.
serial = ['--vulkan','0,1','--device-cap-mib','7680,7680','--dense-topology','primary',
          '--expert-plan','0:/absent/local0','--expert-plan','1:/absent/local1',
          '--expert-native-q8-observation','/absent/expert','--native-moe-combine',
          '--native-hc-serial','/absent/hc-serial','--hc-serial-resident-weights',
          '--native-hc-ffn-write-layer0','/absent/hc0','--layer0-resident-weights',
          '--gdn-resident-weights','--gdn-resident-state','--dense-resident-weights',
          '--native-router','/absent/router','--native-shared','/absent/shared',
          '--native-ple-value-t1','/absent/ple-value','--native-ple-key-t1','/absent/ple-key',
          '--native-ple-gate-t1','/absent/ple-gate','--native-ple-conv-t1','/absent/ple-conv',
          '--native-ple-norm-t1','/absent/ple-norm','--native-ple-hc-read-t1','/absent/ple-hc',
          '--native-gdn-t1','/absent/gdn','--gdn-native-prep-bf16','--native-gdn-prep-t1','/absent/prep',
          '--native-gdn-input-projection-t1','/absent/input','--native-gdn-output-projection-t1','/absent/output',
          '--qsa-norm-rope-resident-weights','/absent/qsa-norm','--native-qsa-norm-rope-t1',
          '--qsa-indexer-resident-weights','/absent/qsa-indexer','--native-qsa-indexer-t1',
          '--qsa-score-select-banks','/absent/qsa-scores','--native-qsa-score-select-t1',
          '--qsa-attention-banks','/absent/qsa-attention','--native-qsa-attention-t1',
          '--native-head-hc-t1','/absent/head','--native-head-vocab-t1','--prompt-head-elision']
if native_enabled:
    combined = run(remote + serial, True)
    assert combined['native_hc_serial'] and combined['native_gdn_t1'] and combined['native_gdn_preparation_t1']
    assert combined['native_gdn_input_projection_t1'] and combined['native_gdn_output_projection_t1']
else:
    assert 'build required' in run(remote + serial, False)
for prohibited in [['--expert-check-rows'], ['--allow-cpu-fallback'], ['--prompt-chunk-size','2'],
                   ['--expert-native-profile','/absent/profile']]:
    run(remote + serial + prohibited, False)

run(['--remote-gpu-worker', 'vulkan-native-q8@192.0.2.1:4567', '--remote-gpu-plan', '/absent/plan',
     '--remote-gpu-tls-ca', '/absent/ca', '--remote-gpu-tls-cert', '/absent/cert',
     '--remote-gpu-tls-key', '/absent/key', '--remote-gpu-tls-peer', 'worker.test'], tls_enabled)
for args in [
    ['--remote-gpu-plan', '/absent/plan'],
    ['--remote-gpu-worker', 'vulkan-native-q8@127.0.0.1:4567'],
    ['--remote-gpu-worker', '127.0.0.1:4567', '--remote-gpu-plan', '/absent/plan'],
    ['--remote-gpu-worker', 'cuda@127.0.0.1:4567', '--remote-gpu-plan', '/absent/plan'],
    ['--remote-gpu-worker', 'vulkan-native-q8@example.com:4567', '--remote-gpu-plan', '/absent/plan'],
    ['--remote-gpu-worker', 'vulkan-native-q8@192.0.2.1:4567', '--remote-gpu-plan', '/absent/plan'],
    remote + ['--remote-gpu-tls-ca', '/absent/ca'],
    remote + ['--remote-gpu-plan', '/second'],
    remote + ['--remote-gpu-deadline-ms', '0'],
    remote + ['--remote-gpu-setup-ms', '600001'],
    remote + ['--prompt-chunk-size', '2'],
    remote + ['--allow-cpu-fallback'],
    remote + ['--expert-check-rows'],
    remote + ['--expert-native-q8-observation', '/absent/receipt'],
]:
    run(args, False)
assert 'remote GPU plan' in run(remote, False, validate=False)
print(f'{count} remote CLI checks passed; no model, network or credential files opened')
