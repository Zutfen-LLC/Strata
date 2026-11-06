"""Real CLI preflight checks; no model, network, or credential access."""
import json
import subprocess
import sys

binary = sys.argv[1]
tls_enabled = len(sys.argv) > 2 and sys.argv[2] == '1'
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
    remote + ['--expert-plan', '0:/absent/local-plan', '--vulkan', '0', '--device-cap-mib', '1024'],
    remote + ['--expert-native-q8-observation', '/absent/receipt'],
]:
    run(args, False)
assert 'remote GPU plan' in run(remote, False, validate=False)
print(f'{count} remote CLI checks passed; no model, network or credential files opened')
