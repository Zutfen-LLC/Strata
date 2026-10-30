#!/usr/bin/env python3
"""Hash-checked launcher for one paired native consumer probe; never a parity waiver."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_q8_capture_tools', ROOT / 'tools/compare_native_q8_stages.py')
captures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(captures)


def checked_bytes(path, expected, *, max_bytes=1 << 20):
    if not re.fullmatch('[0-9a-fA-F]{64}', expected):
        raise ValueError('expected a complete SHA256')
    if path.stat().st_size > max_bytes:
        raise ValueError('input exceeds bounded fixture size')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected.lower():
        raise ValueError('input SHA256 mismatch')
    return data


def validate(fixture_path, native_path, observation_path):
    ff, n, starts, arrays = captures.fixture(fixture_path)
    if (ff, n, starts) != (64, 832, (2, 5, 5, 9)) or arrays[0][160:192] != tuple(range(1, 33)):
        raise ValueError('wrong original fixture shape/block5')
    native, obs = captures.capture(native_path, n), captures.capture(observation_path, n)
    if native['observation_only'] or not obs['observation_only'] or not obs['sentinels_ok']:
        raise ValueError('incorrect capture roles or observation sentinel status')
    a, b = native['input_q8'], obs['input_q8']
    if a[180:184] != bytes(4) or b[180:184] != bytes(4):
        raise ValueError('block5 metadata must be +0/+0')
    if a[184:216] != bytes([127] * 32) or b[184:216] != bytes(32):
        raise ValueError('expected original block5 codes absent')
    if a[:184] != b[:184] or a[216:] != b[216:]:
        raise ValueError('unexpected input-Q8 difference outside block5')
    report = captures.compare(fixture_path, native_path, observation_path)
    if report['sections']['input_q8']['different_bytes'] != 32 or any(
            report['sections'][name]['different_bytes'] for name in captures.NAMES[1:]):
        raise ValueError('incorrect/different capture sections')
    if any(r.get('sentinel_failure') or r['active_separate_vs_fused_different_bytes']
           for r in report['internal_paths'].values()):
        raise ValueError('sentinel or active separate/fused failure')
    if any(int.from_bytes(a[k * 36:k * 36 + 2], 'little') & 0x7c00 == 0x7c00 for k in range(8)):
        raise ValueError('nonfinite prefix activation scale')
    if not any(a[k * 36:k * 36 + 2] not in (bytes(2), bytes([0, 128])) and any(a[k * 36 + 4:k * 36 + 36])
               for k in range(8) if k != 5):
        raise ValueError('common signal blocks missing')
    return report


def validate_hq_capture(directory, retained_oracle, retained_weights):
    """Only a capture whose existing results are unchanged may supply down input."""
    if (directory / 'native-consumer.raw').read_bytes() != retained_oracle:
        raise ValueError('hq capture gate/up/output differs from retained native oracle')
    if (directory / 'native-consumer.raw.weights.bin').read_bytes() != retained_weights:
        raise ValueError('hq capture weights differ from retained native oracle')
    hq = (directory / 'native-consumer.raw.hq.bin').read_bytes()
    if len(hq) != 72:
        raise ValueError('hq capture must contain exactly two actual Q8_1 blocks (72 bytes)')
    return hashlib.sha256(hq).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fixture', 'native', 'observation', 'consumer', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('fixture', 'native', 'observation'):
        p.add_argument('--' + name + '-sha256', required=True)
    for name in ('retained-oracle', 'retained-weights'):
        p.add_argument('--' + name, type=Path)
        p.add_argument('--' + name + '-sha256')
    p.add_argument('--device', required=True)
    p.add_argument('--expect-device', required=True)
    a = p.parse_args()
    if not a.device.isdecimal() or not a.expect_device:
        raise ValueError('verified numeric device and nonempty device substring required')
    sources = {name: checked_bytes(getattr(a, name), getattr(a, name + '_sha256'))
               for name in ('fixture', 'native', 'observation')}
    retained = {}
    supplied = (a.retained_oracle, a.retained_oracle_sha256, a.retained_weights, a.retained_weights_sha256)
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError('hq capture requires both retained artifacts and their verified hashes')
        for name in ('retained-oracle', 'retained-weights'):
            retained[name] = checked_bytes(getattr(a, name.replace('-', '_')), getattr(a, name.replace('-', '_') + '_sha256'))
    a.output.mkdir(parents=False, exist_ok=False)
    # Immutable-by-convention fresh snapshots bind validation and child inputs to
    # the same bytes; originals remain untouched. Hashes are checked before GPU use.
    paths = {}
    for name, data in sources.items():
        paths[name] = a.output / (name + '.bin')
        with paths[name].open('xb') as out:
            out.write(data)
    for name, data in retained.items():
        with (a.output / (name + '.bin')).open('xb') as out:out.write(data)
    comparison = validate(paths['fixture'], paths['native'], paths['observation'])
    (a.output / 'original-comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    exe = a.consumer.resolve(strict=True)
    command = [str(exe), '--device', a.device, '--expect-device', a.expect_device,
               '--native', str(paths['native'].resolve()), '--observation', str(paths['observation'].resolve()),
               '--output', str((a.output / 'native-consumer.raw').resolve())]
    receipt = {'qualified': False, 'fp32_denorm_mode': 'unspecified', 'original_comparator_exit': 1,
               'native_consumer_only': True, 'finite_synthetic_weights': True, 'hq_reuse_verified': False, 'argv': command,
               'denorm_mode_applies_to': 'retained_observation_capture',
               'native_environment': {k: os.environ.get(k) for k in ('STRATA_GROUPED_V1', 'STRATA_KQ_SPLIT',
                   'STRATA_OLD_IQ_MMVQ', 'STRATA_EXPERT_V2', 'STRATA_GU_SWIGLU', 'STRATA_EXPDEC', 'STRATA_EXP_MODE')},
               'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
               'consumer_sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    receipt['source_head'] = revision.stdout.strip() if revision.returncode == 0 else None
    if retained:
        receipt['retained_oracle_sha256'] = hashlib.sha256(retained['retained-oracle']).hexdigest()
        receipt['retained_weights_sha256'] = hashlib.sha256(retained['retained-weights']).hexdigest()
    receipt_path = a.output / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    with (a.output / 'consumer.log').open('wb') as log:
        try:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
                receipt['child_pid'] = child.pid
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    receipt['timeout'] = True
                    child.kill()
                    child.wait()
                receipt['actual_child_exit'] = child.returncode
        except OSError as error:
            receipt['actual_child_exit'] = None
            receipt['launch_error'] = str(error)
    for name in ('native-consumer.raw', 'native-consumer.raw.weights.bin', 'native-consumer.raw.hq.bin'):
        path = a.output / name
        if path.is_file():
            receipt['sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if retained and receipt.get('actual_child_exit') == 0 and not receipt.get('timeout') and not receipt.get('launch_error'):
        try:
            receipt['sha256']['native-consumer.raw.hq.bin'] = validate_hq_capture(a.output, retained['retained-oracle'], retained['retained-weights'])
            receipt['hq_reuse_verified'] = True
        except (OSError, ValueError) as error:
            receipt['hq_reuse_error'] = str(error)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 1 if receipt.get('timeout') or receipt.get('launch_error') or receipt.get('hq_reuse_error') else receipt['actual_child_exit'] if receipt['actual_child_exit'] >= 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
