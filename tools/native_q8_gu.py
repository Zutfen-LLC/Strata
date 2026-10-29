#!/usr/bin/env python3
"""One standalone grouped Q8 gate/up capture against a retained native oracle."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_q8_consumer_preflight', ROOT / 'tools/native_q8_consumer.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def raw_sections(path, magic, counts, status=False):
    data = path.read_bytes()
    (found,), offset = preflight.captures.read_words(data, 0, 1)
    if found != magic:
        raise ValueError('GU: incorrect raw result role/magic')
    sentinel_ok = True
    if status:
        (flag,), offset = preflight.captures.read_words(data, offset, 1)
        if flag not in (0, 1):
            raise ValueError('GU: invalid sentinel status')
        sentinel_ok = bool(flag)
    sections = []
    for count in counts:
        (size,), offset = preflight.captures.read_words(data, offset, 1)
        if size != count:
            raise ValueError('GU: incorrect raw section size/order')
        values, offset = preflight.captures.read_words(data, offset, count)
        sections.append(values)
    if offset != len(data):
        raise ValueError('GU: trailing raw result bytes')
    return sentinel_ok, sections


def validate_oracle(path):
    _, sections = raw_sections(path, 0x3143384e, (64, 64, 512))
    for values in sections:
        if any(v & 0x7f800000 == 0x7f800000 for v in values):
            raise ValueError('GU: nonfinite native oracle')
        if not any(v & 0x7fffffff for v in values):
            raise ValueError('GU: degenerate native oracle')
        if values[:len(values) // 2] != values[len(values) // 2:]:
            raise ValueError('GU: retained native consumer entries differ')
    return sections


def validate_receipt(receipt, hashes):
    if not isinstance(receipt, dict):
        raise ValueError('GU: malformed native receipt')
    if not re.fullmatch('[0-9a-f]{40}', receipt.get('source_head') or '') or not re.fullmatch('[0-9a-f]{64}', receipt.get('consumer_sha256') or ''):
        raise ValueError('GU: missing native source/binary identity')
    if receipt.get('actual_child_exit') != 0 or receipt.get('timeout') or receipt.get('launch_error'):
        raise ValueError('GU: native oracle did not complete successfully')
    if not receipt.get('finite_synthetic_weights') or not receipt.get('native_consumer_only') or receipt.get('qualified') is not False or receipt.get('original_comparator_exit') != 1:
        raise ValueError('GU: wrong native consumer receipt')
    environment = receipt.get('native_environment', {})
    if not isinstance(environment, dict) or 'STRATA_KQ_SPLIT' not in environment or not isinstance(environment['STRATA_KQ_SPLIT'], (str, type(None))) or (environment['STRATA_KQ_SPLIT'] or '').startswith('1'):
        raise ValueError('GU: retained oracle does not select the scoped per-entry branch')
    for field, name in {'fixture': 'fixture', 'native': 'native', 'observation': 'observation',
                        'native-consumer.raw': 'oracle', 'native-consumer.raw.weights.bin': 'weights'}.items():
        if receipt.get('sha256', {}).get(field) != hashes[name]:
            raise ValueError('GU: oracle receipt artifact binding mismatch')


def compare(oracle, result_path):
    sentinels, sections = raw_sections(result_path, 0x3147384e, (64, 64), status=True)
    report = {'exact': bool(sentinels), 'qualified': False, 'quantizer_qualified': False,
              'original_comparator_exit': 1, 'sentinels_ok': sentinels, 'stages': {}}
    for name, expected, got in zip(('gate', 'up'), oracle[:2], sections):
        differences = [{'index': i, 'entry': i // 32, 'row': i % 32,
                        'native_bits': f'{a:08x}', 'vulkan_bits': f'{b:08x}'}
                       for i, (a, b) in enumerate(zip(expected, got)) if a != b]
        nonfinite = [i for i, bits in enumerate(got) if bits & 0x7f800000 == 0x7f800000]
        report['stages'][name] = {'compared': 64, 'different_bits': len(differences),
                                 'differences': differences, 'nonfinite_indices': nonfinite}
        if differences or nonfinite:
            report['exact'] = False
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    names = ('fixture', 'native', 'observation', 'oracle', 'weights', 'native-receipt')
    for name in names:
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--component', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--expect-device', required=True)
    a = p.parse_args()
    if not a.device.isdecimal() or not a.expect_device:
        raise ValueError('GU: verified numeric device and device substring required')
    sources = {name: preflight.checked_bytes(getattr(a, name.replace('-', '_')),
               getattr(a, name.replace('-', '_') + '_sha256')) for name in names}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    receipt = json.loads(sources['native-receipt'])
    validate_receipt(receipt, hashes)
    a.output.mkdir(parents=False, exist_ok=False)
    paths = {}
    for name, data in sources.items():
        paths[name] = a.output / (name + ('.json' if name == 'native-receipt' else '.bin'))
        with paths[name].open('xb') as out:
            out.write(data)
    original = preflight.validate(paths['fixture'], paths['native'], paths['observation'])
    oracle = validate_oracle(paths['oracle'])
    (a.output / 'original-comparison.json').write_text(json.dumps(original, indent=2, allow_nan=False) + '\n')
    exe = a.component.resolve(strict=True)
    raw = (a.output / 'vulkan-gu.raw').resolve()
    command = [str(exe), '--device', a.device, '--expect-device', a.expect_device,
               '--native', str(paths['native'].resolve()), '--observation', str(paths['observation'].resolve()),
               '--weights', str(paths['weights'].resolve()), '--output', str(raw)]
    provenance = {'qualified': False, 'quantizer_qualified': False, 'original_comparator_exit': 1,
                  'sha256': hashes, 'component_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
                  'argv': command, 'native_source_head': receipt.get('source_head'),
                  'native_consumer_sha256': receipt.get('consumer_sha256'),
                  'native_dispatch_basis': 'source and retained environment, not a dynamic launch trace'}
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    provenance['source_head'] = revision.stdout.strip() if revision.returncode == 0 else None
    receipt_path = a.output / 'receipt.json'
    receipt_path.write_text(json.dumps(provenance, indent=2) + '\n')
    with (a.output / 'component.log').open('wb') as log:
        try:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
                provenance['child_pid'] = child.pid
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    provenance['timeout'] = True;child.kill();child.wait()
                provenance['actual_child_exit'] = child.returncode
        except OSError as error:
            provenance['actual_child_exit'] = None;provenance['launch_error'] = str(error)
    result = None
    if raw.is_file():
        try:
            result = compare(oracle, raw)
        except ValueError as error:
            result = {'exact': False, 'qualified': False, 'quantizer_qualified': False,
                      'original_comparator_exit': 1, 'capture_error': str(error)}
        (a.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    for file in (raw, Path(str(raw) + '.views.bin'), Path(str(raw) + '.effective.spv')):
        if file.is_file():provenance['sha256'][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'receipt': provenance, 'comparison': result}))
    return 0 if provenance.get('actual_child_exit') == 0 and not provenance.get('timeout') and result and result['exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
