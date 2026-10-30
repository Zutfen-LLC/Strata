#!/usr/bin/env python3
"""One bounded down-only Vulkan capture using verified actual native packed hq."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_q8_gu_tools', ROOT / 'tools/native_q8_gu.py')
gu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gu)
preflight = gu.preflight


def validate_receipt(receipt, hashes, archive_identity=None):
    gu.validate_receipt(receipt, hashes, archive_identity)
    if receipt.get('hq_reuse_verified') is not True or receipt.get('hq_reuse_error'):
        raise ValueError('down: actual native hq was not verified for reuse')
    if receipt.get('sha256', {}).get('native-consumer.raw.hq.bin') != hashes['hq'] or receipt.get('retained_oracle_sha256') != hashes['oracle'] or receipt.get('retained_weights_sha256') != hashes['weights']:
        raise ValueError('down: hq/retained oracle artifact binding mismatch')


def validate_hq(data):
    if len(data) != 72:
        raise ValueError('down: actual native hq must contain exactly 72 bytes')
    for offset in (0, 36):
        if int.from_bytes(data[offset:offset + 2], 'little') & 0x7c00 == 0x7c00:
            raise ValueError('down: finite native hq scales required')


def compare(oracle, path):
    sentinels, sections = gu.raw_sections(path, 0x3144384e, (1024, 1024, 1024), status=True)
    # Expected values come only from the retained native down output. No dot
    # simulation, hidden reconstruction or Vulkan quantization supplies inputs.
    expected = [list(oracle[2]) + [0xa5a5a5a5] * 512,
                [0xa5a5a5a5] * 512 + list(oracle[2][256:]) + [0xa5a5a5a5] * 256,
                [0xa5a5a5a5] * 1024]
    report = {'exact': sentinels, 'qualified': False, 'quantizer_qualified': False,
              'original_comparator_exit': 1, 'sentinels_ok': sentinels, 'view_sentinels_ok': sentinels, 'stages': {}}
    for name, want, got, active in zip(('down', 'sparse_scatter', 'empty'), expected, sections, ((0, 512), (512, 768), (0, 0))):
        differences = [{'index': i, 'destination': i // 256, 'row': i % 256,
                        'native_or_sentinel_bits': f'{a:08x}', 'vulkan_bits': f'{b:08x}'}
                       for i, (a, b) in enumerate(zip(want, got)) if a != b]
        nonfinite = [i for i, b in enumerate(got) if b & 0x7f800000 == 0x7f800000]
        sentinel_failure = any(b != 0xa5a5a5a5 for i, b in enumerate(got) if i < active[0] or i >= active[1])
        report['sentinels_ok'] &= not sentinel_failure
        report['stages'][name] = {'compared': 1024, 'different_bits': len(differences), 'sentinel_failure': sentinel_failure,
                                 'differences': differences, 'nonfinite_indices': nonfinite}
        if differences or nonfinite:report['exact'] = False
    return report


def main(*, composed_compare=None):
    p = argparse.ArgumentParser(description=("Explicit device-resident composed observation" if composed_compare else __doc__))
    names = ('fixture', 'native', 'observation', 'oracle', 'weights', 'native-receipt', 'hq')
    for name in names:
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--native-source-head', help='Independently verified commit for a null archive receipt only')
    p.add_argument('--native-source-archive-sha256', help='Independently verified source archive hash')
    p.add_argument('--native-consumer-sha256', help='Independently verified native binary hash; must match receipt')
    p.add_argument('--component', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--expect-device', required=True)
    if composed_compare:
        p.add_argument('--observe-unspecified-fp32-denorm', action='store_true', required=True)
    a = p.parse_args()
    if not a.device.isdecimal() or not a.expect_device:
        raise ValueError('down: verified numeric device and device substring required')
    sources = {name: preflight.checked_bytes(getattr(a, name.replace('-', '_')),
               getattr(a, name.replace('-', '_') + '_sha256')) for name in names}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    receipt = json.loads(sources['native-receipt'])
    explicit = (a.native_source_head, a.native_source_archive_sha256, a.native_consumer_sha256)
    archive_identity = None
    if any(value is not None for value in explicit):
        archive_identity = {'source_head': explicit[0], 'source_archive_sha256': explicit[1],
                            'consumer_sha256': explicit[2], 'native_receipt_sha256': hashes['native-receipt']}
    validate_receipt(receipt, hashes, archive_identity)
    validate_hq(sources['hq'])
    a.output.mkdir(parents=False, exist_ok=False)
    paths = {}
    for name, data in sources.items():
        paths[name] = a.output / (name + ('.json' if name == 'native-receipt' else '.bin'))
        with paths[name].open('xb') as out:
            out.write(data)
    original = preflight.validate(paths['fixture'], paths['native'], paths['observation'])
    oracle = gu.validate_oracle(paths['oracle'])
    (a.output / 'original-comparison.json').write_text(json.dumps(original, indent=2, allow_nan=False) + '\n')
    exe = a.component.resolve(strict=True)
    raw = (a.output / ('vulkan-composed.raw' if composed_compare else 'vulkan-down.raw')).resolve()
    command = [str(exe), '--device', a.device, '--expect-device', a.expect_device,
               '--hq', str(paths['hq'].resolve()),
               '--weights', str(paths['weights'].resolve()), '--output', str(raw)]
    if composed_compare:
        command = [str(exe), '--observe-unspecified-fp32-denorm', '--device', a.device,
                   '--expect-device', a.expect_device, '--native', str(paths['native'].resolve()),
                   '--observation', str(paths['observation'].resolve()),
                   '--weights', str(paths['weights'].resolve()), '--output', str(raw)]
    provenance = {'qualified': False, 'quantizer_qualified': False, 'original_comparator_exit': 1, 'native_hq_reuse_verified': True,
                  'sha256': hashes, 'component_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
                  'argv': command, 'native_source_head': receipt.get('source_head'),
                  'native_consumer_sha256': receipt.get('consumer_sha256'),
                  'native_dispatch_basis': 'source and retained environment, not a dynamic launch trace'}
    if composed_compare:
        provenance.update(observation_only=True, fp32_denorm_mode='unspecified', device_resident_intermediates=True)
    if archive_identity is not None:
        provenance['native_archive_identity'] = archive_identity
        provenance['native_archive_identity_basis'] = 'explicit independently verified later inspection; not the original producer receipt'
        provenance['native_source_head'] = archive_identity['source_head']
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
            result = composed_compare(oracle, sources['hq'], raw) if composed_compare else compare(oracle, raw)
        except ValueError as error:
            result = {'exact': False, 'qualified': False, 'quantizer_qualified': False,
                      'original_comparator_exit': 1, 'capture_error': str(error)}
        if composed_compare:
            result.update(observation_only=True, fp32_denorm_mode='unspecified')
        (a.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    modules = ('gu', 'activation', 'down') if composed_compare else ()
    for file in (raw, Path(str(raw) + '.views.bin'), Path(str(raw) + '.effective.spv'),
                 *(Path(str(raw) + '.' + stage + '.effective.spv') for stage in modules)):
        if file.is_file():provenance['sha256'][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    if composed_compare:
        provenance['comparison_exit'] = (0 if result['exact'] else 1) if result else None
    receipt_path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'receipt': provenance, 'comparison': result}))
    return 0 if provenance.get('actual_child_exit') == 0 and not provenance.get('timeout') and result and result['exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
