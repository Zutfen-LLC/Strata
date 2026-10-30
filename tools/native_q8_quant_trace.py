#!/usr/bin/env python3
"""One quantization-only observation; traces interpretable only after exact AMD reproduction."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('q8_geometry_trace_tools', ROOT / 'tools/native_q8_geometry.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
ELEMENTS = (36, 118, 1046, 4479)
FIELDS = ('amax', 'd', 'ratio', 'rounded')
FIXTURE_SHA256 = '26c5ab4747b9957fd2fe9971d6389365ffca53b2f3b069c4c9d72754d35f39c8'
AMD_SHA256 = '31fbc95071ca4d901d7c811b6f59e825c3977c683ed58f660035127c27e8afe5'


def compare(reference, probe):
    retained_ok, sections, invalid = geometry.capture(reference, True)
    if not retained_ok or any(invalid.values()):raise ValueError('quant trace: invalid retained AMD capture')
    canaries, values = geometry.gu.raw_sections(probe, 0x3154384e, (2160, 16), status=True)
    expected = struct.pack('<2160I', *sections[0]);observed = struct.pack('<2160I', *values[0])
    differences = [{'index': i, 'retained_byte': a, 'probe_byte': b}
                   for i, (a, b) in enumerate(zip(expected, observed)) if a != b]
    traces = []
    trace_valid = True
    for slot, element in enumerate(ELEMENTS):
        row = dict(zip(FIELDS, values[1][slot * 4:slot * 4 + 4]))
        finite = all(math.isfinite(struct.unpack('<f', struct.pack('<I', v))[0]) for v in row.values())
        written = all(v != 0xa5a5a5a5 for v in row.values())
        trace_valid &= finite and written
        traces.append({'element': element, 'raw_f32_bits': {k: f'{v:08x}' for k, v in row.items()},
                       'finite': finite, 'written': written})
    reproduced = not differences
    return {'qualified': False, 'quantizer_qualified': False, 'observation_only': True,
            'fp32_denorm_mode': 'unspecified', 'original_comparator_exit': 1, 'retained_amd_comparator_exit': 1,
            'cause_isa_proven': False, 'compared_bytes': len(expected), 'different_bytes': len(differences),
            'differences': differences, 'sentinels_ok': canaries, 'reproduced_retained_amd': reproduced,
            'trace_interpretable': reproduced and canaries and trace_valid, 'traces': traces}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observe-unspecified-fp32-denorm', action='store_true', required=True)
    for name in ('component', 'fixture', 'retained-amd', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--fixture-sha256', required=True)
    p.add_argument('--retained-amd-sha256', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--expect-device', required=True)
    a = p.parse_args()
    if not a.device.isdecimal() or not a.expect_device:raise ValueError('quant trace: verified device index/name required')
    if a.fixture_sha256 != FIXTURE_SHA256 or a.retained_amd_sha256 != AMD_SHA256:
        raise ValueError('quant trace: only the retained fixture/AMD failure identities are supported')
    inputs = {'fixture': geometry.preflight.checked_bytes(a.fixture, a.fixture_sha256),
              'retained-amd': geometry.preflight.checked_bytes(a.retained_amd, a.retained_amd_sha256)}
    a.output.mkdir(parents=False, exist_ok=False)
    for name, data in inputs.items():
        with (a.output / (name + '.bin')).open('xb') as f:f.write(data)
    # Validate the checked snapshot before launch, never a second source read.
    valid, _, invalid = geometry.capture(a.output / 'retained-amd.bin', True)
    if not valid or any(invalid.values()):raise ValueError('quant trace: invalid retained AMD reference')
    exe, raw = a.component.resolve(strict=True), (a.output / 'quant-trace.raw').resolve()
    command = [str(exe), '--observe-unspecified-fp32-denorm', '--device', a.device, '--expect-device', a.expect_device,
               '--fixture', str((a.output / 'fixture.bin').resolve()), '--output', str(raw)]
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    receipt = {'qualified': False, 'quantizer_qualified': False, 'observation_only': True,
               'fp32_denorm_mode': 'unspecified', 'original_comparator_exit': 1, 'retained_amd_comparator_exit': 1,
               'cause_isa_proven': False, 'argv': command, 'component_sha256': geometry.sha(exe.read_bytes()),
               'source_head': revision.stdout.strip() if revision.returncode == 0 else None,
               'source_files_sha256': {name: geometry.sha((ROOT / name).read_bytes()) for name in (
                   'src/kernels/vulkan/native_q8_stages.comp', 'src/kernels/vulkan/native_q8_quant_trace.glsl',
                   'src/core/native_q8_quant_trace.hpp', 'src/core/strata_vk_native_stages.inc')},
               'sha256': {k: geometry.sha(v) for k, v in inputs.items()}}
    path = a.output / 'receipt.json';path.write_text(json.dumps(receipt, indent=2) + '\n')
    with (a.output / 'component.log').open('wb') as log:
        try:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
                receipt['child_pid'] = child.pid
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    receipt['timeout'] = True;child.kill();child.wait()
                receipt['actual_child_exit'] = child.returncode
        except OSError as error:receipt.update(actual_child_exit=None, launch_error=str(error))
    report = None
    if raw.is_file():
        try:report = compare(a.output / 'retained-amd.bin', raw)
        except ValueError as error:
            report = {'trace_interpretable': False, 'qualified': False, 'quantizer_qualified': False,
                      'original_comparator_exit': 1, 'retained_amd_comparator_exit': 1, 'observation_only': True,
                      'fp32_denorm_mode': 'unspecified', 'cause_isa_proven': False, 'capture_error': str(error)}
        if receipt.get('actual_child_exit') != 0 or receipt.get('timeout'):report['trace_interpretable'] = False
        (a.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for suffix in ('', '.views.bin', '.effective.spv'):
        file = Path(str(raw) + suffix)
        if file.is_file():receipt['sha256'][file.name] = geometry.sha(file.read_bytes())
    receipt['comparison_exit'] = (0 if report['trace_interpretable'] else 1) if report else None
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': receipt, 'comparison': report}))
    return 0 if receipt.get('actual_child_exit') == 0 and not receipt.get('timeout') and report and report['trace_interpretable'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
