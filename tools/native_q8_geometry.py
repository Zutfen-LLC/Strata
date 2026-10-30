#!/usr/bin/env python3
"""One bounded H2560/F640 native capture or explicit Vulkan observation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('q8_gu_geometry_tools', ROOT / 'tools/native_q8_gu.py')
gu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gu)
preflight = gu.preflight
WEIGHT_BYTES = 5222400
COUNTS = (2160, 1920, 1920, 1920, 540, 7680)
NAMES = ('input_q8', 'gate', 'up', 'hidden_float_diagnostic', 'hidden_q8', 'output')
SOURCE_FILES = ('src/kernels/cuda/iq_kernels.cu', 'tests/kernels/native_q8_geometry_fixture.hpp',
                'tests/kernels/native_q8_consumer_fixture.hpp', 'tests/kernels/native_q8_geometry_consumer.cu')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packed_sentinel(words, name):
    # Output regions start at this canary. It is not a numerical tolerance.
    return (any(words[i] == 0xa5a5a5a5 for i in range(0, len(words), 9))
            if name in ('input_q8', 'hidden_q8') else 0xa5a5a5a5 in words)


def capture(path, observation):
    sentinels, sections = gu.raw_sections(path, 0x324d384e if observation else 0x314d384e, COUNTS, status=True)
    invalid = {}
    for name, words in zip(NAMES, sections):
        if name in ('input_q8', 'hidden_q8'):
            # Inspect both half metadata fields, not packed int8 words as floats.
            bad = [i * 4 + j for i in range(0, len(words), 9) for j in (0, 2)
                   if (words[i] >> (j * 8)) & 0x7c00 == 0x7c00]
        else:
            bad = [i for i, bits in enumerate(words) if bits & 0x7f800000 == 0x7f800000]
        invalid[name] = bad
        if packed_sentinel(words, name):sentinels = False
        if name not in ('input_q8', 'hidden_q8') and not any(bits & 0x7fffffff for bits in words):
            raise ValueError('geometry: degenerate all-zero ' + name)
    return sentinels, sections, invalid


def compare(native_path, observation_path):
    ns, native, ni = capture(native_path, False)
    vs, observed, vi = capture(observation_path, True)
    report = {'exact': ns and vs and not any(ni.values()) and not any(vi.values()),
              'qualified': False, 'quantizer_qualified': False, 'observation_only': True,
              'fp32_denorm_mode': 'unspecified', 'original_comparator_exit': 1,
              'native_sentinels_ok': ns, 'vulkan_sentinels_ok': vs, 'stages': {}}
    for name, want, got in zip(NAMES, native, observed):
        packed = name in ('input_q8', 'hidden_q8')
        a, b = (struct.pack('<' + 'I' * len(want), *want), struct.pack('<' + 'I' * len(got), *got)) if packed else (want, got)
        differences = [{'index': i, 'native': x, 'vulkan': y,
                        **({'block_byte': i % 36, 'kind': 'metadata' if i % 36 < 4 else 'code'} if packed else {})}
                       for i, (x, y) in enumerate(zip(a, b)) if x != y]
        report['stages'][name] = {'unit': 'byte' if packed else 'raw_f32_bits', 'compared': len(a),
                                 'different': len(differences), 'differences': differences,
                                 'native_nonfinite': ni[name], 'vulkan_nonfinite': vi[name]}
        if differences:report['exact'] = False
    return report


def validate_dispatch(dispatch):
    required = {'h': 2560, 'f': 640, 'entries': 3, 'groups': 3, 'grid_groups': 2,
                'split_off_q8': True, 'kq_split': False, 'gu_split_q8': False,
                'grouped_v1': False, 'old_kernels': False, 'exp_phase': 0,
                'gate_up': 'native_gu_kernel<8>', 'activation': 'swiglu_q8_1_entries_kernel',
                'down': 'native_down_kernel<8>'}
    if any(type(dispatch.get(k)) is not type(v) or dispatch[k] != v for k, v in required.items()):
        raise ValueError('geometry: runtime native selectors do not establish the supported production dispatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--capture-native', action='store_true')
    mode.add_argument('--observe-unspecified-fp32-denorm', action='store_true')
    for name in ('component', 'fixture', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--fixture-sha256', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--expect-device', required=True)
    for name in ('native-raw', 'native-receipt', 'weights'):
        p.add_argument('--' + name, type=Path)
        p.add_argument('--' + name + '-sha256')
    a = p.parse_args()
    if not a.device.isdecimal() or not a.expect_device:
        raise ValueError('geometry: verified numeric device and device name required')
    fixture = preflight.checked_bytes(a.fixture, a.fixture_sha256)
    sources = {name: sha((ROOT / name).read_bytes()) for name in SOURCE_FILES}
    inputs = {'fixture': fixture}
    if not a.capture_native:
        for name in ('native-raw', 'native-receipt', 'weights'):
            path, digest = getattr(a, name.replace('-', '_')), getattr(a, name.replace('-', '_') + '_sha256')
            if path is None or digest is None:raise ValueError('geometry: native reference paths and hashes required')
            inputs[name] = preflight.checked_bytes(path, digest, max_bytes=WEIGHT_BYTES if name == 'weights' else 1 << 20)
        native_receipt = json.loads(inputs['native-receipt'])
        if (native_receipt.get('actual_child_exit') != 0 or native_receipt.get('timeout') or
                native_receipt.get('capture_error') or native_receipt.get('capture_valid') is not True or
                native_receipt.get('capture_native') is not True or native_receipt.get('qualified') is not False or
                native_receipt.get('source_files_sha256') != sources):
            raise ValueError('geometry: unsuccessful native capture or different native source mapping')
        validate_dispatch(native_receipt['dispatch'])
        for key, source in (('fixture', 'fixture'), ('capture.raw', 'native-raw'), ('capture.raw.weights.bin', 'weights')):
            if native_receipt.get('sha256', {}).get(key) != sha(inputs[source]):
                raise ValueError('geometry: native artifact binding mismatch')
        if len(inputs['weights']) != WEIGHT_BYTES:raise ValueError('geometry: exact single-expert weight size required')
    a.output.mkdir(parents=False, exist_ok=False)
    paths = {}
    for name, data in inputs.items():
        paths[name] = (a.output / (name + '.bin')).resolve()
        with paths[name].open('xb') as f:f.write(data)
    exe, raw = a.component.resolve(strict=True), (a.output / 'capture.raw').resolve()
    command = [str(exe)] + ([] if a.capture_native else ['--observe-unspecified-fp32-denorm'])
    command += ['--device', a.device, '--expect-device', a.expect_device, '--fixture', str(paths['fixture'])]
    if not a.capture_native:command += ['--weights', str(paths['weights'])]
    command += ['--output', str(raw)]
    revision = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    receipt = {'qualified': False, 'quantizer_qualified': False, 'original_comparator_exit': 1,
               'capture_native': a.capture_native, 'observation_only': not a.capture_native,
               'fp32_denorm_mode': 'unspecified' if not a.capture_native else 'native CUDA; no Vulkan claim',
               'source_head': revision.stdout.strip() if revision.returncode == 0 else None,
               'source_files_sha256': sources, 'component_sha256': sha(exe.read_bytes()),
               'native_environment': {k: os.environ.get(k) for k in ('STRATA_KQ_SPLIT', 'STRATA_OLD_IQ_MMVQ',
                   'STRATA_GROUPED_V1', 'STRATA_GU_SWIGLU', 'STRATA_NO_SUB16_GU', 'STRATA_EXPERT_V2',
                   'STRATA_EXPERT_V2K', 'STRATA_EXPDEC', 'STRATA_EXP_MODE')},
               'sha256': {k: sha(v) for k, v in inputs.items()}, 'argv': command}
    report_path = a.output / 'receipt.json'
    report_path.write_text(json.dumps(receipt, indent=2) + '\n')
    with (a.output / 'component.log').open('wb') as log:
        try:
            with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as child:
                receipt['child_pid'] = child.pid
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    receipt['timeout'] = True;child.kill();child.wait()
                receipt['actual_child_exit'] = child.returncode
        except OSError as error:receipt.update(actual_child_exit=None, launch_error=str(error))
    comparison = None
    try:
        if receipt.get('actual_child_exit') == 0 and not receipt.get('timeout'):
            sentinels, _, invalid = capture(raw, not a.capture_native)
            receipt['capture_valid'] = sentinels and not any(invalid.values())
            receipt['nonfinite'] = invalid
            if a.capture_native:
                receipt['dispatch'] = json.loads(Path(str(raw) + '.dispatch.json').read_bytes())
                validate_dispatch(receipt['dispatch'])
                weights = Path(str(raw) + '.weights.bin')
                if weights.stat().st_size != WEIGHT_BYTES:raise ValueError('geometry: wrong captured weight size')
            else:
                comparison = compare(paths['native-raw'], raw)
                (a.output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    except (ValueError, OSError) as error:receipt['capture_error'] = str(error)
    for suffix in ('', '.weights.bin', '.dispatch.json', '.views.bin', '.gu.effective.spv', '.activation.effective.spv', '.down.effective.spv'):
        file = Path(str(raw) + suffix)
        if file.is_file():receipt['sha256'][file.name] = sha(file.read_bytes())
    receipt['comparison_exit'] = (0 if comparison['exact'] else 1) if comparison else None
    ok = receipt.get('actual_child_exit') == 0 and not receipt.get('timeout') and receipt.get('capture_valid') and not receipt.get('capture_error')
    if not a.capture_native:ok = ok and comparison and comparison['exact']
    report_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'receipt': receipt, 'comparison': comparison}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
