#!/usr/bin/env python3
"""Explicit composed observation: device GU -> activation/Q8_1 -> down."""
import importlib.util
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_q8_down_tools', ROOT / 'tools/native_q8_down.py')
down = importlib.util.module_from_spec(spec)
spec.loader.exec_module(down)


def compare(oracle, native_hq, path):
    down.validate_hq(native_hq)
    sentinels, sections = down.gu.raw_sections(path, 0x3150384e, (64, 64, 18, 512), status=True)
    report = {'exact': sentinels, 'qualified': False, 'quantizer_qualified': False,
              'observation_only': True, 'fp32_denorm_mode': 'unspecified',
              'original_comparator_exit': 1, 'sentinels_ok': sentinels, 'stages': {}}
    for name, expected, got, rows in zip(('gate', 'up', 'down'), oracle,
                                       (sections[0], sections[1], sections[3]), (32, 32, 256)):
        differences = [{'index': i, 'entry': i // rows, 'row': i % rows,
                        'native_bits': f'{a:08x}', 'vulkan_bits': f'{b:08x}'}
                       for i, (a, b) in enumerate(zip(expected, got)) if a != b]
        nonfinite = [i for i, bits in enumerate(got) if bits & 0x7f800000 == 0x7f800000]
        report['stages'][name] = {'compared': len(got), 'different_bits': len(differences),
                                 'differences': differences, 'nonfinite_indices': nonfinite}
        if differences or nonfinite:
            report['exact'] = False
    hq = struct.pack('<18I', *sections[2])
    differences = [{'index': i, 'entry': i // 36, 'block_byte': i % 36,
                    'kind': 'metadata' if i % 36 < 4 else 'code',
                    'native_byte': a, 'vulkan_byte': b}
                   for i, (a, b) in enumerate(zip(native_hq, hq)) if a != b]
    # Both half metadata fields must be finite, independently of byte parity.
    nonfinite = [i for i in (0, 2, 36, 38)
                 if int.from_bytes(hq[i:i + 2], 'little') & 0x7c00 == 0x7c00]
    report['stages']['hidden_q8_1'] = {'compared_bytes': 72, 'different_bytes': len(differences),
                                       'differences': differences, 'nonfinite_metadata_offsets': nonfinite}
    if differences or nonfinite:
        report['exact'] = False
    return report


if __name__ == '__main__':
    raise SystemExit(down.main(composed_compare=compare))
