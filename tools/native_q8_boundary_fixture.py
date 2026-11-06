#!/usr/bin/env python3
"""Write deterministic finite Q8_1 input blocks; this does not execute a producer."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

# Bit patterns preserve signed zeros and subnormals without host FP arithmetic.
def blocks():
    control = [0x3f800000, 0xbf800000] + [0x3e800000, 0xbe800000] * 15
    signed_subnormal = [0x3f800000, 0xbf800000] + [0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00000000, 0x80000000] * 5
    normal_boundary = [0x04000000, 0x84000000] + [0x00600000, 0x80600000, 0x007fffff, 0x00800000, 0x00800001, 0x80800000] * 5
    # Largest magnitude = 127, so division yields scale 1 in exact arithmetic;
    # examine both sides of +/-0.5 and +/-1.5 quantization ties.
    rounding = [0x42fe0000, 0xc2fe0000] + [0x3effffff, 0x3f000000, 0x3f000001, 0xbeffffff, 0xbf000000, 0xbf000001, 0x3fbfffff, 0x3fc00000, 0x3fc00001, 0xbfbfffff] * 3
    return {'control': control, 'signed_subnormal': signed_subnormal,
            'normal_boundary': normal_boundary, 'rounding': rounding}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new output directory')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = {'schema': 'q8-boundary-input/v1', 'producer_executed': False, 'blocks': {}}
    for name, bits in blocks().items():
        assert len(bits) == 32
        raw = struct.pack('<32I', *bits)
        (args.output / (name + '.f32le')).write_bytes(raw)
        manifest['blocks'][name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'float32_bits': [f'{x:08x}' for x in bits]}
    combined = b''.join(struct.pack('<32I', *values) for values in blocks().values())
    (args.output / 'all.f32le').write_bytes(combined)
    manifest['all_sha256'] = hashlib.sha256(combined).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
