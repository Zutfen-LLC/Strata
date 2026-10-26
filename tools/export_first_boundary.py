#!/usr/bin/env python3
"""Export only window 0/layer 0 and the accepted record-0 CPU rows for offline replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from compare_first_request_trace import read

SRA_SHA256 = '0aaeb0762418bc150d7c4ca2e6c168e38b5069fb2df825cb94dd75312a6d727d'
SRA_BYTES = 8110340
PACKET_BYTES = 287220


def boundary(a, b, sra):
    accepted_sha = SRA_SHA256
    def one(data, kind):
        values = [v for k, v in sorted(data.items()) if k[:3] == (0, 0, 0) and k[3] == kind]
        if not values:
            raise ValueError(f'missing first boundary kind {kind}')
        return np.concatenate(values, axis=0)
    def meta(data, kind):
        values = [v for k, v in data.items() if k[3] == kind]
        if len(values) != 1:
            raise ValueError('missing request binding')
        return values[0]
    if meta(a, 10).tobytes() != meta(b, 10).tobytes() or meta(a, 0).tobytes() != meta(b, 0).tobytes():
        raise ValueError('A0/B request identity differs')
    binding = meta(a, 10).reshape(-1)
    if binding[:4].tolist() != [256, 64, 2, 1024] or binding[4:8].tolist() != [48, 2560, 512, 10]:
        raise ValueError('requires original GEN256/prefill64/spec2/context1024 and Q8 geometry')
    prompt = meta(a, 0)
    if prompt.shape != (1, 38):
        raise ValueError('requires original 38 prompt IDs')
    for kind, shape in ((2, (2, 2560)), (3, (2, 512)), (4, (2, 10)), (5, (2, 10))):
        if one(a, kind).shape != shape or one(a, kind).tobytes() != one(b, kind).tobytes():
            raise ValueError(f'first-boundary A0/B input/router identity mismatch kind {kind}')
    owner_a, owner_b = one(a, 9).reshape(-1), one(b, 9).reshape(-1)
    if not np.all(owner_a == 3) or not np.all((owner_b == 3) | (owner_b == 4)):
        raise ValueError('requires the full CPU-backed first boundary')
    ids = one(a, 4).reshape(-1)
    if len(set(ids.tolist())) != 20 or int(np.count_nonzero(owner_b == 4)) != 13:
        raise ValueError('requires the inventoried 20 distinct entries / 13 CUDA replacements')
    peer_masks = []
    for peer in range(4):
        mask = one(b, 20+peer).reshape(-1)
        if mask.shape != (20,) or not np.isin(mask, [0, 1]).all():
            raise ValueError('invalid original-index peer mask')
        peer_masks.append(mask.astype('u1'))
    if not np.array_equal(np.sum(peer_masks, axis=0), owner_b == 4):
        raise ValueError('peer masks do not cover B CUDA ownership exactly once')
    # SRA1's producer writes record ordinal, then full-unsplit peer ordinal zero,
    # dense original-entry rows, then its ordered partial. Inspect only that prefix.
    if len(sra) != SRA_BYTES or hashlib.sha256(sra).hexdigest() != accepted_sha:
        raise ValueError('accepted SRA1 size/SHA256 mismatch; no recomputation fallback')
    if sra[:12] != b'SRA1'+struct.pack('<II', 0, 0):
        raise ValueError('SRA1 does not start with full record-0 rows')
    rows_end = 12+20*2560*4
    rows, partial = sra[12:rows_end], sra[rows_end:rows_end+2*2560*4]
    unions = [one(data, 12) for data in (a, b)]
    if any(v.shape != (2, 2560) for v in unions):
        raise ValueError('first union shape mismatch')
    packet = (b'SFB1'+struct.pack('<8I', 2, 10, 2560, 38, 256, 0, 0, 0)+bytes.fromhex(accepted_sha)
              +prompt.tobytes()+one(a, 2).tobytes()+ids.astype('<u4').tobytes()+one(a, 5).tobytes()
              +(owner_a == 3).astype('u1').tobytes()+(owner_b == 3).astype('u1').tobytes()
              +b''.join(m.tobytes() for m in peer_masks)+b''.join(v.tobytes() for v in unions)+rows+partial)
    if len(packet) != PACKET_BYTES:
        raise ValueError('first-boundary packet bound')
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a0_trace')
    parser.add_argument('b_trace')
    parser.add_argument('accepted_sra1')
    parser.add_argument('new_packet')
    args = parser.parse_args()
    try:
        sra_path = Path(args.accepted_sra1)
        if sra_path.stat().st_size != SRA_BYTES:
            raise ValueError('accepted SRA1 size mismatch; no recomputation fallback')
        packet = boundary(read(args.a0_trace), read(args.b_trace), sra_path.read_bytes())
        fd = os.open(args.new_packet, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(packet)
    except (OSError, ValueError) as e:
        parser.exit(2, f'first-boundary export refused: {e}\n')
    print(json.dumps({'packet_bytes': len(packet), 'packet_sha256': hashlib.sha256(packet).hexdigest(),
                      'accepted_sra1_sha256': SRA_SHA256, 'cpu_rows': 20, 'cuda_rows': 13,
                      'cuda_blobs': 13, 'cuda_blob_bytes': 67891200, 'historic_record_match': 'checked by replay before CUDA'}))


if __name__ == '__main__':
    main()
