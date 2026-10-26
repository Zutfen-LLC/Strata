#!/usr/bin/env python3
"""Compare bounded first-request traces. Reports magnitude, never causal attribution."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

LIMIT = 64 << 20
NAMES = {0: 'request_ids', 1: 'window_ids', 2: 'moe_input', 3: 'router_scores',
         4: 'router_ids', 5: 'router_weights', 6: 'post_moe', 7: 'residual_before_moe',
         8: 'residual_inject', 9: 'ownership', 10: 'request_binding', 11: 'first_logits', 12: 'network_union'}
# Residual is retained only for the final prompt token. The MoE write is deferred:
# residual_before_moe + inject + post_moe describe that boundary, not a new reduction.
ORDER = {7: 0, 8: 1, 2: 2, 3: 3, 4: 4, 5: 5, 12: 6, 6: 7, 11: 8}


def read(path):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('trace exceeds 64 MiB')
    records = {}
    with path.open('rb') as f:
        if f.read(8) != b'STRFT01\0':
            raise ValueError('unsupported trace magic / endian')
        while raw := f.read(64):
            if len(raw) != 64:
                raise ValueError('truncated header')
            kind, window, pos, layer, tb, rows, cols, size = struct.unpack('<8q', raw)
            if kind not in NAMES and not 20 <= kind < 28:
                raise ValueError(f'unknown record {kind}')
            if rows < 1 or cols < 1 or size != rows * cols * 4 or size > LIMIT:
                raise ValueError('invalid record dimensions')
            payload = f.read(size)
            if len(payload) != size:
                raise ValueError('truncated payload')
            key = (window, pos, layer, kind, tb, rows, cols)
            if key in records:
                raise ValueError(f'duplicate record {key}')
            dtype = '<i4' if kind in (0, 1, 4, 9, 10) or kind >= 20 else '<f4'
            records[key] = np.frombuffer(payload, dtype=dtype).reshape(rows, cols)
    if not records or key[3] != 11:
        raise ValueError('unsealed trace: final first-logit record missing')
    requests = [v for k, v in records.items() if k[3] == 0]
    windows = [k for k in records if k[3] == 1]
    bindings = [v.reshape(-1) for k, v in records.items() if k[3] == 10]
    if len(bindings) != 1 or len(bindings[0]) != 9:
        raise ValueError('missing request/geometry binding')
    _, _, _, _, n_layers, embd, experts, topk, channels = map(int, bindings[0])
    if min(n_layers, embd, experts, topk, channels) < 1:
        raise ValueError('invalid model geometry')
    if len(requests) != 1 or not windows or len(windows) > 32:
        raise ValueError('request/window identity missing')
    for w in windows:
        layers = {k[2] for k in records if k[:2] == w[:2] and k[3] == 2}
        if not layers or layers != set(range(n_layers)):
            raise ValueError('missing input layers')
        for layer in layers:
            for kind in (2, 3, 4, 5, 6):
                matches = [k for k in records if k[:3] == (w[0], w[1], layer) and k[3] == kind]
                if len(matches) != 1 or matches[0][4] != 0 or matches[0][5] != w[6]:
                    raise ValueError('incomplete window/layer tensors')
                width = {2: embd, 3: experts, 4: topk, 5: topk, 6: embd}[kind]
                if matches[0][6] != width:
                    raise ValueError('tensor width differs from model geometry')
            for kind in (9, 12):
                slices = sorted(k for k in records if k[:3] == (w[0], w[1], layer) and k[3] == kind)
                cursor = 0
                for k in slices:
                    if k[4] != cursor or k[6] != (topk if kind == 9 else embd):
                        raise ValueError('ownership/network union slices overlap or leave a gap')
                    cursor += k[5]
                if cursor != w[6]:
                    raise ValueError('ownership/network union slice missing')
    # Every prompt position was actually fed, with its original token identity.
    prompt = requests[0].reshape(-1)
    next_pos = 0
    for w in sorted(windows):
        if w[1] != next_pos or w[1] >= len(prompt):
            raise ValueError('noncontiguous prompt windows')
        values = records[w].reshape(-1)
        n = min(len(values), len(prompt)-w[1])
        if not np.array_equal(values[:n], prompt[w[1]:w[1]+n]):
            raise ValueError('window token IDs differ from original prompt')
        next_pos += len(values)
    if next_pos < len(prompt):
        raise ValueError('trace ended before the final prompt token')
    last = max(windows)
    target = len(prompt)-1-last[1]
    for layer in range(n_layers):
        for kind, width in ((7, channels*embd), (8, channels)):
            k = (last[0], last[1], layer, kind, target, 1, width)
            if k not in records:
                raise ValueError('final prompt residual missing')
    return records


def magnitude(a, b):
    aa, bb = a.astype(np.float64), b.astype(np.float64)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        return {'maxabs': None, 'rel_l2': None, 'nonfinite': True}
    delta = bb-aa
    norm = float(np.linalg.norm(aa))
    diff = float(np.linalg.norm(delta))
    maximum = float(np.max(np.abs(delta)))
    index = np.unravel_index(int(np.argmax(np.abs(delta))), delta.shape)
    return {'maxabs': maximum, 'rel_l2': diff/norm if norm else (0.0 if diff == 0 else None),
            'zero_reference': norm == 0, 'max_index': [int(i) for i in index]}


def compare(a, b):
    # Ownership changes are the experiment; numerical/route/window alignment must match.
    numeric_a = {k: v for k, v in a.items() if k[3] != 9 and k[3] < 20}
    numeric_b = {k: v for k, v in b.items() if k[3] != 9 and k[3] < 20}
    if numeric_a.keys() != numeric_b.keys():
        raise ValueError('trace stage/window/slice alignment differs')
    for k in numeric_a:
        if k[3] in (0, 10) and not np.array_equal(a[k], b[k]):
            raise ValueError('original prompt IDs differ or request binding differs')
    token_changes = {k[0]: np.flatnonzero(a[k].reshape(-1) != b[k].reshape(-1)).tolist()
                     for k in numeric_a if k[3] == 1}
    progression, earliest, routing = [], None, None
    order = lambda k: (k[0], k[2] if k[2] >= 0 else 1 << 30, ORDER.get(k[3], -1), k[4])
    for k in sorted(numeric_a, key=order):
        if k[3] in (0, 1, 10):
            continue
        x, y = a[k], b[k]
        identical = x.tobytes() == y.tobytes()
        item = dict(window=k[0], pos=k[1], layer=k[2], stage=NAMES[k[3]],
                    token_begin=k[4], rows=k[5], bitexact=identical,
                    different_input_token_rows=token_changes.get(k[0], []))
        if k[3] == 4:
            item['changed_entries'] = int(np.count_nonzero(x != y))
            if routing is None and item['changed_entries']:
                routing = item.copy()
        else:
            item.update(magnitude(x, y))
        if earliest is None and not identical:
            earliest = item.copy()
        progression.append(item)
    maxima = {}
    for item in progression:
        if 'maxabs' not in item:
            continue
        name = item['stage']
        current = maxima.get(name)
        if current is None or item['maxabs'] is None or (current['maxabs'] is not None and item['maxabs'] > current['maxabs']):
            maxima[name] = item
    ownership = []
    for k in sorted(set(a) | set(b)):
        if k[3] == 9:
            ownership.append(dict(window=k[0], pos=k[1], layer=k[2], token_begin=k[4],
                                  changed_entries=int(np.count_nonzero(a[k] != b[k])) if k in a and k in b else None))
    return {'request_binding': [v.reshape(-1).tolist() for k, v in a.items() if k[3] == 10],
            'earliest_difference': earliest, 'first_routing_difference': routing,
            'window_token_changes': token_changes,
            'maximum_by_stage': maxima, 'progression': progression, 'ownership_changes': ownership,
            'interpretation': 'Earliest retained difference is a boundary observation, not proof of cause. '
                              'Inspect error growth and routing together. Expert rows are not retained.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a0')
    parser.add_argument('b')
    parser.add_argument('--output', required=True)
    parser.add_argument('--a0-logits', required=True)
    parser.add_argument('--b-logits', required=True)
    args = parser.parse_args()
    try:
        a, b = read(args.a0), read(args.b)
        for data, logits in ((a, args.a0_logits), (b, args.b_logits)):
            final = [v for k, v in data.items() if k[3] == 11]
            if len(final) != 1 or final[0].tobytes() != Path(logits).read_bytes():
                raise ValueError('sealed logits differ from the separate first-logit artifact')
        report = compare(a, b)
    except (ValueError, OSError) as e:
        parser.exit(2, f'trace comparison refused: {e}\n')
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for key in ('earliest_difference', 'first_routing_difference', 'maximum_by_stage', 'interpretation'):
        print(f'{key}: {json.dumps(report[key], allow_nan=False)}')


if __name__ == '__main__':
    main()
