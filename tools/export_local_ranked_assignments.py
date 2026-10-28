"""Export two 480-expert assignment files from a pinned native STRP v1 ranking.

Metadata only. Source order is a placement hypothesis, not measured workload heat.
Use strata-gpu-manifest plan to bind these assignments to the local model manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

LAYERS, EXPERTS, PER_DIE = 48, 512, 480
MAX_BYTES = 1 << 20


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_profile(raw):
    """Read the complete writer format, including its rank table (not frequencies).

    Both make_profile.write_profile and native write_expert_profile store the last
    occurrence's index in the table. Selection below keeps the first occurrence.
    Reject invalid pairs anywhere, including outside the selected prefix.
    """
    if not 24 <= len(raw) <= MAX_BYTES:
        raise ValueError('profile must contain a header and be at most 1 MiB')
    if raw[:4] != b'STRP':
        raise ValueError('profile magic must be STRP')
    version, layers, experts, slots, count = struct.unpack_from('<5I', raw, 4)
    if (version, layers, experts) != (1, LAYERS, EXPERTS):
        raise ValueError('requires native version 1, 48 layers, 512 experts')
    if count > slots:
        raise ValueError('ranked count exceeds slots')
    table_offset = 24 + 4 * count
    if len(raw) != table_offset + 4 * LAYERS * EXPERTS:
        raise ValueError('requires complete pair list and rank table; unexpected length')
    ranked = list(struct.iter_unpack('<HH', raw[24:table_offset]))
    expected = [-1] * (LAYERS * EXPERTS)
    for rank, (layer, expert) in enumerate(ranked):
        if layer >= LAYERS or expert >= EXPERTS:
            raise ValueError(f'out-of-range pair at rank {rank}: {layer} {expert}')
        expected[layer * EXPERTS + expert] = rank
    table = struct.unpack_from(f'<{LAYERS * EXPERTS}i', raw, table_offset)
    if tuple(expected) != table:
        raise ValueError('rank table disagrees with ordered pairs')
    return ranked, slots


def partition(ranked):
    seen, selected = set(), []
    for rank, pair in enumerate(ranked):
        if pair in seen:
            continue
        seen.add(pair)
        if len(selected) < 2 * PER_DIE:
            selected.append((rank, len(selected) % 2, *pair))
    if len(selected) != 2 * PER_DIE:
        raise ValueError('profile has fewer than 960 distinct pairs')
    assignments = [sorted((l, e) for _, owner, l, e in selected if owner == die)
                   for die in range(2)]
    return assignments, selected, len(seen)


def export(profile, expected_sha, out):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha):
        raise ValueError('profile SHA256 must be 64 lowercase hex characters')
    # Bounded read; parse and validate everything before creating any output.
    with Path(profile).open('rb') as source:
        raw = source.read(MAX_BYTES + 1)
    if sha(raw) != expected_sha:
        raise ValueError('profile SHA256 mismatch')
    ranked, slots = parse_profile(raw)
    assignments, selected, distinct = partition(ranked)
    files = {f'assignments-{die}.txt': ''.join(f'{l} {e}\n' for l, e in pairs).encode('ascii')
             for die, pairs in enumerate(assignments)}
    files['selected-ranks.tsv'] = ('source_rank\tdie\tlayer\texpert\n' + ''.join(
        f'{rank}\t{die}\t{l}\t{e}\n' for rank, die, l, e in selected)).encode('ascii')
    report = dict(format='STRP-v1', layers=LAYERS, experts_per_layer=EXPERTS,
                  profile_sha256=expected_sha, profile_bytes=len(raw), profile_slots=slots,
                  ranked_pairs=len(ranked), distinct_pairs=distinct,
                  duplicate_pairs=len(ranked)-distinct, experts_per_die=[PER_DIE, PER_DIE],
                  selection='first 960 distinct source-ranked pairs; alternate owners before sorting',
                  interpretation='source rank only; workload heat and transfer are unproven',
                  source_weight_read=0,
                  output_sha256={name: sha(data) for name, data in sorted(files.items())})
    files['assignments.json'] = (json.dumps(report, indent=2, sort_keys=True)+'\n').encode('utf-8')
    root = Path(out)
    root.mkdir(exist_ok=False)
    for name, data in files.items():
        with (root/name).open('xb') as dest:
            dest.write(data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--profile-sha256', required=True)
    parser.add_argument('--out', required=True, help='new directory; never overwrite an existing placement')
    args = parser.parse_args()
    try:
        print(json.dumps(export(args.profile, args.profile_sha256, args.out), indent=2, sort_keys=True))
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+'\n')


if __name__ == '__main__':
    main()
