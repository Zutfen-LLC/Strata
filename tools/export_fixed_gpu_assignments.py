"""Metadata-only export for the fixed 48x512 Q8_0 primary + one-helper experiment.

Matches generate.cpp's non-per-layer profile fill and RemoteExperts::open's
single fixed-helper scan. Refuses other placement shapes; starts no processes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from make_profile import read_profile

EXPERT_BYTES = 5222400


def export(launch, profile, expected_sha, out):
    argv = launch['argv']
    def value(flag):
        if argv.count(flag) != 1: raise ValueError('expected one ' + flag)
        return argv[argv.index(flag) + 1]
    if value('--expert-cache') != '960' or value('--expert-cache-device1') != '1378':
        raise ValueError('requires fixed primary960/helper1378')
    forbidden = ('--expert-cache-per-layer', '--layer-split', '--peer-device',
                 '--expert-cache-device2', '--expert-cache-device3', '--expert-profile-save')
    if any(flag in argv for flag in forbidden): raise ValueError('unsupported placement transform/tier')
    profile = Path(profile)
    if Path(value('--expert-profile')).resolve() != profile.resolve(): raise ValueError('profile path differs from launch')
    if profile.stat().st_size > (1 << 20): raise ValueError("profile exceeds bounded metadata size")
    raw = profile.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha: raise ValueError('profile SHA256 mismatch')
    if len(raw) < 24: raise ValueError('truncated profile')
    version, layers, experts, slots, count = struct.unpack_from('<5I', raw, 4)
    if (version, layers, experts, count) != (1, 48, 512, 48*512) or count > slots:
        raise ValueError('requires complete version1 48x512 ranking')
    ranked = read_profile(profile)
    if len(set(ranked)) != count or set(ranked) != {(l,e) for l in range(48) for e in range(512)}:
        raise ValueError('duplicate/missing/out-of-range profile pair')
    # Primary fixed admission; helper skips primary and takes its first N unique pairs.
    local = set(ranked[:960+1378]); cpu = [p for p in ranked if p not in local]
    if len(cpu) != 22238: raise ValueError('CPU backing count mismatch')
    root = Path(out); root.mkdir(exist_ok=False)
    def write(name, pairs):
        (root/name).write_text(''.join(f'{l} {e}\n' for l,e in pairs))
    write('local.txt', ranked[:2338]); write('cpu.txt', cpu)
    bitmap = bytearray(48*64)
    for l,e in cpu: bitmap[l*64+e//8] |= 1 << (e%8)
    (root/'cpu.bitmap').write_bytes(bitmap)
    rows=[]; at=0
    for name,gib in [('01',9),('03a',9),('03b',9),('04',18)]:
        cap=(gib*(1<<30))//EXPERT_BYTES; selected=cpu[at:at+cap];at+=cap
        if len(selected)!=cap: raise ValueError('assignments exceed CPU backing')
        write(name+'.txt',selected)
        rows.append(dict(peer=name,experts=cap,assigned_bytes=cap*EXPERT_BYTES,payload_cap_bytes=gib*(1<<30)))
    report=dict(profile_sha256=expected_sha,local_experts=2338,cpu_experts=len(cpu),
                cpu_bytes=len(cpu)*EXPERT_BYTES,cpu_bitmap_sha256=hashlib.sha256(bitmap).hexdigest(),peers=rows,
                source_weight_read=0,selection='fixed profile prefix; disjoint contiguous CPU-rank slices')
    (root/'assignments.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--profile-sha256',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    try: print(json.dumps(export(json.loads(Path(a.launch).read_text()),a.profile,a.profile_sha256,a.out),indent=2))
    except (ValueError,OSError,struct.error) as e: p.exit(1,str(e)+'\n')
