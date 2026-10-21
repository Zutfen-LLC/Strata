#!/usr/bin/env python3
"""PROPOSAL: metadata only; execute only on approved ROLE_ENGINE_A, never ROLE_EDITOR.

Read only pinned metadata files. Never open inventory payload files.
Operational argument bindings and generated outputs remain private.
Create a NEW output directory; SUCCESS.json is written last. No candidate edits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct

CANDIDATE = "bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5"
TREE = "8e0cc2eda89f17a63182d1b273140288a1418b47"
MANIFEST_SHA = "8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318"
PROFILE_SHA = "8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf"
NL, NE, H, FF, BLOB, ROLE, N_GPU_B = 48, 512, 2560, 640, 5222400, 1740800, 3289
MASK = (1 << 64) - 1


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded(path, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= cap, "metadata type/size")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(st.st_size + 1)
        end = os.fstat(fd)
        need(len(data) == st.st_size and
             (st.st_size, st.st_mtime_ns, st.st_ctime_ns) ==
             (end.st_size, end.st_mtime_ns, end.st_ctime_ns), "metadata changed/short read")
        return data
    finally:
        os.close(fd)


def digest(data):
    a, b = 0xcbf29ce484222325, 0x9e3779b97f4a7c15
    for v in data:
        a = ((a ^ v) * 0x100000001b3) & MASK
    for v in reversed(data):
        b = ((b ^ v) * 0x100000001b3) & MASK
    return a, b


class Cursor:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, size):
        need(0 <= size <= len(self.data) - self.pos, "truncated metadata")
        out = self.data[self.pos:self.pos + size]
        self.pos += size
        return out

    def fields(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))


def profile_pairs(raw):
    need(len(raw) >= 24, "short STRP")
    magic, version, nl, ne, slots, nr = struct.unpack_from("<4s5I", raw)
    need((magic, version, nl, ne) == (b"STRP", 1, NL, NE), "STRP geometry/version")
    need(0 < nr <= slots and nr <= NL * NE, "STRP ranked count")
    pairs_end = 24 + 4 * nr
    need(len(raw) == pairs_end + 4 * NL * NE, "STRP trailing rank-table length")
    pairs = list(struct.iter_unpack("<HH", raw[24:pairs_end]))
    need(len(set(pairs)) == nr and all(l < NL and e < NE for l, e in pairs),
         "STRP duplicate/out-of-range pair")
    table = [-1] * (NL * NE)
    for rank, (l, e) in enumerate(pairs):
        table[l * NE + e] = rank
    need(tuple(table) == struct.unpack_from("<" + str(NL * NE) + "i", raw, pairs_end),
         "STRP inverse rank table mismatch")
    return pairs


def manifest_data(raw):
    c = Cursor(raw)
    need(c.take(8) == b"SWLOCAL1", "local manifest magic")
    geometry = c.fields("<6H")
    need(geometry == (NL, NE, H, FF, 8, 8), "local manifest geometry/layout")
    index_digest = c.fields("<2Q")
    nf, = c.fields("<I")
    need(1 <= nf <= 64, "local inventory count")
    files = []
    for _ in range(nf):
        length, = c.fields("<H")
        need(1 <= length <= 255, "local basename length")
        name = c.take(length).decode("ascii")
        size, = c.fields("<Q")
        need(re.fullmatch(r"[A-Za-z0-9_.-]+", name) and name not in (".", "..")
             and 0 < size < (1 << 63), "local basename/size")
        files.append((name, size))
    need(len({name for name, _ in files}) == nf, "duplicate inventory basename")
    layers, spans = [], [[] for _ in files]
    for _ in range(NL):
        ranges = []
        for _ in range(3):
            f, offset, size = c.fields("<IQQ")
            need(f < nf and size == ROLE and offset <= files[f][1] and
                 size * NE <= files[f][1] - offset, "invalid expert source range")
            spans[f].append((offset, offset + size * NE))
            ranges.append((f, offset, size))
        layers.append(ranges)
    for ranges in spans:
        ranges.sort()
        need(all(ranges[i - 1][1] <= ranges[i][0] for i in range(1, len(ranges))),
             "overlapping source tensor ranges")
    blobs = [c.fields("<2Q") for _ in range(NL * NE)]
    roles = [[c.fields("<2Q") for _ in range(3)] for _ in range(NL * NE)]
    need(c.pos == len(raw), "trailing local manifest bytes")
    return geometry, index_digest, files, layers, blobs, roles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--profile-sha256", required=True, help="FULL hash from verified Step A receipt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="NEW absolute output directory; must not exist")
    args = ap.parse_args()
    need(re.fullmatch(r"[0-9a-f]{64}", args.profile_sha256), "full profile SHA-256 required")
    need(args.profile_sha256 == PROFILE_SHA, "profile is not the approved pinned profile")
    out = Path(args.out)
    need(out.is_absolute() and not out.exists(), "output must be a NEW absolute directory")
    profile = bounded(args.profile, 1 << 20)
    manifest = bounded(args.manifest, 8 << 20)
    need(sha(profile) == args.profile_sha256, "profile SHA-256 mismatch")
    need(sha(manifest) == MANIFEST_SHA, "manifest SHA-256 mismatch")
    ranked = profile_pairs(profile)
    geometry, index_digest, files, layers, blobs, roles = manifest_data(manifest)
    primary, quota = set(), [0] * NL
    for l, e in ranked:
        if quota[l] < 22:
            primary.add((l, e))
            quota[l] += 1
    need(quota == [22] * NL and len(primary) == 1056, "primary is not 22/layer")
    remaining = [p for p in ranked if p not in primary]
    need(len(remaining) >= 2263, "helper would need unranked pairs")
    helper = set(remaining[:2263])
    universe = {(l, e) for l in range(NL) for e in range(NE)}
    cpu = universe - primary - helper
    need(not primary & helper and len(helper) == 2263 and len(cpu) == 21257,
         "local ownership/CPU complement count")
    seen = set(ranked)
    priority = ranked + [(l, e) for e in range(NE) for l in range(NL) if (l, e) not in seen]
    assigned_priority = [p for p in priority if p in cpu][:N_GPU_B]
    assigned = set(assigned_priority)
    uncovered = cpu - assigned
    execution = (primary, helper, assigned, uncovered)
    need(len(assigned) == N_GPU_B and assigned <= cpu and
         set.union(*execution) == universe and sum(map(len, execution)) == len(universe),
         "execution ownership is not disjoint/exhaustive")
    outputs, summaries = {}, {}
    for name, pairs in (("primary_A", primary), ("helper_A", helper), ("cpu-full_C", cpu),
                        ("assigned_B", assigned), ("cpu-uncovered_C", uncovered)):
        outputs[name + ".tsv"] = "".join(f"{l} {e}\n" for l, e in sorted(pairs)).encode("ascii")
        summaries[name] = {"count": len(pairs), "bytes": len(pairs) * BLOB,
                           "per_layer": [sum(l == i for l, e in pairs) for i in range(NL)]}
    outputs["assigned_B.priority.tsv"] = "".join(f"{l} {e}\n" for l, e in assigned_priority).encode("ascii")
    bitmap = bytearray(NL * 64)
    for l, e in cpu:
        bitmap[l * 64 + e // 8] |= 1 << (e % 8)
    outputs["cpu-full_C.bitmap"] = bytes(bitmap)
    identity = digest(manifest)
    layout = digest(b"Q8_0-GUD-1" + struct.pack("<4Q6H", *identity, *index_digest, *geometry))
    plan = bytearray(b"SGPUPLN1" + struct.pack("<Q6H2QI", identity[0], *geometry, *layout, N_GPU_B))
    closure = ["layer\texpert\trole\tfile\toffset\tlength\trole_digest_a\trole_digest_b\tblob_digest_a\tblob_digest_b\n"]
    for l, e in sorted(assigned):
        bd = blobs[l * NE + e]
        plan.extend(struct.pack("<HHQ2Q", l, e, BLOB, *bd))
        for role in range(3):
            f, offset, length = layers[l][role]
            rd = roles[l * NE + e][role]
            closure.append(f"{l}\t{e}\t{role}\t{files[f][0]}\t{offset + e * length}\t{length}\t"
                           f"{rd[0]:016x}\t{rd[1]:016x}\t{bd[0]:016x}\t{bd[1]:016x}\n")
    placement = digest(plan)
    plan.extend(struct.pack("<2Q", *placement))
    need(len(plan) == 64 + 28 * N_GPU_B and len(plan) <= 4 << 20, "GPU plan length")
    outputs["plan_B.expected.bin"] = bytes(plan)
    outputs["assigned_B.asset-closure.tsv"] = "".join(closure).encode("ascii")
    summary = {"candidate": CANDIDATE, "tree": TREE, "profile_sha256": sha(profile),
               "manifest_sha256": sha(manifest), "geometry": geometry, "sets": summaries,
               "profile_ranked_pairs": len(ranked), "unranked_pairs_total": NL * NE - len(seen),
               "helper_unranked_taken": 0,
               "selection": "primary first22/layer; helper first2263 nonprimary; GPU first3289 CPU-complement priority",
               "model_id": f"{identity[0]:016x}", "layout_digest_words": [f"{v:016x}" for v in layout],
               "placement_digest_words": [f"{v:016x}" for v in placement],
               "cpu_bitmap_digest_words": [f"{v:016x}" for v in digest(bitmap)],
               "gpu_workspace_bytes": 1598720, "gpu_reserve_mib": 4096,
               "historical_placement_proof": "NOT established by this metadata program",
               "weight_payload_bytes_read": 0, "weight_payload_bytes_transferred": 0}
    summary["artifact_sha256"] = {name: sha(data) for name, data in sorted(outputs.items())}
    outputs["ownership.json"] = (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode("ascii")
    need(sum(map(len, outputs.values())) <= 4 << 20, "metadata output budget")
    outputs["FILES.sha256"] = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(outputs.items())).encode("ascii")
    os.mkdir(out, 0o700)
    directory = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        def publish(name, data):
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=directory)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        for name, data in sorted(outputs.items()):
            publish(name, data)
        publish("SUCCESS.json", (json.dumps({"files_manifest_sha256": sha(outputs["FILES.sha256"]),
                "plan_sha256": sha(plan), "count_B": N_GPU_B, "bytes_B": N_GPU_B * BLOB,
                "cpu_full_bytes": len(cpu) * BLOB, "metadata_only": True}, sort_keys=True) + "\n").encode("ascii"))
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"output": str(out), "count_B": N_GPU_B, "bytes_B": N_GPU_B * BLOB,
                      "plan_sha256": sha(plan), "weight_payload_bytes_read": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
