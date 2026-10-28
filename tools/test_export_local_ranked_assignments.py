import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import export_local_ranked_assignments as M
from make_profile import write_profile


class LocalRankedAssignments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profile = self.root/'profile.bin'
        # Reverse order ensures sorting before owner selection would fail the test.
        self.ranked = [(l, e) for e in reversed(range(25)) for l in reversed(range(48))]

    def raw(self, pairs=None):
        write_profile(self.profile, self.ranked if pairs is None else pairs)
        return self.profile.read_bytes()

    def test_real_writer_format_and_partition_preservation(self):
        raw = self.raw()
        self.assertEqual(M.parse_profile(raw), (self.ranked, len(self.ranked)))
        a, selected, distinct = M.partition(M.parse_profile(raw)[0])
        self.assertEqual(distinct, 1200)
        self.assertEqual([len(p) for p in a], [480, 480])
        self.assertEqual(a[0], sorted(self.ranked[:960:2]))
        self.assertEqual(a[1], sorted(self.ranked[1:960:2]))
        self.assertFalse(set(a[0]) & set(a[1]))
        self.assertEqual(set(a[0]) | set(a[1]), set(self.ranked[:960]))
        self.assertEqual([(l, e) for _, _, l, e in selected], self.ranked[:960])

    def test_duplicates_keep_first_rank_and_do_not_use_slots(self):
        pairs = self.ranked[:3] + self.ranked + [self.ranked[0]]
        a, selected, distinct = M.partition(M.parse_profile(self.raw(pairs))[0])
        self.assertEqual(distinct, 1200)
        self.assertEqual(a[0], sorted(self.ranked[:960:2]))
        self.assertEqual(a[1], sorted(self.ranked[1:960:2]))
        self.assertEqual(selected[3], (6, 1, *self.ranked[3]))
        self.assertEqual(M.partition(M.parse_profile(self.raw(self.ranked[:960]))[0])[2], 960)
        with self.assertRaisesRegex(ValueError, 'fewer than 960'):
            M.partition(M.parse_profile(self.raw(self.ranked[:959] + self.ranked[:1]))[0])

    def test_malformed_headers_and_lengths(self):
        raw = self.raw()
        for bad in (b'', raw[:23], b'FAIL'+raw[4:], raw[:-1], raw+b'\0', b'0'*(M.MAX_BYTES+1)):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                M.parse_profile(bad)
        for offset, value in ((4, 2), (8, 47), (12, 256), (16, 1199), (20, 1201)):
            bad = bytearray(raw)
            struct.pack_into('<I', bad, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                M.parse_profile(bad)

    def test_bounds_checked_even_beyond_selected_prefix_and_table_verified(self):
        raw = self.raw()
        for pair in ((48, 0), (0, 512), (65535, 65535)):
            bad = bytearray(raw)
            struct.pack_into('<HH', bad, 24+4*1100, *pair)
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, 'out-of-range'):
                M.parse_profile(bad)
        bad = bytearray(raw)
        struct.pack_into('<i', bad, 24+4*len(self.ranked), 0)
        with self.assertRaisesRegex(ValueError, 'rank table'):
            M.parse_profile(bad)

    def test_deterministic_files_hash_pinning_and_no_overwrite(self):
        raw = self.raw(self.ranked[:2] + self.ranked)
        digest = hashlib.sha256(raw).hexdigest()
        out = self.root/'out'
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            M.export(self.profile, '0'*64, out)
        self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError, 'lowercase hex'):
            M.export(self.profile, digest.upper(), out)
        report = M.export(self.profile, digest, out)
        self.assertEqual(report['duplicate_pairs'], 2)
        self.assertEqual(report, M.export(self.profile, digest, self.root/'second'))
        self.assertEqual(json.loads((out/'assignments.json').read_text()), report)
        for p in out.iterdir():
            self.assertEqual(p.read_bytes(), (self.root/'second'/p.name).read_bytes())
            if p.name in report['output_sha256']:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), report['output_sha256'][p.name])
        with self.assertRaises(FileExistsError):
            M.export(self.profile, digest, out)
        self.profile.write_bytes(raw[:-1])
        with self.assertRaises(ValueError):
            M.export(self.profile, hashlib.sha256(raw[:-1]).hexdigest(), self.root/'invalid')
        self.assertFalse((self.root/'invalid').exists())

    def test_global_default_files_remain_byte_identical(self):
        raw = self.raw()
        out = self.root/'default'
        M.export(self.profile, M.sha(raw), out)
        # Recorded from the original PR46 exporter before adding the option.
        expected = {
            'assignments-0.txt': 'b7bb46f4ded96bf0cac0b0171ddc4eb4537fa3f24ff6374d78f8008305dad47c',
            'assignments-1.txt': '571a1fecfbe5027b757ca540fb8687da7083bd9259b90a1c506ef33704fc72ac',
            'assignments.json': '299f3646c207d0a9e87b40475df3cbc7e469d4503ac411b261d6ab1d48682262',
            'selected-ranks.tsv': '56008b8633122962def2b809cba207e80a4219ccb62c18169010ca7eca672b3a',
        }
        self.assertEqual({p.name: M.sha(p.read_bytes()) for p in out.iterdir()}, expected)

    def test_per_layer_quota_uses_layer_rank_before_serialization(self):
        pairs = self.ranked[:17] + self.ranked + self.ranked[:3]
        a, selected, distinct = M.partition(M.parse_profile(self.raw(pairs))[0], 2)
        self.assertEqual(distinct, 1200)
        self.assertEqual([len(p) for p in a], [96, 96])
        self.assertFalse(set(a[0]) & set(a[1]))
        for layer in range(48):
            self.assertEqual([e for l, e in a[0] if l == layer], [22, 24])
            self.assertEqual([e for l, e in a[1] if l == layer], [21, 23])
        self.assertEqual(set(a[0]) | set(a[1]), {(l, e) for l in range(48) for e in range(21, 25)})
        self.assertEqual([rank for rank, _, _, _ in selected], sorted(rank for rank, _, _, _ in selected))
        self.assertEqual([(l, e) for _, _, l, e in selected], list(dict.fromkeys(pairs))[:192])

    def test_per_layer_23_counts_hash_pinning_and_determinism(self):
        # Layer-grouped source: a global prefix would miss most layers.
        pairs = [(l, e) for l in reversed(range(48)) for e in reversed(range(50))]
        raw = self.raw(pairs[:11] + pairs)
        digest = M.sha(raw)
        out = self.root/'per-layer'
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            M.export(self.profile, '0'*64, out, 23)
        self.assertFalse(out.exists())
        report = M.export(self.profile, digest, out, 23)
        self.assertEqual(report['experts_per_die'], [1104, 1104])
        self.assertEqual(report['per_device_layer_counts'], [[23]*48, [23]*48])
        self.assertEqual(report['per_layer_per_device'], 23)
        self.assertEqual(report['duplicate_pairs'], 11)
        self.assertEqual(report, M.export(self.profile, digest, self.root/'repeat', 23))
        for p in out.iterdir():
            self.assertEqual(p.read_bytes(), (self.root/'repeat'/p.name).read_bytes())
        a, b = [set(tuple(map(int, row.split())) for row in (out/f'assignments-{die}.txt').read_text().splitlines())
                for die in range(2)]
        self.assertFalse(a & b)
        self.assertEqual(a | b, {(l, e) for l in range(48) for e in range(4, 50)})

    def test_per_layer_bounds_missing_quota_geometry_and_cli(self):
        raw = self.raw()
        for n in (0, -1, 257, 1.5, True, '23'):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, 'integer in 1..256'):
                M.export(self.profile, M.sha(raw), self.root/'invalid-n', n)
            self.assertFalse((self.root/'invalid-n').exists())
        pairs = [(l, e) for l in range(48) for e in range(512)]
        assignments, _, _ = M.partition(M.parse_profile(self.raw(pairs))[0], 256)
        self.assertEqual([len(p) for p in assignments], [12288, 12288])
        short = [p for p in pairs if p[0] != 31 or p[1] < 45]
        raw = self.raw(short)
        with self.assertRaisesRegex(ValueError, 'layer 31 has fewer than 46'):
            M.export(self.profile, M.sha(raw), self.root/'short', 23)
        self.assertFalse((self.root/'short').exists())
        bad = bytearray(raw)
        struct.pack_into('<I', bad, 8, 47)
        self.profile.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, '48 layers'):
            M.export(self.profile, M.sha(bad), self.root/'wrong-geometry', 23)
        self.assertFalse((self.root/'wrong-geometry').exists())
        raw = self.raw(pairs)
        argv = ['export', '--profile', str(self.profile), '--profile-sha256', M.sha(raw),
                '--out', str(self.root/'cli'), '--per-layer-per-device', '23']
        with patch('sys.argv', argv), patch('builtins.print'):
            M.main()
        self.assertEqual(json.loads((self.root/'cli/assignments.json').read_text())['experts_per_die'], [1104, 1104])


if __name__ == '__main__':
    unittest.main()
