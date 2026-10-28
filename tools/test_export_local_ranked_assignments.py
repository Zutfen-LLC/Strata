import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
