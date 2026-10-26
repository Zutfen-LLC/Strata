"""Synthetic export checks; no real traces, rows, weights, devices or services."""
import hashlib
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
import export_first_boundary as export


class BoundaryExportTest(unittest.TestCase):
    def fixture(self):
        a = {(-1, 0, -1, 0, 0, 1, 38): np.arange(38, dtype='<i4').reshape(1, 38),
             (-1, 0, -1, 10, 0, 1, 9): np.array([[256, 64, 2, 1024, 48, 2560, 512, 10, 4]], dtype='<i4')}
        for kind, shape in ((2, (2, 2560)), (3, (2, 512)), (4, (2, 10)), (5, (2, 10)),
                            (9, (2, 10)), (12, (2, 2560))):
            dtype = '<i4' if kind in (4, 9) else '<f4'
            value = np.ones(shape, dtype=dtype)
            if kind == 4:
                value[:] = np.arange(20).reshape(2, 10)
            if kind == 9:
                value[:] = 3
            a[(0, 0, 0, kind, 0, *shape)] = value
        b = {k: v.copy() for k, v in a.items()}
        owner = b[(0, 0, 0, 9, 0, 2, 10)].reshape(-1)
        owner[:13] = 4
        for peer in range(4):
            mask = np.zeros((2, 10), dtype='<i4')
            mask.reshape(-1)[peer:13:4] = 1
            b[(0, 0, 0, 20+peer, 0, 2, 10)] = mask
        sra = bytearray(export.SRA_BYTES)
        sra[:12] = b'SRA1'+struct.pack('<II', 0, 0)
        return a, b, bytes(sra)

    def test_exact_fixed_packet_and_pin(self):
        a, b, sra = self.fixture()
        sha = hashlib.sha256(sra).hexdigest()
        with patch.object(export, 'SRA_SHA256', sha):
            packet = export.boundary(a, b, sra)
        self.assertEqual(len(packet), 287220)
        self.assertEqual(struct.unpack('<8I', packet[4:36]), (2, 10, 2560, 38, 256, 0, 0, 0))
        self.assertEqual(packet[36:68], bytes.fromhex(sha))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            export.boundary(a, b, sra)  # The CLI always requires the real accepted pin.

    def test_identity_and_ownership_refuse_before_reuse(self):
        for kind in (2, 3, 4, 5):
            with self.subTest(kind=kind):
                a, b, sra = self.fixture()
                key = next(k for k in b if k[3] == kind)
                b[key].reshape(-1)[0] += 1
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    export.boundary(a, b, sra)
        a, b, sra = self.fixture()
        b[(0, 0, 0, 20, 0, 2, 10)].reshape(-1)[1] = 1
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            export.boundary(a, b, sra)
        a, b, sra = self.fixture()
        with self.assertRaisesRegex(ValueError, 'start with full record-0'):
            altered = b'BAD!'+sra[4:]
            with patch.object(export, 'SRA_SHA256', hashlib.sha256(altered).hexdigest()):
                export.boundary(a, b, altered)


if __name__ == '__main__':
    unittest.main()
