"""Tests for tools/check_isa_guard.py's policy (#795): which functions may hold wider-ISA instructions.  No binary, no GPU.

    python -m unittest tools.test_check_isa_guard
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_isa_guard as G  # noqa: E402


class WideInstruction(unittest.TestCase):
    def test_evex_prefix_is_wide(self):
        self.assertTrue(G.is_wide(0x62, "vmovups", "zmm0, zmmword ptr [rcx]"))
        self.assertTrue(G.is_wide(0x62, "vpmullq", "xmm0, xmm0, xmm6"))      # EVEX with xmm0-15 operands only

    def test_vex_avx2_is_not(self):
        self.assertFalse(G.is_wide(0xC4, "vpinsrd", "xmm1, xmm1, dword ptr [r13 + r9*4 + 0x405fc0], 3"))
        self.assertFalse(G.is_wide(0xC5, "vmovdqu", "ymm0, ymmword ptr [r14]"))

    def test_avx_vnni_and_registers_above_15(self):
        self.assertTrue(G.is_wide(0xC4, "vpdpbusd", "ymm0, ymm1, ymm2"))
        self.assertTrue(G.is_wide(0xC5, "vpxor", "ymm17, ymm17, ymm17"))
        self.assertTrue(G.is_wide(0xC5, "kmovb", "k1, eax"))


class Policy(unittest.TestCase):
    def test_kernel_in_an_anonymous_namespace_of_a_wide_file_is_fine(self):
        self.assertIsNone(G.judge("??$gu_rows@$0BA@$00@?A0xa1d599f5@cpu@kernels@strata@@YAXXZ",
                                  "strata_kernels_cpu:iq_avx512.cpp.obj", "", False, False, False))
        self.assertIsNone(G.judge("strata::kernels::cpu::(anonymous namespace)::gu_rows<18, 1>(int)", None, "", False,
                                  False, True))

    def test_listed_public_entry_point_is_fine(self):
        self.assertIsNone(G.judge("?act_quant_q8_1@cpu@kernels@strata@@YAXPEBMHAEAUActQ@123@@Z",
                                  "strata_kernels_cpu:expert.cpp.obj", "", False, False, False))

    def test_the_probe_must_not_be_wide(self):
        self.assertIsNotNone(G.judge("?cpu_features@cpu@kernels@strata@@YA?AUCpuFeatures@123@XZ",
                                     "strata_kernels_cpu:expert.cpp.obj", "", False, False, False))
        self.assertIsNotNone(G.judge("strata::kernels::cpu::cpu_features()", None, "", False, False, True))
        self.assertIsNotNone(G.judge("strata::kernels::cpu::iq512_supported(int)", None, "", False, False, True))

    def test_weak_symbol_is_a_violation_even_in_a_wide_file(self):
        why = G.judge("std::vector<float, std::allocator<float> >::_M_realloc_insert(float const&)", None, "", False,
                      True, True)
        self.assertIn("weak", why)

    def test_wide_code_in_another_object_is_a_violation(self):
        self.assertIsNotNone(G.judge("?f@@YAXXZ", "strata_kernels_cpu:pool.cpp.obj", "", False, False, False))

    def test_msvc_vectorizer_guard_is_accepted(self):
        self.assertIsNone(G.judge("?bench@@YAHXZ", "iq_avx2_parity.cpp.obj", "", True, False, False))


class TqPolicy(unittest.TestCase):
    """Deterministic nm/objdump text exercises the real ELF parser and policy, with no tools/binary."""
    entry = "strata::kernels::cpu::tq_detail::tq_rows_avx512(int)"
    helper = "strata::kernels::cpu::tq_detail::(anonymous namespace)::rows_t<8>(int)"

    def elf_result(self, name, *, obj="tq_avx512.cpp.o", weak=False,
                   instruction="62 f1 fd 08 6e c0\tvmovq %rax,%xmm0", dis_name=None):
        # One genuine wide (EVEX) instruction is enough; symbol classification is independent of count.
        nm = f"{obj}:\n00000000 {'W' if weak else 'T'} {name}\n"
        dis = f"{obj}: file format elf64-x86-64\n<{dis_name or name}>:\n\t{instruction}\n"

        def output(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertEqual(command[-1], "fixture.a")
            self.assertIn(command[0], ("nm", "objdump"))
            return SimpleNamespace(stdout=nm if command[0] == "nm" else dis)

        captured = io.StringIO()
        with patch.object(G.shutil, "which", return_value="/fixture/tool"), \
                patch.object(G.subprocess, "run", side_effect=output), \
                contextlib.redirect_stdout(captured):
            result = G.run_elf("fixture.a")
        return result, captured.getvalue()

    def test_tq_anonymous_helper_in_registered_object_is_accepted(self):
        result, output = self.elf_result(self.helper)
        self.assertEqual(result, 0, output)
        self.assertIn("1 hold wide-ISA instructions, 0 not allowed", output)

    def test_tq_public_wrapper_is_accepted_when_compiler_inlines_into_it(self):
        result, output = self.elf_result(self.entry)
        self.assertEqual(result, 0, output)
        self.assertIn("1 hold wide-ISA instructions, 0 not allowed", output)

    def test_tq_pe_entry_uses_the_same_narrow_policy(self):
        name = "?tq_rows_avx512@tq_detail@cpu@kernels@strata@@YAXXZ"
        self.assertIsNone(G.judge(name, "strata_kernels_cpu:tq_avx512.cpp.obj", "", False, False, False))
        self.assertIn("weak / COMDAT", G.judge(name, "strata_kernels_cpu:tq_avx512.cpp.obj", "",
                                             False, True, False))

    def test_weak_tq_entry_helper_and_entry_clone_are_rejected(self):
        for name, dis_name in ((self.entry, None), (self.helper, None),
                               (self.entry, self.entry + " [clone .isra.0]")):
            with self.subTest(name=name, dis_name=dis_name):
                result, output = self.elf_result(name, weak=True, dis_name=dis_name)
                self.assertEqual(result, 1, output)
                self.assertIn("weak / COMDAT symbol", output)

    def test_unlisted_public_helper_in_tq_object_is_still_rejected(self):
        result, output = self.elf_result("strata::kernels::cpu::tq_detail::unrelated_helper(int)")
        self.assertEqual(result, 1, output)
        self.assertIn("unlisted public function", output)

    def test_tq_entry_in_unrelated_object_is_still_rejected(self):
        result, output = self.elf_result(self.entry, obj="pool.cpp.o")
        self.assertEqual(result, 1, output)
        self.assertIn("outside the wide-ISA translation units", output)

    def test_tq_static_initializers_remain_forbidden_for_evex_and_vex(self):
        for instruction in ("62 f1 fd 08 6e c0\tvmovq %rax,%xmm0", "c5 fe 6f c1\tvmovdqu %ymm1,%ymm0"):
            with self.subTest(instruction=instruction):
                result, output = self.elf_result("_GLOBAL__sub_I_tq_fixture", instruction=instruction)
                self.assertEqual(result, 1, output)
                # The VEX-only case is not wide according to is_wide. It must fail the initializer
                # path through INIT_FREE_OBJECTS, not merely the unlisted-public-symbol policy.
                self.assertIn("static initialiser holding AVX-class instructions", output)


if __name__ == "__main__":
    unittest.main()
