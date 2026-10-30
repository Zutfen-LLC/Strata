"""Existing fresh-suite loop with mocked model children; exact parser runs host-only."""
import argparse
import contextlib
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import portable_quality_v1 as P
from test_portable_quality_v1 import Tokenizer
from test_portable_topology_v1 import plan


class NativeMethods(unittest.TestCase):
    def arguments(self, root, enabled):
        binary = root/'binary'; binary.write_bytes(b'mock model, never executed')
        suite = root/'suite'; P.Q.prepare(Tokenizer(), suite)
        placement = plan(root/'plan', (0, 7))
        parser = argparse.ArgumentParser(); P.S.device_arguments(parser); P.native_arguments(parser)
        selection = ['--native-router', '--native-shared', '--native-moe-combine'] if enabled else [
            '--no-native-router', '--no-native-shared', '--no-native-moe-combine']
        args = parser.parse_args(selection+['--expert-native-q8-observation'])
        args.__dict__.update(suite=suite, binary=binary, binary_sha256=P.Q.bundle.sha(binary), out=root/'run',
            expert_plan=['0:'+str(placement)], expert_tensor_mib='256,256', expert_workspace_mib='128,128',
            expert_free_reserve_mib='512,512')
        return args

    def suite(self, root, enabled, failure=None, parser=None):
        args = self.arguments(root, enabled)
        commands = []; children = []; checks = []; expected_prompts, _ = P.frozen(Tokenizer(), args.suite)
        native = P.native_policy(args); binding = P.native_plans(args)[0]

        def launch(argv, stdout, stderr, **_):
            index = len(commands); commands.append(argv)
            child = Mock(pid=1000000+index); child.poll.return_value = 1 if failure == ('exit', index) else 0
            children.append(child)
            tokens = list(map(int, argv[argv.index('--tokens')+1].split(',')))
            self.assertEqual(tokens, expected_prompts[index])
            self.assertEqual(argv[argv.index('--max-new')+1], '32')
            self.assertEqual(argv[argv.index('--context')+1], '256')
            self.assertNotIn('--stdio', argv)
            self.assertNotIn('--timings', argv)
            self.assertNotIn('--allow-cpu-fallback', argv)
            if parser:
                result = subprocess.run([str(parser), *argv[1:]], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                parsed = json.loads(result.stdout)
                self.assertEqual(parsed['tokens'], tokens)
                self.assertEqual((parsed['context'], parsed['max_new']), (256, 32))
                self.assertEqual([parsed[k] for k in P.NATIVE_SWITCHES], [enabled]*3)
                self.assertEqual(parsed['expert'], argv[argv.index('--expert-native-q8-observation')+1])
                for key in ('router', 'shared'):
                    self.assertEqual(parsed[key], argv[argv.index('--native-'+key)+1] if enabled else '')
                checks.append(parsed)
            rows = [dict(position=i, token=token, generated=False, greedy_next=7) for i, token in enumerate(tokens)]
            rows += [dict(position=len(tokens)+i, token=7, generated=True, greedy_next=7) for i in range(32)]
            consumed = len(rows)
            terminal = dict(stop='max_new', position=consumed)
            if enabled:
                for key, label in (('native_router', 'router'), ('native_shared', 'shared')):
                    terminal.update({key+'_method': True, label+'_numerical_parity': False,
                                     key+'_calls_delta': 48*consumed, key+'_rows_delta': 48*consumed})
            if failure == ('counter', index): terminal['native_shared_calls_delta'] = 48
            if failure == ('cumulative', index): terminal['native_shared_calls_delta'] += 48*sum(len(p)+32 for p in expected_prompts[:index])
            if failure == ('off-counter', index): terminal['native_router_calls_delta'] = 0
            for record in rows+[terminal]: stdout.write(json.dumps(record)+'\n')
            stdout.flush()
            entries = 0 if failure == ('participation', index) else 2
            stderr.write(f'expert_device stage=token-complete position={consumed-1} slot=0 backend=Vulkan0 '
                f'numerics=VULKAN_HYBRID_Q8 assigned=1 weight_bytes={binding["assigned_bytes"]} '
                f'entries={entries} row_checks={entries} row_max_abs=0 row_relative_l2=0\n')
            stderr.write(f'expert_dispatch position={consumed-1} gpu_entries={entries} cpu_entries={480*consumed-entries}\n')
            stderr.flush()
            receipt = Path(argv[argv.index('--expert-native-q8-observation')+1]).parent
            modules = [receipt/f'expert.slot-0.{stage}.spv' for stage in ('activation', 'gu', 'down')]
            if enabled:
                for layer in range(48):
                    modules += [receipt/f'{name}.layer-{layer}.t-1.owner-{layer+13}/effective.spv'
                                for name in ('router', 'expert.combine')]
                    modules += [receipt/f'shared.layer-{layer}.t-1.owner-{layer+13}/{name}-effective.spv'
                                for name in ('dense-scalar', 'quant')]
            for module in modules:
                module.parent.mkdir(exist_ok=True)
                module.write_bytes(b'\x03\x02\x23\x07'+bytes(16))
            if failure == ('module', index): modules[-1].unlink()
            if failure == ('module-header', index): modules[-1].write_bytes(bytes(20))
            if failure == ('unrequested', index):
                extra = receipt/'router.layer-0.t-1.owner-0'
                extra.mkdir(); (extra/'effective.spv').write_bytes(b'\x03\x02\x23\x07'+bytes(16))
            if failure == ('plan', index): Path(args.expert_plan[0].split(':', 1)[1]).write_bytes(b'mutated')
            return child

        mapping = [dict(index=i, pci=p, free=9<<30, total=9<<30) for i, p in enumerate(P.S.PCI)]
        patches = [(P.S, 'host_memory', dict(return_value={'effective_available':1<<40})),
                   (P.S, 'model_working_set', dict(return_value={'working_set_bound_bytes':0})),
                   (P.S, 'verified_devices', dict(return_value=mapping)),
                   (P.S, 'vram', dict(side_effect=lambda pci:dict(pci=pci,used=0,total=9<<30))),
                   (P.S, 'rss', dict(return_value={})), (P.S, 'guard_host', {}),
                   (P.S, 'placements', dict(return_value={}))]
        # subprocess.run needs the real Popen for the tiny compiled parser;
        # only this adapter's model launch is substituted through a proxy.
        proxy = Mock(Popen=Mock(side_effect=launch), TimeoutExpired=subprocess.TimeoutExpired)
        with contextlib.ExitStack() as stack:
            for obj, name, kwargs in patches: stack.enter_context(patch.object(obj, name, **kwargs))
            stack.enter_context(patch.object(P, 'subprocess', proxy))
            if failure:
                with self.assertRaises((ValueError, RuntimeError)): P.run_portable(args, Tokenizer())
            else: P.run_portable(args, Tokenizer())
        summary = json.loads((args.out/'summary.json').read_text())
        return summary, commands, children, checks

    def test_twelve_cases_each_arm_fresh_requests_receipts_and_counters(self):
        roots = []
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                summary, commands, children, _ = self.suite(Path(tmp), enabled)
                self.assertEqual(summary['status'], 'complete')
                self.assertEqual([row['index'] for row in summary['requests']], list(range(1, 13)))
                self.assertEqual([summary['native_methods'][key] for key in P.NATIVE_SWITCHES], [enabled]*3)
                self.assertEqual(len(children), 12)  # Each case starts a new session/process, not a continuation.
                self.assertEqual(len({child.pid for child in children}), 12)
                prefixes = [argv[argv.index('--expert-native-q8-observation')+1] for argv in commands]
                self.assertEqual(len(set(prefixes)), 12)
                self.assertTrue(all('case-'+str(i).zfill(2) in prefix for i, prefix in enumerate(prefixes, 1)))
                for row in summary['requests']:
                    self.assertEqual(row['native_receipts']['expert_entries'], 2)
                    self.assertEqual(len(row['native_receipts']['effective_modules']), 195 if enabled else 3)
                    self.assertEqual(bool(row['native_work']), enabled)
                roots.append(summary['suite'])
        self.assertEqual(roots[0], roots[1])

    def test_failures_stop_before_next_case_and_keep_previous_results(self):
        for kind, enabled in (('exit', True), ('counter', True), ('off-counter', False),
                              ('participation', True), ('module', True), ('plan', True), ('cumulative', True),
                              ('module-header', True), ('unrequested', False)):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                summary, commands, _, _ = self.suite(Path(tmp), enabled, (kind, 2))
                self.assertEqual(summary['status'], 'STOP')
                self.assertEqual(len(commands), 3)
                self.assertEqual(len(summary['requests']), 2)
                self.assertIn('error', summary)

    def test_explicit_policy_and_collision_refusals(self):
        parser = argparse.ArgumentParser(); P.native_arguments(parser)
        self.assertIsNone(P.native_policy(parser.parse_args([])))
        for flags in (['--native-router'], ['--expert-native-q8-observation'],
                      ['--native-router', '--native-shared', '--native-moe-combine']):
            with self.assertRaises(ValueError): P.native_policy(parser.parse_args(flags))
        with tempfile.TemporaryDirectory() as tmp:
            args = self.arguments(Path(tmp), True)
            for name, value in (('single_case', True), ('benchmark', True), ('prompt_chunk_size', 2),
                                ('prompt_head_elision', True), ('dump_executed_logits', True)):
                invalid = copy.copy(args); setattr(invalid, name, value)
                with self.assertRaises(ValueError): P.native_policy(invalid)
            case = Path(tmp)/'case'; case.mkdir(); policy = P.native_policy(args)
            P.native_flags(policy, case)
            with self.assertRaises(FileExistsError): P.native_flags(policy, case)

    def test_native_counter_types_and_absence_are_strict(self):
        on = {key:True for key in P.NATIVE_SWITCHES}; on['expert_native_q8_observation'] = True
        row = dict(native_router_method=True,router_numerical_parity=False,native_router_calls_delta=48,
                   native_router_rows_delta=48,native_shared_method=True,shared_numerical_parity=False,
                   native_shared_calls_delta=48,native_shared_rows_delta=48)
        self.assertEqual(P.native_work(row, on, 1), row)
        for key in row:
            bad = row.copy(); bad[key] = 1.0 if type(row[key]) is int else 1
            with self.assertRaises(ValueError): P.native_work(bad, on, 1)
        off = {key:False for key in P.NATIVE_SWITCHES}
        with self.assertRaises(ValueError): P.native_work(row, off, 1)

    def test_adapter_cli_declares_both_arms_explicitly(self):
        for enabled in (False, True):
            flags = ['--'+('' if enabled else 'no-')+name.replace('_', '-') for name in P.NATIVE_SWITCHES]
            argv = ['quality', '--gguf', 'metadata', 'portable', '--binary', 'unused',
                    '--binary-sha256', 'digest', '--suite', 'suite', '--out', 'out',
                    '--expert-plan', '0:plan', '--expert-native-q8-observation', *flags]
            with patch.object(sys, 'argv', argv), patch.object(P.Q, 'tokenizer'), patch.object(P, 'run_portable') as run:
                self.assertEqual(P.main(), 0)
                policy = P.native_policy(run.call_args.args[0])
                self.assertEqual([policy[key] for key in P.NATIVE_SWITCHES], [enabled]*3)

    def test_actual_product_parser_for_all_twenty_four_requests(self):
        compiler = shutil.which('c++')
        if not compiler: self.skipTest('host C++ compiler unavailable')
        source = Path(__file__).resolve().parents[2]
        text = (source/'src/core/portable_model_main.cpp').read_text()
        anchor = '        if(resident) {\n            std::signal'
        self.assertEqual(text.count(anchor), 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'ggml-backend.h').write_text('// No backend API is used before the parser boundary.\n')
            cpp = root/'parser.cpp'; exe = root/'parser'
            cpp.write_text(text[:text.index(anchor)]+'''
        std::cout<<"{\\"context\\":"<<opts.context<<",\\"max_new\\":"<<max_new
            <<",\\"native_router\\":"<<(opts.native_router?"true":"false")
            <<",\\"native_shared\\":"<<(opts.native_shared?"true":"false")
            <<",\\"native_moe_combine\\":"<<(opts.native_moe_combine?"true":"false")
            <<",\\"router\\":"<<std::quoted(opts.native_router_receipt_prefix)
            <<",\\"shared\\":"<<std::quoted(opts.native_shared_receipt_prefix)
            <<",\\"expert\\":"<<std::quoted(opts.expert_native_q8_observation)<<",\\"tokens\\":[";
        for(size_t i=0;i<raw_tokens.size();++i)std::cout<<(i?",":"")<<raw_tokens[i];
        std::cout<<"]}\\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\\n';return 2;}}
''')
            subprocess.run([compiler, '-std=c++20', '-I'+str(root), '-I'+str(source/'include'),
                            '-I'+str(source/'src/core'), str(cpp), '-o', str(exe)], check=True, capture_output=True)
            for enabled in (False, True):
                arm = root/str(enabled); arm.mkdir()
                _, _, _, checks = self.suite(arm, enabled, parser=exe)
                self.assertEqual(len(checks), 12)


if __name__ == '__main__': unittest.main()
