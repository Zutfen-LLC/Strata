"""Synthetic Linux regressions. Run on Hermes only; no GPU/model/native worker required."""
import copy
import json
import os
import pathlib
import runpy
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'harness'
sys.path.insert(0, str(SOURCE))
import bundle
import campaign as C
import run as driver


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='strata-harness-')
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.children = []
        self.addCleanup(self.cleanup_children)
        C.SESSIONS.clear(); C.GUARDIAN = None
        cpu = min(os.sched_getaffinity(0))
        self.arm = dict(name='load', workers=1, loader_threads=4, restart=True, caller_mask=[cpu],
                        pool_masks=[[cpu]], engine_main_mask=[cpu], requests=[], placement_markers=[])
        assets = {}
        for name, data in [('engine', b'#!/bin/sh\nexit 0\n'), ('worker', b'#!/bin/sh\nexit 0\n'),
                           ('manifest', b'synthetic manifest'), ('requests', b'GEN 256 1 2 3\n')]:
            path = self.root / name; path.write_bytes(data); path.chmod(0o700); assets[name] = str(path)
        model = self.root / 'fixture.gguf'; model.write_bytes(b'synthetic')
        env = dict(PATH=os.defpath, LANG='C', LC_ALL='C')
        self.cfg = dict(host=socket.gethostname(), run_root=str(self.root / 'staged'), assets=assets,
            model_paths=[str(model)], argv=['@engine', '--native', str(model), '--remote-cpu-experts', '127.0.0.1:27431',
                                          '--remote-cpu-local-manifest', '@manifest'],
            worker_argv=['@worker', '--port', '27431', '--workers', '1', '--local-load-threads', '4',
                         '--local-manifest', '@manifest', '--residency-forever'], env=env, worker_env=env,
            monitor_env=env, gpu_uuids=['synthetic-gpu'], worker_host='127.0.0.1', port=27431,
            worker_bytes=1024, arms=[self.arm], final_worker='retain', failure_worker='retain',
            startup_seconds=5, worker_start_seconds=5, request_seconds=5, shutdown_seconds=5, cleanup_seconds=2)

    def cleanup_children(self):
        for child in self.children:
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
            if child.stdin is not None: child.stdin.close()
        C.SESSIONS.clear(); C.GUARDIAN = None

    def stage(self):
        config = self.root / 'private-config.json'; config.write_text(json.dumps(self.cfg))
        with patch.object(bundle.shutil, 'which', return_value='/synthetic/nvidia-smi'):
            return bundle.stage(config)

    def child(self, code):
        child = subprocess.Popen([sys.executable, '-B', '-c', code], stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child); return child

    def configure_control(self):
        runtime = dict(self.cfg, worker_count=1, worker_root=str(self.root / 'WORKER'), request_lines=[])
        C.configure(self.root, runtime)
        C.W.mkdir()

    def test_paths_rejected_before_start_or_output(self):
        self.cfg['argv'][0] = 'relative-engine'
        with patch('subprocess.Popen') as spawn, self.assertRaisesRegex(ValueError, 'relative'):
            self.stage()
        spawn.assert_not_called(); self.assertFalse(pathlib.Path(self.cfg['run_root']).exists())

    def test_missing_and_stale_helpers_fail_preflight(self):
        root = self.stage(); (root / 'monitor.py').unlink()
        with patch('subprocess.Popen') as spawn, self.assertRaises(FileNotFoundError):
            bundle.verify(root)
        spawn.assert_not_called()
        (root / 'monitor.py').write_text('pass\n')
        with self.assertRaisesRegex(ValueError, 'staged bytes differ'): bundle.verify(root)

    def test_missing_import_is_found_without_starting_an_owner(self):
        source = self.root / 'source'; shutil.copytree(SOURCE, source)
        (source / 'monitor.py').write_text('import missing_strata_fixture_dependency_12345\n')
        with patch('subprocess.Popen') as spawn, self.assertRaisesRegex(ValueError, 'missing import'):
            bundle.imports(source)
        spawn.assert_not_called()

    def test_reference_binding_schema_and_indices_checked_before_start(self):
        self.arm['reference'] = 'missing'
        with patch('subprocess.Popen') as spawn, self.assertRaisesRegex(ValueError, 'reference'):
            self.stage()
        spawn.assert_not_called()
        ref = self.root / 'reference.json'; ref.write_text('{broken')
        self.cfg['assets']['reference'] = str(ref); self.arm['requests'] = [0]; self.arm['restart'] = False
        self.arm['reference'] = 'reference'
        self.cfg['attach_root'] = str(self.root); self.cfg['attached_worker_idle'] = True
        with patch('subprocess.Popen') as spawn, self.assertRaises(json.JSONDecodeError): self.stage()
        spawn.assert_not_called()
        ref.write_text(json.dumps({'requests': []}))
        with patch('subprocess.Popen') as spawn, self.assertRaises(KeyError): self.stage()
        spawn.assert_not_called()
        self.assertFalse(pathlib.Path(self.cfg['run_root']).exists())

    def test_single_staging_operation_never_overwrites_pins(self):
        root = self.stage(); pins = (root / 'pins.json').read_bytes()
        self.assertEqual(json.loads(pins)['assets/worker'], bundle.sha(self.cfg['assets']['worker']))
        with self.assertRaises(FileExistsError): self.stage()
        self.assertEqual(pins, (root / 'pins.json').read_bytes())

    def test_config_only_selection_preserves_order_and_source(self):
        next_arm = copy.deepcopy(self.arm); next_arm.update(name='next', workers=2, loader_threads=1,
                                                           pool_masks=self.arm['pool_masks'] * 2)
        self.cfg['arms'].append(next_arm)
        root = self.stage(); original = (root / 'config.json').read_bytes()
        cfg = json.loads(original)
        derived = [driver.arm_config(cfg, arm) for arm in cfg['arms']]
        self.assertEqual([row['worker_count'] for row in derived], [1, 2])
        self.assertEqual([row['worker_argv'][row['worker_argv'].index('--local-load-threads') + 1]
                          for row in derived], ['4', '1'])
        self.assertEqual([arm['name'] for arm in cfg['arms']], ['load', 'next'])
        self.assertEqual(original, (root / 'config.json').read_bytes())

    def test_endpoint_mismatch_rejected(self):
        self.cfg['port'] += 1
        with self.assertRaisesRegex(ValueError, 'port mismatch'): self.stage()

    def test_reserved_names_and_missing_monitor_path_fail_before_start(self):
        for name in ('assets', 'WORKER', 'WORKER-before-other'):
            with self.subTest(name=name):
                self.arm['name'] = name
                with self.assertRaisesRegex(ValueError, 'reserved'): self.stage()
        self.arm['name'] = 'load'
        self.cfg['monitor_env'] = {'LANG': 'C'}
        with self.assertRaisesRegex(ValueError, 'monitor PATH'): self.stage()

    def test_attached_arms_cannot_relabel_another_worker_binary(self):
        self.cfg.update(attach_root=str(self.root), attached_worker_idle=True)
        self.arm['restart'] = False
        self.cfg['assets']['worker_other'] = self.cfg['assets']['worker']
        other = copy.deepcopy(self.arm); other.update(name='other', worker='worker_other')
        self.cfg['arms'].append(other)
        with self.assertRaisesRegex(ValueError, 'reuse cannot change'): self.stage()
        self.configure_control(); C.cfg['worker_sha256'] = 'selected-staged-worker'
        (C.W / 'identities.json').write_text(json.dumps({'worker': {'pid': 123}}))
        with patch.object(C, 'ident', return_value={'pid': 123}), patch.object(C, 'tasks', return_value=[]), \
             patch.object(bundle, 'sha', return_value='different-live-worker'), patch.object(os, 'sched_setaffinity') as apply:
            with self.assertRaisesRegex(RuntimeError, 'resident executable differs'): driver.masks(self.arm)
        apply.assert_not_called()

    def test_controls_verify_owned_listener_before_connecting(self):
        self.configure_control()
        with patch.object(C, 'listener_owned', side_effect=RuntimeError('unowned listener')), \
             patch.object(C.socket, 'create_connection') as connection:
            with self.assertRaisesRegex(RuntimeError, 'unowned'): C.wstatus()
        connection.assert_not_called()

    def test_owner_death_before_state_is_reaped(self):
        child = self.child('raise SystemExit(7)')
        with self.assertRaisesRegex(RuntimeError, 'owner exited.*rc=7'):
            C.wait_session(self.root / 'missing-state', lambda row: False, 5, child)
        self.assertEqual(child.returncode, 7)
        with self.assertRaises(ChildProcessError): os.waitpid(child.pid, os.WNOHANG)

    def test_worker_controls_never_queue_behind_engine_session(self):
        self.configure_control(); C.SESSIONS['active'] = object()
        with patch.object(C.socket, 'create_connection') as connection:
            with self.assertRaisesRegex(RuntimeError, 'forbidden'): C.wstatus()
            with self.assertRaisesRegex(RuntimeError, 'forbidden'): C.stop_owned_worker()
        connection.assert_not_called()
        with patch.object(os, 'sched_setaffinity') as apply:
            with self.assertRaisesRegex(RuntimeError, 'forbidden'): driver.masks(self.arm)
        apply.assert_not_called()

    def test_stop_timeout_can_have_executed_and_reaps_retained_child(self):
        self.configure_control()
        child = self.child('import sys; sys.stdin.read(); raise SystemExit(0)')
        C.GUARDIAN = child
        identities = {'guardian': {'pid': child.pid}, 'worker': {'pid': child.pid}}
        (C.W / 'identities.json').write_text(json.dumps(identities))
        def uncertain(*args):
            child.stdin.close()  # STOP executed, but the acknowledgement was lost
            C.atom(C.W / 'state.json', dict(phase='exited', exit_rc=0, failure=None))
            raise TimeoutError('lost STOP acknowledgement')
        with patch.object(C, 'ident'), patch.object(C, 'listener_owned'), \
             patch.object(C.socket, 'create_connection'), patch.object(C.worker_control, 'exchange', side_effect=uncertain) as exchange, \
             patch.object(C, 'port_free'), patch.object(C, 'wait_absent') as absent:
            def after_reap(*args):
                self.assertIsNotNone(child.returncode)
                with self.assertRaises(ChildProcessError): os.waitpid(child.pid, os.WNOHANG)
            absent.side_effect = after_reap
            receipt = C.stop_owned_worker()
        self.assertEqual(exchange.call_count, 1)
        self.assertTrue(receipt['reaped']); self.assertEqual(receipt['guardian_returncode'], 0)
        self.assertEqual(receipt['control_outcome'], 'uncertain')
        self.assertEqual(receipt['outcome'], 'completed')

    def test_uncertain_pending_stop_is_not_reissued(self):
        self.configure_control()
        child = self.child('import sys; sys.stdin.read()'); C.GUARDIAN = child
        identities = {'guardian': {'pid': child.pid}, 'worker': {'pid': child.pid}}
        (C.W / 'identities.json').write_text(json.dumps(identities))
        C.atom(C.W / 'stop.json', dict(identities=identities, outcome='uncertain', control_outcome='uncertain'))
        child.stdin.close()
        C.atom(C.W / 'state.json', dict(phase='exited', exit_rc=0, failure=None))
        with patch.object(C.worker_control, 'exchange') as exchange, patch.object(C, 'wait_absent'), patch.object(C, 'port_free'):
            C.stop_owned_worker()
        exchange.assert_not_called()

    def test_port_probe_matches_time_wait_reuse_and_rejects_live_listener(self):
        listener = socket.socket(); self.addCleanup(listener.close)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0)); listener.listen(); port = listener.getsockname()[1]
        with self.assertRaises(OSError): C.port_free('127.0.0.1', port, 0.05)
        with socket.create_connection(('127.0.0.1', port)) as client:
            accepted, _ = listener.accept(); accepted.close(); client.recv(1)
        listener.close()
        C.port_free('127.0.0.1', port, 0.2)

    def test_port_probe_follows_owned_stop(self):
        self.configure_control()
        arm = self.arm
        events = []
        fake = self.child('import sys; sys.stdin.read()')
        with patch.object(C, 'stop_owned_worker', side_effect=lambda: events.append('stopped-and-reaped')), \
             patch.object(C, 'port_free', side_effect=lambda *args: self.fail('pre-stop bind probe')), \
             patch.object(C.subprocess, 'Popen', return_value=fake), patch.object(C, 'wait_session'), patch.object(driver, 'masks'):
            driver.launch_worker(dict(self.cfg, worker_root=str(C.W)), arm)
        self.assertEqual(events, ['stopped-and-reaped'])

    def fake_runtime(self, worker=False, exit_code=0, monitor_exit=0):
        root = self.root / ('guardian-fixture' if worker else 'broker-fixture'); root.mkdir()
        name = 'worker_owner.py' if worker else 'owner.py'
        shutil.copyfile(SOURCE / name, root / name)
        (root / 'monitor.py').write_text('import pathlib,sys,time\n'
            'while pathlib.Path("/proc",sys.argv[3],"exe").exists(): time.sleep(.02)\n'
            'raise SystemExit(%d)\n' % monitor_exit)
        script = root / 'child.py'
        script.write_text('import sys,time\n' + ('print("listening on 127.0.0.1:27431 ready",flush=True);time.sleep(1)\n'
             if worker else 'print("READY fixture",flush=True)\nsys.stdin.readline()\n') + 'raise SystemExit(%d)\n' % exit_code)
        cfg = dict(self.cfg, argv=[sys.executable, '-u', str(script)], worker_argv=[sys.executable, '-u', str(script)],
                   binary_sha256=bundle.sha(sys.executable), worker_sha256=bundle.sha(sys.executable),
                   manifest_path=self.cfg['assets']['manifest'], manifest_sha256=bundle.sha(self.cfg['assets']['manifest']),
                   worker_root=str(root / 'WORKER'), request_lines=[], max_requests=0,
                   engine_main_mask=self.arm['engine_main_mask'])
        path = root / 'config.json'; path.write_text(json.dumps(cfg))
        return root, path, cfg, name

    def test_monitor_launch_failure_cleans_children_before_state_was_ready(self):
        real_spawn = subprocess.Popen
        for worker in (False, True):
            with self.subTest(worker=worker):
                root, config, cfg, name = self.fake_runtime(worker=worker)
                started = []
                def spawn(argv, *args, **kwargs):
                    if any(str(arg).endswith('/monitor.py') for arg in argv):
                        raise RuntimeError('injected monitor launch failure')
                    child = real_spawn(argv, *args, **kwargs)
                    started.append(child); self.children.append(child); return child
                old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
                argv = [str(root / name), str(config)] if worker else [str(root / name), 'session', str(config)]
                try:
                    with patch.object(sys, 'argv', argv), patch('subprocess.Popen', side_effect=spawn), \
                         self.assertRaisesRegex(RuntimeError, 'injected monitor'):
                        runpy.run_path(str(root / name), run_name='__main__')
                finally:
                    for sig, handler in old.items(): signal.signal(sig, handler)
                self.assertEqual(len(started), 1)
                self.assertIsNotNone(started[0].returncode)
                with self.assertRaises(ChildProcessError): os.waitpid(started[0].pid, os.WNOHANG)

    def test_engine_failure_is_not_a_successful_owner_exit(self):
        root, config, cfg, name = self.fake_runtime(exit_code=7)
        child = subprocess.Popen([sys.executable, '-B', str(root / name), 'session', str(config)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child); C.configure(root, dict(cfg, worker_count=1)); C.SESSIONS['session'] = child
        C.wait_session(root / 'session', lambda row: row['ready'] is not None, 5, child)
        with self.assertRaisesRegex(RuntimeError, 'owner failed'): C.finish_session('session')
        self.assertEqual(json.loads((root / 'session' / 'state.json').read_text())['exit_rc'], 7)

    def test_worker_failure_is_not_a_successful_guardian_exit(self):
        root, config, cfg, name = self.fake_runtime(worker=True, exit_code=7)
        child = subprocess.Popen([sys.executable, '-B', str(root / name), str(config)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child); self.assertNotEqual(child.wait(timeout=5), 0)
        self.assertEqual(json.loads((root / 'WORKER' / 'state.json').read_text())['exit_rc'], 7)

    def test_monitor_nonzero_exit_is_preserved_and_fails_owner(self):
        for worker in (False, True):
            with self.subTest(worker=worker):
                root, config, cfg, name = self.fake_runtime(worker=worker, monitor_exit=9)
                argv = [sys.executable, '-B', str(root / name)]
                argv += [str(config)] if worker else ['session', str(config)]
                child = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.children.append(child)
                if not worker:
                    C.configure(root, dict(cfg, worker_count=1)); C.SESSIONS['session'] = child
                    C.wait_session(root / 'session', lambda row: row['ready'] is not None, 5, child)
                    with self.assertRaisesRegex(RuntimeError, 'owner failed'): C.finish_session('session')
                else: self.assertNotEqual(child.wait(timeout=5), 0)
                final = json.loads((root / ('WORKER' if worker else 'session') / 'state.json').read_text())
                self.assertEqual(final['exit_rc'], 0); self.assertEqual(final['monitor_exit_rc'], 9)


if __name__ == '__main__': unittest.main()
