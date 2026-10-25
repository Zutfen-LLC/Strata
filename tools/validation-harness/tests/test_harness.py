"""Synthetic Linux regressions. Run on Hermes only; no GPU/model/native worker required."""
import copy
import json
import os
import pathlib
import runpy
import select
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

    def test_per_arm_packing_preserves_raw_argv_and_worker(self):
        self.cfg['cpu_route'] = 'group'
        self.cfg['argv'][self.cfg['argv'].index('--remote-cpu-experts')] = '--remote-cpu-worker'
        packed = dict(self.arm, name='packed', restart=False, requests=[0],
                      packed_qualification='abcdef0123456789')
        self.cfg['arms'].append(packed)
        root = self.stage()
        cfg = json.loads((root / 'config.json').read_text())
        raw, candidate = [driver.arm_config(cfg, arm) for arm in cfg['arms']]
        self.assertEqual(raw['argv'], cfg['argv'])
        self.assertEqual(candidate['argv'], raw['argv'] +
                         ['--remote-cpu-packed-qualification', 'abcdef0123456789'])
        self.assertEqual(raw['worker_argv'], candidate['worker_argv'])
        self.assertNotIn('--remote-cpu-packed-qualification', cfg['argv'])

    def test_per_arm_packing_rejects_bad_or_global_qualification(self):
        self.arm['packed_qualification'] = '0000000000000000'
        with self.assertRaisesRegex(ValueError, '16 nonzero'): self.stage()
        self.arm['packed_qualification'] = 'abcdef0123456789'
        with self.assertRaisesRegex(ValueError, 'group route'): self.stage()
        self.cfg['cpu_route'] = 'group'
        self.cfg['argv'][self.cfg['argv'].index('--remote-cpu-experts')] = '--remote-cpu-worker'
        self.cfg['argv'] += ['--remote-cpu-packed-qualification', 'abcdef0123456789']
        with self.assertRaisesRegex(ValueError, 'no global'): self.stage()

    def test_first_logit_reference_requires_one_captured_request(self):
        self.arm['first_logits_reference'] = 'logits'
        with self.assertRaisesRegex(ValueError, 'capture'): self.stage()
        self.cfg['capture_first_logits'] = True
        self.arm['requests'] = [0]
        self.arm['restart'] = False
        self.cfg.update(attach_root=str(self.root), attached_worker_idle=True)
        with self.assertRaisesRegex(ValueError, 'missing first-logit'): self.stage()
        logits = self.root / 'logits'; logits.write_bytes(b'first logits')
        self.cfg['assets']['logits'] = str(logits)
        self.stage()

    def test_first_logit_mismatch_prevents_next_arm(self):
        reference = self.root / 'logits'; reference.write_bytes(b'expected')
        self.cfg['assets']['logits'] = str(reference)
        self.arm.update(name='raw', restart=False, requests=[0], first_logits_reference='logits')
        self.cfg['arms'].append(dict(self.arm, name='packed'))
        (self.root / 'raw-first-logits.0').write_bytes(b'different')
        with patch.object(bundle, 'verify', return_value=self.cfg), \
             patch.object(C, '__file__', str(self.root / 'campaign.py')), \
             patch.object(C.worker_control, '__file__', str(self.root / 'worker_control.py')), \
             patch.object(C, 'configure'), patch.object(driver, 'masks'), \
             patch.object(C, 'run_arm', return_value={}) as execute:
            with self.assertRaisesRegex(RuntimeError, 'first logits diverged'):
                driver.run(self.root)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[0], 'raw')

    def test_endpoint_mismatch_rejected(self):
        self.cfg['port'] += 1
        with self.assertRaisesRegex(ValueError, 'port mismatch'): self.stage()

    def test_cpu_route_legacy_default_preserves_argv(self):
        root = self.stage()
        with patch.object(bundle.shutil, 'which', return_value='/synthetic/nvidia-smi'): cfg = bundle.verify(root)
        self.assertNotIn('cpu_route', cfg)
        derived = driver.arm_config(cfg, cfg['arms'][0])
        self.assertEqual(derived['argv'], cfg['argv'])
        self.assertEqual(derived['argv'].count('--remote-cpu-experts'), 1)
        self.assertNotIn('--remote-cpu-worker', derived['argv'])
        self.assertEqual(bundle.cpu_route_flag(dict(cfg, cpu_route='legacy')), '--remote-cpu-experts')

    def test_cpu_route_single_group_survives_staging_and_arm_selection(self):
        self.cfg['cpu_route'] = 'group'
        self.cfg['argv'][self.cfg['argv'].index('--remote-cpu-experts')] = '--remote-cpu-worker'
        with patch('subprocess.Popen') as spawn: root = self.stage()
        spawn.assert_not_called()
        with patch.object(bundle.shutil, 'which', return_value='/synthetic/nvidia-smi'): cfg = bundle.verify(root)
        original = (root / 'config.json').read_bytes()
        derived = driver.arm_config(cfg, cfg['arms'][0])
        self.assertEqual(derived['cpu_route'], 'group')
        self.assertEqual(derived['argv'], cfg['argv'])
        self.assertEqual(derived['argv'].count('--remote-cpu-worker'), 1)
        self.assertNotIn('--remote-cpu-experts', derived['argv'])
        self.assertEqual(derived['argv'][derived['argv'].index('--remote-cpu-worker') + 1], '127.0.0.1:27431')
        self.assertEqual(derived['argv'][derived['argv'].index('--remote-cpu-local-manifest') + 1], cfg['assets']['manifest'])
        self.assertEqual(original, (root / 'config.json').read_bytes())

    def test_cpu_route_invalid_conflicting_and_duplicate_flags_refused(self):
        baseline = copy.deepcopy(self.cfg)
        cases = [('unknown', 'other', ['--remote-cpu-experts']),
                 ('wrong-legacy', 'legacy', ['--remote-cpu-worker']),
                 ('wrong-group', 'group', ['--remote-cpu-experts']),
                 ('both', 'group', ['--remote-cpu-worker', '--remote-cpu-experts']),
                 ('duplicate', 'group', ['--remote-cpu-worker', '--remote-cpu-worker'])]
        for name, route, flags in cases:
            with self.subTest(name=name):
                self.cfg = copy.deepcopy(baseline); self.cfg['cpu_route'] = route
                index = self.cfg['argv'].index('--remote-cpu-experts')
                del self.cfg['argv'][index:index + 2]
                for flag in flags: self.cfg['argv'] += [flag, '127.0.0.1:27431']
                with patch('subprocess.Popen') as spawn, self.assertRaises(ValueError): self.stage()
                spawn.assert_not_called(); self.assertFalse(pathlib.Path(self.cfg['run_root']).exists())

    def test_cpu_route_exact_endpoint_and_manifest_bindings(self):
        baseline = copy.deepcopy(self.cfg)
        for route in ('legacy', 'group'):
            for fault in ('endpoint', 'engine-manifest', 'worker-manifest'):
                with self.subTest(route=route, fault=fault):
                    self.cfg = copy.deepcopy(baseline); self.cfg['cpu_route'] = route
                    flag = bundle.cpu_route_flag(self.cfg)
                    self.cfg['argv'][self.cfg['argv'].index('--remote-cpu-experts')] = flag
                    if fault == 'endpoint':
                        self.cfg['argv'][self.cfg['argv'].index(flag) + 1] = '127.0.0.1:27432'
                    else:
                        argv = self.cfg['argv'] if fault == 'engine-manifest' else self.cfg['worker_argv']
                        manifest = '--remote-cpu-local-manifest' if fault == 'engine-manifest' else '--local-manifest'
                        argv[argv.index(manifest) + 1] = '@requests'
                    with patch('subprocess.Popen') as spawn, self.assertRaisesRegex(ValueError, 'binding mismatch|endpoint mismatch'):
                        self.stage()
                    spawn.assert_not_called(); self.assertFalse(pathlib.Path(self.cfg['run_root']).exists())

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
             patch.object(C.subprocess, 'Popen', return_value=fake), patch.object(C, 'wait_session'), patch.object(driver, 'masks') as apply:
            driver.launch_worker(dict(self.cfg, worker_root=str(C.W)), arm)
        self.assertEqual(events, ['stopped-and-reaped'])
        apply.assert_called_once_with(arm, phase='setup')

    def test_loader_inherits_setup_union_and_ready_restores_compute_masks(self):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 2: self.skipTest('affinity inheritance regression needs two allowed CPUs')
        self.arm.update(caller_mask=[cpus[0]], pool_masks=[[cpus[1]]], engine_main_mask=[cpus[0]])
        self.configure_control()
        script = '''import json,os,sys,threading
done=threading.Event()
pool=[]
for _ in range(2):
    thread=threading.Thread(target=done.wait,daemon=True)
    thread.start();pool.append(thread)
print(json.dumps({'started':True}),flush=True)
for line in sys.stdin:
    if line.strip()=='SETUP':
        masks=[]
        def load(): masks.append(sorted(os.sched_getaffinity(0)))
        loaders=[threading.Thread(target=load) for _ in range(4)]
        for thread in loaders: thread.start()
        for thread in loaders: thread.join()
        print(json.dumps({'loaders':masks}),flush=True)
    elif line.strip()=='COMPUTE':
        print(json.dumps({'caller':sorted(os.sched_getaffinity(0)),
                          'pool':sorted(os.sched_getaffinity(pool[0].native_id))}),flush=True)
    elif line.strip()=='QUIT': break
done.set()
for thread in pool: thread.join()
'''
        argv = [sys.executable, '-B', '-u', '-c', script, '--workers', '1', '--local-load-threads', '4']
        worker = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.children.append(worker); self.addCleanup(worker.stdout.close)
        def read():
            if not select.select([worker.stdout], [], [], 5)[0]: raise TimeoutError('synthetic worker response')
            return json.loads(worker.stdout.readline())
        def command(line):
            worker.stdin.write(line + '\n'); worker.stdin.flush(); return read()
        self.assertTrue(read()['started'])
        C.cfg['worker_sha256'] = bundle.sha(sys.executable)
        (C.W / 'identities.json').write_text(json.dumps({'worker': {'pid': worker.pid, 'argv': argv}}))
        broker = self.child('import sys; sys.stdin.read()')
        os.sched_setaffinity(broker.pid, set(self.arm['engine_main_mask']))
        session = self.root / 'load'; session.mkdir()
        state = dict(broker={'pid': broker.pid}, engine={'pid': broker.pid}, ready={'mono': 1},
                     active=False, failure=None, phase='idle')
        with patch.object(C, 'ident', side_effect=lambda row: row), patch.object(C, 'topology', return_value={}), \
             patch.object(C.worker_control, 'exchange') as exchange:
            driver.masks(self.arm, phase='setup')
            loaded = command('SETUP')
            self.assertEqual(loaded['loaders'], [cpus[:2]] * 4)
            C.SESSIONS['load'] = broker
            state['active'] = True; (session / 'state.json').write_text(json.dumps(state))
            with patch.object(os, 'sched_setaffinity') as apply:
                with self.assertRaisesRegex(RuntimeError, 'idle READY'): driver.masks(self.arm, 'load', phase='ready')
            apply.assert_not_called()
            state['active'] = False; (session / 'state.json').write_text(json.dumps(state))
            driver.masks(self.arm, 'load', phase='ready')
            self.assertEqual(command('COMPUTE'), {'caller': [cpus[0]], 'pool': [cpus[1]]})
            self.assertEqual(sorted(os.sched_getaffinity(broker.pid)), self.arm['engine_main_mask'])
        exchange.assert_not_called()
        self.assertEqual(json.loads((self.root / 'load-affinity-setup.json').read_text())['phase'], 'setup')
        self.assertEqual(json.loads((self.root / 'load-affinity.json').read_text())['phase'], 'ready')
        worker.stdin.write('QUIT\n'); worker.stdin.flush(); self.assertEqual(worker.wait(timeout=5), 0)

    def test_ready_affinity_precedes_gen_and_also_runs_for_zero_gen_load(self):
        self.configure_control()
        for indices in ([], [0]):
            with self.subTest(indices=indices):
                events = []
                def ready(*args): events.append('READY'); return {'ready': {'mono': 2}, 'started_mono': 1}
                def affinity(session, io, phase='compute'):
                    self.assertEqual((session, io, phase), ('load', False, 'ready'))
                    events.append('compute-mask'); raise RuntimeError('affinity refused before GEN')
                with patch.object(C, 'healthy'), patch.object(C, 'wstatus', return_value={}), patch.object(C, 'gpu_idle'), \
                     patch.object(C, 'launch_session', side_effect=ready), patch.object(C, 'rpc') as rpc:
                    with self.assertRaisesRegex(RuntimeError, 'affinity refused'):
                        C.run_arm('load', self.cfg, indices, affinity)
                rpc.assert_not_called(); self.assertEqual(events, ['READY', 'compute-mask'])

    def test_unbound_worker_startup_failure_stops_and_reaps_owned_hierarchy(self):
        root, config, cfg, name = self.fake_runtime(worker=True)
        (root / 'child.py').write_text('import time\nwhile True: time.sleep(.02)\n')
        cfg['assets']['worker'] = sys.executable
        cfg['worker_argv'] += ['--workers', '1', '--local-load-threads', '4']
        cfg['failure_worker'] = 'stop'
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0)); cfg['port'] = probe.getsockname()[1]
        real_spawn = subprocess.Popen
        owners = []
        def reap_owner(owner):
            if owner.poll() is None: owner.terminate()
            owner.wait(timeout=10)
        def spawn(*args, **kwargs):
            owner = real_spawn(*args, **kwargs)
            owners.append(owner); self.children.append(owner)
            self.addCleanup(reap_owner, owner)
            return owner
        with patch.object(bundle, 'verify', return_value=cfg), \
             patch.object(C, '__file__', str(root / 'campaign.py')), \
             patch.object(C.worker_control, '__file__', str(root / 'worker_control.py')), \
             patch.object(C.subprocess, 'Popen', side_effect=spawn), \
             patch.object(C, 'listener_owned', side_effect=AssertionError('no startup listener')), \
             patch.object(C.worker_control, 'exchange') as exchange:
            with self.assertRaisesRegex(TimeoutError, 'owner deadline') as failure:
                driver.run(root)
        exchange.assert_not_called()
        self.assertEqual(len(owners), 1)
        self.assertIsNotNone(owners[0].returncode)
        with self.assertRaises(ChildProcessError): os.waitpid(owners[0].pid, os.WNOHANG)
        receipt = json.loads((root / 'worker-startup-abort.json').read_text())
        original = json.loads((root / 'failure.json').read_text())
        self.assertEqual(original['error'], repr(failure.exception))
        self.assertEqual(receipt['error'], original['error'])
        self.assertEqual(receipt['outcome'], 'completed'); self.assertTrue(receipt['reaped'])
        self.assertEqual(receipt['guardian_returncode'], owners[0].returncode)
        self.assertEqual(receipt['control_outcome'], 'not-sent')
        self.assertEqual(set(receipt['identities']), {'guardian', 'worker', 'monitor'})
        C.wait_absent(receipt['identities'], 0)
        final = json.loads((root / 'WORKER' / 'state.json').read_text())
        self.assertEqual(final['exit_rc'], -signal.SIGTERM); self.assertEqual(final['monitor_exit_rc'], 0)
        self.assertFalse((root / 'WORKER' / 'listening.json').exists())
        self.assertFalse((root / 'WORKER' / 'stop.json').exists())
        self.assertFalse((root / 'worker-cleanup-uncertain.json').exists())
        self.assertIsNone(C.GUARDIAN)

    def test_startup_abort_refuses_attached_resident_and_existing_stop_intent(self):
        self.configure_control(); owner = self.child('import sys; sys.stdin.read()'); C.GUARDIAN = owner
        C.cfg['attach_root'] = str(self.root)
        with patch.object(owner, 'terminate') as terminate, patch.object(C.worker_control, 'exchange') as exchange:
            with self.assertRaisesRegex(RuntimeError, 'attached'): C.abort_owned_worker_startup(TimeoutError())
            del C.cfg['attach_root']
            (C.W / 'stop.json').write_text(json.dumps({'outcome': 'uncertain'}))
            with self.assertRaisesRegex(RuntimeError, 'STOP intent'): C.abort_owned_worker_startup(TimeoutError())
        terminate.assert_not_called(); exchange.assert_not_called()
        self.assertFalse((self.root / 'worker-startup-abort.json').exists())

    def test_startup_abort_without_complete_identity_receipt_remains_uncertain(self):
        self.configure_control(); owner = self.child('raise SystemExit(0)'); owner.wait(timeout=5)
        C.GUARDIAN = owner
        reason = TimeoutError('startup deadline before identity publication')
        partial = {'guardian': {'pid': owner.pid}, 'worker': {'pid': owner.pid}}
        for identities, expected in [(None, FileNotFoundError), (partial, RuntimeError)]:
            with self.subTest(identities=identities):
                if identities: (C.W / 'identities.json').write_text(json.dumps(identities))
                with self.assertRaises(expected): C.abort_owned_worker_startup(reason)
                receipt = json.loads((self.root / 'worker-startup-abort.json').read_text())
                self.assertEqual(receipt['outcome'], 'uncertain'); self.assertTrue(receipt['reaped'])
                self.assertEqual(receipt['guardian_returncode'], 0)
                self.assertEqual(receipt['error'], repr(reason))
                self.assertIs(C.GUARDIAN, owner)
                with self.assertRaises(ChildProcessError): os.waitpid(owner.pid, os.WNOHANG)

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
