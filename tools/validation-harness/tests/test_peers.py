"""Synthetic per-host guardian tests; no SSH, native binary, GPU or model. Hermes only."""
import copy
import json
import os
import pathlib
import socket
import sys
import time
import unittest
from unittest.mock import patch
import test_harness as fixture_module
import bundle
import campaign as C
import peer
import run as driver


class PeerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.HarnessTests(methodName='runTest');self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.cfg=self.fixture.cfg

    def configuration(self, startup_stall=False):
        host='127.0.0.2'  # another loopback address; proves readiness does not hardcode 127.0.0.1
        with socket.socket() as probe:probe.bind((host,0));port=probe.getsockname()[1]
        worker=self.root/'rpc-worker'
        worker.write_text('#!'+sys.executable+'\n'+'''import argparse,os,socket,struct,threading
p=argparse.ArgumentParser();p.add_argument('--bind-ip');p.add_argument('--port',type=int)
a,_=p.parse_known_args()
done=threading.Event();threads=[threading.Thread(target=done.wait) for _ in range(2)]
for t in threads:t.start()
if os.environ.get('SYNTHETIC_STALL'):done.wait()
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind((a.bind_ip,a.port));s.listen()
print('listening on %s:%d ready'%(a.bind_ip,a.port),flush=True)
def read(c,n):
 b=b''
 while len(b)<n:
  x=c.recv(n-len(b))
  if not x:return None
  b+=x
 return b
stop=False
while not stop:
 c,_=s.accept()
 with c:
  while True:
   head=read(c,16)
   if head is None:break
   magic,msg,_,n=struct.unpack('<IHHQ',head)
   if n:read(c,n)
   if msg==16:
    payload=struct.pack('<II11QI',1,2,*[0,1024,2**64-1,0,0,1024,1024,1024,1,0,0],1);reply=17
   elif msg==9:payload=struct.pack('<7QHB',0,0,0,0,0,0,0,0,1);reply=10
   elif msg==11:payload=b'';reply=10;stop=True
   else:raise RuntimeError('unexpected synthetic control')
   c.sendall(struct.pack('<IHHQ',magic,reply,0,len(payload))+payload)
   if stop:break
s.close();done.set()
for t in threads:t.join()
''');worker.chmod(0o700)
        self.cfg['assets']['worker']=sys.executable
        self.cfg['worker_bytes']=2048
        self.cfg['worker_argv']+=['--memory-budget-mib','1']
        self.cfg['arms'][0].update(cpu_peers=['local'],peer_bytes=[2048])
        cpu=self.cfg['arms'][0]['caller_mask']
        remote=dict(host=socket.gethostname(),run_root=str(self.root/'remote'),
            assets=dict(worker=sys.executable,manifest=self.cfg['assets']['manifest']),
            worker_argv=[sys.executable,'-u',str(worker),'--port',str(port),'--bind-ip',host,'--workers','1','--local-load-threads','4',
                         '--memory-budget-mib','1','--local-root',str(self.root),'--local-manifest','@manifest','--residency-forever','--allow-peer-ip','127.0.0.1'],
            worker_host=host,port=port,workers=1,loader_threads=4,worker_bytes=1024,
            caller_mask=cpu,pool_masks=[cpu],worker_env=self.cfg['worker_env'],monitor_env=self.cfg['monitor_env'],
            monitor_gpu=False,worker_start_seconds=8,shutdown_seconds=5,cleanup_seconds=3)
        definition=dict(name='remote',endpoint=host,port=port,config=remote,
            control_argv=[sys.executable,'-B',str(self.root/'remote/worker_owner.py'),str(self.root/'remote/config.json'),'--rpc'])
        if startup_stall:remote['worker_env']=dict(remote['worker_env'],SYNTHETIC_STALL='1')
        self.cfg['worker_peers']=[dict(name='local',local=True,endpoint='127.0.0.1',port=self.cfg['port']),definition]
        config=self.root/'campaign.json';config.write_text(json.dumps(self.cfg))
        bundle.stage(config,worker='remote')
        with patch.object(bundle.shutil,'which',return_value='/synthetic/nvidia-smi'):root=bundle.stage(config)
        staged=json.loads((root/'config.json').read_text());C.configure(root,dict(staged,worker_count=1))
        return root,staged['worker_peers'][1]

    def client(self,definition,root):
        client=peer.Peer(definition,root)
        def cleanup():
            if client.child is not None and client.child.poll() is None:client.abort()
            elif client.child is not None:client.close_pipes()
        self.addCleanup(cleanup);return client

    def test_remote_role_stages_no_engine_and_binds_lan_style_readiness(self):
        root,definition=self.configuration()
        staged=json.loads((self.root/'remote/config.json').read_text())
        self.assertEqual(set(staged['assets']),{'worker','manifest'});self.assertFalse(staged['monitor_gpu'])
        client=self.client(definition,root);client.prepare()
        self.assertFalse((self.root/'remote/WORKER/identities.json').exists())
        client.start()
        self.assertIn('127.0.0.2:',(self.root/'remote/WORKER/listening.json').read_text())
        client.masks();self.assertEqual(client.status()['status']['resident_bytes'],1024)
        client.stop();self.assertEqual(client.child.returncode,0)
        receipt=json.loads((client.root/'remote-stop.json').read_text())
        self.assertTrue(receipt['remote']['children_reaped']);self.assertTrue(receipt['transport_reaped'])
        self.assertEqual(receipt['remote']['worker_returncode'],0)

    def test_fast_stop_is_reconciled_after_worker_exit_without_second_stop(self):
        root,definition=self.configuration();client=self.client(definition,root);client.start()
        pid=json.loads((client.root/'remote-started.json').read_text())['identities']['worker']['pid']
        original=client.command
        def delayed(op,**fields):
            if op=='wait-stop':
                end=time.monotonic()+3
                while pathlib.Path('/proc',str(pid),'exe').exists():
                    if time.monotonic()>end:raise TimeoutError('synthetic worker did not exit promptly')
                    time.sleep(.02)
            return original(op,**fields)
        with patch.object(client,'command',side_effect=delayed):client.stop()
        self.assertEqual(client.child.returncode,0)
        with self.assertRaisesRegex(RuntimeError,'already attempted'):client.stop()

    def test_stale_remote_binding_refused_before_start(self):
        root,definition=self.configuration();definition=copy.deepcopy(definition)
        definition['config']['pool_masks']=[[min(os.sched_getaffinity(0)),min(os.sched_getaffinity(0))]]
        client=self.client(definition,root)
        with self.assertRaisesRegex(RuntimeError,'binding differs'):client.start()
        self.assertFalse((self.root/'remote/WORKER/identities.json').exists())
        client.abort()
        self.assertIsNotNone(client.child.returncode)

    def test_transport_stderr_quota_preserves_failure_and_reaps(self):
        root,definition=self.configuration();definition=copy.deepcopy(definition)
        definition['control_argv']=[sys.executable,'-B','-c','import sys;sys.stderr.buffer.write(b"x"*8192);sys.stderr.flush();sys.stdin.read()']
        client=self.client(definition,root)
        with patch.object(peer,'TRANSPORT_LOG_MAX',4096):
            with self.assertRaisesRegex(RuntimeError,'byte limit'):client.start()
            with self.assertRaisesRegex(RuntimeError,'remote cleanup uncertain'):client.abort()
        self.assertEqual((client.root/'remote-transport.log').stat().st_size,4096)
        self.assertTrue((client.root/'remote-transport-failure.json').exists())
        self.assertIsNotNone(client.child.returncode)
        self.assertEqual(json.loads((client.root/'remote-abort.json').read_text())['outcome'],'uncertain')

    def test_owned_abort_acknowledges_remote_hierarchy_reaping(self):
        root,definition=self.configuration();client=self.client(definition,root);client.start()
        client.abort()
        receipt=json.loads((client.root/'remote-abort.json').read_text())
        self.assertEqual(receipt['outcome'],'completed')
        self.assertTrue(receipt['remote']['children_reaped']);self.assertTrue(receipt['transport_reaped'])
        self.assertFalse((self.root/'remote/WORKER/stop.json').exists())

    def test_startup_abort_reaps_live_child_before_any_listener(self):
        root,definition=self.configuration(startup_stall=True);client=self.client(definition,root);client.prepare()
        client.child.stdin.write(b'{"op":"start"}\n');client.child.stdin.flush()
        identities=self.root/'remote/WORKER/identities.json';end=time.monotonic()+4
        while not identities.exists():
            if time.monotonic()>end:raise TimeoutError('synthetic worker did not start')
            time.sleep(.02)
        self.assertFalse((self.root/'remote/WORKER/listening.json').exists());client.abort()
        receipt=json.loads((client.root/'remote-abort.json').read_text())
        self.assertEqual(receipt['outcome'],'completed');self.assertTrue(receipt['remote']['children_reaped'])

    def test_membership_config_keeps_exact_total_and_legacy_primary_path(self):
        root,definition=self.configuration();cfg=json.loads((root/'config.json').read_text())
        arm=copy.deepcopy(cfg['arms'][0]);arm.update(name='split',cpu_peers=['local','remote'],peer_bytes=[1024,1024])
        self.assertIn('--remote-cpu-experts',driver.arm_config(cfg,cfg['arms'][0])['argv'])
        selected=driver.arm_config(cfg,arm)
        self.assertEqual(selected['argv'].count('--remote-cpu-worker'),2)
        self.assertEqual(selected['worker_bytes'],1024);self.assertEqual(selected['communication_limit'],2)
        cfg['arms'].insert(1,arm);restore=copy.deepcopy(cfg['arms'][0]);restore['name']='restore';cfg['arms'].append(restore)
        bundle.validate_peers(cfg)
        arm['peer_bytes']=[1024,2048]
        with self.assertRaisesRegex(ValueError,'exact CPU arena total'):bundle.validate_peers(cfg)


if __name__=='__main__':unittest.main()
