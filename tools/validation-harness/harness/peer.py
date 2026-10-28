"""Bounded command channel to the existing per-host guardian, via configured authorized transport."""
import json
import os
import pathlib
import select
import socket
import struct
import subprocess
import threading
import time

import campaign as C
TRANSPORT_LOG_MAX = 32 << 20


class Peer:
    def __init__(self, definition, root):
        self.definition = definition
        self.bundle_root = pathlib.Path(root)
        self.name = definition['name']
        self.root = self.bundle_root/'peers'/self.name
        self.root.mkdir(mode=0o700,parents=True,exist_ok=False)
        self.child = None
        self.buffer = b''
        self.stop_sent = False
        self.binding = None
        self.failure = None
        self.drainer = None

    def read(self, seconds):
        end = time.monotonic() + seconds
        while b'\n' not in self.buffer:
            if self.failure: raise RuntimeError(self.failure)
            remaining = end - time.monotonic()
            if remaining <= 0: raise TimeoutError('peer guardian deadline: ' + self.name)
            if not select.select([self.child.stdout], [], [], min(remaining,.1))[0]: continue
            data = os.read(self.child.stdout.fileno(), 4096)
            if not data: raise RuntimeError('peer guardian EOF: ' + self.name)
            self.buffer += data
            if len(self.buffer) > 65536: raise RuntimeError('peer guardian reply exceeds 64 KiB')
        line, self.buffer = self.buffer.split(b'\n', 1)
        row = json.loads(line)
        if self.failure: raise RuntimeError(self.failure)
        if self.binding is not None and row.get('binding') != self.binding: raise RuntimeError('peer binding changed')
        if 'error' in row: raise RuntimeError('peer guardian: ' + str(row['error']))
        return row

    def command(self, op, **fields):
        data = (json.dumps(dict(op=op, **fields)) + '\n').encode()
        if len(data) > 65536: raise ValueError('peer command exceeds 64 KiB')
        self.child.stdin.write(data); self.child.stdin.flush()
        seconds=self.definition['config']['shutdown_seconds'] if op=='wait-stop' else 15
        if op=='abort':seconds+=self.definition['config']['cleanup_seconds']+60
        return self.read(seconds)

    def prepare(self):
        C.assert_worker_idle()
        self.child = subprocess.Popen(self.definition['control_argv'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        def drain():
            size=0
            try:
                with (self.root/(self.name+'-transport.log')).open('xb') as log:
                    while True:
                        data=os.read(self.child.stderr.fileno(),65536)
                        if not data:break
                        log.write(data[:max(0,TRANSPORT_LOG_MAX-size)]);size+=len(data)
                        if size>TRANSPORT_LOG_MAX and self.failure is None:
                            self.failure='peer transport stderr exceeds its byte limit'
                            C.atom(self.root/(self.name+'-transport-failure.json'),{'error':self.failure})
            except Exception as error:
                self.failure='peer transport drainer failed: '+repr(error)
        self.drainer=threading.Thread(target=drain,daemon=True);self.drainer.start()
        hello = self.read(15)
        local = json.loads((self.bundle_root / 'pins.json').read_text())
        for name in hello['helpers']:
            if local.get(name) != hello['helpers'][name]: raise RuntimeError('peer helper differs: ' + name)
        if set(hello['helpers']) != {name for name in local if '/' not in name and name.endswith('.py')}:
            raise RuntimeError('peer helper inventory differs')
        self.verify_binding(hello,local)

    def start(self):
        if self.child is None:self.prepare()
        self.child.stdin.write(b'{"op":"start"}\n'); self.child.stdin.flush()
        row = self.read(self.definition['config']['worker_start_seconds'])
        C.atom(self.root / (self.name + '-started.json'), row)

    def verify_binding(self, hello, pins):
        expected=self.definition['config'];actual=hello['config']
        fields=('host','run_root','worker_host','port','workers','loader_threads','worker_bytes',
                'caller_mask','pool_masks','worker_env','monitor_env','monitor_gpu',
                'worker_start_seconds','shutdown_seconds','cleanup_seconds')
        if actual.get('role')!='worker' or any(actual.get(k)!=expected.get(k) for k in fields):
            raise RuntimeError('peer configuration binding differs')
        def argv(cfg):
            reverse={path:'@'+key for key,path in cfg['assets'].items()}
            return ['@worker' if i==0 else reverse.get(arg,arg) for i,arg in enumerate(cfg['worker_argv'])]
        if argv(actual)!=argv(expected):raise RuntimeError('peer worker argv/budget differs')
        for key in ('worker','manifest'):
            asset=self.definition.get('asset_bindings',{}).get(key,key)
            if hello['assets'][key]!=pins['assets/'+asset]:raise RuntimeError('peer staged asset differs: '+key)
        self.binding=hello['binding']

    def inspect(self, compute=True):
        row = self.command('inspect', compute=compute)
        C.atom(self.root / (self.name + '-identity.json'), row)
        return row

    def masks(self):
        row = self.command('compute')
        C.atom(self.root / (self.name + '-affinity.json'), row)
        return row

    def status(self):
        C.assert_worker_idle()
        self.inspect(compute=False)  # remote /proc ownership proof; no remote-side TCP control
        with socket.create_connection((self.definition['endpoint'], self.definition['port']), timeout=5) as sock:
            cache = C.worker_control.status(sock)
            values = struct.unpack('<7QHB', C.worker_control.exchange(sock, 9, 10))
        return dict(status=cache, legacy=dict(zip(['session','setup_weight_bytes','infer_in_bytes',
            'infer_out_bytes','requests','failures','cpu_service_ns','last_error','ready'], values)))

    def stop(self):
        C.assert_worker_idle()
        if self.stop_sent: raise RuntimeError('peer STOP already attempted; reconcile original channel')
        self.inspect(compute=False)
        receipt = dict(operation='stop', outcome='pending', control_outcome='pending')
        path = self.root / (self.name + '-stop.json'); C.atom(path, receipt)
        self.stop_sent = True
        try:
            with socket.create_connection((self.definition['endpoint'], self.definition['port']), timeout=5) as sock:
                if C.worker_control.exchange(sock, 11, 10): raise RuntimeError('unexpected peer STOP payload')
            receipt.update(control_outcome='acknowledged')
        except (OSError, RuntimeError, TimeoutError) as error:
            receipt.update(control_outcome='uncertain', error=repr(error))
        C.atom(path, receipt)
        try:
            receipt['remote'] = self.command('wait-stop', control_outcome=receipt['control_outcome'])
            rc = self.child.wait(timeout=self.definition['config']['cleanup_seconds'])
            receipt.update(transport_returncode=rc, transport_reaped=True)
            self.drainer.join(timeout=5)
            if self.drainer.is_alive():raise RuntimeError('peer stderr drainer did not exit')
            self.close_pipes()
            if rc: raise RuntimeError('peer transport exited unsuccessfully')
            receipt['outcome'] = 'completed'
        except Exception as error:
            receipt.update(outcome='uncertain', cleanup_error=repr(error)); C.atom(path, receipt); raise
        C.atom(path, receipt)

    def abort(self):
        # Ask the retained guardian to clean/reap its actual children. Transport exit alone cannot
        # establish remote absence. EOF remains a cleanup fallback, with an explicit uncertain result.
        # No TCP STOP is issued or repeated, including after a timed-out original STOP.
        if self.child is not None:
            receipt={'outcome':'pending'};error=None
            try:
                remote=self.command('abort')
                if not remote.get('aborted') or not remote.get('children_reaped'):
                    raise RuntimeError('peer abort lacks remote child-reap acknowledgement')
                receipt['remote']=remote
            except Exception as caught:error=caught
            if self.child.stdin and not self.child.stdin.closed:self.child.stdin.close()
            try:
                rc=self.child.wait(timeout=self.definition['config']['shutdown_seconds']+self.definition['config']['cleanup_seconds'])
                receipt.update(returncode=rc,transport_reaped=True)
            except Exception as caught:
                error=error or caught
            if self.drainer:
                self.drainer.join(timeout=5)
                if self.drainer.is_alive():error=error or RuntimeError('peer stderr drainer did not exit')
            self.close_pipes()
            receipt['outcome']='uncertain' if error else 'completed'
            if error:receipt['error']=repr(error)
            C.atom(self.root/(self.name+'-abort.json'),receipt)
            if error:raise RuntimeError('peer remote cleanup uncertain: '+self.name) from error

    def close_pipes(self):
        for stream in (self.child.stdin,self.child.stdout,self.child.stderr):
            if stream and not stream.closed:stream.close()
