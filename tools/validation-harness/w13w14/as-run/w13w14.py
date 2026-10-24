#!/usr/bin/env python3
"""w13w14.py — W13/W14 resident CPU worker-count comparison on 02.

Reference P12: __REFERENCE_CAMPAIGN_ROOT__ arm P (post-swap,
turbo, 12 workers): 630 tok / 42.2224 s decode = 14.9210 tok/s.

Per arm (W13 then W14): supported stop of the owned worker -> relaunch with
--workers N and explicit --local-load-threads 4 -> authorized live mask apply
+verify (caller {0}; pool {1..N}; W14 shares physical core 0 with the caller,
disclosed) -> C0 engine session performs the four-thread load (zero GEN,
timed separately) -> one warm engine session with the 3 retained GEN256
requests. Six inference requests total, no warmups. Same PR7 worker ELF
(d897e16c, 4-lane loader; engine 65419a1c unchanged) for both arms; RAM
(256 GB), turbo policy and engine pool (12) fixed. Token streams must match
P12 position-for-position; any mismatch stops the run. Best arm (>= +3%%
decode tok/s vs P12) stays resident; otherwise restore the 12-worker
configuration with a zero-generation four-thread load.
"""
import hashlib, json, os, pathlib, shutil, socket, subprocess, sys, time, traceback

R = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import campaign as C

REF_ROOT = pathlib.Path('__REFERENCE_CAMPAIGN_ROOT__')
OLD_ROOT = pathlib.Path('__OLD_RESIDENT_ROOT__')
WORKER_BYTES = C.WORKER_BYTES
MEANINGFUL = 1.03
HASHES = {
    'campaign.py': 'acf41ef0331b4d706f363c476af4d523aa48f371bc744d71db9eca584bf32b8d',
    'owner.py': '6da70663bebe7e5e8081184df908ed43aa0fafabbc12bd0776b123b06a8c4dcc',
    'worker_owner.py': 'f14d5b941599d82a900626c90b3e70f29df1b479b3061df015d6f14711459dc6',
    'monitor.py': '66411ab44b46988dc5953559fee349459d6a92f46bcc87f23301f2a1b40609f1',
    'worker_control.py': 'eb5b4e905a1367ed221b9dd8054b17e16e21bcc082866d67ef6ce9fa1b784d42',
    'clockmon.py': 'e9a575905a2249b8f901c2edb0b1998a78456b602d30be60f065e2dbdd409d3f',
    'freq_ctl.py': '85ac1ff625d1d8c98ea484bdae6d1df3dd805dca3d43fceba7a26647a32dcc77',
    'strata-engine': '65419a1c85159291c535c03e28321e259a4f50832877b0272c786426a5f0718f',
    'strata-expert-worker-cand': 'd897e16c62a6a3d67d665295a03ae12cb5922b336a66fc7fdc4848d1026ae1be',
    'local.manifest': '8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318',
    'arm-requests.txt': '15162625d52ef5a211490fa1663fc12798bcf86305de24359d39017617ddf824',
}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def mem_avail_kib():
    for l in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if l.startswith('MemAvailable:'):
            return int(l.split()[1])
    raise RuntimeError('no MemAvailable')


def wait_arena_released(before_kib):
    # arena is ~108 GiB; require a >40 GiB MemAvailable rebound (KiB units: 1 GiB = 2^20 KiB)
    for _ in range(90):
        if mem_avail_kib() - before_kib > 40 * 2 ** 20:
            return mem_avail_kib() - before_kib
        time.sleep(2)
    return mem_avail_kib() - before_kib


def port_free():
    # SO_REUSEADDR mirrors the worker listener; worker-side-closed status
    # connections leave TIME_WAIT on this local port for up to 60 s
    end = time.monotonic() + 90
    while True:
        try:
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', C.PORT))
            return True
        except OSError:
            if time.monotonic() >= end:
                raise
            time.sleep(3)


def preflight():
    for f, h in HASHES.items():
        p = R / f
        assert p.exists(), 'missing helper/asset: ' + f
        got = sha(p)
        assert got == h, ('hash mismatch', f, got)
    cfg = json.loads((R / 'config.json').read_text())
    assert cfg['argv'][0] == str(R / 'strata-engine')
    assert cfg['worker_argv'][0] == str(R / 'strata-expert-worker-cand')
    assert cfg['binary_sha256'] == HASHES['strata-engine']
    assert cfg['worker_sha256'] == HASHES['strata-expert-worker-cand']
    assert cfg['manifest_sha256'] == HASHES['local.manifest']
    assert cfg['argv'][cfg['argv'].index('--pool-workers') + 1] == '12'
    assert cfg['worker_argv'][cfg['worker_argv'].index('--workers') + 1] == '12'
    assert cfg['worker_argv'][cfg['worker_argv'].index('--local-load-threads') + 1] == '4'
    old_ids = json.loads((OLD_ROOT / 'WORKER' / 'identities.json').read_text())
    old_alive = pathlib.Path('/proc', str(old_ids['worker']['pid']), 'exe').exists()
    if old_alive:
        # the receipt-bound old resident owns the listener until the authorized
        # stop inside main(); stop_owned_worker proves release with its own bind
        C.atom(R / 'port-held-by-old-resident.json', {'wall': time.time(),
                'old_worker_pid': old_ids['worker']['pid'], 'note': 'expected pre-stop'})
    else:
        assert port_free()
    C.gpu_idle()
    topo = C.topology()
    assert len({(v['physical_package_id'], v['core_id']) for v in topo.values()}) == 14
    assert mem_avail_kib() > 130 * 2 ** 20
    ref = load_reference()
    C.atom(R / 'preflight.json', {'wall': time.time(), 'hashes': 'verified (11 files)',
                                  'engine_pool': '12 fixed', 'load_threads': '4 pinned',
                                  'mem_available_kib': mem_avail_kib(),
                                  'distinct_physical_cores': 14,
                                  'reference_tok_per_s': ref['tok_per_s']})


def load_reference():
    s = json.loads((REF_ROOT / 'arm-P-summary.json').read_text())
    assert s['tokens_total'] == 630 and abs(s['decode_s_sum'] - 42.2224) < 5e-5
    r = s['tokens_total'] / s['decode_s_sum']
    assert abs(r - 14.9210) < 5e-5, r
    rec = {'root': str(REF_ROOT), 'report_sha256_prefix': '26a39d26',
           'tokens_total': s['tokens_total'], 'decode_s_sum': s['decode_s_sum'],
           'prompt_s_sum': s['prompt_s_sum'], 'warm_wall_s': s['warm_wall_s'],
           'worker_service_delta_s': s['worker_service_delta_s'], 'tok_per_s': r,
           'worker_elf_at_reference': 'cf35cdd185cc2d44aeb793b255138ca38cbe9c14b4761d64df6e1d3c804cf7ec'}
    C.atom(R / 'reference-P12.json', rec)
    return rec


def stop_owned_worker(tag, identities):
    wpid = identities['worker']['pid']
    avail_before = mem_avail_kib()
    r = subprocess.run(['python3', str(R / 'worker_control.py'), 'stop', '--port', str(C.PORT)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and 'stop_acknowledged' in r.stdout, (r.returncode, r.stdout, r.stderr)
    deadline = time.monotonic() + 420
    while time.monotonic() < deadline and any(pathlib.Path('/proc', str(v['pid'])).exists() for v in identities.values()):
        time.sleep(2)
    for k, v in identities.items():
        assert not pathlib.Path('/proc', str(v['pid']), 'exe').exists(), (k, v['pid'])
    recovered = wait_arena_released(avail_before)
    assert recovered > 40 * 2 ** 20, ('arena not released', recovered)
    assert port_free()
    return {'wall': time.time(), 'method': 'control opcode 11 (idle, no active session)',
            'identities': identities, 'mem_recovered_kib': recovered}


def stop_old_resident():
    ids = json.loads((OLD_ROOT / 'WORKER' / 'identities.json').read_text())
    wpid = ids['worker']['pid']
    if not pathlib.Path('/proc', str(wpid)).exists():
        assert (R / 'old-worker-stop-receipt.json').exists(), 'old worker gone without receipt'
        return
    rec = stop_owned_worker('old', ids)
    rec.update({'old_root': str(OLD_ROOT),
                'note': 'authorized replacement: T4-resident PR7 worker (0 requests, idle) -> W13 arm'})
    C.atom(R / 'old-worker-stop-receipt.json', rec)


def phys_of(logicals):
    out = {}
    for c in logicals:
        p = pathlib.Path('/sys/devices/system/cpu', 'cpu' + str(c), 'topology', 'core_id')
        out[c] = p.read_text().strip()
    return out


def apply_masks(name, n):
    """Authorized explicit live masks on the owned worker: caller -> {0},
    pool TIDs -> {1..N}. Verify by re-reading /proc after the set."""
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    wpid = ids['worker']['pid']
    for _ in range(60):
        wt = C.tasks(wpid)
        if len(wt) >= n + 2:
            break
        time.sleep(0.5)
    wt = C.tasks(wpid)
    assert len(wt) == n + 2, ('unexpected task count', len(wt), n + 2)
    before = [{'tid': t['tid'], 'mask': t['mask'], 'comm': t['comm']} for t in wt]
    os.sched_setaffinity(wpid, {0})
    for i, t in enumerate(wt[1:1 + n]):
        os.sched_setaffinity(t['tid'], {i + 1})
    wt2 = C.tasks(wpid)
    assert wt2[0]['mask'] == [0], wt2[0]
    assert [t['mask'] for t in wt2[1:1 + n]] == [[c] for c in range(1, n + 1)], wt2
    pool_phys = phys_of(range(1, n + 1))
    caller_phys = phys_of([0])
    shared = sorted({c for c, p in pool_phys.items() if p == caller_phys[0]})
    distinct_phys = len(set(pool_phys.values()))
    assert distinct_phys == n, ('pool does not cover N distinct physical cores', pool_phys)
    C.atom(R / ('masks-%s.json' % name), {'wall': time.time(), 'pid': wpid,
            'arm': name, 'workers': n,
            'before': before,
            'after': [{'tid': t['tid'], 'mask': t['mask'], 'comm': t['comm']} for t in wt2],
            'pool_logical_to_physical': pool_phys, 'caller_logical_to_physical': caller_phys,
            'pool_distinct_physical_cores': distinct_phys,
            'pool_sharing_caller_physical_core': shared,
            'method': 'os.sched_setaffinity on owned caller+pool TIDs (authorized explicit mask arm)'})


def load_arm(name, n):
    """Stop+archive the current owned worker, relaunch with N workers / 4 load
    lanes, apply+verify masks, then ONE C0 engine session performs the load
    (zero GEN) with startup timed separately."""
    if (R / 'WORKER').exists():
        C.state('w-stop-before-' + name)
        rec = stop_owned_worker('before-' + name, json.loads((R / 'WORKER' / 'identities.json').read_text()))
        C.atom(R / ('worker-stop-before-' + name + '.json'), rec)
        k = 1
        while (R / ('WORKER-arch%d' % k)).exists():
            k += 1
        shutil.move(str(R / 'WORKER'), str(R / ('WORKER-arch%d' % k)))
    cfg = json.loads((R / 'config.json').read_text())
    i = cfg['worker_argv'].index('--workers')
    cfg['worker_argv'][i + 1] = str(n)
    assert cfg['worker_argv'][cfg['worker_argv'].index('--local-load-threads') + 1] == '4'
    (R / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    C.gpu_idle()
    C.state('w-launch-' + name)
    C.mark(phase='load', arm=name, req=0)
    t_spawn = time.monotonic()
    subprocess.Popen(['python3', '-B', str(R / 'worker_owner.py')],
                     stdin=subprocess.DEVNULL, stdout=(R / 'worker-owner.log').open('ab'),
                     stderr=subprocess.STDOUT, start_new_session=True)
    C.waitfile(R / 'WORKER' / 'listening.json', lambda v: True, 120)
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    (R / 'clockmon-worker-pid').write_text(str(ids['worker']['pid']) + '\n')
    apply_masks(name, n)
    fresh = C.wstatus()
    assert fresh['status']['loaded_bytes'] == 0 and fresh['status']['requests'] == 0
    C.listener_owned()
    sess = 'C0-' + name
    assert not (R / sess).exists()
    eowner = subprocess.Popen(['python3', '-B', str(R / 'owner.py'), sess],
                              stdin=subprocess.DEVNULL,
                              stdout=(R / ('engine-owner-%s.log' % name)).open('ab'),
                              stderr=subprocess.STDOUT, start_new_session=True)
    st = C.wait_session(R / sess, lambda v: v['ready'] is not None, 2450, owner_pid=eowner.pid)
    engine_to_ready = st['ready']['mono'] - st['started_mono']
    spawn_to_ready = st['ready']['mono'] - t_spawn
    text = (R / sess / 'stderr.log').read_text()
    markers = {k: (k in text) for k in ['cache_hit=0', 'setup network=0 B', '116135731200',
                                        'remote CPU experts: 22238 of 24576 experts']}
    C.rpc(sess, 'send', 'QUIT')
    eowner.wait(timeout=300)
    v = json.loads((R / sess / 'state.json').read_text())
    assert v.get('phase') == 'exited' and v.get('exit_rc') == 0 and v['requests_sent'] == 0, v
    for _ in range(30):
        if all(not pathlib.Path('/proc', str(x['pid']), 'exe').exists()
               for x in json.loads((R / sess / 'identities.json').read_text()).values()):
            break
        time.sleep(1)
    else:
        for x in json.loads((R / sess / 'identities.json').read_text()).values():
            assert not pathlib.Path('/proc', str(x['pid']), 'exe').exists(), x
    C.gpu_idle()
    ws = C.wstatus()
    s = ws['status']
    assert s['complete'] == 1 and s['resident_bytes'] == s['loaded_bytes'] == s['verified_bytes'] == WORKER_BYTES
    assert s['arena_allocations'] == 1 and s['requests'] == 0 and s['network_weight_bytes'] == 0
    werr = (R / 'WORKER' / 'stderr.log').read_text(errors='replace')
    lane_lines = [l for l in werr.splitlines() if 'lane' in l.lower() or 'local loading' in l.lower()]
    (R / 'worker-identities.json').write_bytes((R / 'WORKER' / 'identities.json').read_bytes())
    rec = {'arm': name, 'workers': n, 'load_threads': 4,
           'engine_to_ready_s': round(engine_to_ready, 3),
           'spawn_to_ready_s': round(spawn_to_ready, 3),
           'engine_stderr_markers': markers, 'worker_lane_lines': lane_lines,
           'worker_status': s}
    C.atom(R / ('startup-%s.json' % name), rec)
    C.state('w-load-' + name + '-complete', engine_to_ready_s=rec['engine_to_ready_s'])
    return rec


def make_affinity(n):
    def affinity(arm, tag, need_io=False):
        wi = C.ident(json.loads((R / 'worker-identities.json').read_text())['worker'])
        wt = C.tasks(wi['pid'])
        assert len(wt) == n + 2 and wt[0]['mask'] == [0], 'worker caller mask changed'
        assert [x['mask'] for x in wt[1:1 + n]] == [[c] for c in range(1, n + 1)], 'pool masks changed'
        ei = json.loads((R / arm / 'state.json').read_text())['engine']
        C.ident(ei)
        et = C.tasks(ei['pid'])
        assert et[0]['mask'] == [0], 'engine main not on CPU0'
        io = [x for x in et if x['comm'] == 'strata-cpu-io']
        if need_io:
            assert len(io) == 1, io
        out = {'wall': time.time(), 'worker': wi, 'worker_pool': wt[1:1 + n],
               'worker_caller': wt[0], 'worker_auxiliary': wt[1 + n:], 'engine': ei,
               'engine_main': et[0], 'engine_pool': et[1:13] if len(et) > 13 else [],
               'communication_threads': io, 'topology': C.topology(),
               'workers': n}
        C.atom(R / (tag + '-affinity.json'), out)
        return out
    return affinity


def compare_tokens(rows, ref_rows, name):
    per = []
    for r, b in zip(rows, ref_rows):
        per.append({'index': r['index'], 'output_count': r['output_count'],
                    'worker_entries': r.get('worker_entries'), 'match': r['tokens'] == b['tokens']})
    ok = len(rows) == 3 and all(p['match'] for p in per)
    C.atom(R / ('tokens-%s.json' % name), {'wall': time.time(), 'identical_to_P12': ok,
                                           'per_request': per})
    return ok, per


def rate(summary):
    return summary['tokens_total'] / summary['decode_s_sum']


def main():
    C.state('w13w14-preflight')
    preflight()
    C.freq_ctl('snapshot', str(R / 'freq-snapshot.json'))
    clockmon = subprocess.Popen(['python3', '-B', str(R / 'clockmon.py'), str(R), '10800'],
                                stdin=subprocess.DEVNULL,
                                stdout=(R / 'clockmon.log').open('ab'),
                                stderr=subprocess.STDOUT, start_new_session=True)
    C.atom(R / 'clockmon-launch.json', {'pid': clockmon.pid, 'wall': time.time()})
    time.sleep(2)
    try:
        stop_old_resident()
        ref = json.loads((R / 'reference-P12.json').read_text())
        ref_rows = json.loads((REF_ROOT / 'arm-P-summary.json').read_text())['requests']
        results, startups, tokchecks = {}, {}, {}
        for name, n in [('W13', 13), ('W14', 14)]:
            startups[name] = load_arm(name, n)
            C.affinity = make_affinity(n)
            C.mark(phase='arm', arm=name, req=0)
            summary = C.run_arm(name, idxs=(0, 1, 2))
            ok, per = compare_tokens(summary['requests'], ref_rows, name)
            assert ok, 'token stream diverges from P12 in ' + name
            results[name] = summary
            tokchecks[name] = per
            C.state('w-arm-' + name + '-complete',
                    tok_per_s=round(rate(summary), 4),
                    decode_s_sum=summary['decode_s_sum'])
        p12 = ref['tok_per_s']
        w13, w14 = rate(results['W13']), rate(results['W14'])
        best = None
        if w13 >= p12 * MEANINGFUL and w13 >= w14:
            best = 'W13'
        elif w14 >= p12 * MEANINGFUL and w14 > w13:
            best = 'W14'
        if best is None:
            verdict = 'no meaningful improvement; restoring 12-worker reference config'
            startups['R12'] = load_arm('R12', 12)
            resident, resident_n = 'R12', 12
        elif best == 'W14':
            verdict = 'W14 best and already resident'
            resident, resident_n = 'W14', 14
        else:
            verdict = 'W13 best; reloading it as the resident configuration'
            startups['W13R'] = load_arm('W13R', 13)
            resident, resident_n = 'W13R', 13
        C.state('decision', verdict=verdict, resident=resident,
                w13_tok_per_s=round(w13, 4), w14_tok_per_s=round(w14, 4), p12_tok_per_s=p12)
        C.atom(R / 'decision.json', {'wall': time.time(), 'verdict': verdict, 'resident': resident,
                                     'p12': p12, 'W13': w13, 'W14': w14,
                                     'meaningful_threshold': MEANINGFUL})
    finally:
        C.mark(phase='final-check')
        (R / 'clockmon.STOP').write_text('done\n')
        try:
            clockmon.wait(timeout=60)
        except subprocess.TimeoutExpired:
            clockmon.terminate()
    fin = C.wstatus()
    C.healthy()
    C.atom(R / 'worker-final.json', fin)
    write_report(results, startups, resident, resident_n, verdict, fin, tokchecks)
    C.state('completed-' + resident + '-resident')
    print('W13W14 COMPLETE', flush=True)


def write_report(results, startups, resident, resident_n, verdict, fin, tokchecks):  # noqa: E821
    ref = json.loads((R / 'reference-P12.json').read_text())
    lines = []
    lines.append('W13-W14-POOL-COUNT-20261024 — worker-count comparison vs P12')
    lines.append('')
    lines.append('REFERENCE (bound live from ' + ref['root'] + ', REPORT sha prefix ' + ref['report_sha256_prefix'] + ')')
    lines.append('  P12 (12 workers, post-swap 256 GB, turbo): %.4f tok/s = %d tok / %.4f s decode; warm wall %.2f s; prompt %.4f s' % (
        ref['tok_per_s'], ref['tokens_total'], ref['decode_s_sum'], ref['warm_wall_s'], ref['prompt_s_sum']))
    lines.append('')
    lines.append('ARMS (same 3 retained GEN256 requests, 6 requests total, no warmups; engine 65419a1c,')
    lines.append('worker ELF d897e16c = PR7 loader with explicit --local-load-threads 4 BOTH arms;')
    lines.append('engine pool 12 fixed; caller pinned {0}; pool pinned {1..N} live and verified)')
    for name in ['W13', 'W14']:
        s = results[name]
        st = startups[name]
        n = st['workers']
        lines.append('  %s (%d workers): %.4f tok/s = %d tok / %.4f s decode; warm wall %.2f s; prompt %.4f s' % (
            name, n, rate(s), s['tokens_total'], s['decode_s_sum'], s['warm_wall_s'], s['prompt_s_sum']))
        lines.append('    worker service delta %.3f s; startup engine-to-READY %.1f s (four-thread load, timed separately)' % (
            s['worker_service_delta_s'], st['engine_to_ready_s']))
        entries = [str(r.get('worker_entries')) for r in s['requests']]
        lines.append('    worker_entries per request: ' + '/'.join(entries) + ' (P12: 102724/108052/94299)')
        lines.append('    token streams identical to P12: %s' % all(p['match'] for p in tokchecks[name]))
        mk = json.loads((R / ('masks-%s.json' % name)).read_text())
        if mk['pool_sharing_caller_physical_core']:
            lines.append('    DISCLOSED: pool logical %s share physical core %s with the caller' % (
                mk['pool_sharing_caller_physical_core'], mk['caller_logical_to_physical']['0']))
        else:
            lines.append('    pool on %d distinct physical cores; caller physical core not shared (sibling logical 14 idle)' % n)
    lines.append('')
    lines.append('DISCLOSURES')
    lines.append('- Worker ELF changed vs P12: cf35cdd1 (pre-PR7 serial loader) -> d897e16c (PR7 4-lane')
    lines.append('  loader, launched with explicit --local-load-threads 4). Loader implementation/default')
    lines.append('  change only — NOT a simultaneous PR6 optimization; PR6 remains unmerged and the engine')
    lines.append('  binary is unchanged. Weights byte-identical: all arms reproduce P12 tokens exactly.')
    lines.append('- P12 cold load was 513.4 s serial pre-PR7; W-arm startups are warm-cache four-thread')
    lines.append('  loads (~60-70 s expected) and are NOT comparable to that 513.4 s figure.')
    lines.append('- Turbo policy, governor and RAM population unchanged throughout; no frequency writes.')
    lines.append('')
    lines.append('VERDICT: ' + verdict)
    lines.append('RESIDENT: %s (%d workers) — arena %d B complete, no expiry' % (
        resident, resident_n, fin['status']['resident_bytes']))
    ids = json.loads((R / 'WORKER' / 'identities.json').read_text())
    lines.append('  worker pid %d (start_ticks %d), guardian %d, monitor %d; port %d' % (
        ids['worker']['pid'], ids['worker']['start_ticks'], ids['guardian']['pid'],
        ids['monitor']['pid'], C.PORT))
    lines.append('  status: python3 %s/tools/worker_control.py status --host 127.0.0.1 --port %d' % (C.ref_root_hint() if hasattr(C, 'ref_root_hint') else str(R), C.PORT))
    lines.append('  stop:   python3 %s/worker_control.py stop --port %d' % (str(R), C.PORT))
    lines.append('')
    lines.append('EVIDENCE: ' + str(R) + ' (startup-*.json, masks-*.json, arm-*-summary.json,')
    lines.append('tokens-*.json, decision.json, worker-final.json, clock-journal.jsonl)')
    (R / 'REPORT.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        C.atom(R / 'w13w14-failure.json', {'wall': time.time(), 'error': repr(e),
                                           'traceback': traceback.format_exc()})
        try:
            for arm in ['C0-W13', 'C0-W14', 'W13', 'W14', 'C0-W13R', 'W13R', 'C0-R12', 'R12']:
                if (R / arm / 'input.sock').exists():
                    st = json.loads((R / arm / 'state.json').read_text())
                    if st.get('active'):
                        C.rpc(arm, 'send', 'STOP')
                    C.rpc(arm, 'send', 'QUIT')
        except Exception:
            pass
        try:
            C.mark(phase='aborted')
            (R / 'clockmon.STOP').write_text('abort\n')
        except Exception:
            pass
        C.state('failed')
        raise
