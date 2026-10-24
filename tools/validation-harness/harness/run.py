"""Fixed runtime sequence extracted from W13/W14 and PR7; arms come only from staged config."""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

import bundle
import campaign as C


def option(argv, flag, value):
    argv = list(argv)
    if argv.count(flag) != 1: raise ValueError('expected one option: ' + flag)
    argv[argv.index(flag) + 1] = str(value)
    return argv


def arm_config(cfg, arm):
    result = dict(cfg)
    result['worker_count'] = arm['workers']
    result['worker_argv'] = option(option(cfg['worker_argv'], '--workers', arm['workers']),
                                   '--local-load-threads', arm['loader_threads'])
    result['worker_argv'][0] = cfg['assets'][arm.get('worker', 'worker')]
    result['argv'] = list(cfg['argv']); result['argv'][0] = cfg['assets'][arm.get('engine', 'engine')]
    result['binary_sha256'] = bundle.sha(result['argv'][0])
    result['worker_sha256'] = bundle.sha(result['worker_argv'][0])
    result['manifest_path'] = cfg['assets']['manifest']
    result['manifest_sha256'] = bundle.sha(result['manifest_path'])
    result['max_requests'] = len(arm['requests'])
    result['placement_markers'] = arm['placement_markers']
    result['engine_main_mask'] = arm['engine_main_mask']
    return result


def masks(arm, session=None, need_io=False):
    if session is None: C.assert_worker_idle()
    ids = json.loads((C.W / 'identities.json').read_text())
    worker = C.ident(ids['worker']); rows = C.tasks(worker['pid'])
    if session is None and bundle.sha(pathlib.Path('/proc', str(worker['pid']), 'exe')) != C.cfg['worker_sha256']:
        raise RuntimeError('resident executable differs from selected worker asset')
    for flag, expected in [('--workers', arm['workers']), ('--local-load-threads', arm['loader_threads'])]:
        if worker['argv'].count(flag) != 1 or worker['argv'][worker['argv'].index(flag) + 1] != str(expected):
            raise RuntimeError('resident worker argv differs from configured arm: ' + flag)
    if len(rows) != arm['workers'] + 2: raise RuntimeError('unexpected worker task count')
    expected = [arm['caller_mask'], *arm['pool_masks']]
    for row, mask in zip(rows, expected):
        if session is None: os.sched_setaffinity(row['tid'], set(mask))
    rows = C.tasks(worker['pid'])
    if [row['mask'] for row in rows[:len(expected)]] != [sorted(set(mask)) for mask in expected]:
        raise RuntimeError('configured worker masks changed')
    engine_rows = []
    if session is not None:
        engine = json.loads((C.R / session / 'state.json').read_text())['engine']; C.ident(engine)
        engine_rows = C.tasks(engine['pid'])
        if engine_rows[0]['mask'] != sorted(set(arm['engine_main_mask'])): raise RuntimeError('engine main mask changed')
    io = [row for row in engine_rows if row['comm'] == 'strata-cpu-io']
    if need_io and len(io) != 1: raise RuntimeError('missing communication task')
    receipt = {'worker_tasks': rows, 'engine_tasks': engine_rows, 'communication_threads': io, 'topology': C.topology()}
    C.atom(C.R / (arm['name'] + '-affinity.json'), receipt)
    return receipt


def launch_worker(configuration, arm):
    C.assert_worker_idle()
    if C.W.exists():
        C.stop_owned_worker()
        shutil.move(str(C.W), str(C.R / ('WORKER-before-' + arm['name'])))
    else:
        C.port_free(configuration['worker_host'], configuration['port'], configuration['cleanup_seconds'])
    path = C.R / ('worker-' + arm['name'] + '.config.json')
    with path.open('x') as out: json.dump(configuration, out, indent=2)
    C.GUARDIAN = subprocess.Popen([sys.executable, '-B', str(C.R / 'worker_owner.py'), str(path)],
        stdin=subprocess.DEVNULL, stdout=(C.R / ('worker-owner-' + arm['name'] + '.log')).open('xb'),
        stderr=subprocess.STDOUT, start_new_session=True)
    C.wait_session(C.W, lambda row: (C.W / 'listening.json').exists() and (C.W / 'health.json').exists(),
                   configuration['worker_start_seconds'], C.GUARDIAN)
    masks(arm)


def run(root):
    root = pathlib.Path(root).resolve()
    cfg = bundle.verify(root)  # complete preflight before import binding or any Popen
    if pathlib.Path(C.__file__).resolve() != root / 'campaign.py': raise RuntimeError('stale campaign import')
    if pathlib.Path(C.worker_control.__file__).resolve() != root / 'worker_control.py': raise RuntimeError('stale control import')
    C.configure(root, {**cfg, 'worker_count': cfg['arms'][0]['workers']})
    results = []
    try:
        for arm in cfg['arms']:  # no implicit warmup, recovery, extra serial arm or performance decision
            bundle.verify(root)
            configuration = arm_config(cfg, arm)
            C.configure(root, configuration)
            if arm['restart']: launch_worker(configuration, arm)
            else: masks(arm, session=None)  # explicit masks are also checked on attached/reused workers
            summary = C.run_arm(arm['name'], configuration, arm['requests'],
                                 lambda session, io: masks(arm, session, io))
            if arm.get('reference'):
                reference = json.loads(pathlib.Path(cfg['assets'][arm['reference']]).read_text())['requests']
                expected = {row['index']: row for row in reference}
                for row in summary['requests']:
                    prior = expected[row['index']]
                    if row['tokens'] != prior['tokens'] or row['worker_entries'] != prior['worker_entries']:
                        raise RuntimeError('reference tokens/worker entries diverged')
            results.append(summary)
        if cfg['final_worker'] == 'stop': C.stop_owned_worker()
        C.atom(root / 'summary.json', {'arms': results, 'final_worker': cfg['final_worker']})
    except BaseException as error:
        C.atom(root / 'failure.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        for session in list(C.SESSIONS):
            try: C.finish_session(session, abort=True)
            except Exception as cleanup: C.atom(root / (session + '-cleanup-uncertain.json'), {'error': repr(cleanup)})
        if cfg['failure_worker'] == 'stop' and not C.SESSIONS and C.GUARDIAN is not None:
            # Never retry an uncertain STOP; the retained child may still process it later.
            receipt = C.W / 'stop.json'
            if not receipt.exists(): C.stop_owned_worker()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    run(parser.parse_args().run_root)
