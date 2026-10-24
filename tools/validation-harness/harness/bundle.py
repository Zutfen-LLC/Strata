"""One source/config/asset staging operation. No process launch or remote transport."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import socket
import sys

HELPERS = ('bundle.py', 'run.py', 'campaign.py', 'owner.py', 'worker_owner.py', 'monitor.py', 'worker_control.py')
NAME = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,31}\Z')


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def absolute(path, executable=False):
    p = pathlib.Path(path)
    if not p.is_absolute() or '__' in str(p) or not p.exists():
        raise ValueError('missing, relative or placeholder path: ' + str(p))
    if not os.access(p, os.R_OK) or (p.is_dir() and not os.access(p, os.X_OK)):
        raise ValueError('unreadable/inaccessible path: ' + str(p))
    if executable and (not p.is_file() or not os.access(p, os.X_OK)):
        raise ValueError('not an executable file: ' + str(p))
    return p


def imports(directory):
    for name in HELPERS:
        path = directory / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                first = module.split('.')[0]
                if (directory / (first + '.py')).is_file(): continue
                if importlib.util.find_spec(first) is None:
                    raise ValueError('missing import in ' + name + ': ' + first)


def validate(cfg, source):
    imports(source)  # parse sources and resolve dependencies; never import the side-effecting owners
    if re.search(r'__[A-Z0-9_]+__', json.dumps(cfg)): raise ValueError('unfilled config placeholder')
    absolute(sys.executable, True)
    if cfg['host'] != socket.gethostname(): raise ValueError('config host differs from execution host')
    root = pathlib.Path(cfg['run_root'])
    if not root.is_absolute() or '__' in str(root): raise ValueError('output root must be absolute and configured')
    if not 1 <= cfg['port'] <= 65535 or cfg['worker_host'] != '127.0.0.1':
        raise ValueError('retained harness requires a numeric loopback worker endpoint')
    if not 0 < cfg['worker_bytes'] < (1 << 50): raise ValueError('invalid expected arena bytes')
    for key, path in cfg['assets'].items():
        if not NAME.fullmatch(key): raise ValueError('invalid asset name')
        p = absolute(path, key.startswith(('engine', 'worker')))
        if not p.is_file(): raise ValueError('staged assets must be files')
    for key in ('engine', 'worker', 'manifest', 'requests'):
        if key not in cfg['assets']: raise ValueError('missing required asset: ' + key)
    for path in cfg['model_paths']: absolute(path)
    for argv in (cfg['argv'], cfg['worker_argv']):
        if not argv or not isinstance(argv[0], str): raise ValueError('empty executable argv')
        for index, arg in enumerate(argv):
            if not isinstance(arg, str) or '\0' in arg: raise ValueError('invalid argv string')
            if arg.startswith('@'):
                if arg[1:] not in cfg['assets']: raise ValueError('unknown asset binding: ' + arg)
            elif index == 0 or '/' in arg or arg.endswith(('.gguf', '.json', '.manifest', '.bin', '.txt')):
                absolute(arg, index == 0)
        for flag in ('--workers', '--local-load-threads') if argv is cfg['worker_argv'] else ():
            if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
                raise ValueError('missing/duplicate configurable option: ' + flag)
    def value(argv, flag):
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv): raise ValueError('invalid option: ' + flag)
        return argv[argv.index(flag) + 1]
    if value(cfg['worker_argv'], '--port') != str(cfg['port']): raise ValueError('worker/control port mismatch')
    if '--bind-ip' in cfg['worker_argv'] and value(cfg['worker_argv'], '--bind-ip') != cfg['worker_host']:
        raise ValueError('worker/control address mismatch')
    if value(cfg['argv'], '--remote-cpu-experts') != cfg['worker_host'] + ':' + str(cfg['port']):
        raise ValueError('engine/control endpoint mismatch')
    def resolve(arg): return cfg['assets'][arg[1:]] if arg.startswith('@') else arg
    for argv, flag in [(cfg['argv'], '--remote-cpu-local-manifest'), (cfg['worker_argv'], '--local-manifest')]:
        if resolve(value(argv, flag)) != cfg['assets']['manifest']: raise ValueError('manifest binding mismatch')
    if cfg['worker_argv'].count('--residency-forever') != 1 or '--residency-ms' in cfg['worker_argv']:
        raise ValueError('retained runtime sequence requires explicit service-owned residency')
    if not isinstance(cfg['gpu_uuids'], list) or not 1 <= len(cfg['gpu_uuids']) <= 8 or any(not isinstance(v,str) or not v for v in cfg['gpu_uuids']):
        raise ValueError('explicit GPU visibility list required')
    if any(arg.startswith('--remote-gpu-') or arg == '--gpu' for arg in cfg['argv']):
        raise ValueError('retained engine broker does not support network GPU modes')
    if 'STRATA_ARENA_MMAP' in cfg['env'] or 'STRATA_PRIMARY_DEVICE' in cfg['env']:
        raise ValueError('unsupported engine environment override')
    allowed = os.sched_getaffinity(0)
    lines = [line for line in pathlib.Path(cfg['assets']['requests']).read_text().splitlines() if line.startswith('GEN ')]
    if not lines or len(lines) > 64 or any(not line.startswith('GEN ') or len(line) > 16000 for line in lines):
        raise ValueError('request file must contain 1..64 bounded retained GEN lines')
    names = set(); running_worker = None
    for arm in cfg['arms']:
        if not NAME.fullmatch(arm['name']) or arm['name'] in names: raise ValueError('invalid/duplicate arm name')
        if arm['name'] in ('assets', 'WORKER') or arm['name'].startswith('WORKER-before-'):
            raise ValueError('arm name collides with reserved runtime paths')
        if len(os.fsencode(root / arm['name'] / 'input.sock')) >= 108:
            raise ValueError('output path exceeds Linux Unix-socket bound')
        names.add(arm['name'])
        if not 1 <= arm['workers'] <= 256 or not 1 <= arm['loader_threads'] <= 4:
            raise ValueError('worker/loader count outside bounds')
        if len(arm['pool_masks']) != arm['workers']: raise ValueError('one configured pool mask per worker required')
        for mask in [arm['caller_mask'], arm['engine_main_mask'], *arm['pool_masks']]:
            if not mask or len(mask) > 4096 or any(type(cpu) is not int or cpu < 0 for cpu in mask):
                raise ValueError('invalid affinity mask')
            if not set(mask) <= allowed: raise ValueError('configured CPU outside current allowed set')
        if len(arm['requests']) > 64 or any(type(i) is not int or not 0 <= i < len(lines) for i in arm['requests']):
            raise ValueError('invalid request selection')
        for field, default in [('engine', 'engine'), ('worker', 'worker')]:
            if arm.get(field, default) not in cfg['assets']: raise ValueError('missing arm asset')
            absolute(cfg['assets'][arm.get(field, default)], True)
        if not isinstance(arm['restart'], bool): raise ValueError('restart must be explicit boolean')
        if arm['restart']:
            if arm['requests']: raise ValueError('cold-load entries must have no requests; select a separate warm entry')
            running_worker = (arm['workers'], arm['loader_threads'], arm.get('worker', 'worker'))
        elif running_worker is None:
            running_worker = (arm['workers'], arm['loader_threads'], arm.get('worker', 'worker'))
        elif running_worker != (arm['workers'], arm['loader_threads'], arm.get('worker', 'worker')):
            raise ValueError('reuse cannot change worker binary/count/loader settings')
        if not isinstance(arm['placement_markers'], list) or any(not isinstance(v, str) for v in arm['placement_markers']):
            raise ValueError('placement markers must be configured strings')
        if arm.get('reference'):
            if arm['reference'] not in cfg['assets']: raise ValueError('missing reference asset')
            reference = json.loads(pathlib.Path(cfg['assets'][arm['reference']]).read_text())['requests']
            indexed = {row['index']: row for row in reference}
            if len(indexed) != len(reference): raise ValueError('duplicate reference request index')
            for index in arm['requests']:
                row = indexed[index + 1]
                if not isinstance(row['tokens'], list) or 'worker_entries' not in row:
                    raise ValueError('invalid reference request')
    if not cfg['arms'] or len(cfg['arms']) > 32: raise ValueError('arm count must be 1..32')
    for key in ('startup_seconds', 'worker_start_seconds', 'request_seconds', 'shutdown_seconds', 'cleanup_seconds'):
        if not 0 < cfg[key] <= 14400: raise ValueError('invalid finite deadline: ' + key)
    if cfg['final_worker'] not in ('retain', 'stop') or cfg['failure_worker'] not in ('retain', 'stop'):
        raise ValueError('explicit retain/stop policy required')
    if cfg.get('attach_root'):
        absolute(cfg['attach_root'])
        if not cfg.get('attached_worker_idle') or cfg['final_worker'] != 'retain' or cfg['failure_worker'] != 'retain':
            raise ValueError('attached resident requires explicit idle confirmation and retain policies')
        if any(arm['restart'] for arm in cfg['arms']): raise ValueError('cannot restart an attached guardian')
    elif not cfg['arms'][0]['restart']: raise ValueError('first arm must launch or explicitly attach a worker')
    for key in ('env', 'worker_env', 'monitor_env'):
        if not isinstance(cfg[key], dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in cfg[key].items()):
            raise ValueError('environments must contain only string bindings')
    if not cfg['monitor_env'].get('PATH'): raise ValueError('monitor PATH must be explicitly configured')
    if shutil.which('nvidia-smi', path=cfg['monitor_env'].get('PATH')) is None:
        raise ValueError('retained independent monitor requires nvidia-smi on the execution host')
    return lines


def stage(config_path):
    source = pathlib.Path(__file__).resolve().parent
    if pathlib.Path(config_path).stat().st_size > 2 << 20: raise ValueError('config exceeds 2 MiB')
    cfg = json.loads(pathlib.Path(config_path).read_text())
    lines = validate(cfg, source)  # all paths/imports checked before creating output or starting owners
    root = pathlib.Path(cfg['run_root']); root.mkdir(mode=0o700, exist_ok=False)
    (root / 'assets').mkdir(mode=0o700)
    pins = {}
    for name, src in [(name, source / name) for name in HELPERS] + [
            ('assets/' + name, pathlib.Path(path)) for name, path in cfg['assets'].items()]:
        destination = root / name
        with src.open('rb') as inp, destination.open('xb') as out: shutil.copyfileobj(inp, out, 1 << 20)
        shutil.copymode(src, destination)
        before = sha(src); after = sha(destination)
        if before != after: raise RuntimeError('source changed during staging: ' + name)
        pins[name] = after
    cfg['request_lines'] = lines
    cfg['assets'] = {name: str(root / 'assets' / name) for name in cfg['assets']}
    cfg['worker_root'] = str(pathlib.Path(cfg['attach_root']) / 'WORKER') if cfg.get('attach_root') else str(root / 'WORKER')
    cfg['argv'] = [cfg['assets'][arg[1:]] if arg.startswith('@') else arg for arg in cfg['argv']]
    cfg['worker_argv'] = [cfg['assets'][arg[1:]] if arg.startswith('@') else arg for arg in cfg['worker_argv']]
    with (root / 'config.json').open('x') as out: json.dump(cfg, out, indent=2)
    pins['config.json'] = sha(root / 'config.json')
    with (root / 'pins.json').open('x') as out: json.dump(pins, out, indent=2)
    verify(root)
    return root


def verify(root):
    root = pathlib.Path(root)
    pins = json.loads((root / 'pins.json').read_text())
    required = set(HELPERS) | {'config.json', 'assets/engine', 'assets/worker', 'assets/manifest', 'assets/requests'}
    if not required <= pins.keys(): raise ValueError('incomplete staging manifest')
    for name, expected in pins.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()): raise ValueError('invalid staged path')
        if sha(path) != expected: raise ValueError('staged bytes differ: ' + name)
    imports(root)
    cfg = json.loads((root / 'config.json').read_text())
    validate(cfg, root)
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(stage(args.config))
