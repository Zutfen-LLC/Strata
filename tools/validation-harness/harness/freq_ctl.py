#!/usr/bin/env python3
"""freq_ctl.py — snapshot / transition / verify cpufreq policies on the target host.

Eric's post-boot baseline configuration (to restore on ALL exits):
  intel_pstate passive, no_turbo=0, driver intel_cpufreq, governor schedutil,
  scaling_min_freq=1200000, scaling_max_freq=3000000 on all 28 CPUs.
Capped arm: scaling_max_freq=2000000 on all 28 CPUs (turbo stays enabled at the
global level; the per-policy ceiling enforces the 2 GHz software limit).
"""
import json, os, pathlib, subprocess, sys, time

CPUS = list(range(28))
BASE = {'min': 1200000, 'max': 3000000}


def cpusys(cpu, name):
    return pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}')


def read_policy():
    row = {
        'wall': time.time(),
        'pstate_status': pathlib.Path('/sys/devices/system/cpu/intel_pstate/status').read_text().strip(),
        'no_turbo': pathlib.Path('/sys/devices/system/cpu/intel_pstate/no_turbo').read_text().strip(),
        'driver': cpusys(0, 'scaling_driver').read_text().strip(),
        'governor': cpusys(0, 'scaling_governor').read_text().strip(),
        'governors_all': sorted({cpusys(c, 'scaling_governor').read_text().strip() for c in CPUS}),
        'drivers_all': sorted({cpusys(c, 'scaling_driver').read_text().strip() for c in CPUS}),
        'cpus': {},
    }
    for c in CPUS:
        row['cpus'][str(c)] = {
            'min': int(cpusys(c, 'scaling_min_freq').read_text()),
            'max': int(cpusys(c, 'scaling_max_freq').read_text()),
            'cpuinfo_min': int(cpusys(c, 'cpuinfo_min_freq').read_text()),
            'cpuinfo_max': int(cpusys(c, 'cpuinfo_max_freq').read_text()),
            'cur': int(cpusys(c, 'scaling_cur_freq').read_text()),
        }
    return row


def write_max(value):
    """One sudo bash loop writes scaling_max_freq on all 28 CPUs."""
    script = 'for c in $(seq 0 27); do echo %d > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_max_freq; done' % value
    return subprocess.run(['sudo', '-n', 'bash', '-c', script], capture_output=True, text=True)


def verify(value, deadline_s=8.0):
    """Repeated stable readback: all 28 scaling_max_freq == value exactly."""
    started = time.monotonic()
    attempts = []
    while time.monotonic() - started < deadline_s:
        cur = {c: int(cpusys(c, 'scaling_max_freq').read_text()) for c in CPUS}
        ok = all(v == value for v in cur.values())
        attempts.append({'elapsed': round(time.monotonic() - started, 3), 'ok': ok,
                         'bad': {str(c): v for c, v in cur.items() if v != value} if not ok else {}})
        if ok:
            return {'ok': True, 'attempts': attempts, 'elapsed': round(time.monotonic() - started, 3)}
        time.sleep(0.3)
    return {'ok': False, 'attempts': attempts, 'elapsed': round(time.monotonic() - started, 3)}


def equal_baseline(snap, pol):
    """pol matches snapshot on every controlled knob."""
    bad = {}
    if pol['no_turbo'] != snap['no_turbo']:
        bad['no_turbo'] = (snap['no_turbo'], pol['no_turbo'])
    if pol['driver'] != snap['driver'] or pol['drivers_all'] != snap['drivers_all']:
        bad['driver'] = (snap['driver'], pol['driver'], pol['drivers_all'])
    if pol['governors_all'] != snap['governors_all']:
        bad['governor'] = (snap['governors_all'], pol['governors_all'])
    for c in CPUS:
        a, b = snap['cpus'][str(c)], pol['cpus'][str(c)]
        for k in ('min', 'max', 'cpuinfo_min', 'cpuinfo_max'):
            if a[k] != b[k]:
                bad[f'cpu{c}.{k}'] = (a[k], b[k])
    return bad


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'read'
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == 'read':
        print(json.dumps(read_policy(), indent=2))
        return 0
    if cmd == 'snapshot':
        snap = read_policy()
        snap['expected_baseline'] = BASE
        out.write_text(json.dumps(snap, indent=2) + '\n')
        print(json.dumps({'snapshot': str(out), 'no_turbo': snap['no_turbo'],
                          'governor': snap['governor'], 'driver': snap['driver'],
                          'max_set': sorted({v['max'] for v in snap['cpus'].values()}),
                          'min_set': sorted({v['min'] for v in snap['cpus'].values()})}))
        return 0
    if cmd == 'set':
        value = int(sys.argv[3])
        before = read_policy()
        r = write_max(value)
        time.sleep(0.2)
        v = verify(value)
        after = read_policy()
        rec = {'wall': time.time(), 'target': value, 'write_rc': r.returncode,
               'write_stderr': r.stderr.strip()[:500] if r.returncode else '',
               'verify': v, 'before_max': {str(c): before['cpus'][str(c)]['max'] for c in CPUS},
               'after': after}
        if out:
            out.write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps({'ok': v['ok'] and r.returncode == 0, 'target': value,
                          'rc': r.returncode, 'verify_attempts': len(v['attempts']),
                          'bad': v['attempts'][-1]['bad'] if not v['ok'] else {}}))
        return 0 if (v['ok'] and r.returncode == 0) else 1
    if cmd == 'check-restored':
        snap = json.loads(out.with_name('freq-snapshot-baseline.json').read_text()
                          if out.name != 'freq-snapshot-baseline.json' else out.read_text())
        pol = read_policy()
        bad = equal_baseline(snap, pol)
        rec = {'wall': time.time(), 'equal_baseline': not bad, 'bad': bad, 'policy': pol}
        res = pathlib.Path(out).parent / 'freq-final-restored.json'
        res.write_text(json.dumps(rec, indent=2) + '\n')
        print(json.dumps({'equal_baseline': not bad, 'bad': bad}))
        return 0 if not bad else 1
    print('usage: freq_ctl.py read|snapshot F|set F VALUE|check-restored F', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
