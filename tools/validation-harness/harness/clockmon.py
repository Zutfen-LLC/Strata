#!/usr/bin/env python3
"""clockmon.py — 1 Hz read-only clock/thermal/power sampler (no new software).

Records per tick: per-CPU scaling_cur_freq (28), per-CPU /proc/stat busy deltas,
coretemp temperatures, package/core thermal throttle counters, RAPL package
energy (if readable unprivileged), host swap + owned worker VmSwap, plus the
current arm/phase marker written by campaign.py. Self-exits on STOP file or
after max_s seconds. Never writes to anything except its own journal.
"""
import json, os, pathlib, sys, time

D = pathlib.Path(sys.argv[1])
D.mkdir(parents=True, exist_ok=True)
STOP = D / 'clockmon.STOP'
JOURNAL = D / 'clock-journal.jsonl'
MARKER = D / 'clockmon-marker.json'
MAX_S = float(sys.argv[2]) if len(sys.argv) > 2 else 5400.0
CPUS = list(range(28))

# static labels
hw = {}
for h in pathlib.Path('/sys/class/hwmon').iterdir():
    try:
        name = (h / 'name').read_text().strip()
    except Exception:
        continue
    if name == 'coretemp':
        for t in sorted(h.glob('temp*_input')):
            lbl = t.with_name(t.name.replace('_input', '_label'))
            hw[str(t)] = lbl.read_text().strip() if lbl.exists() else t.name

rapl_paths = [pathlib.Path('/sys/class/powercap/intel-rapl:0/energy_uj')]
rapl_readable = None
worker_pid = None
wpid_file = D / 'clockmon-worker-pid'


def marker():
    try:
        return json.loads(MARKER.read_text())
    except Exception:
        return {}


def stat_snapshot():
    txt = pathlib.Path('/proc/stat').read_text()
    out = {}
    for line in txt.splitlines():
        if line.startswith('cpu') and line[3:4].isdigit():
            f = line.split()
            out[int(f[0][3:])] = [int(x) for x in f[1:]]
    return out


def throttle():
    row = {}
    p = pathlib.Path('/sys/devices/system/cpu/cpu0/thermal_throttle')
    row['package_throttle_count'] = int((p / 'package_throttle_count').read_text())
    row['package_throttle_total_time_ms'] = int((p / 'package_throttle_total_time_ms').read_text())
    core_cnt = 0
    core_ms = 0
    for c in CPUS:
        q = pathlib.Path(f'/sys/devices/system/cpu/cpu{c}/thermal_throttle')
        core_cnt += int((q / 'core_throttle_count').read_text())
        core_ms += int((q / 'core_throttle_total_time_ms').read_text())
    row['core_throttle_count_sum'] = core_cnt
    row['core_throttle_total_time_ms_sum'] = core_ms
    return row


def rapl():
    global rapl_readable
    if rapl_readable is False:
        return None
    try:
        return int(rapl_paths[0].read_text())
    except PermissionError:
        rapl_readable = False
        return 'EACCES'
    except Exception as e:
        rapl_readable = False
        return repr(e)


def worker_swap():
    global worker_pid
    if worker_pid is None and wpid_file.exists():
        try:
            worker_pid = int(wpid_file.read_text().strip())
        except Exception:
            worker_pid = None
    if worker_pid is None:
        return None
    try:
        for l in pathlib.Path(f'/proc/{worker_pid}/status').read_text().splitlines():
            if l.startswith('VmSwap:'):
                return int(l.split()[1])
    except Exception:
        return 'gone'
    return None


prev = stat_snapshot()
started = time.monotonic()
n = 0
with JOURNAL.open('a') as f:
    while True:
        if STOP.exists() or (time.monotonic() - started) > MAX_S:
            break
        t0 = time.monotonic()
        cur = {}
        for c in CPUS:
            try:
                cur[str(c)] = int(pathlib.Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq').read_text())
            except Exception:
                cur[str(c)] = None
        stat = stat_snapshot()
        busy = {}
        for c in CPUS:
            a, b = prev.get(c), stat.get(c)
            if a and b:
                da = sum(b) - sum(a)
                d0 = b[3] - a[3]  # idle column
                busy[str(c)] = round(100.0 * max(0, da - d0) / da, 2) if da > 0 else 0.0
        prev = stat
        temps = {}
        for pth, lbl in hw.items():
            try:
                temps[lbl] = int(pathlib.Path(pth).read_text()) // 1000
            except Exception:
                pass
        swapfree = None
        for l in pathlib.Path('/proc/meminfo').read_text().splitlines():
            if l.startswith('SwapFree:'):
                swapfree = int(l.split()[1])
        row = {'n': n, 'mono': round(t0, 6), 'wall': time.time(), 'mark': marker(),
               'cur_khz': cur, 'busy_pct': busy, 'temps_c': temps,
               'throttle': throttle(), 'rapl_energy_uj': rapl(),
               'swap_free_kib': swapfree, 'worker_vmswap_kib': worker_swap()}
        f.write(json.dumps(row) + '\n')
        f.flush()
        n += 1
        delay = 1.0 - ((time.monotonic() - t0) % 1.0)
        time.sleep(max(0.05, delay))
(D / 'clockmon-exit.json').write_text(json.dumps({'wall': time.time(), 'samples': n}) + '\n')
