"""Fixed worker deadline/socket repetitions under one-CPU contention; no hosts or model assets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', required=True)
    ap.add_argument('--out', required=True, help='new directory for logs (existing evidence is never replaced)')
    args = ap.parse_args()
    build = Path(args.build).absolute()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cpu = min(os.sched_getaffinity(0))
    load = subprocess.Popen(['taskset', '-c', str(cpu), sys.executable, '-c', 'while True: pass'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    try:
        cases = [(f'deadline-{i}', ['expert_worker_test', '--deadline-only'], 30)
                 for i in range(1, 4)] + [(f'socket-{i}', ['worker_socket_test'], 15)
                                        for i in range(1, 4)]
        cases += [(name, [name], 120) for name in ('expert_worker_test', 'expert_tier_test')]
        for name, argv, limit in cases:
            start = time.monotonic()
            try:
                with (out / f'{name}.log').open('w') as log:
                    result = subprocess.run(['taskset', '-c', str(cpu), str(build / argv[0]), *argv[1:]],
                                            stdout=log, stderr=subprocess.STDOUT, timeout=limit)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
            results.append(dict(case=name, exit=code, seconds=round(time.monotonic() - start, 3), cpu=cpu))
            # Each case runs its fixed count once. A failure is retained, never retried to obtain PASS.
    finally:
        load.terminate()
        load.wait(timeout=5)
        (out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return int(any(row['exit'] != 0 for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
