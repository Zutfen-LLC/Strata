#!/usr/bin/env python3
"""timed_run.py — stdlib wrapper: monotonic elapsed, exit code, RUSAGE_CHILDREN.ru_maxrss.
Usage: timed_run.py OUTBASE CMD [ARGS...]  -> OUTBASE.log, OUTBASE.time.json"""
import json, resource, subprocess, sys, time

outbase, cmd = sys.argv[1], sys.argv[2:]
t0 = time.monotonic()
with open(outbase + '.log', 'wb') as out:
    rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
el = time.monotonic() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
open(outbase + '.time.json', 'w').write(json.dumps(
    {'cmd': cmd, 'rc': rc, 'elapsed_s': round(el, 3),
     'max_rss_kib': ru.ru_maxrss, 'note': 'ru_maxrss = max individual descendant RSS'}, indent=2) + '\n')
print(json.dumps({'rc': rc, 'elapsed_s': round(el, 3), 'max_rss_kib': ru.ru_maxrss}))
sys.exit(rc)
