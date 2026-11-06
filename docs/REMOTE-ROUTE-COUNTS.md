# Opt-in remote placement counts

For one native `--serve` request, set `STRATA_REMOTE_ROUTE_COUNTS` to a new
private output file. The normal request continues through prompt and decode;
no activation, token or weight values are saved. Each output record aggregates
identical `(layer, T, K, original expert IDs)` dispatches and their count. The
collector is off by default, caps unique patterns at 65,536, and marks overflow
or invalid input incomplete. It reports write failures without changing inference.
An interrupted or failed request may leave an unsealed file. Counters include
executed speculative verification rows, including drafts that were not accepted;
they are not counts of generated tokens. Collecting them has overhead and the
instrumented run is not a latency benchmark.

Score one or more candidate text assignments (one `layer expert` pair per line)
against the actual frozen CPU-worker partition in the same format:

```sh
python3 -B tools/score_remote_route_counts.py "$COUNTS" \
  --cpu-assignments "$CPU_PARTITION" "$CURRENT_PLAN_PAIRS" "$CANDIDATE_PAIRS"
```

The scorer refuses incomplete/unfinished counts and assignments outside the
supplied CPU partition. It returns CPU-complement frequency and the predicted
number of nonempty dispatch slices and selected entries for each candidate.
Counts are for one destination. Use the exact frozen partition and preserve
existing independent plan, source, and local GPU ownership validation. These
are observed routing opportunities, not measured remote calls or latency.
