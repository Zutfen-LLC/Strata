# Issue #14: one live raw / packed decision

Hermes handoff; **fleet execution NOT RUN here**. Run only prompt index 0, once raw and once
packed, on the established 02+05 1 Gb/s split with exact entry-row replies. No warmup or tuning.
The existing runner needs only per-arm `packed_qualification` and `first_logits_reference`.
Absent fields preserve its old argv/behavior. No product defaults change.

Before dispatch, tell the user: the old full02 worker cannot receive packed requests. After the
small build checks, stop its identity-checked owned hierarchy, load one half on02 and one on05,
keep those same residents for both arms, then stop both halves and restore full02. Preserve all
weights/files. Never load a half alongside the old full arena. The exact old PID need not survive.

## Build only what this comparison uses

Use the published handoff commit as `PIN`. Archive it into a fresh02 source root; do not change
`/home/hermes/strata-pr10-9ed478e-src` or its CUDA cache. For example use
`/home/hermes/strata-pr10-ab14-src`, `-cuda` and `-out` (all initially absent).
Seed the new CUDA cache from the **user cache entries** of
`/home/hermes/strata-pr10-9ed478e-build-cuda/CMakeCache.txt`: retain its generator, compilers,
CUDA architectures, native/optimization/ISA options, linker options and existing ggml dependency
binding. Relocate only source/build/output paths; enable tests and fully disconnected dependencies.
Use its already corrected dependency prefix. Keep CUDA ON; no CPU-OFF/PIC repair is needed here.
Do not copy a CMakeCache verbatim into a different build directory.

On05 reuse the isolated successful `/home/hermes/strata-pr10-wire05-20261025-r1-build` and its
`-src` product source. The retained role05 product source is unchanged by this harness-only patch.
Build its actual receiver plus the two tiny qualification targets; no eight-test repeat.
Use the current published helpers when staging the05 peer, so helper pins agree with02.

```sh
# 02: SRC02/BUILD02/OUT02 are the fresh roots above; CUDA_PRELOAD holds the retained values above.
timeout --signal=TERM --kill-after=5s 90s cmake -S "$SRC02" -B "$BUILD02" \
  -C "$CUDA_PRELOAD" -DSTRATA_BUILD_TESTS=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON
timeout --signal=TERM --kill-after=5s 900s cmake --build "$BUILD02" --parallel 2 \
  --target strata strata-expert-worker cpu_packed_wire_test cpu_packed_activation_proof
# 05: BUILD05 is the retained r1-build. Preserve its configured runtime output directory.
timeout --signal=TERM --kill-after=5s 600s cmake --build "$BUILD05" --parallel 2 \
  --target strata-expert-worker cpu_packed_wire_test cpu_packed_activation_proof
```

Keep normal build logs, actual ELF hashes, CMake cache/compiler/native flags. No packages or other
backends. Before fleet dispatch the only build binding still needed is `CUDA_PRELOAD`, generated
from that retained cache, including its actual existing `STRATA_GGML_DIR`; no new dependency search.
Bind `BIN02/BIN05` to the configured executable output directories, not guessed `build/bin` paths.

## Minimal qualification for these actual binaries

Run `cpu_packed_wire_test` once from each actual cache under the established 180-second fixture
guard. It checks real framed raw/packed transport, native scratch and exact replies. Then use the
existing production proof (190-second outer guard per invocation):

```sh
"$BIN02/cpu_packed_activation_proof" --pack "$CAP02" "$OUT02/activations.spa"
"$BIN02/cpu_packed_activation_proof" --rows "$CAP02" "$MAN02" /srv/models/qwen38-q8 \
  "$OUT02/activations.spa" "$OUT02/rows.sra"
# Transfer only the small SPA1 through the existing channel; SPA02_ON05 names that copy.
"$BIN05/cpu_packed_activation_proof" --pack "$CAP05" "$OUT05/activations.spa"
"$BIN05/cpu_packed_activation_proof" --rows "$CAP05" /home/hermes/strata-wl-metadata/local.manifest \
  /srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66 \
  "$SPA02_ON05" "$OUT05/rows.sra"
```

`MAN02=/home/hermes/strata-pr10-split-opt-20261025-02/s1/L/assets/manifest`.
`CAP05=/home/hermes/strata-pr10-micro-bc8e5754-20261025-05/cpu-capture.bin`; `CAP02` is the
already relayed identical capture from the completed cross-host proof. Its exact retained02 path
must be supplied, or copy this 37,544,692-byte capture once through the existing channel.
Use the existing fixture admission (10 GiB available, 8 GiB active floor, 2 GiB owned limit,
zero owned swap/new OOM); do not pass its small limits to production workers.

Require both wire checks successful, both SPA1 byte-identical, both SRA1 byte-identical and all
48 row/partial/capture gates successful. Use `cmp` on controller copies. These checks cover the
sender's native build, local02 receiver and05 receiver. Only then explicitly assign a fresh
nonzero 16-lowercase-hex `QUAL` to this exact engine02/worker02/worker05 binary tuple and record
those hashes/native configurations. No source-only qualification or old fixture ID.

## One existing-runner cohort

Derive a fresh private config from the retained split `s1/COHORT.source.json`, using current
helpers and new ELF assets. Preserve model, original request file, masks, 12 workers/4 loaders,
primary960/helper1378, turbo/OMP policy, deadlines and monitors. Keep
`--remote-cpu-entry-rows`; add `--remote-cpu-rpc-timing` identically to both inference arms.
Both workers get `--packed-qualification QUAL` from initial load onward. Raw clients omit the
probe, which these workers accept. No worker restart or load between raw and packed.

Use exactly these four arms in one `run.py` invocation:

| Arm | restart | requests | peers / bytes | extra arm fields |
| --- | --- | --- | --- | --- |
| load | true | [] | 02+05 / 58067865600 each | none |
| raw | false | [0] | same | reference + first_logits_reference |
| packed | false | [0] | same | same references + packed_qualification=QUAL |
| restore | true | [] | 02 / 116135731200 | none |

Set both inference arms' `reference` to an asset containing the original
`/home/hermes/strata-pr10-9ed478e-20261025-run/arm-A0-summary.json`, and
`first_logits_reference` to an asset containing its `A0-first-logits.0`. Enable
`capture_first_logits=true`. Exact token/entry and first-logit checks run after each arm;
a mismatch prevents the next arm. Expected original prompt0: 213 tokens/102724 entries.
The packed opt-in belongs only to its arm, never global engine argv. Keep cold loads separate.

In peer config retain the exact forced `control_argv` and fixed peer root
`/home/hermes/strata-pr10-9ed478e-20261025-05-peer`. Verify the retained approved worker route
27431 at current05 address10.0.0.123 from02 address10.0.0.180; preserve control restrictions.
Do not append SSH commands, use fixture17531, or change firewall/keys. Stage current helpers and
the05 worker there after identity-checked stopped-root archival, as in the existing split recipe.
If05 ELF differs from02, add a distinct local `worker05` asset on02 and set the peer's
`asset_bindings.worker="worker05"`; the remote worker binding must hash to that asset.

Stage a separate zero-GEN full02 recovery bundle before releasing worker809314. Use the existing
[identity-checked replacement](CPU-REPLY-CODEC-LIVE-20261025.md#release-the-old-owned-hierarchy-once)
and [split cleanup/restoration](CPU-ROWS-SPLIT-20261025.md#finishing-after-a-failed-b-controller)
path. Adapt only candidate/staged paths and hashes; do not source old scripts' fixed source-SHA
or W/R staging assertions unchanged. Keep its one-STOP/no-retry ownership checks. Fresh admission
must cover each half plus the unchanged reserve (05 budget61440MiB); no swap as capacity.

Execute the staged `run.py --run-root COHORT_ROOT` once. If an arm fails, the existing runner
cleans the owned cohort and skips later arms: reconcile cleanup and run the already staged
zero-GEN full02 recovery once. Do not repeat GEN. Normal restore stops05 before02's half.
Finish with full02 complete/forever idle,116135731200B, one allocation, zero owned swap/new OOM,
and05 owned subset absent. Preserve original files and all failure receipts.

## Decision and stop

Return one short table: raw/packed prompt seconds, decode seconds, request wall
(`done_mono-sent_mono`), exact token/entry identity, first-logit `cmp`, per-peer inference
request/reply bytes, actual `packed_requests`/`raw_fallbacks` with reasons, group `pack_calls`,
`quantized_rows`, `pack_ns`, plus setup and final restoration status. Retain stdout/stderr and
resource receipts; these counters already exist. `pack_ns` is included in wall, not added again.
Do not sum overlapping peers' elapsed times.

Require packed requests on both active peers. No gain claim for unqualified/unsupported-peer raw
fallback; record any unsupported-token/scale fallback and the actual packed share. If packing is
inactive or exactness fails, stop and report failure, restore, and do not tune/retry. A flat/slower
result or a tiny difference without a clear useful gain means stop #14 and prioritize #20.
A useful improvement supports keeping #14 as an opt-in research result, with this one-sample limit.
No merge, deployment, default change, production qualification or automatic next session.
