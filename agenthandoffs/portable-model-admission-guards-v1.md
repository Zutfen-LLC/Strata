# PR42 additive admission corrections — v1

Corrects independent review findings against
`ba4c426739777b28004e784e90caefe767026fd1`. Work is confined to the existing
isolated task-8/strata branch. The deployed ba4c426 source/build and running
Hermes stages1/2 were not touched. No fleet command or real-model run occurred.

## Physical device identity

The CLI rejects repeated Vulkan indices before backend initialization.
Session admission compares `ggml_backend_get_device` identities, so separate
backend objects for one physical die cannot receive separate memory ledgers.
It does not compare names or descriptions; distinct V340 device identities
remain legal. The existing split and other admission checks remain.

`portable_device_admission_test` constructs distinct handles for one device
and checks rejection before architecture/manifest/weight admission. It runs
with CPU handles by default and actual Vulkan handles with `--vulkan INDEX`.
On a two-device host, `--vulkan INDEX,OTHER` additionally checks that the two
different identities reach the next guard using a tiny deliberately
unsupported-architecture GGUF fixture. There is no model workload in this
admission fixture.

## Host working-set admission and loading-time stops

The existing supervisor sums six unique shard stat sizes without scanning
payloads. For the retained model,188225033248B plus the explicit32GiB CLI
anonymous cap and8GiB reserve requires231174706208B (about215.3GiB) of
MemAvailable and any applicable inherited cgroup allowance before Popen.
All terms and baselines are retained even on admission failure.

The32GiB product cap bounds admitted anonymous/runtime allocations, not
mapped file residency. File size is a conservative worst-case working-set
bound; mappings do not mean that all those pages are resident.

From process loading onward, every0.2s sample records MemAvailable, global
swap used/free/total, pswpin/pswpout, child VmSwap/RSS/high-water RSS and
visible finite ancestor cgroup-v2 memory.max/current allowances. Only the
owned child process group is terminated on an8GiB host/effective-cgroup
reserve breach, any new child swap, or system swap-out growth from the
pre-launch baseline. Historical system swap need not be zero. No cgroup is
changed. Unsupported/unreadable applicable accounting stops preflight.
Sampled guards are not atomic RAM reservations or a transient-OOM guarantee.

## Verification

Passed:
- Eight targeted CPU CTests in the Vulkan-linked build,35.37s, including the
  full48-layer synthetic session/head/provider and new device guard.
- New duplicate-device fixture in the CPU-only build and with two distinct
  Vulkan0 handles on local llvmpipe. Physical two-device acceptance is not
  claimed from this single-device software backend.
- CPU-only and Vulkan CLIs reject --vulkan0,0 before initialization.
- Twelve Python tests, including low-memory preflight preventing Popen,
  finite ancestor cgroup ceilings, child swap/system swap-out triggers,
  historical swap allowance, and pressure during loading terminating only
  the mocked owned process group while preserving its failing sample.
- Python compilation and diff checks. No math/session/expert layout changes.

An initial build invocation hit the stale generated target list for the new
fixture; explicit CMake regeneration and rebuild succeeded. The fixture was
not reported as passed until it existed and ran. No validation failure remains.

Logs:
- `/tmp/portable-model-admission-guards-{config,build2,ctest,python}.log`
- `/tmp/portable-model-device-admission-vulkan-final.log`
- `/tmp/portable-model-admission-cpu-{config,build,only-test}.log`

Updated operational stages: `portable-model-hermes-v2.md` supersedes v1 for
the corrected source. Parent must use the new exact PR head in fresh isolated
source/build/evidence paths; this correction does not itself dispatch a run.
No merge or deployed-tree mutation.
