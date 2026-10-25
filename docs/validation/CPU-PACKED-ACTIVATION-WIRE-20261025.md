# Issue #14: opt-in packed activation request contract

This stage follows the [sampled proof](CPU-PACKED-ACTIVATION-PROOF-20261025.md).
Raw F32 stays the default. Replies and original entry order stay unchanged. No streaming,
compression, placement, socket policy, resident reload or live model run is included.

Before inference, an opted-in client probes a separate session-bound capability. Both operators
must supply the same nonempty qualification ID for the exact sender/receiver build pair whose
native configuration and sampled byte/row gates they have verified. Matching source, CPU names,
format IDs, or an arbitrary matching ID alone are not evidence. IDs must be retired when either
build/compiler/native configuration changes. No build is qualified automatically.
The probe is a 16-byte STATUS extension: session u64 and qualification u64, little endian.
A capable peer echoes it with a version u32 suffix; an old peer returns its normal validated
59-byte ready status. That old status or an explicit declined capability deliberately selects raw;
malformed replies, I/O failure and cancellation fail the setup. Default clients send no probe.

The new request messages use SEW1 framing and unchanged identity/routing metadata, followed by
version 1, format Q8_0, width 2560, 80 blocks (four little-endian u16 fields), then one 2720-byte
row per token having any owned entry, in token order. Each block is finite FP16 scale bits in
little endian and 32 signed-byte bit patterns. No structs, padding, SPA1 headers or file digests
are transported. The receiver checks exact length, binary masks, version/format/geometry,
finite scales and negotiated session, then uses zero-filled native scratch directly. It never
re-quantizes packed input. Existing session, sequence, ownership, count, size, finite-weight,
readiness and deadline checks still apply. Malformed packed packets fail; they never retry raw.

Only Linux x86-64 native Q8_0 gate/up/down, 2560/640, no rotation/tail, T1/T2 is eligible.
Unsupported geometry/build/qualification or scales select raw before sending. Mixed groups
retain raw for unsupported peers. A group packs the union of active eligible tokens once per
window, shares immutable owned fields through all peer completions, and reports pack calls,
quantized rows, elapsed packing time, packed requests and raw fallback reasons. Single tiers
use the same packer. A scale failure falls back for that window without changing quantization.

Desktop validation is static only. Builds, executable checks and placement measurements are
NOT RUN here. No speedup is claimed. Requests are about 18.7% of remote bytes; even optimistic
combined savings are under 13.71% and replies do not shrink. Offline handoff follows below.

## Operator interface and accounting

The engine accepts `--remote-cpu-packed-qualification 16_HEX` with a fixed CPU group.
An offline test worker accepts `--packed-qualification 16_HEX`. The ID is 16 lowercase
hex digits, nonzero. Leave both absent for raw. The example IDs in synthetic tests are
fixtures, never deployment qualifications. A capability is cleared on setup/disconnect;
a successful reply binds the current session and version. Probe status bytes do not count
as inference traffic. Request messages are 23 (weighted) and 24 (entry rows); ACK schemas
remain 8 and 21. A receiver accepts raw even after packed negotiation.

Opt-in group/tier request wall time includes sender packing. `pack_ns` measures native
quantization, field serialization and allocation together; it is a subset of that wall
interval, not extra time to add again. Group `pack_calls` counts attempts per logical window,
`quantized_rows` counts actual native quantizer calls (including work before a scale refusal).
Peer packing counters stay zero under a group. Inference timing records actual request/reply
wire bytes; use `--remote-cpu-rpc-timing` for those counters. Group reports the precise last
pack refusal (`unsupported_token_count`, `unsupported_scale`, etc.); peer fallback counters
also identify missing capability. A zero-work peer sends no inference and adds no fallback.
No peers eligible means no sender quantization. One unsupported token scale selects raw for
all eligible peers in that window; no partially packed request is sent.

## Prescriptive offline Hermes handoff (NOT RUN)

Parent dispatch only. Do not contact the fleet runtime at e145df6e or worker 809314 on02.
No model GEN, service reload/replacement, install, firewall, deployment or live A/B.
Do not set a real deployment qualification ID or start a worker binary against model assets.
Synthetic integration tests create their own in-process services on ephemeral loopback ports;
the capture proof opens no sockets and reads one digest-verified expert blob at a time.

1. Bind the exact new published commit as `PIN`. Preserve all retained sources/builds/ELFs.
   Use fresh isolated source, build and evidence directories. Reuse the role-specific archive,
   retained compiler/cache bindings, native flag equality gate and ggml-base PIC correction from
   [role05 proof handoff](CPU-PACKED-ACTIVATION-PROOF-20261025.md) and
   [role02 handoff](CPU-PACKED-ACTIVATION-ROLE02-20261025.md). Do not infer role02 flags from05.
   Only the fresh offline build disables GPU backends. Record compiler identity and effective
   native commands for all five retained targets, with `warning_only=0` before execution.
2. Build these targets in that verified fresh CPU build (bounded job count 2, 600 s limit):

   ```sh
   timeout --signal=TERM --kill-after=5s 600s cmake --build "$NEW_BUILD" --parallel 2 --target \
     cpu_packed_wire_test cpu_worker_group_test cpu_packed_activation_proof expert_worker_test \
     expert_tier_test worker_residency_test worker_service_residency_test
   ```

   Keep full build logs and ELF hashes. Build failures stop the stage. Separately compile the
   touched engine CLI in isolated CUDA, HIP and SYCL builds before review; record unavailable
   backends as NOT RUN. Do not execute those engine binaries or mutate shared runtime builds.
3. Reuse the prior admission checks (>=10 GiB available and cgroup headroom), bounded execution
   wrapper (2 GiB address space/RSS, zero owned swap, unchanged host OOM count), and parent
   watchdog. Run each gate serially with a fresh log, 180 s timeout and no tiny inherited file-size
   limit. Synthetic tests create small temporary Q8 fixtures, with no downloaded model:

   ```sh
   ctest --test-dir "$NEW_BUILD" --output-on-failure --no-tests=error -j1 \
     -R '^(cpu_packed_wire_test|cpu_worker_group_test|cpu_worker_group_packed_test|cpu_packed_activation_check|expert_worker_test|expert_tier_test|worker_residency_test|worker_service_residency_test)$'
   ```

   Require explicit Passed for every named test; a missing test or skip is not a pass. The new
   wire gate mutates version, format, width, block count, masks, IDs, lengths and nonfinite scales;
   exercises receiver negotiation, stale identity, replay, ownership, count and finite-weight
   refusals; verifies raw scalar bytes and exact packed/native scratch. The packed group gate
   checks weighted and entry-row replies against existing independent exact native references,
   one/two peers, mixed capability, T4/T8 raw fallback, resident reconnect, old-status compatibility,
   malformed capability refusal and once-per-union packing. Existing default tests remain required.
4. With the SHA-verified original capture/manifest and read-only roots, rerun the bounded `--pack`
   and `--rows` handoffs on both retained native role builds. The same proof now routes every
   supported fixture through the production codec/restorer before its existing exact native-row,
   ordered-partial and retained-capture gates. Keep every old SPA1/SRA1 artifact. New artifact
   bytes must `cmp` exactly with the retained proof artifacts and with the other role's new
   artifacts; preserve hashes and comparison logs. No gate or tolerance changes. A failure stops.
5. Retain the `PACKED_PLACEMENT` lines from the synthetic packed group test: wall time, pack_ns,
   calls, rows, actual request/reply bytes and per-peer raw/packed counts. They show offline
   placement cost and sharing only; synthetic loopback timings are not LAN latency or TPS.
   Report each role/build separately, with raw versus packed workload/configuration and all
   fallbacks. Do not average across different native builds or add overlapping peer timings.

Passed here: `git diff --check` and focused source review only. Review found missing edge-fixture
routing metadata and timing that excluded sender packing; both were corrected. These are static
checks, not compiler or runtime evidence. Executable gates, placement measurements and backend
builds are NOT RUN on this desktop. Live end-to-end latency remains outside this handoff and
requires a separately specified parent-controlled session after all offline gates.
