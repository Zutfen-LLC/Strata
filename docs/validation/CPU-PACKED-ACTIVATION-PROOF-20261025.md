# Packed activation equivalence: source-level plan

Tracks [#14](https://github.com/Zutfen-LLC/Strata/issues/14), separately from socket issue #12.
Planning only: no implementation, protocol change, worker contact, reload, GEN or test dispatch.

The current wire codec (`src/worker/expert_wire.cpp::encode_infer`) sends F32 activations. Worker
`execute` (`src/worker/expert_worker.cpp`) calls `native_quant_act` before native expert jobs consume
those bytes. `NativeFmt.gu_act` and `act_bytes` define the dot-product input; `kNativeActBytes=4096`
is only a scratch stride. For the current Q8_0 2560-value row, useful bytes are 2720 versus 10240 F32
bytes. Moving this existing quantization boundary could preserve downstream arithmetic; it does not
preserve or reconstruct the original F32 bits. It requires proof, not a same-source/CPU assumption.

The smallest proof has two gates using the SAME 16 real records selected by the completed compression
diagnostic, including their original masks, weights and phase/position labels:

1. A small offline target, linked separately with each role's retained native CPU configuration,
   checks Q8_0/2560 geometry, `gu_act`, `act_bytes`, and the current absence of Hadamard rotation.
   Derive each peer mask from the original full ownership bitmap, quantize only its active tokens
   and keep inactive scratch zeroed. Serialize the existing function's defined scale/quantized-byte
   fields, restore into zeroed existing-stride scratch, and compare all useful bytes across both builds.
   Do not send C structs/padding, reinterpret packed bytes as floats, or quantize a second time.
   At most 32 token rows require 87040 useful packed bytes; this stage needs no weight reads.
2. Reuse the independent native-row reconstruction already present in
   `tests/worker/cpu_reply_compression_bench.cpp::rows_for`: original quantized inputs versus restored
   packed inputs must yield bitwise native expert output rows and original ordered weighted partials, and
   partials must bitmatch capture. Read and digest-check one existing expert blob at a time, at most
   5222400 bytes. No arena or worker process. A format mismatch ends the proof before interpretation.

Focused synthetic checks should cover active/inactive token masks, zero work, signed zero, finite
edge values, malformed lengths/blocks/scales and explicit endianness. Retain finite F32 validation
before quantization and observe existing behavior on extreme finite inputs rather than silently
changing acceptance. Keep entry order, IDs, routing weights and the GPU/helper merge unchanged.
Scope stays Q8_0; other activation formats/rotations are refused by this proof, not generalized.

Use existing assets and native caches with isolated proof outputs, no engine/worker product rebuild
or launch. A later executable handoff can reuse the current 2 GiB/180-second offline bounds; it is
not supplied or dispatched by this planning note. Coverage remains first-logit/prefill, not long decode.
Only after cross-build packed-byte AND row/partial gates pass should a separate opt-in wire capability
be considered. No candidate adoption or protocol work follows automatically.

The supplied request/reply totals limit the opportunity: requests are 18.67% of combined bytes.
73.4375% fewer activation bytes gives an optimistic 13.71% combined-byte ceiling before unchanged
request metadata, about 2.160 ideal serialized 1 Gb/s seconds. These are upper bounds, not live TPS.
See [the retained arithmetic and compression assessment](CPU-REPLY-COMPRESSION-20261025.md).
