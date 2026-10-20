# Independent scoped review

One independent reviewer inspected the candidate against exact base
`7a590706f20bc976d5bf094be3dfd0c461cf8869`. Scope: correctness, identity/integrity,
assigned-only memory and reads, compatibility, deadlines/cancellation, actual engine
integration, focused fixtures and the Hermes recipe.

Two material findings were corrected:

- A native pack could contain a stale same-size `experts.bin`. Local mode now requires
  `--native`, `--mmap-experts` and the actual GGUF source. It refuses `experts.bin` before
  GPU expert-cache fills.
- The whole-setup budget was first checked after negotiation. A reviewer loopback probe
  observed a 50 ms setup budget wait 300 ms for a withheld setup ACK. Setup control now
  starts before connection and covers controlled connect/send/header/payload waits,
  uploads and readiness. Regression probes cover withheld headers, delayed payloads,
  cancellation during negotiation and cancellation before connection.

The reviewer found no outstanding material issue after inspecting the corrections.
A nonblocking timing note was also corrected: engine startup now times the manifest
read/index validation separately from tier connection/setup.

The reviewer independently reproduced the original timeout defect. The reviewer did
not rerun the final six CPU suites, compile CUDA or perform hardware operations.
The primary agent ran the retained final CPU and compatibility checks.
