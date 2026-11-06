# Portable model remote expert boundary

The portable model API has an opt-in `PortableModelOptions::remote_experts` single-peer placement. Supply an existing `GpuWorkerTier::Open` with a validated immutable GPU plan and explicit Vulkan, native-Q8 and indexed-row flags. The session checks the plan against its local manifest and supplies verified source blobs for setup. The existing client enforces worker negotiation, ownership, session binding, indexed rows, readiness and transport policy, including mTLS for non-loopback Vulkan endpoints.

This slice supports serial T1 model dispatch with no local resident expert plans. It rejects prompt chunks, mixed local resident placement, local native-Q8 observation, CPU fallback mode and row-oracle mode during preflight. Those restrictions are explicit: they are not a CPU fallback after remote failure. Experts not in the remote plan use the existing CPU provider. The returned rows retain original token/slot order and remain unweighted; the existing FFN performs its normal combine.

`remote_expert_status` queries worker counters while the session is open; `remote_expert_stats` preserves client counters. Call `close_remote_experts` explicitly on a healthy idle session to send the bounded TLS close notification. It does not stop the shared worker; an error or destruction uses the existing nonblocking cleanup. The portable CLI does not yet construct this option. Model-level scheduling, mixed local/remote placements and prompt batching remain future work.

A terminal remote transport failure or an explicit remote close makes the session unusable. `reset` refuses and leaves it poisoned; reopen the model session rather than relying on reconnection or fallback.

CPU mock and loopback tests verify wiring and protocol behavior, not native-Q8 hardware arithmetic, cross-backend parity, full-model parity or performance. A later hardware gate needs fresh same-device direct-versus-worker row evidence for the exact build, input and placement, followed by a separately scoped model test.
