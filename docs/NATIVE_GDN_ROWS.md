# Standalone native GDN rows

The standalone `PortableNativePleGdnComposition` connects materialized PLE rows to HC and native GDN for one or two consecutive tokens. It does not enable model-level native chunks. Experts and QSA are outside this path.

The caller constructs the existing PLE composition and the bounded GDN input, core and output owners. The composition borrows the PLE and projection owners and takes ownership of the core owner. The backend, role tensors and their storage must outlive all of them. Do not use the borrowed owners independently while they are attached. The path is single-threaded and does not permit reentrant state changes during consumption.

GDN stages candidate history and state in separate banks. The synchronous consumer receives a checked view containing projected rows, token IDs, positions, HC residual rows and injection rows. The optional preparation callback may still fail. Only after both callbacks and the PLE commit succeed does the composition perform the non-throwing GDN state-bank swap. A failed attempt invalidates its view and poisons the composition until reset. Previously published views remain valid until a successful replacement or reset. Consumer side effects outside these owners are not rolled back.

The component budgets and `host_bound` functions describe the listed owned tensor and host payloads. They are not process-memory limits. The caller must also account for borrowed role buffers, caller-held copies, allocator overhead and opaque backend and driver allocations. The existing PLE factory, paths and diagnostic proof limits still apply.

This uses the established serial native Q8_1/F32 projection and recurrence method with causal rows. It is not a claim of numerical equivalence to native prefill's FP16/BF16 GEMM input conventions. CPU mocks and shader validation do not establish GPU correctness. No throughput or model-level prefill claim follows from this standalone path.
