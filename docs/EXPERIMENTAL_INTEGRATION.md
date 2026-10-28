# Experimental integration qualification

This dependency stack is intended for experimental research use. The available evidence does not establish production qualification. Hosted checks, statuses and workflow runs were absent in the integration audit; that is not CI green. Reported hardware results below are supplied execution evidence, not a new hardware campaign performed during documentation cleanup.

## CPU and network dependency chain

PR10 adds fixed CPU workers and packed request selection. Qualified peers automatically select the packed request format, with raw fallback and opt-out; this separately authorized default change is an exception to the stack's otherwise opt-in features. Actual CUDA controls and a complete 213-token exact comparison were reported. The latest tested-source-to-head difference was documentation only.

PR25, PR28 and PR29 reported acceptance on both AMD V340 dies. PR30 and PR31 have host, client, CUDA-contract and mTLS regressions. A separately established physical loopback receipt and coordinator CUDA compilation receipt were not available in the prerequisite audit. PR31 incorporates the relevant PR27 implementation; PR27 is not a separate prerequisite merge.

PR34–37 add validation and reporting. PR37 reported 75 host tests passing after an intermittent guardian EOF, which remains part of the execution history.

## Portable model and expert residency

PR38–41 introduce opt-in FFN, GDN, PLE and QSA modules. Their public component documentation and tests define the supported shapes, state/reset behavior, placement contracts and refusal cases. Component fixtures do not establish complete-model numerical equivalence.

PR42 and PR45 completed a frozen 12-case functional screen: native and portable each passed 10/12, failed the same two cases, and produced identical output-token sequences on all 12. The unchanged scorer was inconclusive because the baseline missed. The original strict whole-model logit parity gate remains **FAILED**. Matching these functional outputs is not a broad quality qualification or numerical-parity pass.

PR43 reported byte identity across 41 real portable logit rows and stdout with expert reuse. This establishes equivalence of that portable optimization, not parity with native CUDA. PR44 adds optional local phase timing. Timing scopes overlap and must not be summed.

PR45 reported 480 resident experts per AMD die, actual GPU expert execution on both dies and a functional screen matching the native outputs. PR46 and PR51 add explicit ranked placement metadata, leaving placement opt-in. PR47 passed the NVIDIA complete-model mechanism checkpoint; it did not establish cross-backend numeric parity.

PR48 includes local A/B/A and reset coverage and actual AMD and NVIDIA resident passes. Four requests completed in a retained process; warm CPU verification, copy, digest and invalidation work was zero. These observations are workload-specific, not a general speedup claim.

PR49 passed actual CPU pool phase timing. Pool phases are contained in pool elapsed; enclosing and enclosed intervals must not be added. Work outside the pool cannot all be attributed to GPU or dense compute.

PR50 passed the guarded head-credit runtime checkpoint with the existing 480-per-die plans. The unchanged 1152-per-die pair remains excluded.

PR51's 1104-per-die metadata candidate passed one token and one frozen case with the PR50 runtime. Each die had 1104 assigned experts; the token recorded 7 and 6 GPU entries and corresponding row checks, with maximum row errors 2.38e-7 and 2.98e-8. The frozen case returned the correct four-token answer, with 1132/978 GPU entries and 11810 CPU entries. Wall time was 124.289 seconds, including 39.512 seconds setup, 72.899 seconds prompt and 10.685 seconds decode. This is not a speedup claim. Full quality for this placement remains **untested**; 1152 remains excluded.

PR52 adds default-off coarse expert-provider and executor host-clock timing. Seven focused local CTests and 54 portable Python tests were reported passing; the CLI also compiled without Vulkan. Its physical timing run remains **pending**. Executor elapsed includes host, transfer and backend wait work; it is not pure GPU execution. Provider, pool and executor intervals overlap. No optimization or causal performance conclusion follows from the instrumentation alone.

## Documentation preservation

Machine-specific operating instructions, private communication references and exact operator receipts are preserved in the private agent handoff repository with original source commits, paths and SHA-256 hashes. Current public notes use portable variables or engineering summaries. Existing public Git history is unchanged by this additive cleanup. No pinned hardware test tree or running job is changed by documentation publication or integration.
