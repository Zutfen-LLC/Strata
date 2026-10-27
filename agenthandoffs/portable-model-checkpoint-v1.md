# Portable model: first implementation checkpoint

Scope authority: delegated product milestone, stacked on corrected PR41 commit
`1d8eae1c3906962932b77e366f324c51aa89819d`. No fleet access, merges, deployment,
installation, defaults change, or optimization campaign is part of this task.

Checkout: `/home/zutfen/Documents/Codex/2026-10-27/task-8/strata`.
Branch: `codex/portable-model-session`. Other task worktrees remain untouched.

## Implemented at this checkpoint

- GDN construction and GGUF binding admit every non-QSA layer in 0..47.
- QSA construction and GGUF binding admit layers 3/7/.../47. All weight-name
  parsing uses the requested prefix length, including two-digit layer IDs.
- GDN's GGUF factory reads the PLE configuration before mapping the table.
  Absent/empty configuration means no PLE. The fleet's configured list [1]
  enables layer 1 only. Unexpected PLE weight tensors are rejected even when
  metadata did not enable PLE. Unsupported PLE lists still fail explicitly.
- Tests check valid/invalid layer families, exact late-layer residual parity,
  correct expert callback IDs, layer 1 without PLE, and late-layer QSA GGUF
  binding. Existing independent math/state/failure/reset tests remain.

This is a foundation checkpoint, not a working full-model product. No
portable_model executable, model-wide fail-stop session, portable head, or
product expert provider exists yet. No full-model numerical claim is made.

## Next implementation sequence

1. Add an opt-in portable_model target requiring the existing portable FFN,
   GDN, QSA and native CPU expert support. Keep all defaults OFF.
2. Add a lifetime-owned GgufModel, six-shard fleet validation, complete tensor
   inventory/shape/type/bounds checks and comparison to the existing model
   identity manifest. Keep embedding, PLE, and expert payloads read-only mapped.
   No full model hashing/weight scan as a setup gate.
3. Add a host Q8 embedding row gather and portable ggml output head. Reuse
   scalar dequantizers and current HC math: per-stream RMS/weighted norm,
   down/4, SiLU, sigmoid-up gates, stream mean, output.weight. Head executes
   on the final assigned die in accelerator mode.
4. Add in-process CPU expert dispatch through native_fmt, native_quant_act,
   and ExpertPool::run_split_multi_native. Validate each GGUF projection's
   layout against NativeFmt row/offset/size helpers. Stage only selected
   experts in a bounded arena; do not treat canonical packed weights as raw
   GGUF, expand experts to F32, or add a service/network listener. Return
   unweighted rows in route order; PortableFfn remains the sole reducer.
5. Add the serial session: gather and freshly broadcast embedding for every
   token, execute all 48 layers, one authoritative contiguous position from
   zero, poison after any token-step failure, reset all histories, explicit
   EOS/context/max-new handling, stable greedy token-ID generation. Preserve
   full-vocabulary teacher-forced logits for comparison.
6. Add per-backend actual scheduler reservation accounting (the existing
   graph_bytes aggregates backend allocations), host weight copies, state,
   candidate/scratch, staging and head allocations. Admit fixed contiguous
   layer placement under separate die budgets with explicit headroom.
   Document recurrent host/device transfers and backend fallback honestly.
7. Run meaningful synthetic session/head/expert/reference tests and prior
   regressions; verify strict Vulkan placement on available local software
   Vulkan. Publish one draft PR stacked on PR41 with exact checks/limits and
   a versioned Hermes recipe. Parent owns real AMD and reference runs.

## Local verification

Build directory `/tmp/strata-portable-model-build`; dependency:
`/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`.
CMake `/tmp/cmake-3.31.6-linux-x86_64/bin/cmake`.
CPU configure uses STRATA_BUILD_TESTS, STRATA_NATIVE_EXPERTS,
STRATA_PORTABLE_FFN/GDN/QSA=ON and STRATA_GGML_DIR above. CUDA/HIP/SYCL OFF.
An initial extra STRATA_GPU=OFF argument was unused (not a project option);
GPU control is the existing STRATA_ENABLE_CUDA/HIP/SYCL defaults.

Targeted build: portable_gdn_test, portable_qsa_test,
portable_qsa_gguf_test, portable_ple_test, portable_ffn_test.

CTest command:
```
/tmp/cmake-3.31.6-linux-x86_64/bin/ctest --test-dir /tmp/strata-portable-model-build --output-on-failure -R '^portable_(gdn|qsa|qsa_gguf|ple|ffn)_test$' -j 1
```

Logs: `/tmp/portable-model-checkpoint-build.log`,
`/tmp/portable-model-checkpoint-rebuild.log`,
`/tmp/portable-model-checkpoint-ctest.log`.

## Narrow parent input needed before final model admission/recipe

Provide the existing six-shard model manifest path and format (or its bounded
metadata contents) for the named issue300 Q8 model. No new hash campaign is
requested. Later the exact known prompt token IDs and retained02 reference
logit positions/tolerance contract are needed for the operational recipe;
synthetic implementation can proceed independently.

## Access outcomes

The first git worktree command could not write the source checkout's shared
Git metadata under the read-only sandbox. The authorized escalated command
succeeded and created this isolated branch/worktree. No auto-review rejection,
pending access request, uncertain mutation, or fleet operation occurred.
