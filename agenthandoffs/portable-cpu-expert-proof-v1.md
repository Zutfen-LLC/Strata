# Portable CPU expert reuse proof — v1

Code base: PR42 documentation commit `59ce0bc5aae81b8cd4f3f17924e8b1cbb16a243d`;
functional portable baseline `ae6489205c9eca5c072610dbacbc78c8489dffda`.
This change preserves portable arithmetic. The failed native-CUDA parity
screen remains failed and belongs to the separate numerical investigation.

## Local evidence and build checkpoint

The changed `portable_model.cpp` compiled with GCC, C++20, `-O2` and native
experts enabled. The new row test and enhanced model test linked against the
existing `/tmp/strata-portable-model-build` native CPU libraries, using retained
GGML source `3cf03257f219afbe7334045ff7c6a06ac68c627d`. No dependency or other
worktree was modified. This incremental check is not a fresh all-backend build.
The first row-test link omitted the PLE library and failed with undefined
`ple_artifact_consts`; the corrected link with existing portable libraries passed.

Passed:

- `portable_cpu_experts_test`: real Q8 native conversion/pool, byte-identical
  unweighted rows against the original staging oracle, repeated/reordered hits,
  new expert misses, layer and model isolation, changed-source corruption,
  first-verification rejection, failed-entry retry, and retained mapping use
  after caller path strings are destroyed. Receipt: hits=20, verifications=43,
  copied_bytes=digest_bytes=224563200, native_batches=6. A ten-expert repeat
  adds zero copy/digest bytes while executing the same native batch.
- `portable_model_test`: all 48 layers, full vocabulary; tokens 5 and 6 have
  byte-identical logits versus sequential original-provider execution, and
  the second token records 480 hits. Existing scalar, generation, EOS,
  context, layer-10 corruption, poison/reset/replay checks passed. Report:
  mapped=188214088960, host_weights=4831936000, staging=52224000 bytes.
- Focused source diff/whitespace inspection.

Pending at this checkpoint: clean CMake builds, Vulkan/CUDA/HIP/SYCL compilation,
physical V340 operation and real-model before/after timing. Parent supervises
long builds and owns all fleet dispatch. No fleet access or model launch here.

For a clean isolated build, retain existing toolchain/dependency configuration
from `portable-model-hermes-v4.md`, and add `portable_cpu_experts_test` to the
explicit target list. Run the two tests once, with no retries:

```sh
ctest --test-dir "$BUILD" -R '^portable_(cpu_experts|model)_test$' --output-on-failure
```

## One matched real-model comparison, parent dispatched

This document does not dispatch a run. Parent/Hermes supplies the exact reviewed
candidate SHA, fresh source/build/evidence paths and explicit one-pair runtime
authority. Preserve `/home/hermes/portable-model-pr42-ae64892-real` and every
previous failure. Never overwrite raw rows, binaries, models or prior logs.
Use the existing host/swap/per-die supervisor unchanged for each arm.

1. Build candidate once in an isolated directory with the pinned GGML dependency.
   Run both focused tests once. Record source SHA, binary SHA256, build flags,
   dependency SHA and immutable six-shard/manifest identity. Stop on any failure.
2. Use that same candidate binary for both arms. The **before/oracle** arm adds
   only `--expert-dispatch-oracle`; the **after** arm omits it. Hold the exact
   retained 38-token prompt, context 64, max-new 3, worker/thread counts, two
   Vulkan indices, split, caps, environment and supervisor settings fixed.
   Make separate empty logit/output directories. No automatic retry, sweep or
   concurrent arm. Parent chooses the fixed order and records it; page-cache
   order is a timing confounder, so one pair is descriptive evidence only.
3. Wrap each existing supervised command in `/usr/bin/time -v -o ARM/time.txt`.
   Keep its stdout, stderr, per-token cumulative `expert_work`, placements,
   memory samples and all raw `pos-N.f32` dumps. The existing successful
   portable baseline has IDs [71093,12305,198]; record IDs from both arms.
4. Compare **every** consumed position, requiring identical file sets, lengths
   and bytes. Also require exact stdout/token/stop records. For example, with
   BEFORE and AFTER pointing only to the newly created raw-logit directories:

```sh
python3 - "$BEFORE" "$AFTER" <<'PY'
import pathlib, sys
before, after = map(pathlib.Path, sys.argv[1:])
a = sorted(p.name for p in before.glob('pos-*.f32'))
b = sorted(p.name for p in after.glob('pos-*.f32'))
assert a and a == b, 'missing or mismatched position sets'
for name in a:
    x, y = (before / name).read_bytes(), (after / name).read_bytes()
    assert x and len(x) % 4 == 0 and x == y, name
print('PASS byte-identical portable logits:', len(a), 'positions')
PY
cmp "$BEFORE_STDOUT" "$AFTER_STDOUT"
```

5. Oracle verifications should equal selected experts dispatched (480/token).
   Candidate native_batches must equal oracle native_batches (48/token).
   With unchanged files, candidate verifications equal unique layer/expert
   pairs encountered; each hit removes 5,222,400 copied bytes and the same
   digest input bytes (10,444,800 paired FNV logical byte accesses). Report
   actual hits/misses and source_invalidations, never assume route reuse.
   Compare elapsed/user/system time and supervisor memory observations without
   attributing all elapsed difference to FNV or claiming a general speedup.

Any byte mismatch, digest failure, source metadata drift, placement failure,
health/memory stop or unexpected process state ends the pair. Preserve and
report it, without weakening limits or retrying. This test neither changes nor
passes the previously failed CUDA numerical screen. Both arms assume immutable
source files; metadata checks are not protection against adversarial mutation.
