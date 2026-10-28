> Machine locations below are portable placeholders for retained artifacts. Detailed operator receipts are preserved privately with original source hashes. Historical public commits remain public.

# First HC projection controlled discriminator — v1

Run exactly one bounded real-weight operation comparison on05, not another full
prompt. This test-only target is behind existing portable GDN/build-tests/Vulkan
flags. Product session/provider/weights/defaults and comparator limits are unchanged.
The retained row analysis rules out a mere common offset: centered relative L2
0.0713219994, affine residual relative L2 0.0712544205, correlation0.9974865982;
top10 overlap10/10 and winner71093 do not waive the failed original screen.

## One controlled boundary

Use first prompt token248045, fresh four-stream embedding, layer0
`hc_attn_norm.weight` and `hc_attn_down.weight`. Independently compute one
common normalized input (F64 RMS sum, F32 normalization/gain multiplication).
Feed the **same F32 input vector** to all arms. This deliberately isolates the
first down projection; it does not claim to trace backend RMS or a complete layer.

Compare the actual GGUF Q8 matrix on CPU and strict Vulkan, retaining each
backend's default Q8 matvec policy, then compare both with exactly dequantized
F32 coefficients, explicit F32 accumulation/source precision, and an independent
F64 scalar dot product rounded once to F32. Also independently pack the common
input with the pinned CPU Q8_0 quantizer and scalar-multiply the decoded input
against the same coefficients. This predicts CPU activation-quantization error
without using the CPU dot-product kernel as its own oracle. No BF16 pack
reconstruction or native-CUDA equivalence is claimed.

All arms use identical dimensions/order/coefficients/input, with no expert
callback, routing, nonlinearities, histories, head or token loop. The scheduler
must assign the one actual MUL_MAT to the requested backend; hidden CPU fallback
is rejected. This uses the product's existing portable scheduler policy.
The test reports eight vector metric groups (max abs, RMS, relative L2, worst
channel), no raw arrays or model output files. Success means valid finite strict
execution, **not** a new numerical/quality pass. No limits are fitted to results.

## Versioned operator command

Parent supplies the exact published commit containing this handoff as SOURCE_SHA.
Use a separate Git checkout and build directory, preserving deployed source,
binaries and all prior results. Existing pinned dependency only; no install.

```sh
set -euo pipefail
: "${MODEL_CMAKE05:?existing cmake executable version3.24 or newer}"
: "${SOURCE_SHA:?full reviewed PR42 diagnostic commit}"
: "${SOURCE:?separate existing source checkout at SOURCE_SHA}"
: "${BUILD:?fresh diagnostic-only build directory}"
: "${EVIDENCE:?fresh private evidence directory}"
: "${LLAMA_SRC:?existing llama.cpp at 3cf03257f219afbe7334045ff7c6a06ac68c627d}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$SOURCE_SHA"
test "$(git -C "$LLAMA_SRC" rev-parse HEAD)" = 3cf03257f219afbe7334045ff7c6a06ac68c627d
test ! -e "$BUILD"
test -z "$(git -C "$SOURCE" status --porcelain --untracked-files=no)"
mkdir "$EVIDENCE"
"$MODEL_CMAKE05" -S "$SOURCE" -B "$BUILD" \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_PORTABLE_FFN=ON -DSTRATA_PORTABLE_GDN=ON \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DGGML_VULKAN=ON -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DSTRATA_GGML_DIR="$LLAMA_SRC" \
  -DCMAKE_PREFIX_PATH="${MODEL_CMAKE_PREFIX05:-}" \
  -DCMAKE_CXX_FLAGS="${MODEL_CXX_FLAGS05:-}" \
  > "$EVIDENCE/configure.log" 2>&1
"$MODEL_CMAKE05" --build "$BUILD" --target portable_hc_boundary -j4 \
  > "$EVIDENCE/build.log" 2>&1
sha256sum "$BUILD/portable_hc_boundary" > "$EVIDENCE/binary.sha256"
# One synthetic mechanism check; not a numerical acceptance calibration.
timeout --signal=TERM --kill-after=5s 120s \
  "$BUILD/portable_hc_boundary" --synthetic 0 \
  > "$EVIDENCE/synthetic.json" 2> "$EVIDENCE/synthetic.log"
# One real-weight boundary: die0, which owns layer0 in the completed product run.
timeout --signal=TERM --kill-after=5s 120s \
  "$BUILD/portable_hc_boundary" --model \
  ${PRIVATE_ARTIFACT_ED0D8ADE} 0 \
  > "$EVIDENCE/real.json" 2> "$EVIDENCE/real.log"
```

Execute commands sequentially and stop on any nonzero result. Confirm the device
initialization line identifies the intended V340, not llvmpipe/CPU. Keep the
existing physical Vulkan environment/dependency configuration from the passed
build; do not install or alter driver settings. No new source options forcing
CPU tensor fallback are allowed. Source-only metadata/split/bounds admission uses
GgufModel; payload reads are limited to a~2.7KiB embedding row,40KiB norm and
3481600B matrix. Exact widened matrix is13107200B. Each arm destroys its
allocations before the next; workspaces are small and bounded by the existing
256MiB graph cap. The175GiB file mapping is virtual, not a whole-model working
set requirement. No huge PLE/expert/head payload is touched or hashed.

## Interpretation, fixed before real output

- If Q8 CPU/Vulkan differs, but the F32 controls approach the independent scalar
  oracle and CPU Q8 matches the independently quantized-input oracle, the Q8
  activation/dot arithmetic explains that **boundary's** discrepancy. This is a
  controlled arithmetic-policy contribution, not proof it explains position37.
- If an F32 arm departs from its scalar oracle materially beyond the other arm,
  retain the exact metrics and investigate this single projection's backend/
  integration path. Do not attribute it to Q8 quantization or run the full model.
- If CPU Q8 does not follow its independently quantized-input oracle, investigate
  that CPU dot/packing path before using CPU output as a parity reference.
- If all controls are close, the first HC projection is not a useful explanation
  of the final discrepancy. Report that exclusion; this test does not examine
  native BF16 HC, later nonlinearities, GDN/QSA histories or expert arithmetic.

There is no new universal tolerance or quality verdict in these descriptions.
Report all raw metric groups, physical backend identity and source/binary hashes.
The original full-model logit/probability/KL gates remain failed and unchanged.

## Local checks completed

Target built at pinned ggml. Strict llvmpipe synthetic execution and JSON/finite/
shape checks passed. On this **synthetic software Vulkan** fixture, Q8 CPU versus
Vulkan relative L2 was0.0073347019; matched F32 CPU/Vulkan2.2631900e-6.
CPU quantized-input scalar oracle versus CPU Q8 was8.7068413e-7, while its
predicted activation-quantization effect was0.0073399589. These validate the
mechanism and illustrate distinguishable signatures; they are not real-weight
or AMD results and do not set acceptance thresholds. Logs:
`${BUILD_ARTIFACT_0A1812F1},synthetic}.log` and
`${BUILD_ARTIFACT_82BAE760}`.

## Physical result supplied by parent — receipt1793153523.065029

operator ran the synthetic and real operation checks at
`9c476e014d5eeb3861d0e9eac85d7975649895c4`; both exited0. Real execution
was strict Vulkan0 UUID `00000000-0700-0000-0000-000000000000`, with no
remaining process. Evidence05 `${PRIVATE_ARTIFACT_8F6A6DD8}`;
binary SHA256 `cb62f177f41c1eb7c92fcfe8d6358d2dd2c8985f624a0a78d4096441fb71ae09`.
Slack C0C5MK91H3L, thread1793152694.143009, message1793153523.065029.
These are parent-supplied runtime observations, not execution by this task.

| Real-weight comparison | Relative L2 |
| --- | ---: |
| Q8 CPU versus Vulkan | 0.00128030826017 |
| F32 CPU versus Vulkan | 1.30667864648e-7 |
| Scalar versus F32 CPU | 1.04784679909e-7 |
| Scalar versus F32 Vulkan | 8.34281813208e-8 |
| Quantized-input oracle versus Q8 CPU | 1.4024089929e-7 |
| Scalar versus quantized-input oracle | 0.00127945581917 |
| Scalar versus Q8 CPU | 0.00127947595217 |
| Scalar versus Q8 Vulkan | 5.68286042261e-8 |

Q8 CPU/Vulkan maximum absolute difference was0.122528076172.
The independent CPU input-quantization control reproduces the discrepancy at
this projection, while matched F32 CPU/Vulkan and Vulkan Q8 follow the scalar
oracle much more closely. This demonstrates an activation-quantization
contribution **at this boundary**. It does not explain all later layers or
position37, establish native HC pack format, or establish quality parity.
Original whole-model numerical gates remain failed and unchanged.

## Single recommended next qualification

Run one paired **existing frozen12-case answer screen** from
`tools/validation-harness/quality.py` on the native reference and the functional
portable model, using its existing template, tokenizer, gold answers and
`correct()` rules unchanged. Do not substitute the100-case v2 suite or tune
prompts after outputs. This is a small functional regression screen, not a
broad evaluation framework or another attempt to calibrate numerical limits.
It is the next recommendation, not a fleet dispatch from this task.

Freeze the exact token IDs before either arm. Use serial requests with a fresh
session/reset for each, max-new32 and context256 (existing screen admits at most
224 prompt tokens). Preserve each product's existing stop semantics; record stop
reason and IDs. Use the already qualified binaries: portable functional source
`ae6489205c9eca5c072610dbacbc78c8489dffda`, binary SHA256
`06831fc5743559190ce56ec148187aa16f09dd3360410667adad13f6f4589c0a`, and the
retained02 native binary/settings. Keep both V340 dies, genuine CPU experts,
host PLE and strict placement/host guards for the portable arm. Do not fold the
separate PR43 optimization into this comparison before its independent proof.
No new build, weights, archive or full-logit dump is needed. Keep only bounded
IDs/decoded answers/scoring/runtime receipts; use existing shared tokenizer and
semantic scorer, not the old single-expert participation predicate in the
historical quality harness. Its transport/placement adapter is not automatically
compatible with the full-model CLI, so those old campaign commands must not be
run unmodified. Parent can reuse exact matching retained native case outputs if
available; otherwise one baseline pass is needed.

Decision outcomes fixed before running:

- Native12/12 and portable12/12, with healthy strict execution: narrow functional
  screen passes. Keep the full-model numerical parity failure explicitly open;
  this is not broad model-quality equivalence or a tolerance waiver.
- Native12/12 and any portable semantic/format failure: actual bounded task
  regression. Stop expansion and retain the first failing prompt/IDs for one
  targeted investigation; do not optimize or dismiss it as backend noise.
- Native below12/12: inconclusive reference for this screen; report both scores
  and per-case outcomes without declaring portable quality acceptance or changing
  gold answers. A portable runtime/placement/guard failure is independently a
  product failure regardless of answer score.

The12-case screen distinguishes observable task regressions from the already
measured vector discrepancy. It does not by itself classify the mathematical
cause of every differing logit. Disk05 has16GiB free per parent; do not create
new heavy artifacts. No run, new acceptance threshold or product code change
was made in this documentation checkpoint.
