# Native-OFF correction: independent static reviews

Scope: `16d065e23211f2d422c56a3497a8be200c1e1d4d` through correction commit
`62cabbdde396db1eb84d357d194354fced232954`, followed by the additive source/document corrections
recorded below. Two independent reviewers used the code-review skill for spec and quality reviews.
Both inspected source only. Neither ran configure, builds, tests, test binaries, benchmarks, models or
host operations. The user's explicit Geekom restriction overrides the repository backend-build
requirement for this handoff. These reviews do not establish successful linking or runtime behavior.

## Findings and source corrections

- **Spec P3 / quality P2: early refusal misclassified option values.** The first OFF-only scan examined
  every raw argument. A relative pack or token-file path equal to `--remote-cpu-resident` would be
  mistaken for a worker option. The scan now skips values using the arities of the main and Qwen3.6
  parsers, including both values of `--control-vector-layer-range`, and stops at help/version/families.
  The new unrun CLI regression source checks all six option-shaped names as pack, synthetic token-file,
  dump and layer-range values against the real OFF engine. The arity list must track future value-taking
  options in both parsers; this avoids moving the family dispatch or changing native-ON parsing.
- **Quality P3: stale validation wording.** README's final caveat still described the whole candidate as
  lacking CUDA and real-host GPU-release evidence. It now explicitly attributes those 16d065e results
  to the user receipt and reserves the unvalidated status for the new correction. Earlier failed and
  interrupted coding-task logs and the unsafe pre-fix binary warning remain intact.
- **Spec follow-up P3: family-parser consumption differs in serve mode.** The union arity scan can
  treat a worker flag as a main-parser option value while Qwen3.6 treats it as an actual unknown flag
  and ignores it. Qwen3.6 now explicitly refuses actual worker options in native-OFF before engine
  construction, model payload loading or GPU initialization. The existing family metadata probe may
  precede that refusal. Unrun CLI source uses a tiny tensor-free GGUF header to select that path and
  check all six worker flags without real model assets. The new refusal code is excluded from
  native-ON builds.

The initial reviews found no other material issue in the compile guards, opaque worker refusal,
native-ON ownership query, regression source, shared CUDA/HIP scope or attributed evidence. No HIP
validation or SYCL worker support was inferred. Both independent follow-up reviews found the source
corrections resolved, with no remaining material finding. The quality reviewer noted that the duplicated
arity list needs maintenance when either parser changes; the synchronization comment and handoff make
that obligation explicit. The tensor-free GGUF fixture was inspected, not executed.

## Required runtime gate

The [Hermes correction handoff](../NATIVE_OFF_HERMES.md) requires a separate serial native-OFF CUDA
engine link and refusal checks on 03, preserving its known-good native-ON build, then native-ON dispatch
and small 05 worker fixtures. No correction-head runtime check has been executed here. Worker runtime,
wire protocol and resident lifecycle sources remain unchanged from user-tested 16d065e; a new full-model
load is not required by this correction diff.
