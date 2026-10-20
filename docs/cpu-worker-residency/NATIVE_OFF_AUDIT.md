# Common CPU and native-expert dependency boundary

This is static source inspection after the user-reported [79f4da0 link failure](evidence/hermes-79f4da0-failure.md).
It is not a successful compile/link claim. The coding task ran no configure, build, tests or binaries.
The [independent spec and quality review](evidence/common-kernel-review.md) records scope and findings.

| Consumer compiled with native experts OFF | Required implementation | Source ownership after correction |
| --- | --- | --- |
| `expert_layout.cpp`: `q2_rows_any`, legacy rows and `act_quant_any` | AVX-512 Q2/activation in `expert.cpp`, AVX2 in `q2_avx2.cpp`, AVX1 in `q2_avx1.cpp` | All common `strata_kernels_cpu` sources |
| `RouterLookahead` in shared `expert_source.cpp` and migrated SYCL source | BF16 AVX2/FMA dot and AVX1 dot; existing scalar fallback below AVX | Common `router_avx2.cpp` and `kq_avx1.cpp` |
| `pool.cpp`, shared dispatcher, remote/peer expert source paths | Canonical Q2 kernels and native format abstraction | Q2 remains real math; existing OFF `native_fmt`/layout errors and aborting unreachable native computation stubs remain |
| Native-only `native_expert.cpp` | IQ/K-quant/NVFP4/trellis kernels, ggml type traits and quantizers | Still conditional on `STRATA_NATIVE_EXPERTS`; no implicit ggml enablement |
| Shared `mtp.cpp` | Optional ggml requantization | Existing `STRATA_NATIVE_EXPERTS` guards and unavailable-build error path remain |
| Shared dispatcher and driver | Worker tier construction, ownership, post/join and reporting | Earlier native guards and early unavailable-worker CLI refusal remain unchanged |

The two BF16 AVX2 functions and their reduction helper moved from `kq_avx2.cpp` without arithmetic
edits. The rest of that file uses `ggml-common.h` and stays native-only. Q2 AVX1 and BF16 AVX1 need no
ggml header/library. Each implementation appears once in the common source list and was removed from
the native-only list. These are real kernels, not success-returning placeholders or zero-output stubs.

Top-level CMake covers CPU-only, CUDA, shared wave32 HIP and the gfx906 compatibility route. MSVC keeps
`/arch:AVX2` for the extracted router and `/arch:AVX` for the two AVX1 files. GCC/Clang retain the prior
AVX2/FMA/F16C and AVX1 flags; Q2 AVX1 still explicitly disables AVX2/FMA/F16C. Flags remain per source,
not on probes, the whole library or the driver. The ISA scanner includes the new router file in the
no-static-initializer list. Native-ON numerical behavior is intended to stay the same; source-identical
math and flags do not establish binary identity or numerical parity after recompilation.

`STRATA_PORTABLE` and native older-CPU `STRATA_ISA_FLOOR` policy are unchanged. The latter's ggml floor
definitions remain in the native-only block. This fix does not newly permit native-OFF engines to run
on SSE-only CPUs: the existing engine CPU checks still apply. GPU MMQ configurations that explicitly
require ggml/native experts still reject OFF rather than silently enabling it. Tests OFF/ON and
`STRATA_BUILD_TESTS=OFF` use the same common source ownership.

The separate SYCL CMake list had the same common CPU dependency gap and is kept synchronized. Its
migrated driver/dispatcher is separate and has no claim of persistent worker support. No SYCL build
or runtime has been validated. Metal is a separate Apple ARM build with its existing kernel stubs and
native ggml configuration; it does not compile these x86 translation units and is unchanged. No Metal
worker-support or backend validation claim is added.

Regression sources now make the existing synthetic `router_dot_parity` and forced-AVX variant available
with native OFF as well as ON. Q2 bit-plane/legacy numerical checks also get a forced-AVX case. Actual
`strata` and `worker_disabled_dispatch_test` links remain mandatory, followed by CLI refusal tests.
The bounded [Hermes handoff](NATIVE_OFF_HERMES.md) exercises these gates. Only that execution can confirm
generated build/link behavior for the selected toolchain. Other backends and CPU floors remain explicit
gaps until tested in their own environments.
