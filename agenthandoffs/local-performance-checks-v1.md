# Local timing preparation checks

Scope: c76d03c7 plus the CLI timing/benchmark changes in this commit. Local
Valinor Linux x86_64, GCC16.2.1, retained CMake3.31.6. No fleet SSH, inference,
clock operations, certificate operations or services. No performance result.

Passed:

- Fresh Release CMake builds in `validation/cpu` and `validation/vulkan`, each
  using two jobs and target `portable_timing_cli_test` (depends on portable_model).
  Native experts, portable MODEL/FFN/GDN/QSA and tests ON, CUDA/HIP/SYCL OFF,
  FetchContent disconnected, retained GGML dependency at the existing path
  `/tmp/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d`.
  Vulkan additionally uses `/tmp/strata-spirv-headers-prefix` prefix/include flags.
  As in the prior build, the local GGML source snapshot lacks Git metadata;
  its path identifies retained provenance, not a newly verified source commit.
- `ctest --test-dir validation/cpu --output-on-failure -R '^portable_timing_cli_test$'`:
  1/1 passed,19.93s. Same command with validation/vulkan, run alone:1/1 passed,18.25s.
  Real six-shard sparse model, two prompt steps, early EOS after one generated
  step despite max-new2. Timed/default stdout and all three full logit dumps
  byte-identical. Finite nonnegative phase/selection timers and actual counts.
- `python -m unittest discover -s tools/validation-harness -p 'test_portable*py'`:
  27/27 host-only tests passed. Includes frozen request mutation refusal, owned
  lifecycle/occupied-port/failure cleanup controls, benchmark four-trial count,
  no native first-logit dump, original DONE timings, excluded warmup receipt,
  nonfinite timing/count rejection and original quality/scoring tests.
- Adapter `--help`, native/portable `--help`, both CLI `--help`, and diff checks.
- quality.py SHA01a1a6c4dc6c1a5f94f005316e34b7983699362fb5f3a4edb3811597892e38f1;
  tokenizer SHAe0cd8d6ae38dda0a4ad7c14f812eb64d5fd6f4fed9c920d4427a14bf585dc9b3
  match PR42's frozen v2 recipe. Complete adapter/test copied from fb518b1b
  before the narrow opt-in benchmark changes; the original seven tests pass.
- Retained driver5 source was read from Slack F0CCH9D1CPP; local exact source
  SHA b5f1a5c10a0337d6254776ad01a19dd787076f41fd114538ea01f28708a1ae2f
  confirmed. Static timing-boundary inspection also used fd35f803 generate.cpp.

Failed attempt retained: running both heavy synthetic checks concurrently caused
one Vulkan-linked CLI child to exit during inference without an engine error.
The concurrent CPU check passed. A subsequent isolated Vulkan-linked check
passed. Memory pressure is suspected, not established: kernel logs were unreadable
and the original child wait status was not printed. The recipe requires the
synthetic test to run alone. `validation/vulkan-ctest-concurrent-failed.log` keeps
the failed output. No inference arithmetic was changed to obtain the isolated pass.
Compiler warnings from retained CPU/GGML code did not fail either build.

Not run: physical Vulkan inference, native CUDA model load/inference, fleet
thermal/clock trial, CUDA/HIP/SYCL builds (their source is untouched). The
Vulkan-linked synthetic CLI test executes the CPU path, not Vulkan hardware.
The PR remains draft for parent review/dispatch and physical validation.

Raw local configure/build/CTest/Python logs remain in this checkout's untracked
`validation/` directory with both fresh build trees, approximately411MiB total.
No logs, binaries, models or generated source archives are committed. See
[the handoff](local-performance-hermes-v1.md) for exact bounded fleet commands,
phase definitions, guard/restoration conditions and retention requirements.
