# Resident stdio local checkpoint

Implementation tested: `45513fd885babcf81d252b73804a14d2a751a74a`, stacked directly on PR47
`fb2ea3c92093eacc8663d2b90b88813ec2bc4076`. No complete model payload or fleet
execution was used. The process protocol and restrictions are in
`docs/PORTABLE_MODEL.md`; the complete-source exact-commit build and guarded
one-warmup/three-measurement recipe is `portable-resident-hermes-v1.md`.

Passed on validation host:

- Fresh Release Vulkan-linked build with CMake3.31.6, retained GGML
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`, retained SPIRV headers;
  CUDA/HIP/SYCL disabled. No downloads/installs.
- Nine serial CTests in92.76s: portable_stdio_test, portable_stdio_cli_test,
  portable_timing_cli_test, portable_model_test, portable_cpu_experts_test,
  portable_local_experts_test, portable_gdn_test, portable_ple_test,
  portable_qsa_test. The real sparse-model CLI test passed in31.68s.
- Actual CLI one-shot/resident token and greedy-row equality; A/B/A reuse;
  early generated EOS; correct first-use and warm verification/hit deltas;
  EOF, QUIT and SIGTERM; malformed token refusal before rows; source metadata
  mutation exits without another result. Provider/model tests cover poisoned
  partial-step failure/no retry, reset/replay, CPU row parity, plan preparation
  once, cumulative GPU/CPU entries and release of both owned fake executors.
- 51 portable Python tests in0.727s, including six new finite-process tests:
  four GENs on one fake child, timeout/reap, process failure/reap, guard STOP/reap,
  four retained request receipts with guarded main, failing-sample STOP receipt.
- Portable CLI translation unit compiled without Vulkan using C++17 and the
  exact pinned GGML headers. `git diff --check`, Python compilation and harness
  `--help` passed.

The primary logs are in the task workspace beside the resident worktree:
`configure.log`, `build-final.log`, `ctest-final.log`, `python-tests-final.log`.
These are local working receipts, not physical AMD evidence. No performance or
quality change is claimed. Documentation impact is the explicit resident CLI
contract and executable guarded handoff; accepted campaign/quality status is
unchanged. The subsequent [AMD physical result](portable-resident-amd-results-v1.md)
records operator execution separately from these local checks.

Limitations: POSIX stdio only; finite serial requests, no network/concurrency.
Signals clean up cooperatively at idle/load/token boundaries; the supervisor
bounds stalled backend operations. Opening size/mtime identity plus existing
expert digests assumes immutable files and is not a new whole-file hash campaign.
CUDA/HIP/SYCL native builds and the assembled full-model physical GPU path were
not run locally. Original strict numerical parity remains FAILED.
