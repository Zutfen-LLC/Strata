# CUDA worker private LAN listener

This opt-in listener change is stacked on PR #10 at
`ce30b1dc7fc6a93a878476e97d673bb8d82b77a2`, including the qualified hybrid
arithmetic implementation `9e7cf13e6ec0b96a465eb687c17075559d611650`.
It permits a matched SSH/direct TCP experiment. No timing result is claimed.

## Worker flags

Keep the existing plan, cache root, port, device, budgets, deadlines and environment.
For SSH forwarding to loopback, omit the new flags; the default remains `127.0.0.1`.
For direct TCP, append exactly:

```sh
--bind-ip "$WORKER_PRIVATE_IPV4" --allow-peer-ip "$COORDINATOR_PRIVATE_IPV4"
```

Both values must be numeric IPv4 addresses. The bind must name a local RFC1918
interface address (10/8, 172.16/12 or 192.168/16); the OS refuses an unassigned
address. Hostnames, wildcard binding, CIDRs, malformed addresses, multicast peers
and private binding without an exact peer allowlist are refused. Repeat
`--allow-peer-ip` only for additional intended source addresses. Peer checks use
the kernel-reported source address before any protocol handling. This is source
filtering, not authentication or encryption; use the already trusted private LAN.
No firewall, SSH access or interface changes are part of this patch.

## Reuse the qualified build

In an isolated source checkout at this patch, reuse the existing qualified CUDA
configure/build directory with the same compiler, CUDA SDK, architecture, native
CPU/ggml ISA, flags and runtime bindings. Do not point a concurrent task's build
directory at another source tree. If the source path changes, reproduce its
recorded configure in an isolated build directory instead. Build only:

```sh
cmake --build "$QUALIFIED_CUDA_BUILD" --target strata-cuda-expert-worker -j"$BUILD_JOBS"
```

Retain the binary hash and use that exact patched ELF for **both SSH and direct
arms** on all four workers. Keep `STRATA_CUDA_Q8_CPU_REFERENCE=1`, plans, cache
roots, devices and budgets as qualified. The coordinator engine and existing
harness need no rebuild. No GPU kernel, nonlinear evaluation, native packing,
protocol codec, cache implementation, inference thread or socket NODELAY policy
changed. Incremental rebuilding still needs the compatible system/CUDA runtime.

For the parent's eventual A/B/A: A0 SSH, B direct private TCP, A1 SSH. Change only
transport/endpoints and the necessary worker listener flags. Preserve request,
model, placement, reference gate and each arm's cache/readiness lifecycle. Compare
complete tokens and raw first logits under the unchanged gate; retain per-arm
timing and cleanup evidence. A worker listening only on its private address cannot
be reached by an SSH forward whose remote target remains `127.0.0.1`.

If keeping the same worker listener alive across all arms, use the private bind
for every arm, point the existing SSH forward's remote destination at that same
private address, and explicitly allow the worker-local numeric source used by
that forwarded connection in addition to the coordinator's direct source. Do
not infer that this source is loopback or broaden the allowlist. The parent must
select the arrangement that fits the existing harness; no new runner is needed.

## Local validation

Linux x86-64, Release, host execution only. A fresh isolated build used native
experts ON, tests ON, CUDA/HIP/SYCL OFF, with the existing pinned local ggml source.

Passed:

```sh
cmake --build build-listener --target gpu_worker_test gpu_worker_group_test worker_socket_test worker_lan_test -j4
ctest --test-dir build-listener --output-on-failure -R '^(gpu_worker_test|gpu_worker_group_test|worker_socket_test|worker_lan_test)$'
c++ -std=c++17 -Iinclude -Isrc/worker -fsyntax-only src/worker/gpu_worker_main.cpp
git diff --check
```

Final focused CTest result: **4/4 passed, 10.59 seconds**.

The GPU service tests use the existing fake executor with real loopback sockets.
They cover unchanged unrestricted loopback default, RFC1918 range boundaries,
missing allowlist, malformed/wildcard/public binds, malformed peers including
embedded NUL, finite deadlines, explicit selected loopback binding, multiple
allowed sources, denied-source EOF/reset and subsequent allowed setup/inference.
Valid private policy is checked before a deliberate invalid-cache sentinel;
no physical NIC is opened by these tests.

The actual worker main was also linked to a temporary no-GPU executor and tested:
2 successful default/explicit launches and orderly stops, plus 9 rejected CLI
settings (including missing values and repeated malformed peers). This verifies
CLI routing, not a CUDA executable. Initial sandbox test attempts failed because
socket creation was blocked; the same tests passed with local socket permission.

Not run: CUDA compilation/link/execution (`nvcc` unavailable), physical private
LAN binding, SSH forwarding, hardware inference or the A/B/A experiment. The PR
remains draft pending the targeted CUDA build and parent-run qualification.

## Python harness follow-up

Hermes supplied a successful CUDA worker build on role03 (311 seconds), ELF
SHA256 `38cd6b15a161f74a3145fb02c81b895a489fdf805e93bc112dc7893ca68a53e8`,
and direct LAN reachability without firewall changes. These are supplied results,
not local hardware reproduction. **Preserve that ELF and the existing coordinator;
no CUDA or coordinator rebuild is needed for this Python-only follow-up.**

The changed Python files are exactly:

- `tools/validation-harness/harness/bundle.py` (runtime/staging helper)
- `tools/validation-harness/tests/test_harness.py` (synthetic regressions only)

`validate_network_gpu()` now admits explicit numeric RFC1918 IPv4 endpoints as
well as the existing `127.0.0.1` forwards. Uniqueness uses `(address, numeric port)`:
four distinct worker hosts may all use port 27432, while duplicate pairs (even
with different decimal port spellings) fail. Hostnames, IPv6, CIDRs, wildcards,
public/link-local/multicast addresses and malformed/out-of-range ports fail.
All existing external-row, retained CPU, peer-count and fallback guards remain.

Staging/run-path audit: `stage()` calls `validate()` before creating output and
`verify()` after copying. `run.py` calls `verify()` at startup and before each arm;
`owner.py` calls the same `validate_network_gpu()` before its engine Popen. These
all consume the updated staged `bundle.py`; no edit to `run.py` or `owner.py` is
needed. Remaining loopback enforcement in `validate()` is for the retained CPU
worker/control endpoint, which stays unchanged. GPU argv and plans pass through
asset resolution and `arm_config()` unchanged. Read-only inspection confirms the
existing coordinator parser accepts numeric IPv4 and GPU-group duplicate checking
already uses `(host, port)`; no additional GPU loopback gate was found there.

Stage from the patched source helper into a **new unique run root**, using the
existing direct-arm private config with only its GPU endpoint/required output-path
bindings adjusted. Keep all binary asset paths/hashes, plans, cache roots, retained
CPU attachment, requests, numerical references and lifecycle policies as qualified:

```sh
python3 -B "$PATCH_SOURCE/tools/validation-harness/harness/bundle.py" --config "$DIRECT_CONFIG"
```

This stages all pinned helpers and unchanged configured assets, then verifies the
bundle. Use that new bundle's `run.py` for the already authorized experiment;
staging alone launches no owners. Do not overwrite `bundle.py` or alter `pins.json`
in an existing staged run: its hashes must continue to protect the old bundle.
Only `bundle.py` needs updating in the source helper set; the test file need not
be staged on fleet hosts. Use the same already-built patched CUDA ELF in SSH and
direct arms. This follow-up performed no hardware experiment or fleet staging.

Focused synthetic command:

```sh
python3 -m unittest discover -s tools/validation-harness/tests -p test_harness.py -k external_gpu -v
```

All 3 focused tests passed, including four private hosts sharing port 27432 through
real staging, actual staged helper validation, repeated `run.py` preflight and arm
configuration (owners/hardware mocked), plus staged-byte tamper rejection.

Broader validation: all **41/41** tests in `test_harness.py` passed (14.873 s).
The full synthetic suite passed 51/52 (24.970 s); the existing CPU peer test
`test_remote_role_stages_no_engine_and_binds_lan_style_readiness` failed with
`peer guardian EOF`, also in isolation. Its captured transport receipt showed
`campaign.ident()` rejecting an argv identity mismatch. The untouched prior
revision's peer suite also failed 1/8 with guardian EOF in a different existing
peer test (`test_fast_stop_is_reconciled_after_worker_exit_without_second_stop`).
This establishes a prior peer-suite instability, not a fully green overall suite;
no identity/lifecycle check was bypassed or changed. No CUDA rebuild, hardware run
or performance claim is part of these Python results.
