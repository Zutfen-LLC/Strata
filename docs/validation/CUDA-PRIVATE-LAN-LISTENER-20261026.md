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
