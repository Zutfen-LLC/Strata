# The two-host remote CPU-expert run: inferswarm05 + inferswarm03 (inferswarm #347/#349)

A self-contained recipe for the first bounded full-model trial of `--remote-cpu-experts`.  Nothing here
needs an unpublished file: both hosts build from the candidate branch of
`https://github.com/Zutfen-LLC/Strata`.  Execute from Hermes' dedicated host with SSH access to both
machines.  **Every step that downloads, installs or changes a driver needs its own bounded operations
approval - this recipe assumes the hosts are already built up to the checks below and only builds
software and runs the experiment.**

The outcome under test: the real Strata full-model runtime on **inferswarm03** uses its two RTX 3060s
(GPU0: dense/routing/attention/KV/shared + its expert cache; GPU1: the `--expert-cache-device1` helper
tier) while the expert complement executes on **inferswarm05**'s RAM/CPU through the resident worker.

## 0. Fresh host checks (record the numbers, refuse on drift)

On **inferswarm05** (the worker):

```sh
lscpu | grep -E "Model name|Flags" | head -2      # CPU ISA: the worker needs AVX2 (ggml-cpu's floor)
free -g                                            # RAM: must hold the assigned expert set + headroom
df -h /home                                        # disk for the build
```

On **inferswarm03** (the engine):

```sh
lscpu | grep "Model name"; free -g; df -h /home
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv   # BOTH GPU UUIDs + free VRAM
nvidia-smi | grep "CUDA Version"                                                  # driver/toolkit
nvcc --version | tail -2                                                           # CUDA >= 12.x for the sm_86 build
```

Both GPUs must be the same model (two RTX 3060s).  Free VRAM after the desktop/idle must hold CUDA0's
dense weights + KV + expert cache AND CUDA1's helper cache, with the margins the engine prints at
startup; if the profile-driven caches do not fit, lower the cache sizes (below) rather than the reserve.

## 1. Model and pack (reuse existing assets; no transfer without approval)

The mode needs a **native Q8_0/Q8_0 pack** (every layer plain `Q8_0` gate/up and `Q8_0` down, one
geometry) - the worker's one fixture-tested layout - plus the model's GGUF shards the engine loads
anyway.  Check what is already on inferswarm03's model disk:

```sh
ls <model-dir>          # the pack's native_experts.txt must say gu=8 d=8 on EVERY layer
grep -v '^#' <model-dir>/native_experts.txt | awk '{print $2, $3}' | sort -u   # expect a single line: 8 8
```

- If a Q8_0/Q8_0 pack of the model already exists on the shared/local storage, reuse it.
- **A smaller or different-quant pack is NOT a code change away**: the worker accepts only Q8_0/Q8_0.
  Re-quantizing or re-packing the model is a new multi-GB artifact and needs its own bounded approval.
- inferswarm03 keeps the model files on disk (the dense weights and the setup upload read them); what it
  must NOT need is a RAM mirror of the experts - the mode refuses `--resident-cpu-experts` precisely so
  the claim "no hidden full cold-weight RAM mirror on inferswarm03" stays checkable (`free -g` before
  and during the run, and the engine's own startup lines report the staging bound).
- inferswarm05 needs NO model files: every owned blob arrives over the wire at setup.

## 2. Build the CPU-only worker on inferswarm05

```sh
git clone -b <candidate-branch> https://github.com/Zutfen-LLC/Strata && cd Strata
cmake -S . -B build-worker -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-worker --target strata-expert-worker
# the focused checks (a few minutes, CPU only):
cmake -S . -B build-worker -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON
cmake --build build-worker --target expert_worker_test expert_worker_e2e_test expert_tier_test
./build-worker/expert_worker_test && ./build-worker/expert_worker_e2e_test && ./build-worker/expert_tier_test
```

No CUDA/HIP toolkit is involved.  Size the worker's memory budget to the ASSIGNED SET (the engine prints
it at startup, step 5 - run the engine's `--plan`-style startup once without serving, or size from the
check below) plus headroom; it refuses an assigned set over budget (typed `over-budget`).

## 3. Build the CUDA sm_86 engine on inferswarm03

```sh
git clone -b <candidate-branch> https://github.com/Zutfen-LLC/Strata && cd Strata
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86            # the RTX 3060 is sm_86; NOT the default sm_120
cmake --build build --target strata            # plus the dispatch test where a toolkit exists:
cmake -S . -B build -DSTRATA_BUILD_TESTS=ON && cmake --build build --target worker_dispatch_test
./build/worker_dispatch_test
```

## 4. The tunnel (owned by the engine host, loopback only)

The worker binds 127.0.0.1 only, by policy.  From inferswarm03 (one owned process, one known PID):

```sh
ssh -f -N -L 7431:127.0.0.1:7431 inferswarm05     # record the PID; kill it in cleanup
```

## 5. Size the placement (fixed, within budgets)

Run the engine once to startup with the mode on; it derives and PRINTS the frozen partition ("N of M
experts are no GPU tier's (X GiB at Q8_0)") before uploading.  Choose the cache sizes so that:

- CUDA0: dense + KV + its expert cache + the session/drafter fit its 12 GB with the engine's margins;
- CUDA1 (`--expert-cache-device1 N`): N experts of the complement fit its own 12 GB;
- the worker (`--memory-budget-mib` on inferswarm05) holds the REST of the complement plus headroom
  (roughly 5 MiB per expert at this geometry; the engine prints the exact GiB).

The roughly 136 GB all-RAM Q8 figure is the EVERYTHING-on-CPU bound, not a requirement here: with two
3060 caches holding, say, ~2 x ~1700 experts, inferswarm05 needs ~(24576 - 3400) x 5 MiB ~ 104 GiB - size
`--expert-cache-device1` (and inferswarm05's RAM) first, then let the complement fall out.

## 6. Freeze the experiment settings

- Model + pack: as checked in step 1; quant **Q8_0/Q8_0** throughout.
- CPU kernel/window policy: the worker's `--workers N` (0 = its default plan for the machine); the
  engine's native Q8_0 kernels on its own local share; verify windows for prompt and decode
  (`--short-read 64`, serial requests, no `--batch`).
- Speculation: keep the serving default MTP settings FIXED for the run and record them (the startup log
  prints them); do not A/B speculation inside this experiment.
- KV precision and cache placement: the engine defaults, recorded from the startup log.
- Fixed placement: `--adapt-every 0 --adapt-swaps 0 --pcie-frac 0`, `STRATA_ROUTE_TAIL_SKIP=0` (the mode
  enforces the last one and refuses the adaptive tiers outright).
- Deadlines: `--remote-cpu-deadline-ms 10000` (the default) unless the first windows show the worker
  needs longer - record any change.

## 7. Run

On inferswarm05 (the worker, one process, verbose on a first run):

```sh
./build-worker/strata-expert-worker --port 7431 --memory-budget-mib <sized-in-step-5> --verbose \
   > worker.log 2>&1 &      # record the PID
```

On inferswarm03 (the engine; the startup prints setup seconds, the partition, then serves):

```sh
STRATA_ROUTE_TAIL_SKIP=0 ./build/strata generate --serve --native <model-dir> \
  --remote-cpu-experts 127.0.0.1:7431 \
  --expert-cache <n0> --expert-cache-per-layer --expert-profile <profile> \
  --expert-cache-device1 <n1> --gpu 0,1 \
  --adapt-every 0 --adapt-swaps 0 --pcie-frac 0 --short-read 64 <frozen validation flags>
```

Then ONE short text request through the serve API (a few hundred tokens at most - anything longer than
`--short-read` fresh tokens is rejected by design), and the frozen output/numerical check: the same
request against a matching local control if one is feasible on the same host (the same build without
`--remote-cpu-experts`, caches holding everything they can), comparing tokens; where a control is not
feasible, record logit sanity (finite, plausible distribution) and say so - do not manufacture an
apples-to-apples speedup across different quants or hosts.

## 8. What the evidence must show (all of it, or the run does not count)

- Both GPU tiers AND the worker participated: the DONE line's ` worker_entries=N` (> 0 every request),
  the cache-hit/lookback counters for GPU0, the helper tier's startup + per-request counters for GPU1,
  and the worker's own STATUS line (requests > 0, CPU service seconds > 0).  Residency alone is not
  computation.
- No local expert reads on inferswarm03 after readiness: the request summary's file-blob deltas for the
  decode are zero for worker-owned work, and `free -g` shows no expert-sized RAM growth during the run.
- Setup vs inference apart: the startup line's setup seconds (upload) vs the DONE line's
  `worker_req_wall` (per-request client wall) vs the worker STATUS's CPU service seconds (its kernels) -
  client wait is never reported as GPU or worker kernel time.
- Failure behavior: one deliberate probe - stop the worker mid-run once - must abort the request with
  the worker named in the error, no partial output, and the engine exits (a restart re-uploads).

## 9. Cleanup and the result

Kill the engine, the worker and the tunnel (the recorded PIDs), keep both logs and the engine's stdout
(PP/DONE lines) as the retained record - including a failed run, which is PRESERVED, not retried to
green.  Report: request outcome, tokens vs control (or the stated limitation), setup/upload seconds,
per-request worker wall and CPU service, RAM/VRAM peaks, network bytes (the worker STATUS's MiB in/out),
and cleanup confirmation.  The retain/change/stop decision belongs to issue #347, not to this recipe.
