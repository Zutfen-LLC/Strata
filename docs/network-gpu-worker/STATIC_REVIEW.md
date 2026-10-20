# Independent static review receipt

Base: `34a59d83a51bdb3f0299d65306740bd357818715` (merged PR #1).
Initial reviewed candidate: `afe1f47074303825d78a0537eae34058a709f909`.
Corrected code candidate: `9b0c7b124b7353ca5fc47c467580a4709e6132e3`.
This receipt is a later documentation-only commit; it does not change reviewed code.

Two independent review agents inspected the initial diff and then the additive
correction. They were given the full user specification, AGENTS.md, CONTRIBUTING.md,
the implementation and fixture sources, and the absolute Geekom no-execution rule.
They used read-only text/git inspection. No reviewer configured, compiled, ran tests
(including Python), executed binaries/benchmarks, loaded models or operated a remote
host. Agent judgments are static review, not validation of compiled or running code.

## Spec review

Two P2 gaps were identified and resolved additively:

1. Capacity refusal originally substituted an unconditional fake executor failure.
   The CUDA fixture now first requires successful real CUDA execution, then sends an
   impossible reserve through SGW1 to the production executor's available-VRAM fit
   check. It checks that the tier never becomes READY, and reads zero weight/workspace
   allocation counters only after the owned service stops and joins. This covers
   pre-allocation refusal; it does not claim an observed allocation failure.
2. The mixed-tier fixture originally compared the merge only to the same received
   partials. It now independently dequantizes all requested weights with GGML,
   computes double dot products and float expert outputs, and measures relative L1,
   maximum absolute error, RMSE and cancellation for the full mixed result. The
   exact owner/merge check remains separate from the `3e-2` float-reference gate.

The reviewer rechecked `9b0c7b1` and reported both findings resolved, with no remaining
spec blockers. Selective acquisition/cache, frozen ownership, retained CPU partition,
reply association, failure handling and native-OFF boundaries matched the requested
slice on static inspection. Runtime acceptance remains pending Hermes.

## Quality and standards review

One P1 and one P2 finding were identified and resolved additively:

1. SETUP and READY inherited a short frame deadline while the server verified/loaded
   an entire assigned set. Their first reply byte now waits within the remaining
   whole-setup budget. Once a frame starts, its remaining header and payload keep
   the short frame deadline. The server starts its setup clock before verification.
   Delayed-executor sources cover successful long READY waiting and client/server
   setup expiry. Ordinary filesystem calls and kernels remain unpreemptible; the
   Hermes outer process timeout supplies the hard wall bound.
2. Malformed replies originally stopped at codec unit sources. Five dishonest SGW1
   peer cases now feed truncated, duplicated, wrong-binding, nonfinite and wrong-
   sequence replies through the real client transport. They check rejection, dead
   state, cleared partials, and absence of a retry frame. Thread joins precede reads
   of peer state and service counters.

The reviewer rechecked `9b0c7b1` and reported no additional static blockers in the
correction. The initial pass found no further blockers in CMake native-OFF/CPU-only
boundaries, CUDA buffer sizing, cache containment/publication, or setup-keepalive
thread ownership. No hard documented style violations were identified. Compressed
statements/naming in transport sources remain a nonblocking readability concern.

## Author static checks and remaining gate

Canonical origin was verified as `git@github.com:Zutfen-LLC/Strata.git`. GitHub main
was checked before base selection and before publication at the requested exact
base SHA, with no drift. The existing clean residency checkout at `466b01d` is an
ancestor with the identical tree; the feature uses a separate branch/worktree.

`git diff --check` is clean. CPU wire/service/client/tier implementations, CPU kernels,
the ISA checker and historical residency evidence have no diff against the base.
No history was rewritten. The first three additive commits separate assets, worker
and routing; later commits contain documentation and static-review fixes.

All build/test/benchmark/model-load results are **not run**. No performance, full-
model parity, HIP/SYCL/Windows or real-model acceptance is claimed. The PR remains a
draft for the bounded [Hermes fixture gate](HERMES.md). A real-model 03+04+05 window
requires separate authorization after fixtures pass. Do not merge at this gate.
