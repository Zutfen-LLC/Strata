# One metadata computation, then execution approval

This is a sanitized planning artifact. All role, path, accelerator-selection and endpoint identifiers are defined semantically in [README.md](README.md); their operational values must be supplied privately. Never commit those bindings or generated private outputs.

Pinned implementation: `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, tree `8e0cc2eda89f17a63182d1b273140288a1418b47`. Expected main: `34a59d83a51bdb3f0299d65306740bd357818715`. PR #2 remains draft/unmerged. This artifact revision changes planning text and neutral planner labels only. It is not a tested implementation revision or permission to run a model.

## Source and identity reasoning

The selection rules in the primary fill, helper fill and CPU complement are unchanged between historical `16d065e23211f2d422c56a3497a8be200c1e1d4d` and the pinned implementation. Source blob pins: `expert_cache.cpp` is `bd98ad711dd71fa15e196e8dda194645fc202de7`; `remote_experts.cpp` is `6e684959557155a113f85612952170d55b97626b`. The profile blob is `a4ed7e5feb68e68fa76df66b9109e99c1ef143b8`, 196632 bytes, SHA-256 `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.

Historical exact ownership can be derived without saved owner lists only when complete profile identity, per-layer policy, realized 22 primary slots per layer, single-helper realized count 2263, no unranked appends, and absence of peer/stage/adaptive reordering are established from private receipts. Matching counts or the global asset manifest alone do not prove historical placement. If the premises are missing or different, still compute the new exact partition once, label history NOT ESTABLISHED or DIFFERENT, and request explicit approval of that known partition. Do not silently claim continuity or alter an existing resident arena.

Eligible CPU-complement identities are `24576-1056-2263=21257`; 3319 is the local accelerator total. The proposed remote set contains 3289 experts, `17176473600` bytes. Workspace is `1598720` bytes; requested weight plus workspace is `17178072320` bytes. Actual free-memory observations remain private and later CUDA admission is authoritative.

`strata-gpu-manifest plan` reads only the small trusted manifest and assignments, then writes placement metadata. It opens no tensor payload file and initializes no GPU context. The independent planner encoding must match its output byte-for-byte. No actual assignment or asset-plan hash is frozen until the approved metadata computation succeeds.

## Bounded metadata procedure: separate approval required

1. Retrieve only the four small planning files at the exact artifact commit using existing authorized GitHub authentication. This narrow retrieval exception permits neither a repository clone nor source/dependency/model downloads. Verify supplied file sizes and hashes, then stop unless metadata computation is separately approved. The reviewed sanitized planner SHA-256 is `801206ff8e93fdc8141299121b65f7371f609bf9b868731a1d3c181c0281caf7`; use this artifact's source rather than an earlier private copy.
2. Bind identifiers privately. Run no planner, Python, configure, builds, tests, binaries, benchmarks or model loads on ROLE_EDITOR. The separately approved metadata step may use ROLE_ENGINE_A and read-only provenance inspection on ROLE_GPU_B only. Do not clone/fetch/advance sources, rebuild, install, contact services, transfer weights, read tensor payloads, regenerate a manifest, start workers/engines or modify existing assets/evidence. Preserve previous evidence and disclose any earlier preparation side effects privately, without repeating passing assertions for presentation.
3. Use one monotonic 300-second wall including cleanup; reserve the final 15 seconds and clip nested operations to remaining time. At most 16 MiB of source/metadata/report transfer, planner output at most 4 MiB, profile at most 1 MiB and manifest at most 8 MiB per read. Logs and RCs are new and append-only. Failed/incomplete outputs must not be accepted, overwritten or rerun just to obtain success.
4. Resolve STEP_A_PACKET, PROFILE_A, MANIFEST_A and GPU_MANIFEST_A. Require the complete pinned profile SHA above and manifest SHA `8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318`. Verify the live metadata binary against EXPECTED_GPU_MANIFEST_A_SHA and existing build/configuration provenance. Do not execute a binary merely to identify it.
5. Run the supplied planner once on ROLE_ENGINE_A. OUT_A must be a new absolute directory. The virtual-memory limit applies only to this metadata subprocess:

```sh
(
  ulimit -v 262144
  exec timeout --signal=TERM --kill-after=5s 115 python3 -I "$PLANNER_A" \
    --profile "$PROFILE_A" \
    --profile-sha256 8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf \
    --manifest "$MANIFEST_A" --out "$OUT_A"
)
```

Require RC 0, SUCCESS.json, and verification of every FILES.sha256 entry from inside OUT_A. The source validates the STRP trailing inverse-rank table and the complete SWLOCAL1 schema/ranges/digest inventory. It opens only the two pinned metadata files. It computes primary_A=1056, helper_A=2263, cpu-full_C=21257, assigned_B=3289 and cpu-uncovered_C=17968; disjoint/exhaustive execution ownership; the unchanged full CPU bitmap; per-layer counts; per-asset source-range/digest closure; canonical expected plan metadata and hashes. Digest128 is the existing noncryptographic integrity identity, not authentication.

6. Run the provenance-verified metadata CLI once, with a new output file:

```sh
timeout --signal=TERM --kill-after=5s 30 "$GPU_MANIFEST_A" plan \
  "$MANIFEST_A" "$OUT_A/assigned_B.tsv" "$OUT_A/plan_B.bin"
cmp -- "$OUT_A/plan_B.expected.bin" "$OUT_A/plan_B.bin"
```

Require source_weight_read=0, 3289 assets, 17176473600 assigned bytes, exact byte equality, and full plan/model/layout/placement/CPU-bitmap/list hashes. A mismatch stops acceptance without retry. Closure is a statement about trusted metadata; it does not establish current tensor payload validity or historical placement by itself.

7. Complete historical proof from the private receipt inputs. On ROLE_GPU_B, read-only verify BIN_GPU_B's actual full hash/size/path against EXPECTED_BIN_GPU_B_SHA and the existing validated bundle/configuration. An archived hash alone is insufficient. Bound this single non-model binary hash to 256 MiB. Do not execute, extract, install or rebuild it. Missing/mismatched provenance blocks all later model launches but does not prevent independent metadata work. Inspect proposed endpoint occupancy passively, without service connections. Include existing ROLE_ENGINE_A memory/cgroup observations; fresh execution admission is a later gate.
8. Return one PRIVATE execution-approval packet containing the raw exact plan/owner files, full hashes and identities, historical-proof status, live binary provenance, actual resource observations, resolved commands/bindings, fixed request IDs, and the authoritative [execution-budget proposal](EXECUTION-BUDGET-bf51f09e.md). Record the CPU-after-trial retention choice and specific prerequisites. Stop before expert reads/transfers or worker/engine/model starts. No broader planning cycle, implementation change, merge or changed assignment is authorized.

Static review of this artifact is not execution evidence. The separate implementation fixture receipts remain distinct from this planning revision.
