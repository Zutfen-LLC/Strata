# Sanitized fixed-worker planning artifacts

This directory contains planning/harness delivery artifacts, not a new tested engine implementation. The implementation remains `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, tree `8e0cc2eda89f17a63182d1b273140288a1418b47`, based on main `34a59d83a51bdb3f0299d65306740bd357818715`. PR #2 stays draft/unmerged. Artifact history changes only this directory.

The original four-file publication is pinned at `a6ede26d536e20f1f9b7db5a9601e23f3b995060`. Its planner, metadata handoff and execution budget remain unchanged as historical records. Current resource-monitor amendment v1 adds standalone replacement source and authored synthetic fixtures, NOT EXECUTED. The reported later placement is frozen; do not run the old planner again. See [amendment](RESOURCE-MONITOR-AMENDMENT-v1.md), [future synthetic handoff](HERMES-RESOURCE-MONITOR-v1.md) and [static review](RESOURCE-MONITOR-STATIC-REVIEW-v1.md). Faithful integration awaits sanitized original driver functions; no trial retry is authorized.

- [Planner](METADATA-PLANNER-bf51f09e.py): sanitized comments and neutral output labels; selection, format validation, digest and plan-encoding logic are retained. It has not been executed for this publication.
- [Metadata handoff](METADATA-HANDOFF-bf51f09e.md): historical bounded metadata procedure; the later frozen plan is not to be recomputed.
- [Execution budget](EXECUTION-BUDGET-bf51f09e.md): unchanged historical limits/CLI templates. Its old approval does not authorize a retry; monitor semantics are clarified by the separately named amendment.

Publication/retrieval authorizes no planner, Python, configure, build, test, benchmark, model operation, worker launch, merge or remote-service control. Static source/sanitization review is not runtime validation. No operational bindings, private receipts, generated ownership results or model assets are included. Keep all such material private.

## Semantic identifiers

Identifiers below have no public machine, account, address, device or filesystem binding. Supply real values only in the private operator handoff. Freeze the values in the exact approval packet; do not discover or change them dynamically.

| Identifier | Semantic purpose |
| --- | --- |
| ROLE_ENGINE_A | Engine role owning the local primary and helper CUDA accelerators and selective source metadata/files. |
| ROLE_GPU_B | Fixed network CUDA expert-worker role with one selected accelerator and bounded cache. |
| ROLE_CPU_C | Persistent CPU expert-worker role retaining the complete approved CPU residency partition. |
| ROLE_CONTROLLER | Operator orchestration, immutable artifact retrieval and private evidence collection role. |
| ROLE_EDITOR | Editing/static-review role; prohibited from planner/Python/build/test/binary/benchmark/model execution. |
| STEP_A_PACKET | Private prior metadata/provenance/resource packet and its named receipts. |
| PROFILE_A | Absolute path of the pinned trusted STRP profile on ROLE_ENGINE_A. |
| MANIFEST_A / MANIFEST_C | Absolute paths of byte-identical trusted SWLOCAL1 manifests on the engine/CPU roles. |
| PLANNER_A | Absolute path of this verified sanitized planner copied into a new private metadata workspace. |
| OUT_A | New absolute metadata-output directory; must not already exist. |
| GPU_MANIFEST_A | Absolute path of the existing candidate strata-gpu-manifest metadata CLI. |
| EXPECTED_GPU_MANIFEST_A_SHA | Full SHA-256 from its verified private build/bundle provenance. |
| BIN_ENGINE_A | Absolute path of the existing candidate strata engine executable. |
| BIN_GPU_B | Absolute path of the intended live candidate strata-cuda-expert-worker executable. |
| EXPECTED_BIN_GPU_B_SHA | Full SHA-256 from its verified private build/bundle provenance; compare with the live file. |
| BIN_CPU_C | Absolute path of the existing candidate strata-expert-worker executable. |
| NATIVE_SHARD_A / NATIVE_PACK_A | Existing immutable native reference input and pack paths used by the engine. |
| LOCAL_ROOT_C | Existing immutable local CPU tensor-source root; no network upload fallback. |
| GPU_PLAN_A / GPU_PLAN_B | OUT_A/plan_B.bin and its later separately approved identical metadata copy on the GPU-worker role. |
| CACHE_B | New initially empty, exclusively owned bounded GPU disk-cache directory. |
| RUN_A / RUN_B / RUN_C | New private append-only evidence directories for the three execution roles. |
| REQUESTS_A | Private input file containing the fixed request bytes in the budget. |
| GPU_SELECT_A | Verified engine accelerator-selection list, ordered as primary then helper. |
| GPU_SELECT_B | Verified single GPU-worker accelerator selection for CUDA_VISIBLE_DEVICES; visible ordinal zero is used afterward. |
| CPU_PORT_C / GPU_PORT_B | Privately approved unoccupied loopback listener ports for the CPU/GPU worker roles. |
| ENDPOINT_CPU_C_ON_A / ENDPOINT_GPU_B_ON_A | Numeric loopback host:port endpoints visible to the engine through existing approved private transport. |
| GPU_PHASE_CAP_S / PHASE | Controller-selected phase hard cap including grace and unique G0/G1 evidence label. |

All path variables are private absolute paths. Endpoint/selection variables are private values, not suggested network or physical-device mappings. The public documents intentionally provide no relay route, host naming convention or operational username. Full source/profile/manifest hashes are integrity pins, not credentials or placement proof.

Output labels primary_A/helper_A identify the engine's local sets; assigned_B is the fixed remote GPU set; cpu-full_C is the preserved residency set; cpu-uncovered_C is its execution complement after removing assigned_B. The source uses count_B/bytes_B for remote-assignment summary fields and plan_B.expected.bin for independent canonical metadata. These neutral labels do not alter the encoded expert identities or selection order. Generated file inventories, closure basenames/offsets, output paths and future observed plan hashes remain private; do not commit them here.

## Small-file retrieval and integrity

The original narrow delivery exception applies to the four original files at their pinned publication: at most16 KiB each/64 KiB total. The new monitor handoff proposes separately approved retrieval/fixture validation of its explicitly named ten-file set, at most64 KiB/file/256 KiB total. Use existing authorized GitHub authentication and exact immutable revisions. Neither route permits a repository clone, branch checkout, engine/dependency/model download, installation, credential setup or model/runtime authority. Stop on hash/size/identity mismatch; no moving-branch imports or automatic retry.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| METADATA-PLANNER-bf51f09e.py | 11110 | `801206ff8e93fdc8141299121b65f7371f609bf9b868731a1d3c181c0281caf7` |
| METADATA-HANDOFF-bf51f09e.md | 7437 | `80b4a90b84369919878fdc1b449dcbd926796909ebb1c66b7aa1fad220476884` |
| EXECUTION-BUDGET-bf51f09e.md | 15063 | `8801a57e48c290b9f9e78e0f27556d1bfa90f5ea17e1f296a8d56b48e15c8085` |

The publication handoff supplies the checksum file's own SHA/size to avoid a circular self-hash. `SHA256SUMS.resource-monitor-v1` binds this README and the eight other monitor files. Historical four-file hashes still apply at their pinned revision. The pinned implementation's reported fixture evidence is distinct from the new unexecuted monitor sources. The later reported plan hash/counts appear in the amendment as user-reported evidence; they were not recomputed here. Keep all generated owner lists and operational bindings private.
