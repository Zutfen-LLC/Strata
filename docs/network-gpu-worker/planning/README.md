# Sanitized fixed-worker planning artifacts

These four files are planning delivery artifacts, not a new tested implementation. The implementation remains `bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5`, tree `8e0cc2eda89f17a63182d1b273140288a1418b47`, based on main `34a59d83a51bdb3f0299d65306740bd357818715`. PR #2 stays draft/unmerged. The dedicated artifact commit is identified by the immutable retrieval handoff; its extra commit contains only this directory.

- [Planner](METADATA-PLANNER-bf51f09e.py): sanitized comments and neutral output labels; selection, format validation, digest and plan-encoding logic are retained. It has not been executed for this publication.
- [Metadata handoff](METADATA-HANDOFF-bf51f09e.md): one separately approved bounded metadata computation, then STOP for exact execution approval.
- [Execution budget](EXECUTION-BUDGET-bf51f09e.md): authoritative proposed limits and candidate CLI templates; no runtime authority or PASS claim.

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

The narrow delivery exception permits authenticated retrieval of ONLY README.md and the three linked files from this directory at the exact immutable artifact commit. Use existing authorized GitHub authentication; fetch files individually, at most 16 KiB each and 64 KiB total. No repository clone, branch checkout, dependency/source/model download, installation, credential setup or runtime authority is included. Stop on a hash/size/identity mismatch. Retrieval ends after verification unless a separate metadata-computation approval exists.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| METADATA-PLANNER-bf51f09e.py | 11110 | `801206ff8e93fdc8141299121b65f7371f609bf9b868731a1d3c181c0281caf7` |
| METADATA-HANDOFF-bf51f09e.md | 7437 | `80b4a90b84369919878fdc1b449dcbd926796909ebb1c66b7aa1fad220476884` |
| EXECUTION-BUDGET-bf51f09e.md | 15063 | `8801a57e48c290b9f9e78e0f27556d1bfa90f5ea17e1f296a8d56b48e15c8085` |

The retrieval handoff supplies this README's own hash/size to avoid a circular self-hash. Sanitation changes the planner's source hash and artifact labels; the pinned implementation's reported fixture evidence does not become evidence that this artifact was executed. Exact assignment IDs/plan hashes remain uncomputed until the separately approved metadata procedure succeeds.
