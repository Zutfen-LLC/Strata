# Source map — namespace-preflight-v1

Published sanitized handoff for the blocked a7f54f3 no-model validation
(admission PermissionError on /proc/1/exe). Branch:
handoff/namespace-preflight-v1 (base a7f54f3ade61d8b712489d28f689bf92cd141aa5).

| Published file | Private origin (controller evidence root, retained locally) | Sanitization |
| --- | --- | --- |
| FAILED-PREFLIGHT-SANITIZED.md | evidence/preflight.py, evidence/preflight.stderr, evidence/preflight.command.json (executed argv) | verbatim source; PID 1 of the remote host shown as generic /proc/1; ROLE_* identifiers only; no hostnames/users/ns inodes/raw logs |
| CORRECTION-SPEC.md | findings derived from: evidence/preflight.*, published a7f54f3 planning sources (resource_telemetry.py verify_initial_namespaces/OwnedProcess.pin, live_probe.py, README.md private-binding contract), read-only ROLE_CPU_C observation of 2026-10-21 | observation result summarized; host named only as ROLE_CPU_C; boot_id and namespace inode values withheld from GitHub, retained locally |
| SHA256SUMS | computed over the two files above at publish time | n/a |

Private mappings and original hashes are retained only in the controller
evidence root (see REPORT.md SHA-256
f20ce2ed7218de78303ffea491705277ce58d9a1ccb0a6e3146611a79d3818d3 and
private-bindings.json); they are intentionally not reproduced here.

Related published code at base revision (unchanged by this handoff):
docs/network-gpu-worker/planning/supervisor-integration-v1/ — in particular
resource_telemetry.py (namespace verifier, OwnedProcess), live_probe.py
(consumes `namespaces` pairs), README.md (private binding contract: pairs are
admin-trusted, "not a receipt inferred from an untrusted container's PID 1").
