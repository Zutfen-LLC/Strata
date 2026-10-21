"""One bounded GPU-only completion; no product or I/O occurs at import."""
from pathlib import PurePosixPath

from resource_guard import CoverageError

PHASES = ("G1",)
WINDOW_S = 1200
SETUP_MS = 1800000
ENGINE_CAPS = {p: WINDOW_S for p in PHASES}
GPU_CAPS = {p: WINDOW_S for p in PHASES}
ASSIGNED_BYTES = 17176473600
HISTORICAL_ACTIVE_S = 5744.2
ARCHIVE_SHA = "35a57b3a4c84aec93ece3c1b255b4704f58d140abf9f07dc779e8a83a723f41d"
G0_ARCHIVE_SHA = "511471cd5cbdbee193426be21dd8ffdb6c89755d822294b6594633ef9040a552"
IMPLEMENTATION = "bf51f09e286a7a5ce89fbff67b3ee07cb044a5e5"
PLAN_SHA = "a51c6e4acc74cd84f5333d3b13b4cd58abf1ddf6276b59fd757fa815f8045578"


def validate_baseline(receipt, verify_file):
    """Use retained B outputs/logits; metadata comes from the trusted old report.

    Hash only these seven small files, not models/binaries or old archives.
    Numerical/output interpretation remains the existing evidence workflow.
    """
    if not isinstance(receipt, dict) or receipt.get("archive_sha256") != ARCHIVE_SHA or \
            receipt.get("g0_archive_sha256") != G0_ARCHIVE_SHA or \
            receipt.get("implementation") != IMPLEMENTATION or receipt.get("plan_sha256") != PLAN_SHA:
        raise CoverageError("completion-baseline-source-mismatch")
    paths = set()
    for phase, tokens in (("B0", [64]), ("B1", [64, 64]), ("G0", [64])):
        row = receipt.get(phase)
        if not isinstance(row, dict) or type(row.get("rc")) is not int or row["rc"] != 0 or \
                row.get("done_tokens") != tokens or not isinstance(row.get("logits"), list) or \
                len(row["logits"]) != len(tokens):
            raise CoverageError("completion-baseline-incomplete:" + phase)
        if phase == "G0" and (row.get("product_complete") is not True or
                              row.get("supervisor_status") != "FAILED"):
            raise CoverageError("g0-product-reference-must-preserve-supervisor-failure")
        for artifact in [row.get("output"), *row["logits"]]:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or \
                    not PurePosixPath(artifact["path"]).is_absolute() or artifact["path"] in paths:
                raise CoverageError("completion-baseline-artifact-path")
            paths.add(artifact["path"])
            verify_file(artifact["path"], artifact.get("sha256"), 8 * 1048576)
    return receipt


def cache_accounting(phase, *, reused, transferred, persisted, source_read, cached, assigned):
    """Check actual successful product counters after setup, not a disk inventory.

    Hermes calls this on its existing parsed evidence. It is not product validation
    of the cache itself; the product's missing bitmap and digest checks do that.
    """
    values = (reused, transferred, persisted, source_read, cached, assigned)
    if phase not in ("G0", "G1") or any(type(v) is not int or not 0 <= v <= ASSIGNED_BYTES for v in values):
        raise CoverageError("completion-cache-counter-schema")
    if reused % 5222400 or transferred % 5222400 or \
            assigned != ASSIGNED_BYTES or cached != assigned or reused + transferred != assigned or \
            source_read != transferred or persisted != transferred:
        raise CoverageError("completion-cache-plan-not-reconciled")
    if phase == "G1" and (reused != assigned or transferred != 0):
        raise CoverageError("completion-warm-cache-transferred-weights")
    return {"phase": phase, "cache_state": "resumed" if phase == "G0" else "warm",
            "assigned_bytes": assigned, "validated_reuse_bytes": reused,
            "new_assigned_payload_bytes": transferred}
