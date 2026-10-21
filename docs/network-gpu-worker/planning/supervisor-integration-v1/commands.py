"""Frozen product argv, environment and request bytes from historical drivers.

Binding values remain private. Rendering is not launch authorization.
"""
from resource_guard import CoverageError
from policy import ENGINE_CAPS, GPU_CAPS

REQUESTS = b"GEN 64 9419\nGEN 64 14773,6353\nGEN 64 9419\nQUIT\n"


def product(role, phase, b):
    env = {"OMP_NUM_THREADS": "2"}
    if role == "ROLE_CPU_C" and phase == "CPU":
        argv = [b["BIN_CPU_C"], "--port", b["CPU_PORT_C"], "--workers", "2",
                "--memory-budget-mib", "105870", "--deadline-ms", "10000",
                "--local-setup-ms", "600000", "--local-root", b["LOCAL_ROOT_C"],
                "--local-manifest", b["MANIFEST_C"], "--residency-ms", "1800000"]
        return argv, env, 7200
    if role == "ROLE_GPU_B" and phase in GPU_CAPS:
        env["CUDA_VISIBLE_DEVICES"] = b["GPU_SELECT_B"]
        argv = [b["BIN_GPU_B"], "--plan", b["GPU_PLAN_B"], "--cache-root", b["CACHE_B"],
                "--port", b["GPU_PORT_B"], "--device", "0", "--cache-budget-mib", "18432",
                "--reserve-mib", "4096", "--deadline-ms", "10000", "--setup-ms", "1200000"]
        return argv, env, GPU_CAPS[phase]
    if role != "ROLE_ENGINE_A" or phase not in ENGINE_CAPS:
        raise CoverageError("role-phase-mismatch")
    env.update(STRATA_IO_PF_THREADS="2", STRATA_ROUTE_TAIL_SKIP="0", STRATA_LOOKAHEAD="0",
               STRATA_LOOKAHEAD_STATS="0", STRATA_FS_SLOTS="0",
               STRATA_DUMP_FIRST_LOGITS=b["RUN_A"] + "/" + phase + ".first-logits")
    argv = [b["BIN_ENGINE_A"], "--serve", "--native", b["NATIVE_SHARD_A"], "--pack",
            b["NATIVE_PACK_A"], "--gpu", b["GPU_SELECT_A"], "--mmap-experts", "--greedy",
            "--pool-workers", "2", "--remote-cpu-experts", b["ENDPOINT_CPU_C_ON_A"],
            "--remote-cpu-deadline-ms", "10000", "--remote-cpu-local-manifest", b["MANIFEST_A"],
            "--remote-cpu-resident", "--remote-cpu-setup-ms", "600000", "--expert-cache", "1056",
            "--expert-cache-per-layer", "--expert-profile", b["PROFILE_A"],
            "--expert-cache-device1", "2263", "--remote-expert-opt", "--vram-reserve-mib", "1152",
            "--adapt-every", "0", "--adapt-swaps", "0", "--pcie-frac", "0", "--short-read", "64",
            "--prefill", "64", "--prompt-cache", "0", "--spec", "2", "--suffix-draft", "0",
            "--lookup-chain", "0", "--pipeline-windows", "0", "--max-context", "1024",
            "--dump-routing", b["RUN_A"] + "/" + phase + ".routing.bin"]
    if phase.startswith("G"):
        argv += ["--remote-gpu-experts", b["ENDPOINT_GPU_B_ON_A"], "--remote-gpu-plan", b["GPU_PLAN_A"],
                 "--remote-gpu-setup-ms", "1200000", "--remote-gpu-deadline-ms", "10000"]
    return argv, env, ENGINE_CAPS[phase]
