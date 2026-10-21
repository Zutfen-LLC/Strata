"""Frozen product argv, environment and request bytes from historical drivers.

Binding values remain private. Rendering is not launch authorization.
"""
from resource_guard import CoverageError
from policy import ENGINE_CAPS, GPU_CAPS
import gpu_completion as completion

REQUESTS = b"GEN 64 9419\nGEN 64 14773,6353\nGEN 64 9419\nQUIT\n"
B0_REQUESTS = b"GEN 64 9419\nQUIT\n"
COMPARISON_REQUESTS = b"GEN 64 9419\nGEN 64 14773,6353\nQUIT\n"


def requests(phase):
    """B0/G0 single-A smoke; B1/G1 identical two-prompt comparisons."""
    if phase not in ENGINE_CAPS:
        raise CoverageError("request-phase-mismatch")
    return B0_REQUESTS if phase in ("B0", "G0") else COMPARISON_REQUESTS


def product(role, phase, b, *, gpu_completion=False):
    engine_caps = completion.ENGINE_CAPS if gpu_completion else ENGINE_CAPS
    gpu_caps = completion.GPU_CAPS if gpu_completion else GPU_CAPS
    setup_ms = str(completion.SETUP_MS if gpu_completion else 1200000)
    env = {"OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin", "LANG": "C",
           "LC_ALL": "C", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if role == "ROLE_CPU_C" and phase == "CPU":
        argv = [b["BIN_CPU_C"], "--port", b["CPU_PORT_C"], "--workers", "2",
                "--memory-budget-mib", "106428", "--deadline-ms", "10000",
                "--local-setup-ms", "600000", "--local-root", b["LOCAL_ROOT_C"],
                "--local-manifest", b["MANIFEST_C"], "--residency-ms",
                str(completion.WINDOW_S * 1000) if gpu_completion else "1800000"]
        return argv, env, 7200
    if role == "ROLE_GPU_B" and phase in gpu_caps:
        env["CUDA_VISIBLE_DEVICES"] = b["GPU_SELECT_B"]
        argv = [b["BIN_GPU_B"], "--plan", b["GPU_PLAN_B"], "--cache-root", b["CACHE_B"],
                "--port", b["GPU_PORT_B"], "--device", "0", "--cache-budget-mib", "18432",
                "--reserve-mib", "4096", "--deadline-ms", "10000", "--setup-ms", setup_ms]
        return argv, env, gpu_caps[phase]
    if role != "ROLE_ENGINE_A" or phase not in engine_caps:
        raise CoverageError("role-phase-mismatch")
    env.update(STRATA_IO_PF_THREADS="2", STRATA_ROUTE_TAIL_SKIP="0", STRATA_LOOKAHEAD="0",
               STRATA_LOOKAHEAD_STATS="0", STRATA_FS_SLOTS="0",
               STRATA_DUMP_FIRST_LOGITS=b["RUN_A"] + "/" + phase + ".first-logits")
    env["CUDA_VISIBLE_DEVICES"] = b["GPU_SELECT_A"]
    argv = [b["BIN_ENGINE_A"], "--serve", "--native", b["NATIVE_SHARD_A"], "--pack",
            b["NATIVE_PACK_A"], "--gpu", b["GPU_SELECT_A"], "--mmap-experts", "--greedy",
            "--pool-workers", "2", "--remote-cpu-experts", b["ENDPOINT_CPU_C_ON_A"],
            "--remote-cpu-deadline-ms", "10000", "--remote-cpu-local-manifest", b["MANIFEST_A"],
            "--remote-cpu-resident", "--remote-cpu-setup-ms", "600000", "--expert-cache", "960",
            "--expert-cache-per-layer", "--expert-profile", b["PROFILE_A"],
            "--expert-cache-device1", "2247", "--remote-expert-opt", "--vram-reserve-mib", "1152",
            "--adapt-every", "0", "--adapt-swaps", "0", "--pcie-frac", "0", "--short-read", "64",
            "--prefill", "64", "--prompt-cache", "0", "--spec", "2", "--suffix-draft", "0",
            "--lookup-chain", "0", "--pipeline-windows", "0", "--max-context", "1024",
            "--dump-routing", b["RUN_A"] + "/" + phase + ".routing.bin"]
    if phase.startswith("G"):
        argv += ["--remote-gpu-experts", b["ENDPOINT_GPU_B_ON_A"], "--remote-gpu-plan", b["GPU_PLAN_A"],
                 "--remote-gpu-setup-ms", setup_ms, "--remote-gpu-deadline-ms", "10000"]
    return argv, env, engine_caps[phase]
