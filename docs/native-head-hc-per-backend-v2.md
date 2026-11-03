# Head HC negative SiLU overflow candidate, version 2

This extends only the opt-in T1 output-head mixer and its per-backend checker. It does not alter shared HC mixer admission, the earlier version 1 result, the original exact-comparison STOP, or a model default. No physical qualification or full-model parity is implied.

The unchanged native down projection computes `x / (1 + __expf(-x))`. In one retained position-26 Vulkan refusal, the captured finite preactivation was `0xc2c803e5` (about −100.0076), its exponential was positive infinity and its calculated output was negative zero. The associated offline source-order control reproduced all 320 captured preactivations. CUDA intrinsic results for that input have not been measured.

The head-only candidate retains the same exponential, rounded addition and RNE division. For a finite negative preactivation at or below −89, it admits the result only if the actual exponential and denominator are both positive infinity and the actual result is exactly negative zero. It neither clamps nor substitutes values. The transition above −89, subnormal inputs, nonfinite inputs and unexpected exponential, denominator or output classes remain refused. The shared mixer uses its unchanged admission path.

The updated head checker uses contract `native-head-hc-own-primitives-rne-overflow-v2`. It checks the same per-backend operation order, primitive bounds, captured Vulkan dots and own-gate association as version 1, with the added head-only signed-zero class. Native sum, dot, exponential and gate intermediates are still not exported. Cross-backend differences remain diagnostic; prior version 1 receipts are not reclassified.

Before a model continuation, independently review the source and build. Run bounded native-first and Vulkan physical checks with the actual captured input and boundary controls, with source, role, module, input, lifecycle and cleanup bindings. Native/Vulkan results, including the native sign of zero, must be observed rather than inferred from source. If any required class or existing guard differs, stop and retain the evidence.
