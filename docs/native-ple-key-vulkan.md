# Layer-1 PLE key projection

`--native-ple-key-t1 PREFIX` is an experimental, default-off Q8_0 key
projection for serial, eager layer-1 PLE. It requires a strict accelerator,
resident GDN attention weights and a fresh receipt prefix. The existing
native Q8_1 quantizer consumes the 2,560-value gathered embedding, and the
existing Q8_0 MMVQ consumer produces 10,240 key values. This is the native
`PleWeights::key_native_data` Q8_0 path. It is not the native S2 key fallback
or the separate BF16 key path.

The PLE graph consumes the actual producer-owned device output for its
existing grouped key normalization. Its value projection, gate, grouped
normalization, convolution, residual and both histories remain in the
existing portable path. The key projection runs before the graph and does not
advance the token or normalized-convolution history. A later projection,
graph or FFN failure cannot publish the candidate state; the existing layer
transaction remains responsible for commit, reset and retry.

The Q8_0 role stays resident in its original format. The component borrows its
existing device bank and accounts separately for its own input, packed and
output banks. It checks the resident role against its original bytes and
rejects nonfinite weight scales. Input admission is the existing bounded Q8
producer domain, including its finite and normal-or-zero restrictions. These
limits are diagnostic support boundaries, not claims about all CUDA inputs.

`portable_native_ple_key_test` checks owner and buffer identity, admission,
pre-dispatch refusal and retry with host providers. The layer integration test
checks the actual normalization input bank, gathered input, native-format
host reference, FFN failure noncommit, token/history ordering and reset.
Existing Q8 producer and row-layout controls also run. These are CPU tests;
CUDA/Vulkan numerical comparison and model execution remain necessary. The
option does not enable native PLE post-operations or claim full PLE parity.
