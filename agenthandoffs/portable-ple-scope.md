# Portable PLE layer 1 scope (v1)

Stack: PR39 a06902e452257824e454e1d00e00928161a3ef79.
Implement configured PLE before HC/GDN/FFN in layer 1. Keep Q8_0 table
host mapped; gather only sixteen rows per token. Reuse Strata hash and Q8_0
row decoding. Preserve exact upstream grouped norms, signed square root gate,
dilation-three normalized convolution history, and EOS predecessor semantics.
Commit token history, PLE history, GDN history/state and residual together only
after successful expert dispatch. Explicit reset clears every history.
QSA, full-model embedding/head/all-layer loop, fleet operations and CUDA changes
are excluded.

Tests: independent small-table hash/lookup and EOS vectors; malformed metadata
and binding; actual H2560 layer1 multiple tokens; scalar PLE intermediates;
callback failure/throw/nonfinite rollback and retry; reset; CPU/software Vulkan
parity and strict actual placement; unchanged prior portable FFN/GDN regressions.
Actual V340L checks remain parent dispatched. No long build started at scope
checkpoint. Local retained ggml exists at the required pin.
