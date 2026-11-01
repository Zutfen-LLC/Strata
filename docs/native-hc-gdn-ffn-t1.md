# Actual serial GDN-to-FFN HC read

`PortableGdnOptions::native_hc_ffn_read` connects the actual layer-0 GDN
producer to a native-method FFN HC read. It requires the existing composed
attention read, strict accelerator, resident dense weights and a receipt prefix.
It refuses PLE, chunks, production policy, diagnostic prior-attention tuples and
the legacy observer schema. Defaults and existing model CLI flags are unchanged;
there is no new model flag or manufactured prior write at the first attention
entry. The C++ coordinator path is executable with existing model-loader views;
real FFN role and same-input numerical qualification remain required before
exposing it as a model option.

Native `verify.cpp` calls `gr_read_group(1, true, inj_, inj2_)` after the actual
GDN projection. `layer.cpp` likewise skips the eager attention write in fused
mode and supplies original R, block output and attention injection to half 1.
The portable opt-in retains the actual projected tensor as a graph output and
reads the tuple before releasing the shared arena. It does not reconstruct BO
from a rounded residual or substitute a fixture BO at this boundary.

The FFN owner receives immutable original R, actual BO and actual attention
injection. Existing pending sigmoid/FMA, norm/injection and down/up/mixed
components produce a disjoint R-prime and half-1 HC outputs. Both FFN HC and
the final residual consumer use that same candidate. Router/shared/expert
selection and provider contracts retain their configured methods. The existing
final FFN residual write is still materialized by GGML; deferred FFN-to-next-layer
carry and its arithmetic are **not** implemented by this slice.

The nine-tensor FFN loader replaces only its three HC Q8 banks with the original
compat-BF16 conversion, streaming one conversion at a time. There is no second
resident Q8 bank or retained host role copy. Real aligned weights, owner scratch,
component metadata and both scheduler arenas are admitted under existing budgets.
The FFN ledger adds a conservative 8 MiB host conversion/preflight/trace envelope.
No quota, cap or physical reserve is changed; model placement must be replanned
from the complete resulting ledger rather than historical counts.

State/history/token metadata and public last residual commit after the whole
FFN/provider transaction succeeds. Failure leaves public candidates uncommitted;
paid successful component counters remain cumulative across reset. Every call
uploads a fresh complete tuple, and the caller's original operands remain
unchanged. Stable HC owners outlive the referencing schedulers. The legacy
batch and implicit no-tuple FFN APIs explicitly refuse in this mode.

Focused host tests execute the actual GDN/FFN graphs with mocked HC device
arithmetic. They inspect the actual projected graph output and FFN mixed consumer,
use a separate scalar FMA-before-normalization reference, and compare the final
FFN output against an independently materialized candidate control. They cover
host/resident state, private/shared arena, reset/retry, stale tuple prevention,
component/provider failure, shape/nonfinite refusal, accounting and disabled
output equivalence. A deliberate old-R final-consumer mutant fails. Mocks do
not qualify physical arithmetic or the real FFN roles.

`portable_gdn_native_hc_probe OPERANDS FRESH_OUTPUT EXPECT_DEVICE ffn` uses the
approved attention inputs with deterministic synthetic remaining GDN/FFN weights.
BO is the actual projection of that synthetic GDN execution, not an invented
prior tuple. It exports bounded FFN observations, checks reset/rollback and a
sequential same-backend FFN control fed the actual materialized R-prime. Whole
state equality remains source-enforced rather than an exported state-array audit.
No full-HC, actual-model-pending, CUDA/AMD identity or quality claim follows.
The existing numerical contracts remain unchanged. No hardware execution is
implied by compiling this probe.
