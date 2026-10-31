# Native layer-0 pending input snapshot

The opt-in native layer-0 observer writes STRL004. Three required fields are
copied immediately before `gr_read_group(1, true, inj_, inj2_)`: original R
[10240 F32], actual GDN projection BO [2560 F32] and previous attention injection
[4 F32]. These are exact stream-ordered device copies before the fused read can
write R. They are not reconstructed from a rounded post-read residual.

The existing observer reservation remains 8 MiB, admitted before expert sizing.
The added fields consume 51,216 payload bytes plus three eight-byte record
headers inside that reservation. Serialization/download still happens only at
the existing successful window completion. There is no additional synchronization,
new numerical execution, state commit, native mode or scheduling selection.
T>1 graphs do not take a T1 snapshot; the configured speculative capacity stays
independent of the actual T1 graph. Absent observer request creates no owner or
copy nodes. A disabled direct owner also allocates nothing.

Portable writers continue STRL003. Readers preserve STRL001/002/003 scope and
report the new tuple unavailable there. A STRL004 file must provide all three
finite, exact-shaped fields and its normal seal. Comparing older/native versus
portable traces does not substitute another boundary for a missing tuple.
Equal final output or model provenance does not establish role byte identity.

Host tests cover graph-copy failure, null/bounds/version/required-field refusal,
original operands remaining unchanged, serialized pre-mutation values surviving
source-buffer reuse, disabled copies/allocations, legacy reader scope and exact
source placement before the destructive FFN read. The portable full-GDN test
continues to check enabled/disabled numerical equivalence on its available CPU
backend. CUDA calls in the native owner test are mocked: actual native compile
and hardware enable/disable equality remain unverified until the separately
prepared native build/capture gate. The observer itself supplies diagnostics,
not model numerical qualification.
