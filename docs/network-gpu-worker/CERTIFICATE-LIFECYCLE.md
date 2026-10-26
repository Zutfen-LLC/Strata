# Native worker certificates and supervisor lifecycle

## Implemented transport

CPU and GPU workers share `TlsConfig` and `TlsSession`. Build native workers with
`STRATA_WORKER_TLS=ON` and installed OpenSSL 3.5 or newer. Supply a dedicated CA
bundle, the node's certificate chain, its private-key path and the intended peer's
exact SAN DNS name. This does not read or modify system trust. TLS 1.3, mutual
chain verification and exact SAN verification are required; CN and wildcard
fallback, resumption, early data and plaintext retry are disabled.

Use distinct stable identities for each worker process, including two workers on
one machine. Map each numeric endpoint to one approved SAN DNS name in endpoint
order. A worker pins the approved coordinator SAN. Routing addresses do not grant
certificate identities. Source-IP admission remains an additional filter.

Both worker CLIs accept these arguments alongside existing listener/model options:

```text
--tls-ca /operator/ca.pem
--tls-cert /operator/node-chain.pem
--tls-key /operator/node-key.pem
--tls-peer coordinator.example.internal
```

The coordinator takes the corresponding `--remote-cpu-tls-ca`,
`--remote-cpu-tls-cert`, `--remote-cpu-tls-key`, and repeated
`--remote-cpu-tls-peer worker-a.example.internal` arguments. Use the GPU equivalents
`--remote-gpu-tls-*` for GPU endpoints. Single legacy endpoint syntax takes one
peer; groups take exactly one name per endpoint in the same order. The coordinator
uses its own certificate/key for each worker class. Keep keys node-local, readable
only by the process account. Encrypted PEM keys are rejected without prompting.
Never put private keys in arguments, logs, source control, Slack or test bundles.

Configuration validation rejects missing/incomplete paths and expired/not-yet-valid
local certificates. Handshake diagnostics report certificate verification reasons
(such as expiry, issuer or identity failure) or a TLS/missing-certificate/deadline
error. They do not print keys, certificate bodies or inference payloads. Operators
must independently monitor public certificate expiry and fingerprints: this PR has
no renewal scheduler or proactive expiry alert.

One handshake precedes existing CPU SEW1 or GPU SGW1 frames. Persistent connections
carry setup, upload/local attachment, cache status and many inference requests.
Finite monotonic I/O deadlines and cancellation remain active. Failure closes the
connection; the engine aborts that session rather than retrying plaintext. A new
explicit engine session repeats authenticated setup. Compatible verified CPU
resident weights and GPU disk caches may be reattached under their existing rules.
There is no automatic reconnect during an inference request.

## Rotation and removal today

Files are opened on each new connection/handshake, including each worker accept.
Replacing the certificate/key/CA files does not update existing TLS sessions.
Publish matching certificate and key together using operator-owned versioned paths
or an atomic directory/symlink switch; do not expose a half-updated pair. Retain
the previous public CA during an explicitly bounded overlap if an issuer changes.
Check new certificate validity, chain, identity, key match and expiry before making
it available. Then drain the affected engine session at a request boundary and
start a new authenticated session. A failed replacement refuses new connections;
there is no downgrade. Keep rollback material protected under the node's policy.

Leaf expiry is checked at handshake, not continuously on an open connection.
Short-lived certificates alone do not revoke already-open sessions. Replacing a
CA bundle or removing a certificate file also does not terminate an open session.
This PR implements no CRL, OCSP, serial denylist, connection maximum age or online
membership check. An idle/frame deadline is not a maximum connection lifetime.

For immediate removal, the operator must stop/drain the affected coordinator or
worker connections and remove the node's endpoint/approved identity from the
operator configuration. For a compromised issuer or leaf that could impersonate a
still-authorized identity, replace the trusted issuer with a new approved bundle,
reissue remaining node certificates, and close all sessions using the old trust.
Stopping renewal alone only prevents later handshakes after expiry; it is not
immediate revocation. These are operator procedures, not actions this PR executes.

## Planned supervisor ownership (separate implementation)

The coordinator/supervisor should own approved membership, certificate issuance and
automatic renewal. Keep its signing authority separated from the inference data
path, constrained to the approved node registry and roles, with an auditable
operator-controlled admission step. Each node generates and retains its own private
key; it submits a CSR and receives only public certificates/chain material. Never
transfer private keys between nodes or give arbitrary requesters a certificate.
Initial admission needs an authenticated bootstrap approved by an operator, such
as an out-of-band verified node identity and public-key fingerprint. The bootstrap
mechanism, signing-key custody and recovery procedure need their own scoped design
and authorization before implementation.

The issuer derives SAN identity and permitted client/server role from that registry,
not from an unchecked CSR. Renewal proves possession of an admitted node's key and
valid current identity, checks that membership remains active, and preserves the
stable node identity. Key rotation requires proof of old and new key possession
or explicit recovery admission. A CSR's requested identity cannot expand access.

Start qualification with a 24-hour leaf lifetime and renewal around half-life,
with bounded jitter (for example one hour) and capped exponential retry backoff.
This is a proposed starting policy, not a runtime default or a claim that 24 hours
is universally sufficient. It gives an overnight outage recovery window while
limiting stale credentials compared with long-lived leaves. Choose final timings
from the required outage tolerance, clock-skew margin and removal objectives.
Renew earlier than the last usable window, retain overlapping valid certificates,
and avoid synchronized renewals across a cluster. Retry must never extend expiry,
accept an untrusted issuer or silently re-enroll a removed node.

During a supervisor outage, admitted nodes can establish connections with their
still-valid certificates and existing sessions continue under current transport
rules. Alert on failed renewal and decreasing validity. Once expiry is reached,
new handshakes fail closed. The future lifecycle controller should drain sessions
at safe request boundaries before a configurable connection-age/expiry bound,
allowing a documented bounded in-flight grace interval. That controller and its
clock/session policy do not exist in this PR. Do not force a handshake for every
inference request. Without such a bound, busy existing sessions can outlive expiry
indefinitely, including during an outage; operators must account for this today.

Removal must disable issuance/renewal immediately, remove routing and approved peer
mapping, and trigger closure of existing sessions. A future authenticated membership
update or revocation mechanism must propagate this decision to workers, with clear
fail-closed behavior when membership freshness expires. Test issuer rotation,
overlap, jitter/backoff, clock skew, supervisor outage, restart and recovery before
making enrollment or renewal a product feature. No CA service, enrollment API,
renewal daemon, persistent credentials or fleet trust changes are implemented here.
