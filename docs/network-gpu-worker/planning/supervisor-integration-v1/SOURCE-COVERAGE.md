# Historical source to replacement mapping

Input: `ad921e0f09fa340bc138cc230202c1b7a6e0674d`, verified eight files,
48869 bytes, against all eight supplied SHA-256s before inspection. That
directory and its manifest remain unchanged. This is a new implementation;
static correspondence below does not claim runtime equivalence or PASS.

| Historical section | New source | Protection / limits |
| --- | --- | --- |
| CPU driver admission, lines27–48 | Session.admit / RoleGuard / original monitor cgroup_snapshot | New local stamps; fixed SwapFree/events anchor; actual membership/full ancestors; KiB→bytes fixed; required missing blocks |
| CPU launch/identity, lines50–73 | commands.product / Job / linux_owned | Same worker flags; blocked exec until coverage; pidfd/start/exe/ns proof; owned listener only |
| CPU loop/stop, lines75–113 | monitor_child / Job.stop / agent | Same host/RSS/swap limits; diagnostics added; independent parent watchdog; escalation and genuine reaping |
| Engine baseline/admission, lines24–44 | persistent Session / ROLE_ENGINE_A policy | One baseline across four phases; own12/4GiB and24/8GiB policies, never CPU limits |
| Engine sampler, lines46–65 | monitor_child / OuterGuards | Host/cgroup/RSS/anon plus selected GPU floors; missing required coverage stops |
| Engine run/after_phase, lines67–101 | commands / Job / pending_cleanup | Same requests/options/caps; normal terminal observation distinct from guard failure; up-to30s baseline accounting bounded by original clocks |
| Engine four phase gates, lines103–151 | Controller / TrialOrder | CPU covered before B0/B1; GPU listener coverage before G0/G1; any required peer failure aborts; unchanged serial phases and gaps |
| GPU driver plan/admission, lines16–50 | agent verified_file / OuterGuards | Frozen exact plan SHA; live binary hash; new cache; actual cache disk; fit formula and explicit UUID binding |
| GPU launch/readiness, lines51–77 | commands / Job / owned_listener | Same flags/caps; covered owned listener before engine; product protocol alone declares GPU READY after load |
| GPU sampler/stop, lines78–114 | role policy / outer / Job.stop | Same RSS/disk; enforced4GiB runtime reserve; parent TERM/KILL/reap; no arbitrary PID stop |
| Controller forwards/abort, lines19–53 | Forward / Peer / close_all / launch | At most four newly owned existing transport commands; all-owner cleanup; no preexisting service/forward termination |
| Controller CPU/phase sequencing, lines55–125 | Controller.run / agent challenge protocol | No standalone preload; all-role failure propagation; no hidden retries; global and nested durations clipped |
| Controller retention, lines127–133 | public cache_status / finite detached CPU guardian | Unknown private helper removed; read-only public16/17 exchange verifies payload100/complete/arena/idle TTL; earliest three caps |

## Remaining boundaries

The private historical worker_control.py was not provided or inspected. Its
internals are not needed by this replacement: passive owned-listener proof
avoids admission service contact; public pinned CACHE_STATUS supplies read-only
idle TTL after engines close; pidfd parent cleanup replaces unload/stop helper
calls. Faithful equivalence to unknown helper internals is not claimed.

Existing private transport argv and namespace/binary/config provenance must be
bound and reviewed before later use. This code deliberately cannot infer private
routes or existing-service ownership. No transport or endpoint was contacted in
this work. The planned harmless-child gate does not validate those dependencies.

Old reported B0 failure remains failed evidence: cancellation during fresh arena
loading, no completed load, B1/G0/G1 not launched, no GPU-worker execution or
remote cache. Missing diagnostics prevent swap attribution. This integration
does not reinterpret that outcome or claim completed-load deadline margin.
