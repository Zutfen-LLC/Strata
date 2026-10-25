# Issue #14: opt-in packed activation request contract

## Terminal role05 offline result: PASS; implementation session stopped

The parent reports PASS in the [canonical Hermes receipt](https://zutfen.slack.com/archives/C0C5MK91H3L/p1792971990345319),
which was read for this update. These are supplied execution results, not desktop execution or an
independent audit of private artifacts. Product source remained
`fa39c2f60d0e3b6c3154469b40b569dcd43f8479` (implementation
`976d0538c320bbcfbd486335572f980cf3a2044c`).

All eight named CPU tests passed: `cpu_packed_wire_test`, `cpu_worker_group_test`,
`cpu_worker_group_packed_test`, `cpu_packed_activation_check`, `expert_worker_test`,
`expert_tier_test`, `worker_residency_test`, and `worker_service_residency_test`.
Incremental build took 87.54 s; tests about 52 s combined. Capture pack took 0.16 s;
rows took 9.03 s. Both exact `cmp` gates against retained SPA1/SRA1 passed, including
48 native-row/ordered-partial/capture metric lines. Artifact SHA256 values remain
`531d113d5d110fa4a53fa464ca8e504f913945d8af233113040f2d0dd6af2e0a` and
`0aaeb0762418bc150d7c4ca2e6c168e38b5069fb2df825cb94dd75312a6d727d`.

The synthetic wire fixture sent 20560 raw versus 2808 packed request bytes; replies stayed
20512 bytes. The fixture omits an inactive token row. Its ratio is not a production traffic
ratio. Group packing totals were 76613–129219 ns over 7–8 rows, roughly 11–16 microseconds
per row. These are offline samples, not LAN latency, live TPS or a speedup claim. Deliberately
unsupported T4/T8 requests passed raw fallback. Default remains raw F32.

The earlier process-monitor and Windows archive line-ending failures were handoff failures;
they produced no product or numerical verdict. The direct continuation completed with fresh
`direct-*` evidence and preserved earlier receipts. No model service contact, GEN, reload,
install or firewall change was reported. Role02 wire, backend compilation and live A/B remain
**NOT RUN**. This focused implementation session is complete. Commands below are archived
history, not instructions to start another validation campaign.

For one pragmatic Linux CUDA raw-versus-packed comparison, actual blockers are a sender build
and packed-capable receiver on the intended path, a minimal exact packet/native-row check for
that actual build pair before assigning its explicit qualification ID, and a parent-owned receiver
arrangement that preserves the retained resident state and fits memory. Worker 809314 at the
old runtime cannot exercise packed requests: it would negotiate raw fallback. Do not replace or
reload it automatically. Once those bindings exist, reuse the existing runner/model and unchanged
numerical gates for one fixed raw/packed workload; verify packed requests really occurred, report
fallbacks, bytes, packing cost and end-to-end time, and retain complete output comparisons.

An all-backend matrix, universal/compiler proof, long-decode sweep, new runner, additional
wrapper, or broad production rollout qualification is not a prerequisite for that bounded Linux
CUDA research comparison. Given the under-13.71% optimistic combined-byte ceiling, proceed only
if the capable receiver can be made available cheaply without disturbing residency. Otherwise the
practical stopping point is this offline PASS. The user approved one bounded raw-versus-packed comparison using the existing runner. A fresh
focused parent session owns that execution; this closeout starts no experiment or automatic follow-up.

This stage follows the [sampled proof](CPU-PACKED-ACTIVATION-PROOF-20261025.md).
Raw F32 stays the default. Replies and original entry order stay unchanged. No streaming,
compression, placement, socket policy, resident reload or live model run is included.

Before inference, an opted-in client probes a separate session-bound capability. Both operators
must supply the same nonempty qualification ID for the exact sender/receiver build pair whose
native configuration and sampled byte/row gates they have verified. Matching source, CPU names,
format IDs, or an arbitrary matching ID alone are not evidence. IDs must be retired when either
build/compiler/native configuration changes. No build is qualified automatically.
The probe is a 16-byte STATUS extension: session u64 and qualification u64, little endian.
A capable peer echoes it with a version u32 suffix; an old peer returns its normal validated
59-byte ready status. That old status or an explicit declined capability deliberately selects raw;
malformed replies, I/O failure and cancellation fail the setup. Default clients send no probe.

The new request messages use SEW1 framing and unchanged identity/routing metadata, followed by
version 1, format Q8_0, width 2560, 80 blocks (four little-endian u16 fields), then one 2720-byte
row per token having any owned entry, in token order. Each block is finite FP16 scale bits in
little endian and 32 signed-byte bit patterns. No structs, padding, SPA1 headers or file digests
are transported. The receiver checks exact length, binary masks, version/format/geometry,
finite scales and negotiated session, then uses zero-filled native scratch directly. It never
re-quantizes packed input. Existing session, sequence, ownership, count, size, finite-weight,
readiness and deadline checks still apply. Malformed packed packets fail; they never retry raw.

Only Linux x86-64 native Q8_0 gate/up/down, 2560/640, no rotation/tail, T1/T2 is eligible.
Unsupported geometry/build/qualification or scales select raw before sending. Mixed groups
retain raw for unsupported peers. A group packs the union of active eligible tokens once per
window, shares immutable owned fields through all peer completions, and reports pack calls,
quantized rows, elapsed packing time, packed requests and raw fallback reasons. Single tiers
use the same packer. A scale failure falls back for that window without changing quantization.

Desktop validation is static only. Role05 execution is reported above; no builds, executable
checks or placement measurements were run on this desktop. No speedup is claimed. Requests are about 18.7% of remote bytes; even optimistic
combined savings are under 13.71% and replies do not shrink. Offline handoff follows below.

## Operator interface and accounting

The engine accepts `--remote-cpu-packed-qualification 16_HEX` with a fixed CPU group.
An offline test worker accepts `--packed-qualification 16_HEX`. The ID is 16 lowercase
hex digits, nonzero. Leave both absent for raw. The example IDs in synthetic tests are
fixtures, never deployment qualifications. A capability is cleared on setup/disconnect;
a successful reply binds the current session and version. Probe status bytes do not count
as inference traffic. Request messages are 23 (weighted) and 24 (entry rows); ACK schemas
remain 8 and 21. A receiver accepts raw even after packed negotiation.

Opt-in group/tier request wall time includes sender packing. `pack_ns` measures native
quantization, field serialization and allocation together; it is a subset of that wall
interval, not extra time to add again. Group `pack_calls` counts attempts per logical window,
`quantized_rows` counts actual native quantizer calls (including work before a scale refusal).
Peer packing counters stay zero under a group. Inference timing records actual request/reply
wire bytes; use `--remote-cpu-rpc-timing` for those counters. Group reports the precise last
pack refusal (`unsupported_token_count`, `unsupported_scale`, etc.); peer fallback counters
also identify missing capability. A zero-work peer sends no inference and adds no fallback.
No peers eligible means no sender quantization. One unsupported token scale selects raw for
all eligible peers in that window; no partially packed request is sent.

## First-role 05 offline handoff (desktop NOT RUN; first attempt stopped)

The parent reports that the first attempt passed configuration gates, then stopped during the
build monitor's process-disappearance race. For the existing r1 roots, use the exact resume
below. Do not rerun this fresh-root block or discard the preserved attempt.

Run only this marked block for the first useful result. It contains the full role05 source,
configuration, build, per-test guards and production-codec capture comparison. It does not
invoke the old setup/relay recipes. Role02 and CUDA/HIP/SYCL compilation remain separate
later gates; they do not block this first-role CPU result. Live A/B remains separately scoped.

The bindings are the verified successful role05 proof archive/cache and existing ggml source.
The block never treats the proof archive as a git checkout, changes its cache, installs a
package, launches a model worker or contacts a retained service. It fetches the exact `PIN`
only into the verified clean compression git source, then creates a fresh source archive.
Do not execute this block from the desktop.

Export the full published commit containing this document as `PIN`. The three default fresh
paths are `/home/hermes/strata-pr10-wire05-20261025-r1-src`, the matching `-build`, and `-out`.
They may be explicitly rebound before extraction; all must be absent with existing parents.
An occupied path is a stop, not permission to delete, resume or invent another attempt.
Extract between the exact BEGIN/END markers into a fresh private file and run it with bash.

Configuration comes from `/home/hermes/strata-pr10-packed-06bbd809-build/CMakeCache.txt`,
whose source is the retained proof archive. No global or target PIC correction is added:
role05 already succeeded. Native flags, source-specific flags/defines and compiler identity
must match all five retained targets before building. The only permitted effective flag
difference is the new worker codec's `third_party/ggml` include directory. Warning waivers
are zero. All cache differences are saved explicitly, including source/build/output relocation,
the unchanged external ggml binding, disabled GPU backends and disconnected dependency mode.
No mismatch triggers automatic tuning or continuation.

Each executable test runs alone with an absolute 180-second process watchdog. CTest's default
is also 180 seconds; shorter registered test timeouts remain, and the watchdog caps a longer
registered timeout (such as expert_worker_test's 600 seconds). Capture pack/rows retain their
180-second internal guard and 190-second outer limit.
The wrapper reuses the proof's 10 GiB admission, 8 GiB active floor, 2 GiB address-space/RSS,
zero owned swap and unchanged OOM gates, checks cgroup ancestry, and records per-invocation
logs/receipts. It signals only positive PIDs descended from its own child with matching start
identities. The capture proof keeps its existing internal guard too. No small inherited file-size
limit is applied to synthetic fixtures. Any failure stops the block and retains partial evidence.

```sh
# BEGIN PACKED_WIRE_ROLE05
set -euo pipefail
umask 077
: "${PIN:?export the full published commit containing this block}"
export PIN
export NEW_SRC="${NEW_SRC:-/home/hermes/strata-pr10-wire05-20261025-r1-src}"
export NEW_BUILD="${NEW_BUILD:-/home/hermes/strata-pr10-wire05-20261025-r1-build}"
export RUN_ROOT="${RUN_ROOT:-/home/hermes/strata-pr10-wire05-20261025-r1-out}"
export GIT_OPTIONAL_LOCKS=0
python3 -B - <<'PY'
import hashlib,json,os,pathlib,re,resource,shlex,signal,subprocess,time
P=pathlib.Path; env=os.environ; pin=env['PIN']
assert re.fullmatch('[0-9a-f]{40}',pin), 'full commit PIN required'
git=P('/home/hermes/strata-pr10-compress-74ffe391-src')
base_pin='e5685da112f9e75262176635743b0790586d88c3'
oldsrc=P('/home/hermes/strata-pr10-packed-06bbd809-src') # archive, never use git here
old=P('/home/hermes/strata-pr10-packed-06bbd809-build')
ggml=P('/home/hermes/strata-cand-66235df1-build-cpu/_deps/strata_llamacpp-src')
cap=P('/home/hermes/strata-pr10-micro-bc8e5754-20261025-05/cpu-capture.bin')
manifest=P('/home/hermes/strata-wl-metadata/local.manifest')
model=P('/srv/models/issue300-qwen38-q8-38bb39ee97821de2c9009abb7e93950eec396e66')
retained=P('/home/hermes/strata-pr10-packed-06bbd809-20261025-05')
src,build,out=(P(env[k]) for k in ('NEW_SRC','NEW_BUILD','RUN_ROOT'))
for p in (src,build,out):
    assert p.is_absolute() and p.parent.is_dir() and not p.exists() and not p.is_symlink(), 'fresh absolute paths required'
assert len({src,build,out})==3 and not any(a in b.parents for a in (src,build,out) for b in (src,build,out) if a!=b)
assert model.is_dir() and oldsrc.is_dir()
out.mkdir(mode=0o700)
def save(name,value):
    with (out/name).open('x') as f: json.dump(value,f,indent=2)
def output(argv): return subprocess.check_output([str(x) for x in argv],text=True,timeout=30).strip()
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''): h.update(data)
    return h.hexdigest()
inputs={str(cap):'218bd2e9210ea7cbe641e3e09d7be5f060c5691d630ee6357935eb240f983208',
    str(manifest):'8cc9a1ca01ded6b747628522f3fc896c6ca315583461e9c3826edda8b8f62318',
    str(retained/'activations.spa'):'531d113d5d110fa4a53fa464ca8e504f913945d8af233113040f2d0dd6af2e0a',
    str(retained/'rows-same-build.sra'):'0aaeb0762418bc150d7c4ca2e6c168e38b5069fb2df825cb94dd75312a6d727d'}
assert cap.stat().st_size==37544692
for path,want in inputs.items(): assert digest(P(path))==want, 'retained input hash: '+path
save('input-hashes.json',inputs)
assert output(['git','-C',git,'rev-parse','HEAD'])==base_pin
assert not output(['git','-C',git,'status','--porcelain'])
assert output(['git','-C',ggml,'rev-parse','HEAD'])=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert not output(['git','-C',ggml,'status','--porcelain'])
assert (ggml/'ggml/include/ggml.h').is_file()
# Same admission/active limits as the successful proof; no inherited arithmetic overrides.
overrides=('STRATA_FORCE_ISA STRATA_FORCE_AVX2 STRATA_NO_AVXVNNI STRATA_CPU_YMM STRATA_KQ256 '
    'STRATA_NO_IQ128 STRATA_NO_IQ256 STRATA_NO_IQ512 STRATA_CPU_IQ_ALL_EXACT1 STRATA_IQ_MT_MIN '
    'STRATA_IQ3S_MT1 STRATA_NO_IQ4NL STRATA_IQ_GATHER STRATA_IQ256_GATHER STRATA_IQ_PREFETCH '
    'STRATA_NO_Q8K_AVX1 STRATA_NO_Q8K_AVX2 STRATA_NO_TQ_KERNELS STRATA_NO_NVFP4_512 STRATA_NO_NVFP4_256').split()
assert not any(k in env for k in overrides), 'inherited native override requires a new binding'
assert not any(k in env for k in ('LD_PRELOAD','LD_LIBRARY_PATH','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','CMAKE_ARGS'))
def vm(path,key):
    return int(re.search(r'^'+re.escape(key)+r':?\s+(\d+)',P(path).read_text(),re.M)[1])
oom=vm('/proc/vmstat','oom_kill')
def capacity(need):
    available=vm('/proc/meminfo','MemAvailable')*1024
    assert available>=need, 'host memory admission/active floor'
    def unescape(s):return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m[1],8)),s)
    mounts=[]
    for line in P('/proc/self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1); fields=left.split()
        if right.split()[0]=='cgroup2' and unescape(fields[3])=='/':mounts.append(P(unescape(fields[4])))
    assert len(mounts)==1, 'full cgroup2 ancestry must be visible'
    membership=[s.split(':',2)[2] for s in P('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')]
    assert len(membership)==1 and membership[0].startswith('/') and '..' not in P(membership[0]).parts
    mount=mounts[0];p=mount.joinpath(*P(membership[0]).parts[1:]);rows=[]
    while p!=mount:
        current=int((p/'memory.current').read_text());row={'path':str(p),'current':current}
        for key in ('memory.max','memory.high'):
            value=(p/key).read_text().strip();assert value=='max' or int(value)-current>=need, 'cgroup headroom'
            row[key]=value
        rows.append(row)
        p=p.parent
    assert vm('/proc/vmstat','oom_kill')==oom, 'new host OOM event'
    return {'available':available,'cgroups':rows,'oom_kill':oom}
def process(pid):
    try:
        fields=P(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19],int(fields[1]) # birth tick and PPID
    except (FileNotFoundError,ProcessLookupError):return None
def identity(pid):
    info=process(pid);return info[0] if info else None
def descendants(root,root_stamp):
    found=[];todo=[(root,root_stamp)];seen=set()
    while todo:
        pid,stamp=todo.pop()
        if stamp is None or (pid,stamp) in seen or identity(pid)!=stamp:continue
        seen.add((pid,stamp));found.append((pid,stamp))
        # Include children forked by scheduler threads, pin birth/PPID before traversal.
        try:tasks=list(P(f'/proc/{pid}/task').iterdir())
        except (FileNotFoundError,ProcessLookupError):continue
        for task in tasks:
            try:candidates=[int(x) for x in (task/'children').read_text().split()]
            except (FileNotFoundError,ProcessLookupError):continue
            for candidate in candidates:
                info=process(candidate)
                if info and info[1]==pid and identity(pid)==stamp:todo.append((candidate,info[0]))
    return found
# One bounded invocation at a time, tracking only its own positive PIDs/start identities.
# No process-group signals, service discovery, resident checks or external PID actions.
def run(name,argv,seconds,guarded=False):
    admission=capacity(10<<30);save(name+'-admission.json',admission)
    owned=set();peak=0;start=time.monotonic();failure=None
    def limits():
        if guarded:resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
        soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE)
        resource.setrlimit(resource.RLIMIT_FSIZE,(hard,hard))
    with (out/(name+'.log')).open('xb') as log:
        child=subprocess.Popen([str(x) for x in argv],stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
        root_stamp=identity(child.pid);assert root_stamp is not None, 'cannot pin child identity'
        owned.add((child.pid,root_stamp))
        def collect():
            live=set()
            for pid,stamp in list(owned):live.update(descendants(pid,stamp))
            owned.update(live);return live
        try:
            while True:
                live=collect()
                if time.monotonic()-start>seconds:raise RuntimeError('absolute timeout')
                if vm('/proc/vmstat','oom_kill')!=oom:raise RuntimeError('new host OOM event')
                if guarded:
                    capacity(8<<30);rss=0
                    for pid,stamp in live:
                        if identity(pid)!=stamp:continue
                        try:
                            status=P(f'/proc/{pid}/status').read_text()
                            assert not re.search(r'^VmSwap:\s+[1-9]',status,re.M), 'owned swap'
                            match=re.search(r'^VmRSS:\s+(\d+)',status,re.M)
                            rss+=(int(match[1])*1024 if match else 0)
                        except (FileNotFoundError,ProcessLookupError):pass
                    peak=max(peak,rss);assert rss<=2<<30, 'aggregate owned RSS above 2 GiB'
                rc=child.poll()
                if rc is not None:
                    assert rc==0, f'exit {rc}'
                    assert not any(identity(pid)==stamp and pid!=child.pid for pid,stamp in owned), 'owned child survived command'
                    break
                time.sleep(.05)
        except BaseException as exc:
            failure=str(exc)
            # Refresh children before stopping; never signal a PID whose identity changed.
            collect()
            for sig in (signal.SIGTERM,signal.SIGKILL):
                collect()
                for pid,stamp in sorted(owned,reverse=True):
                    if identity(pid)==stamp:
                        try:os.kill(pid,sig)
                        except ProcessLookupError:pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
                if sig==signal.SIGTERM:time.sleep(.1)
            raise
        finally:
            save(name+'-receipt.json',{'argv':[str(x) for x in argv],'elapsed_s':time.monotonic()-start,
                'returncode':child.poll(),'failure':failure,'peak_owned_rss_bytes':peak,
                'owned_pid_birth_pairs':sorted(owned),'oom_kill_before':oom,'oom_kill_after':vm('/proc/vmstat','oom_kill')})
# Fetch only into the already verified clean git object source; do not checkout there.
run('fetch',['git','-C',git,'fetch','origin',pin],90)
assert output(['git','-C',git,'rev-parse',pin+'^{commit}'])==pin
archive=out/'source.tar'
with archive.open('xb') as f:subprocess.run(['git','-C',str(git),'archive',pin],stdout=f,check=True,timeout=30)
src.mkdir(mode=0o700)
subprocess.run(['tar','-xf',str(archive),'-C',str(src)],check=True,timeout=30)
def cache(path):
    result={}
    for line in path.read_text().splitlines():
        m=re.match(r'([^:#]+):([^=]+)=(.*)',line)
        if m:result[m[1]]=(m[2],m[3])
    return result
ret=cache(old/'CMakeCache.txt')
assert ret['CMAKE_HOME_DIRECTORY'][1]==str(oldsrc)
for key,want in {'CMAKE_BUILD_TYPE':'Release','STRATA_NATIVE_EXPERTS':'ON','STRATA_BUILD_TESTS':'ON',
                 'STRATA_ENABLE_CUDA':'OFF'}.items():assert ret[key][1]==want,(key,ret.get(key))
assert ret['CMAKE_GENERATOR'][1]=='Unix Makefiles', 'flags.make equality recipe is fixed to retained generator'
for k,(kind,value) in ret.items():
    if kind not in ('INTERNAL','STATIC') and re.fullmatch(r'CMAKE_(TOOLCHAIN_FILE|SYSROOT.*|SYSTEM_.*|OSX_.*|STAGING_PREFIX|LINKER_TYPE|PROJECT_.*INCLUDE.*)',k):
        assert not value, 'unsupported retained cross/toolchain/include setting: '+k
keys=[k for k,(kind,value) in ret.items() if kind not in ('INTERNAL','STATIC') and
    (k.startswith(('STRATA_','GGML_')) or
     re.fullmatch(r'CMAKE_(C|CXX)_(COMPILER.*|FLAGS.*|STANDARD.*|EXTENSIONS|LINKER_LAUNCHER|LINK_WHAT_YOU_USE|VISIBILITY_PRESET)',k) or
     re.fullmatch(r'CMAKE_(INTERPROCEDURAL_OPTIMIZATION.*|(EXE|SHARED|MODULE|STATIC)_LINKER_FLAGS.*|AR|RANLIB|LINKER|NM|OBJDUMP|STRIP|POSITION_INDEPENDENT_CODE|BUILD_TYPE|MAKE_PROGRAM|VISIBILITY_INLINES_HIDDEN|LINK_WHAT_YOU_USE|BUILD_RPATH|INSTALL_RPATH|SKIP_RPATH|SKIP_BUILD_RPATH|BUILD_WITH_INSTALL_RPATH|INSTALL_RPATH_USE_LINK_PATH|LINK_SEARCH_START_STATIC|LINK_SEARCH_END_STATIC)',k))]
values={k:(ret[k][0] if ret[k][0] in ('BOOL','STRING','PATH','FILEPATH') else 'STRING',ret[k][1]) for k in keys}
values.update(STRATA_GGML_DIR=('PATH',str(ggml)),STRATA_ENABLE_CUDA=('BOOL','OFF'),
    STRATA_ENABLE_HIP=('BOOL','OFF'),STRATA_ENABLE_SYCL=('BOOL','OFF'),
    FETCHCONTENT_FULLY_DISCONNECTED=('BOOL','ON'),CMAKE_EXPORT_COMPILE_COMMANDS=('BOOL','ON'),
    CMAKE_RUNTIME_OUTPUT_DIRECTORY=('PATH',str(out/'bin')),CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE=('PATH',str(out/'bin')))
with (out/'initial.cmake').open('x') as f:
    for k,(kind,value) in sorted(values.items()):
        assert ']===]' not in value
        f.write(f'set({k} [===[{value}]===] CACHE {kind} "" FORCE)\n')
save('configuration-inputs.json',{'pin':pin,'retained_archive':str(oldsrc),'retained_build':str(old),
    'ggml':str(ggml),'changes':{k:{'retained':ret.get(k),'candidate':v} for k,v in values.items() if ret.get(k)!=v},
    'intentional_effective_flag_delta':{'strata_expert_worker':'add only -I@SRC/third_party/ggml'},
    'warning_waivers':0,'pic_override':'none; exact retained flags required'})
(out/'retained-CMakeCache.txt').write_bytes((old/'CMakeCache.txt').read_bytes())
(out/'uname.txt').write_text(output(['uname','-a'])+'\n');(out/'lscpu.txt').write_text(output(['lscpu'])+'\n')
run('configure',['cmake','-S',src,'-B',build,'-G',ret['CMAKE_GENERATOR'][1],'-C',out/'initial.cmake'],90)
actual=cache(build/'CMakeCache.txt')
for k,v in values.items():assert actual.get(k)==v, 'requested cache value changed: '+k
save('configuration-differences.json',{k:{'retained':ret.get(k),'candidate':actual.get(k)}
    for k in sorted(set(ret)|set(actual)) if ret.get(k)!=actual.get(k)})
(out/'candidate-CMakeCache.txt').write_bytes((build/'CMakeCache.txt').read_bytes())
replacements={str(oldsrc):'@SRC',str(src):'@SRC',str(old):'@BUILD',str(build):'@BUILD',str(ggml):'@GGML'}
for k in ('CMAKE_RUNTIME_OUTPUT_DIRECTORY','CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE'):
    if ret.get(k,('',''))[1]:replacements[ret[k][1]]='@BIN'
replacements[str(out/'bin')]='@BIN'
def normalize(s):
    for a,b in sorted(replacements.items(),key=lambda kv:len(kv[0]),reverse=True):s=s.replace(a,b)
    return s
report={}
for target in ('strata_kernels_cpu','strata_expert_worker','strata_worker_tier','ggml-cpu','ggml-base'):
    pair=[]
    for root in (old,build):
        paths=list(root.rglob(f'CMakeFiles/{target}.dir/flags.make'));assert len(paths)==1,(target,paths)
        pair.append([normalize(line.strip()) for line in paths[0].read_text().splitlines() if '=' in line])
    adjusted=[line.replace(' -I@SRC/third_party/ggml','') if line.startswith(('C_INCLUDES =','CXX_INCLUDES =')) else line for line in pair[1]] if target=='strata_expert_worker' else pair[1]
    report[target]={'retained':pair[0],'candidate':pair[1],'equal_after_declared_include_delta':pair[0]==adjusted}
save('native-flags-comparison.json',report)
assert all(v['equal_after_declared_include_delta'] for v in report.values()), 'native flag mismatch; no warning/PIC waiver or continuation'
for lang in ('C','CXX'):
    identities=[]
    for root in (old,build):
        paths=list((root/'CMakeFiles').glob(f'*/CMake{lang}Compiler.cmake'));assert len(paths)==1
        text=paths[0].read_text();identities.append({k:re.search(r'^set\('+k+r'\s+(.*?)\)$',text,re.M)[1]
            for k in (f'CMAKE_{lang}_COMPILER',f'CMAKE_{lang}_COMPILER_ID',f'CMAKE_{lang}_COMPILER_VERSION')})
    save(lang+'-compiler-identities.json',identities);assert identities[0]==identities[1]
targets=['cpu_packed_wire_test','cpu_worker_group_test','cpu_packed_activation_proof','expert_worker_test',
         'expert_tier_test','worker_residency_test','worker_service_residency_test']
run('build',['cmake','--build',build,'--parallel','2','--verbose','--target',*targets],600)
(out/'compile_commands.json').write_bytes((build/'compile_commands.json').read_bytes())
for target in targets:
    paths=list(build.rglob(f'CMakeFiles/{target}.dir/link.txt'));assert len(paths)==1
    text=paths[0].read_text();(out/(target+'-link.txt')).write_text(text)
    assert not re.search(r'(?:lib)?(?:cuda\w*|cublas\w*|amdhip\w*|sycl\w*|nvrtc\w*|strata_engine)(?:[.\s/-]|$)|libstrata_kernels\.a',text,re.I), 'GPU/engine link dependency'
save('ELF-hashes.json',{str(p):digest(p) for p in (out/'bin').iterdir() if p.is_file()})
# The former proof artifact gate remains exact, including all existing numerical assertions.
tests=['cpu_packed_wire_test','cpu_worker_group_test','cpu_worker_group_packed_test','cpu_packed_activation_check',
       'expert_worker_test','expert_tier_test','worker_residency_test','worker_service_residency_test']
listing=json.loads(output(['ctest','--test-dir',build,'--show-only=json-v1']))
assert all(sum(t['name']==name for t in listing['tests'])==1 for name in tests)
for name in tests:
    run(name,['ctest','--test-dir',build,'-V','--no-tests=error','--timeout','180','-j1','-R','^'+name+'$'],180,True)
    text=(out/(name+'.log')).read_text()
    assert re.search(re.escape(name)+r'.*Passed',text) and not re.search('Skipped|Not Run|FAILED',text)
proof=out/'bin/cpu_packed_activation_proof'
run('pack',[proof,'--pack',cap,out/'activations.spa'],190,True)
assert re.search(r'^PACKED_DONE ok=1 records=16 bytes=92884 weights_read=0$',(out/'pack.log').read_text(),re.M)
run('cmp-pack',['cmp',out/'activations.spa',retained/'activations.spa'],30,True)
run('rows',[proof,'--rows',cap,manifest,model,retained/'activations.spa',out/'rows-same-build.sra'],190,True)
lines=(out/'rows.log').read_text().splitlines();metrics=[x for x in lines if x.startswith('PACKED_ROWS ')];done=[x for x in lines if x.startswith('PACKED_ROWS_DONE ')]
assert len(done)==1 and 'records=16 ' in done[0] and len(metrics)==48
fields=[dict(word.split('=',1) for word in line.split()[1:]) for line in metrics]
assert len({(v['record'],v['peer']) for v in fields})==48
assert all(v['row_bits']=='1' and v['partial_bits']=='1' and v['capture_bits']==str(int(v['peer']=='-1')) for v in fields)
assert done[0]=='PACKED_ROWS_DONE ok=1 records=16 packed_reference_bits=1 same_build_rows_bits=1 retained_capture_bits=1 cross_build_rows=REQUIRES_CMP sockets=0 worker_loads=0 model_GEN=0'
run('cmp-rows',['cmp',out/'rows-same-build.sra',retained/'rows-same-build.sra'],30,True)
assert digest(out/'activations.spa')==inputs[str(retained/'activations.spa')]
assert digest(out/'rows-same-build.sra')==inputs[str(retained/'rows-same-build.sra')]
assert output(['git','-C',git,'rev-parse','HEAD'])==base_pin and not output(['git','-C',git,'status','--porcelain'])
assert not output(['git','-C',ggml,'status','--porcelain'])
save('terminal-result.json',{'pin':pin,'role':'05','tests':tests,'production_codec_retained_pack_cmp':True,
    'production_codec_retained_rows_cmp':True,'artifact_hashes':{n:digest(out/n) for n in ('activations.spa','rows-same-build.sra')},
    'native_flags_equal_after_declared_include_delta':True,'warning_waivers':0,
    'role02':'NOT RUN','backend_compilation':'NOT RUN','live_AB':'NOT RUN'})
print('PACKED_WIRE_ROLE05 PASS pin='+pin+' evidence='+str(out),flush=True)
PY
# END PACKED_WIRE_ROLE05
```

A terminal role05 PASS establishes this build's focused CPU wire/default regressions and exact
production-codec agreement with its retained SPA1/SRA1 artifacts. It does not establish role02
wire compatibility, universal native equivalence, long decode, LAN latency or TPS. Keep the
`PACKED_PLACEMENT` and `PACKED_WIRE_COST` lines as offline cost samples with their workload
and build provenance; packing is included in opt-in wall time. No speedup follows from byte
reduction alone. Return `terminal-result.json`, native/configuration comparisons, per-test logs,
resource receipts and artifact hashes to the parent. Do not proceed to another role or live A/B.

Desktop checks remain static: whitespace, block syntax and focused review. No block, compiler,
test executable or fleet command has been run here.

## Resume role05 after the compiler-child disappearance race

Parent-supplied result: fetch, archive, configuration, all five native-flag comparisons and both
compiler-identity gates passed at `fa39c2f60d0e3b6c3154469b40b569dcd43f8479`. The build monitor
then raised `ProcessLookupError` (ESRCH) while reading an exited compiler child's `/proc/PID/stat`.
The failure handler terminated the build. No test, pack or rows gate ran. This is a handoff-monitor
failure, not a product-code failure or numerical outcome. Preserve all original logs and receipts.

The correction catches only ENOENT/ESRCH for transient process stat, task-directory, children and
status reads. Permission, I/O, parse and other errors still fail. Birth identity, parent relationship,
memory, cancellation and timeout checks are unchanged. The resume below loads those corrected
helpers and the identical build/test/artifact tail from the published block above. Only log names
get `resume1-` prefixes; numerical assertions and target/test lists are unchanged. No new monitor,
source extraction, reconfiguration or repeat of passed stages is involved.

Export the full correction commit as `RESUME_PIN`, then run this exact block. It uses the existing
`r1-src`, `r1-build`, and `r1-out` paths. It fetches only the correction document through the verified
git object source; the configured product source stays at the original archive pin. A prior resume,
missing passed-stage receipt, cache mismatch, or surviving original owned PID stops before building.
The original configuration/failure evidence is hashed into fresh resume provenance, never replaced.

```sh
# BEGIN PACKED_WIRE_ROLE05_RESUME
set -euo pipefail
umask 077
: "${RESUME_PIN:?export the full published correction commit}"
export RESUME_PIN
python3 -B - <<'PY'
import ast,hashlib,json,os,pathlib,re,subprocess
P=pathlib.Path; correction=os.environ['RESUME_PIN']
assert re.fullmatch('[0-9a-f]{40}',correction), 'full correction PIN required'
git=P('/home/hermes/strata-pr10-compress-74ffe391-src')
assert subprocess.check_output(['git','-C',str(git),'rev-parse','HEAD'],text=True).strip()=='e5685da112f9e75262176635743b0790586d88c3'
assert not subprocess.check_output(['git','-C',str(git),'status','--porcelain'],text=True).strip()
subprocess.run(['git','-C',str(git),'fetch','origin',correction],check=True,timeout=90)
assert subprocess.check_output(['git','-C',str(git),'rev-parse',correction+'^{commit}'],text=True).strip()==correction
text=subprocess.check_output(['git','-C',str(git),'show',correction+':docs/validation/CPU-PACKED-ACTIVATION-WIRE-20261025.md'],text=True,timeout=30)
# Exact anchored markers distinguish the original block from this resume's own source text.
block=re.search(r'^# BEGIN PACKED_WIRE_ROLE05\n(.*?)^# END PACKED_WIRE_ROLE05$',text,re.M|re.S)
assert block
body=block[1].split("python3 -B - <<'PY'\n",1)[1].rsplit('\nPY\n',1)[0]
ast.parse(body)
# Reuse bindings, pure helpers and corrected monitor; omit fresh-root/fetch/configure stages.
prefix=body[:body.index('for p in (src,build,out):')]
helpers=body[body.index('def save(name,value):'):body.index('assert cap.stat().st_size==37544692')]
guards=body[body.index('# Same admission/active limits'):body.index('# Fetch only into')]
tail=body[body.index("targets=['cpu_packed_wire_test'"):]
assert guards.count('except (FileNotFoundError,ProcessLookupError):')==4
assert 'except FileNotFoundError:' not in guards
assert guards.count('except ProcessLookupError:')==1 # existing positive-PID signal race only
# Name-only changes. Preserve the exact build, test, pack, rows and cmp invocations/gates.
assert tail.count("text=(out/(name+'.log')).read_text()") == 1
assert tail.count("(out/'pack.log').read_text()") == 1
assert tail.count("(out/'rows.log').read_text()") == 1
tail=tail.replace("text=(out/(name+'.log')).read_text()","text=(out/('resume1-'+name+'.log')).read_text()")
tail=tail.replace("(out/'pack.log').read_text()","(out/'resume1-pack.log').read_text()")
tail=tail.replace("(out/'rows.log').read_text()","(out/'resume1-rows.log').read_text()")
os.environ.update(PIN='fa39c2f60d0e3b6c3154469b40b569dcd43f8479',
    NEW_SRC='/home/hermes/strata-pr10-wire05-20261025-r1-src',
    NEW_BUILD='/home/hermes/strata-pr10-wire05-20261025-r1-build',
    RUN_ROOT='/home/hermes/strata-pr10-wire05-20261025-r1-out')
checks=r'''
assert all(p.is_dir() for p in (src,build,out)), 'existing configured r1 roots required'
prior=['configuration-inputs.json','configuration-differences.json','native-flags-comparison.json',
       'C-compiler-identities.json','CXX-compiler-identities.json','candidate-CMakeCache.txt',
       'build.log','build-admission.json','build-receipt.json','source.tar']
assert all((out/n).is_file() for n in prior), 'preserved passed-stage/failure receipts required'
assert list(out.glob('resume1-*'))==[out/'resume1-command.py'], 'resume1 is single-use; preserve any prior attempt'
assert (build/'CMakeCache.txt').read_bytes()==(out/'candidate-CMakeCache.txt').read_bytes(), 'configured cache changed'
configuration=json.loads((out/'configuration-inputs.json').read_text())
assert configuration['pin']==pin and configuration['warning_waivers']==0
flags=json.loads((out/'native-flags-comparison.json').read_text())
assert set(flags)=={'strata_kernels_cpu','strata_expert_worker','strata_worker_tier','ggml-cpu','ggml-base'}
assert all(v['equal_after_declared_include_delta'] for v in flags.values())
for language in ('C','CXX'):
    pair=json.loads((out/(language+'-compiler-identities.json')).read_text())
    assert len(pair)==2 and pair[0]==pair[1]
# Read-only binding checks against the passed receipts, without reconfiguration/compilers.
for path,want in inputs.items():assert digest(P(path))==want, 'original input binding changed: '+path
assert output(['git','-C',ggml,'rev-parse','HEAD'])=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert not output(['git','-C',ggml,'status','--porcelain'])
import tarfile
objects={}
for item in output(['git','-C',git,'ls-tree','-r','-z',pin]).split('\0'):
    if item:
        meta,name=item.split('\t',1);mode,kind,oid=meta.split();objects[name]=(mode,kind,oid)
archive_paths=set()
with tarfile.open(out/'source.tar') as archive:
    assert archive.pax_headers.get('comment')==pin, 'retained archive pin changed'
    for member in archive.getmembers():
        if member.isdir():continue
        assert member.isfile() or member.issym(), 'unexpected archive member'
        relative=P(member.name);assert not relative.is_absolute() and '..' not in relative.parts
        path=src/relative;archive_paths.add(member.name)
        data=member.linkname.encode() if member.issym() else archive.extractfile(member).read()
        mode,kind,oid=objects[member.name];assert kind=='blob'
        assert hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()==oid, 'archive differs from pinned git blob'
        if member.issym():assert mode=='120000' and path.is_symlink() and os.readlink(path)==member.linkname
        else:
            assert mode in ('100644','100755') and not path.is_symlink()
            assert path.resolve().is_relative_to(src) and path.read_bytes()==data, 'configured source changed: '+member.name
assert {str(p.relative_to(src)) for p in src.rglob('*') if p.is_file() or p.is_symlink()}==archive_paths, 'source extras/missing files'
replacements={str(oldsrc):'@SRC',str(src):'@SRC',str(old):'@BUILD',str(build):'@BUILD',str(ggml):'@GGML'}
for target in flags:
    paths=list(build.rglob(f'CMakeFiles/{target}.dir/flags.make'));assert len(paths)==1
    lines=[line.strip() for line in paths[0].read_text().splitlines() if '=' in line]
    for before,after in sorted(replacements.items(),key=lambda kv:len(kv[0]),reverse=True):
        lines=[line.replace(before,after) for line in lines]
    assert lines==flags[target]['candidate'], 'generated flags changed: '+target
for language in ('C','CXX'):
    paths=list((build/'CMakeFiles').glob(f'*/CMake{language}Compiler.cmake'));assert len(paths)==1
    text=paths[0].read_text()
    fields={k:re.search(r'^set\('+k+r'\s+(.*?)\)$',text,re.M)[1]
        for k in (f'CMAKE_{language}_COMPILER',f'CMAKE_{language}_COMPILER_ID',f'CMAKE_{language}_COMPILER_VERSION')}
    assert fields==json.loads((out/(language+'-compiler-identities.json')).read_text())[1], 'generated compiler identity changed'
# No later stage ran. Every output this unchanged tail may create must still be absent.
for name in ('compile_commands.json','ELF-hashes.json','terminal-result.json','activations.spa','rows-same-build.sra',
             'cpu_packed_wire_test.log','cpu_worker_group_test.log','cpu_worker_group_packed_test.log',
             'cpu_packed_activation_check.log','expert_worker_test.log','expert_tier_test.log',
             'worker_residency_test.log','worker_service_residency_test.log','pack.log','rows.log'):
    assert not (out/name).exists(), 'later-stage output already present: '+name
for target in ('cpu_packed_wire_test','cpu_worker_group_test','cpu_packed_activation_proof','expert_worker_test',
               'expert_tier_test','worker_residency_test','worker_service_residency_test'):
    assert not (out/(target+'-link.txt')).exists(), 'later-stage link receipt already present'
original=json.loads((out/'build-receipt.json').read_text())
assert original['failure'] is not None and original['returncode'] is not None, 'interrupted build cleanup unconfirmed'
assert original['returncode']!=0, 'expected interrupted build receipt'
assert not any(identity(int(pid))==stamp for pid,stamp in original['owned_pid_birth_pairs']), 'original owned process still alive; stop without signalling it'
save('resume1-provenance.json',{'product_pin':pin,'correction_pin':os.environ['RESUME_PIN'],
    'preserved_evidence_sha256':{name:digest(out/name) for name in prior},
    'existing_source':str(src),'existing_build':str(build),'reconfigure':False,'repeat_passed_stages':False})
_original_run=run
def run(name,argv,seconds,guarded=False):
    return _original_run('resume1-'+name,argv,seconds,guarded)
'''
program=prefix+helpers+guards+checks+tail
ast.parse(program)
resume=P(os.environ['RUN_ROOT'])/'resume1-command.py'
with resume.open('x') as f:f.write(program)
# Same monitor and unchanged tail; begin with the interrupted incremental build.
exec(compile(program,str(resume),'exec'),{'__name__':'__main__'})
PY
# END PACKED_WIRE_ROLE05_RESUME
```

The resume is **NOT RUN** on this desktop. Static syntax/extraction and focused review are separate
from Hermes execution. Report the old monitor failure and the resume outcome separately. Retain
`resume1-command.py`, `resume1-provenance.json`, all `resume1-*.log`/receipts and unchanged artifact
gates. No automatic second resume, role02, backend campaign or live A/B follows.
