// src/worker/expert_tier.cpp - inferswarm #349: the engine side of the remote CPU-expert mode.
//
// The lifecycle the header states, against #348's client exactly as published: no second protocol, no
// direct socket work, no re-implementation of the worker's kernels.  Setup streams each owned blob
// through ONE blob-sized staging buffer (`Open::fetch` fills it), so the engine host never mirrors the
// expert set in RAM.  One persistent communication thread carries INFER frames while the engine's dispatch
// runs its local CPU jobs; finish() waits before any downstream consumer may commit, and every failure latches dead.
#include "strata/worker/expert_tier.hpp"
#include "strata/worker/cpu_capture.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::worker {
namespace {

uint64_t mono_ms() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

uint64_t mono_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

uint64_t current_thread_id() {
#ifdef _WIN32
    return (uint64_t) GetCurrentThreadId();
#elif defined(__APPLE__)
    uint64_t id = 0;
    pthread_threadid_np(nullptr, &id);
    return id;
#elif defined(__linux__)
    return (uint64_t) syscall(SYS_gettid);
#else
    return (uint64_t) std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
}

void name_current_thread() {
#if defined(__APPLE__)
    pthread_setname_np("strata-cpu-io");
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), "strata-cpu-io");
#endif
}

/// The manifest's model identity: a digest of everything that changes the bytes the worker would compute
/// (name, layout, geometry).  The worker only carries and checks it; the value itself is the engine's.
uint64_t model_digest(const char* name, uint16_t gu, uint16_t d, uint16_t layers, uint16_t experts,
                      uint16_t embd, uint16_t ff) {
    const char* parts[] = {name, "gu", "d", "layers", "experts", "embd", "ff"};
    const uint64_t v[] = {0, gu, d, layers, experts, embd, ff};
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*) p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    for (int i = 0; i < 7; ++i) {
        mix(parts[i], std::strlen(parts[i]));
        if (i > 0) mix(&v[i], sizeof v[i]);
    }
    return h ? h : 1;   // 0 would collide with the manifest default
}

}  // namespace

ExpertWorkerTier::~ExpertWorkerTier() {
    close();
}

bool ExpertWorkerTier::open(const Open& in, std::string& err) {
    if (dead_) { err = "the remote CPU-expert worker tier is dead: " + dead_why_; return false; }
    if (ready_) { err = "the remote CPU-expert worker tier is already open"; return false; }
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        stopping_ = pending_ = busy_ = posted_ = served_ = false;
    }
    if (in.resident && !in.local) {
        err = "persistent weight reuse requires a trusted local manifest";
        return false;
    }
    st_ = {};
    rpc_timing_.reset();
    rpc_timing_.set_enabled(in.rpc_timing);
    if (in.fetch == nullptr && (!in.local || in.local_fallback)) {
        err = "the remote CPU-expert tier needs a blob fetcher";
        return false;
    }
    if (in.n_layers < 1 || in.n_expert < 1 || in.n_embd < 1 || in.n_ff < 1 ||
        in.n_layers > kMaxLayers || in.n_expert > kMaxExperts || in.n_embd > kMaxNEmbd ||
        in.n_ff > kMaxNFf) {
        err = "the remote CPU-expert tier's geometry is outside the worker's compiled bounds";
        return false;
    }
    const size_t stride = ((size_t) in.n_expert + 7) / 8;
    if (in.partition.size() != (size_t) in.n_layers * stride) {
        err = "the remote CPU-expert tier's partition bitmap does not match its geometry";
        return false;
    }
    const auto t0 = mono_ms();
    auto check_setup = [&] {
        const auto elapsed = mono_ms() - t0;
        if ((in.cancelled && in.cancelled()) ||
            (in.local && (in.setup_ms <= 0 || in.deadline_ms <= 0 || elapsed >= (uint64_t)in.setup_ms))) {
            err = "remote CPU setup cancelled or exceeded its total budget";
            return false;
        }
        if (in.local)
            client_.set_deadline_ms((int)std::min<uint64_t>(in.deadline_ms, in.setup_ms - elapsed));
        return true;
    };
    if (!check_setup()) {
        die(err);
        return false;
    }
    const auto control = [&] { return !check_setup(); };
    const bool connected =
        in.local
            ? client_.connect_local(in.host, in.port, std::min(in.deadline_ms, in.setup_ms), control, err)
            : client_.connect_host(in.host, in.port, in.deadline_ms, err);
    if (!connected) {
        err = "connecting the remote CPU-expert worker (" + in.host + ":" + std::to_string(in.port) + "): " + err;
        return false;
    }
    struct ResetControl {
        ExpertWorkerClient &client;
        ~ResetControl() { client.set_setup_cancelled({}); }
    } reset{client_};
    if (in.local || in.cancelled)
        client_.set_setup_cancelled(control);
    SetupManifest m;
    if (in.resident_reuse_only) {
        WorkerCacheInfo cache;
        if (!in.local || !in.resident || !client_.cache_status(cache,err) || !cache.complete || cache.session ||
            cache.idle_remaining_ms!=kResidencyNoExpiry) {
            err="diagnostic requires an idle complete forever-resident cache";client_.close();return false;
        }
    }
    m.nonce = (uint64_t) t0 ^ ((uint64_t) (uintptr_t) this << 32);
    m.model_id = model_digest(in.model_name, in.gu_type, in.d_type, in.n_layers, in.n_expert, in.n_embd, in.n_ff);
    std::strncpy(m.model_name, in.model_name, sizeof m.model_name - 1);
    std::strncpy(m.quant_name, "Q8_0", sizeof m.quant_name - 1);
    m.gu_type = in.gu_type;
    m.d_type = in.d_type;
    m.n_layers = in.n_layers;
    m.n_expert = in.n_expert;
    m.n_embd = in.n_embd;
    m.n_ff = in.n_ff;
    m.partition = in.partition;
    m.partition_id = blob_digest(in.partition).a;
    if (in.local)
        m.model_id = in.local->identity.a;
    if (in.local && (!in.local->matches(m) || in.setup_ms <= 0 || in.deadline_ms <= 0)) {
        err = "local manifest does not match engine geometry, quantization or setup budget";
        client_.close();
        return false;
    }
    const bool setup_ok =
        in.local ? client_.setup_local(m, in.local->identity, in.local_fallback, err, in.resident) : client_.setup(m, err);
    if (!setup_ok) {
        err = std::string("the remote CPU-expert worker refused setup (") + status_name(client_.last_status()) +
              "): " + err;
        client_.close();
        die(err);
        return false;
    }
    const uint32_t blob = client_.blob_bytes();
    if (blob == 0) { err = "the remote CPU-expert worker reported a zero blob size"; client_.close(); die(err); return false; }
    st_.connect_setup_ms = (double)(mono_ms() - t0);
    std::vector<uint8_t> staging; // allocated only for upload; at most one blob
    uint64_t owned = m.owned_count();
    auto failed = [&] {
        client_.set_setup_cancelled({});
        client_.close();
        die(err);
        return false;
    };
    auto upload = [&](int l, int e) {
        const auto t = mono_ms();
        if (!check_setup())
            return false;
        staging.resize(blob);
        if (!in.fetch || !in.fetch(l, e, staging.data(), blob)) {
            err = "expert source could not produce the missing upload blob";
            return false;
        }
        const Digest128 *expected = in.local ? &in.local->digests[(size_t)l * in.n_expert + e] : nullptr;
        if (!check_setup() || !client_.upload(l, e, staging.data(), blob, err, expected))
            return false;
        st_.setup_network_bytes += blob;
        st_.fetch_upload_ms += (double)(mono_ms() - t);
        return true;
    };
    st_.cache_hit = client_.cache_hit();
    if (in.resident_reuse_only && !st_.cache_hit) {
        err="diagnostic requires a resident cache hit";return failed();
    }
    st_.reused_bytes = client_.reused_bytes();
    st_.session = client_.session();
    for (int l = 0; !st_.cache_hit && l < in.n_layers; ++l) {
        std::vector<std::pair<uint16_t, uint16_t>> slots;
        for (int e = 0; e < in.n_expert; ++e)
            if (m.owns(l, e))
                slots.push_back({(uint16_t)l, (uint16_t)e});
        if (!in.local) {
            for (const auto &s : slots)
                if (!upload(s.first, s.second))
                    return failed();
            continue;
        }
        for (size_t i = 0; i < slots.size(); i += kLocalBatchMax) {
            if (!check_setup())
                return failed();
            const auto t = mono_ms();
            std::vector<std::pair<uint16_t, uint16_t>> batch(
                slots.begin() + i, slots.begin() + std::min(slots.size(), i + kLocalBatchMax));
            std::vector<uint8_t> hits;
            if (!client_.load_local(batch, hits, err))
                return failed();
            st_.local_ms += (double)(mono_ms() - t);
            for (size_t j = 0; j < hits.size(); ++j)
                if (!hits[j]) {
                    if (!in.local_fallback) {
                        err = "local expert is absent and upload fallback was not requested";
                        return failed();
                    }
                    if (!upload(batch[j].first, batch[j].second))
                        return failed();
                }
        }
    }
    if (!check_setup())
        return failed();
    const auto rt = mono_ms();
    if (owned == 0) {
        err = "the remote CPU-expert worker owns nothing: every expert is on a GPU tier already";
        client_.close();
        return false;   // a configuration error, not a worker failure: not latched
    }
    if (!client_.mark_ready(err)) {
        err = std::string("the remote CPU-expert worker is not ready (") + status_name(client_.last_status()) +
              "): " + err;
        client_.close();
        die(err);
        return false;
    }
    if (!check_setup())
        return failed();
    st_.ready_ms = (double)(mono_ms() - rt);
    client_.set_deadline_ms(in.deadline_ms);
    const auto &ls = client_.local_stats();
    st_.local_bytes = ls.bytes;
    st_.local_hits = ls.hits;
    st_.local_misses = ls.misses;
    st_.local_read_ns = ls.read_ns;
    st_.local_verify_ns = ls.verify_ns;
    partition_ = std::move(m.partition);
    n_layers_ = in.n_layers;
    n_expert_ = in.n_expert;
    n_embd_ = in.n_embd;
    stride_ = stride;
    model_id_ = m.model_id;
    partition_id_ = m.partition_id;
    sequence_ = 0;
    ready_ = true;
    st_.setup_blobs = owned;
    st_.setup_weight_bytes = owned * (uint64_t) blob;
    st_.setup_ms = (double) (mono_ms() - t0);
    return true;
}

void ExpertWorkerTier::close() {
    {
        std::lock_guard<std::mutex> lock(request_mutex_);
        stopping_ = true;
        request_wake_.notify_all();
    }
    if (th_.joinable()) th_.join();
    client_.close();
    std::lock_guard<std::mutex> lock(request_mutex_);
    ready_ = false;
    posted_ = false;
    pending_ = false;
    busy_ = false;
}

void ExpertWorkerTier::die(const std::string& why) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (!dead_.load(std::memory_order_relaxed)) {
        dead_why_ = why;
        ++st_.failures;
        dead_.store(true, std::memory_order_release);
    }
}

bool ExpertWorkerTier::post(int64_t layer, const float* x, const int32_t* ids, const float* weights,
                            const uint8_t* owned, int n_tok, int k, std::string& err) {
    uint64_t prepare_start = 0;
    posted_ = false;
    if (dead_) { err = "the remote CPU-expert worker tier is dead: " + dead_why_; return false; }
    if (!ready_) { err = "the remote CPU-expert worker tier is not open"; return false; }
    if (n_tok < 1 || n_tok > kMaxT || k < 1 || k > kMaxK) {
        err = "a remote CPU-expert request outside the worker's compiled bounds (T or K)";
        die(err);
        return false;
    }
    std::unique_lock<std::mutex> lock(request_mutex_);
    // A post() before the previous finish() still orders behind that exchange before reusing the slot.
    request_done_.wait(lock, [this] { return !busy_; });
    if (dead_) { err = "the previous remote CPU-expert request failed: " + dead_why_; return false; }
    if (rpc_timing_.enabled()) prepare_start = mono_ns();
    served_ = false;
    req_.session = client_.session();
    req_.model_id = model_id_;
    req_.partition_id = partition_id_;
    req_.sequence = ++sequence_;
    req_.layer = (uint16_t) layer;
    req_.T = (uint8_t) n_tok;
    req_.K = (uint8_t) k;
    req_.mask.assign((size_t) n_tok * k, 0);
    req_.ids.assign((size_t) n_tok * k, 0);
    req_.weights.assign((size_t) n_tok * k, 0.f);
    int marked = 0;
    for (int i = 0; i < n_tok * k; ++i) {
        req_.ids[(size_t) i] = (uint16_t) ids[i];
        req_.weights[(size_t) i] = weights[i];
        if (owned[i] == 0) continue;
        // the engine may mark only entries the frozen partition holds; anything else is its own bug and
        // the worker would refuse the whole frame (`not-owned`)
        if (ids[i] < 0 || !holds(layer, ids[i])) {
            err = "the dispatch marked entry " + std::to_string(i) + " (layer " + std::to_string(layer) +
                  ", expert " + std::to_string(ids[i]) + ") for the remote CPU-expert worker, which the partition does not own";
            lock.unlock();
            die(err);
            return false;
        }
        req_.mask[(size_t) i] = 1;
        ++marked;
    }
    if (marked == 0 && !(capture_ && capture_->active())) return true;
    req_.activations.assign(x, x + (size_t) n_tok * n_embd_);
    if (capture_ && capture_->active()) {
        if (!capture_->begin(req_) || (!marked && !capture_->finish(nullptr))) {
            err=capture_->error();lock.unlock();die(err);return false;
        }
    }
    if (!marked) return true; // diagnostic zero-work event, still no network request
    reply_.partials.clear();
    posted_ = true;
    served_ = false;
    st_.requests += 1;
    st_.entries += (uint64_t) marked;
    posted_ns_ = mono_ns();
    if (rpc_timing_.enabled())
        rpc_timing_.add_phase(RpcTimingPhase::ClientPrepare, posted_ns_ - prepare_start);
    pending_ = busy_ = true;
    if (!th_.joinable()) {
        try {
            th_ = std::thread([this] { run_requests(); });
        } catch (const std::exception& e) {
            pending_ = busy_ = posted_ = false;
            --st_.requests;
            st_.entries -= (uint64_t) marked;
            lock.unlock();
            err = std::string("could not start the remote CPU-expert communication thread: ") + e.what();
            die(err);
            return false;
        }
    }
    request_wake_.notify_one();
    return true;
}

void ExpertWorkerTier::run_requests() {
    name_current_thread();
    std::unique_lock<std::mutex> lock(request_mutex_);
    st_.communication_thread_id = current_thread_id();
    ++st_.communication_thread_starts;
    for (;;) {
        request_wake_.wait(lock, [this] { return pending_ || stopping_; });
        if (stopping_ && !pending_) break;
        pending_ = false;
        // post() waited for the previous exchange and invalidated its result under this mutex.
        // Transfer that capacity to the socket owner before unlocking; finish() waits for !busy_.
        InferReply r;
        r.partials.swap(reply_.partials);
        const uint64_t exchange_start = mono_ns();
        const double queue_ms = (double) (exchange_start - posted_ns_) / 1e6;
        lock.unlock();

        bool ok = false;
        std::string e;
        RpcExchangeTiming measured;
        try {
            ok = client_.infer(req_, r, e, rpc_timing_.enabled() ? &measured : nullptr);
        } catch (const std::exception& ex) {
            e = std::string("exception in communication thread: ") + ex.what();
        } catch (...) {
            e = "unknown exception in communication thread";
        }
        const uint16_t status = client_.last_status();
        const uint64_t exchange_end = mono_ns();
        if (!ok)
            die("the remote CPU-expert worker refused or dropped the request (" +
                std::string(status_name(status)) + "): " + e);

        lock.lock();
        st_.queue_ms += queue_ms;
        st_.exchange_ms += (double) (exchange_end - exchange_start) / 1e6;
        if (rpc_timing_.enabled()) {
            rpc_timing_.add_phase(RpcTimingPhase::ClientQueueWake, exchange_start - posted_ns_);
            rpc_timing_.add_phase(RpcTimingPhase::ClientEncodeSend, measured.encode_send_ns);
            rpc_timing_.add_phase(RpcTimingPhase::ClientReceiveDecode, measured.receive_decode_ns);
            const auto shape = rpc_request_shape(req_.mask, req_.ids);
            rpc_timing_.finish_call(req_.T, shape.entries, shape.distinct_experts,
                                    measured.request_bytes, measured.reply_bytes, ok);
        }
        if (ok) reply_ = std::move(r);
        busy_ = false;
        request_done_.notify_all();
        if (!ok) break;
    }
    pending_ = busy_ = false;
    request_done_.notify_all();
}

bool ExpertWorkerTier::finish(std::string& err) {
    if (!posted_) {
        if (dead_) { err = dead_why_; return false; }
        return true;
    }
    const auto w0 = mono_ns();
    std::unique_lock<std::mutex> lock(request_mutex_);
    request_done_.wait(lock, [this] { return !busy_; });
    const uint64_t now = mono_ns();
    st_.wait_ms += (double) (now - w0) / 1e6;           // what the caller actually waited after its own work
    st_.request_ms += (double) (now - posted_ns_) / 1e6; // full post -> finish wall; overlaps caller work
    if (rpc_timing_.enabled())
        rpc_timing_.add_phase(RpcTimingPhase::ClientFinishWait, now - w0);
    posted_ = false;
    if (dead_) { err = dead_why_; return false; }
    if (reply_.partials.size() != (size_t) req_.T * n_embd_) {
        err = "the remote CPU-expert worker returned a short partial";
        lock.unlock();
        die(err);
        return false;
    }
    if (capture_ && capture_->active() && !capture_->finish(reply_.partials.data())) {
        err=capture_->error();lock.unlock();die(err);return false;
    }
    served_ = true;
    return true;
}

bool ExpertWorkerTier::status(WorkerStatusInfo& out, std::string& err) {
    if (dead_) { err = "the remote CPU-expert worker tier is dead: " + dead_why_; return false; }
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (posted_ || busy_) { err = "a remote CPU-expert request is in flight"; return false; }
    if (!client_.status(out, err)) return false;
    return true;
}

bool ExpertWorkerTier::keepalive_if_idle(std::string& err) {
    std::lock_guard<std::mutex> lock(request_mutex_);
    if (posted_ || busy_) return true;
    WorkerStatusInfo out;
    if (!ready_ || dead_ || !client_.status(out, err)) return false;
    if (out.session != st_.session || out.ready != 1) {err="CPU session lost while idle";return false;}
    return true;
}

void ExpertWorkerTier::report(std::string& out) const {
    std::lock_guard<std::mutex> lock(request_mutex_);
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "%s%llu blobs / %.2f GiB ready in %.1f s; %llu requests, %llu entries; "
                  "aggregate timings (overlap; not a partition): request wall %.1f ms, queue %.1f ms, "
                  "exchange %.1f ms, caller finish wait %.1f ms; "
                  "communication thread name=strata-cpu-io id=%llu starts=%llu",
                  dead_ ? "DEAD after its last failure; " : "", (unsigned long long) st_.setup_blobs,
                  (double) st_.setup_weight_bytes / 1073741824.0, st_.setup_ms / 1e3,
                  (unsigned long long) st_.requests, (unsigned long long) st_.entries,
                  st_.request_ms, st_.queue_ms, st_.exchange_ms, st_.wait_ms,
                  (unsigned long long) st_.communication_thread_id,
                  (unsigned long long) st_.communication_thread_starts);
    out = buf;
    std::snprintf(buf, sizeof buf,
                  "; setup network=%llu B local=%llu B hits=%llu misses=%llu; phases connect/setup=%.1f ms "
                  "local=%.1f ms (read=%.1f verify=%.1f) fetch/upload=%.1f ms ready=%.1f ms",
                  (unsigned long long)st_.setup_network_bytes, (unsigned long long)st_.local_bytes,
                  (unsigned long long)st_.local_hits, (unsigned long long)st_.local_misses,
                  st_.connect_setup_ms, st_.local_ms, st_.local_read_ns / 1e6, st_.local_verify_ns / 1e6,
                  st_.fetch_upload_ms, st_.ready_ms);
    out += buf;
    std::snprintf(buf, sizeof buf, "; cache_hit=%d reused=%llu B session=%llx", st_.cache_hit,
                  (unsigned long long)st_.reused_bytes, (unsigned long long)st_.session);
    out += buf;
    const std::string timing = rpc_timing_.report("client");
    if (!timing.empty()) out += "; " + timing;
}

}  // namespace strata::worker
