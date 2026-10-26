#include "strata/worker/gpu_worker.hpp"
#include "wire_sock.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

namespace strata::worker {
namespace {
uint64_t now_ms() { return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
uint64_t now_ns() { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }
}
GpuWorkerService::GpuWorkerService(GpuWorkerOptions o, std::unique_ptr<GpuExecutor> e)
    : options_(std::move(o)), executor_(std::move(e)) {}
GpuWorkerService::~GpuWorkerService() {
    reset();
    sock_fd fd = (sock_fd)listener_; sock::close_fd(fd);
}
bool GpuWorkerService::listen(uint16_t& port, std::string& err) {
    if (!sock::listener_policy(options_.bind_ip, options_.allow_peer_ips, allowed_peers_, err)) return false;
    in_addr address{};
    ::inet_pton(AF_INET, options_.bind_ip.c_str(), &address); // parsed by listener_policy
    const uint32_t ip = ntohl(address.s_addr);
    if ((ip >> 24) != 127 && (ip >> 24) != 10 && (ip >> 20) != 0xac1 &&
        (ip >> 16) != 0xc0a8) {
        err = "CUDA bind IP must be loopback or one specific RFC1918 private interface address";
        return false;
    }
    if (!executor_ || options_.deadline_ms < 1 || options_.setup_ms < 1 ||
        (options_.vulkan ? options_.runtime_reserve != 0 : options_.runtime_reserve < (64ull << 20)) ||
        !validate_gpu_plan(options_.plan, err)) {
        if (err.empty()) err = options_.vulkan ? "invalid Vulkan worker options or nonzero reserve" :
            "invalid CUDA worker options or reserve (minimum 64 MiB)";
        return false;
    }
    if(options_.vulkan && (options_.plan.n_embd != 2560 || options_.plan.n_ff != 640)) {
        err="Vulkan worker requires Q8_0 H2560/FF640"; return false;
    }
    if (!cache_.open(options_.cache_root, options_.cache_budget, err)) return false;
    sock_fd fd = kBadFd;
    const bool listening = options_.bind_ip == "127.0.0.1" && allowed_peers_.empty()
        ? sock::listen_loopback(port, fd, err)
        : sock::listen_ipv4(options_.bind_ip, port, fd, err);
    if (!listening) return false;
    listener_ = (uintptr_t)fd;
    return true;
}
void GpuWorkerService::reset() {
    if (executor_) executor_->release(counters_);
    negotiated_ = setup_ = ready_ = rows_ = false;
    session_ = sequence_ = 0; missing_.clear();
    std::vector<uint8_t>().swap(upload_); uploading_ = SIZE_MAX;
}
void GpuWorkerService::run() {
    while (!stop_.load()) {
        sock_fd peer = kBadFd;
        const bool accepted = allowed_peers_.empty()
            ? sock::accept_one((sock_fd)listener_, peer, 100)
            : sock::accept_allowed((sock_fd)listener_, peer, 100, false, allowed_peers_);
        if (!accepted) continue;
        reset(); serve((uintptr_t)peer); sock::close_fd(peer); reset();
    }
    reset();
    sock_fd fd = (sock_fd)listener_; sock::close_fd(fd); listener_ = ~(uintptr_t)0;
    std::fprintf(stderr,"%s-worker stopped assigned=%llu cached=%llu transferred=%llu persisted=%llu retained_disk=%llu verified=%llu reused=%llu staging_peak=%llu "
      "gpu_allocated=%llu gpu_peak=%llu gpu_weights=%llu gpu_scratch=%llu requests=%llu entries=%llu failures=%llu service_ns=%llu\n",
      options_.vulkan?"vulkan":"cuda",(unsigned long long)counters_.assigned_bytes,(unsigned long long)counters_.cached_bytes,
      (unsigned long long)counters_.transferred_bytes,(unsigned long long)counters_.persisted_bytes,
      (unsigned long long)counters_.retained_disk_bytes,(unsigned long long)counters_.verified_bytes,
      (unsigned long long)counters_.reused_bytes,(unsigned long long)counters_.host_staging_peak,
      (unsigned long long)counters_.gpu_allocated_bytes,(unsigned long long)counters_.gpu_peak_bytes,
      (unsigned long long)counters_.gpu_weight_bytes,(unsigned long long)counters_.gpu_scratch_bytes,
      (unsigned long long)counters_.requests,(unsigned long long)counters_.entries,
      (unsigned long long)counters_.failures,(unsigned long long)counters_.service_ns);
}
void GpuWorkerService::serve(uintptr_t fd) {
    const auto cancel = [&] { return stop_.load() || (setup_ && !ready_ &&
        now_ms() - setup_started_ >= (uint64_t)options_.setup_ms); };
    while (!cancel()) {
        WireFrame in, out; std::string err;
        if (!gpu_codec::receive(fd,in,kGpuFrameMax,options_.deadline_ms,cancel,err)) {
            ++counters_.failures; return;
        }
        if (in.status != ST_OK) { ++counters_.failures; return; }
        const bool ok = dispatch(in,out,err);
        if (!ok) {
            ++counters_.failures; out.status = out.status == ST_OK ? ST_BAD_REQUEST : out.status;
            std::fprintf(stderr,"%s-worker refused msg=%u status=%s model=%llx placement=%llx:%llx reason=%s\n",
                options_.vulkan?"vulkan":"cuda",in.msg,status_name(out.status),(unsigned long long)options_.plan.model_id,
                (unsigned long long)options_.plan.placement_id.a,(unsigned long long)options_.plan.placement_id.b,err.c_str());
        }
        if (!gpu_codec::send(fd,out.msg,out.status,out.payload,options_.deadline_ms,cancel,err)) {
            ++counters_.failures; return;
        }
        // A refused operation ends this connection/session. No partial result or retry is possible.
        if (!ok) return;
        if (in.msg == GPU_STOP) { stop_.store(true); return; }
    }
}
bool GpuWorkerService::dispatch(const WireFrame& in, WireFrame& out, std::string& err) {
    using namespace gpu_codec;
    auto cache_stats = [&] {
        const auto& s=cache_.stats();
        counters_.cached_bytes=s.cached_bytes; counters_.persisted_bytes=s.persisted_bytes;
        counters_.retained_disk_bytes=s.retained_disk_bytes; counters_.verified_bytes=s.verified_bytes;
        counters_.reused_bytes=s.reused_bytes;
        counters_.host_staging_peak=std::max(counters_.host_staging_peak,s.host_staging_peak);
    };
    out.msg = in.msg + 1; out.status = ST_OK;
    auto refuse = [&](uint16_t status, const char* why) { out.status = status; err = why; return false; };
    Cursor c{in.payload};
    if (in.msg == GPU_HELLO) {
        const auto version = c.get(4), capability = c.get(4), numerics = c.get(4);
        const auto expected_backend=options_.vulkan ? kGpuCapabilityVulkan : kGpuCapabilityCuda;
        const auto expected_numerics=options_.vulkan ? kGpuNumericsVulkanHybridQ8 : kGpuNumericsQ8_1;
        const bool valid_capability=options_.vulkan ? capability == (expected_backend|kGpuCapabilityEntryRows) :
            (capability == expected_backend || capability == (expected_backend|kGpuCapabilityEntryRows));
        if (negotiated_ || !c.done() || version != kGpuWireVersion || !valid_capability ||
            numerics != expected_numerics) return refuse(ST_PROTOCOL,"unsupported GPU capability/numerics");
        rows_ = (capability & kGpuCapabilityEntryRows)!=0;
        if(rows_ && !executor_->supports_rows()) return refuse(ST_PROTOCOL,"GPU entry rows unsupported");
        put(out.payload,kGpuWireVersion,4); put(out.payload,capability,4);
        put(out.payload,expected_numerics,4); put(out.payload,kUploadChunkMax,4);
        negotiated_ = true; return true;
    }
    if (!negotiated_) return refuse(ST_WRONG_STATE,"CUDA HELLO required");
    if (in.msg == GPU_SETUP) {
        if (setup_ || in.payload.size() <= 8) return refuse(ST_WRONG_STATE,"frozen setup already exists");
        setup_ = true; setup_started_ = now_ms();
        const auto nonce = c.get();
        GpuPlan proposed;
        if (!nonce || !decode_gpu_plan(in.payload.data()+8,in.payload.size()-8,proposed,err))
            return refuse(ST_BAD_REQUEST,"invalid CUDA plan");
        std::vector<uint8_t> trusted, request;
        if (!encode_gpu_plan(options_.plan,trusted,err) || !encode_gpu_plan(proposed,request,err) || trusted != request)
            return refuse(ST_STALE_SESSION,"operator placement/model/layout mismatch");
        if (!cache_.prepare(options_.plan,missing_,err)) {out.status=ST_OVER_BUDGET;return false;}
        cache_stats();
        if(stop_.load() || now_ms()-setup_started_ >= (uint64_t)options_.setup_ms)
            return refuse(ST_DEADLINE,"CUDA cache verification completed after setup budget or stop");
        // Nonce plus process monotonic timestamp/serial gives new bindings on reconnect; no session reuse.
        session_ = nonce ^ now_ns() ^ (++session_serial_ * 0x9e3779b97f4a7c15ull);
        if (!session_) session_ = ++session_serial_;
        counters_.assigned_bytes = options_.plan.assigned_bytes();
        counters_.cached_bytes = cache_.stats().cached_bytes;
        counters_.host_staging_peak = std::max(counters_.host_staging_peak,cache_.stats().host_staging_peak);
        out.payload = binding(session_,options_.plan);
        put(out.payload,missing_.size(),4); out.payload.insert(out.payload.end(),missing_.begin(),missing_.end());
        return true;
    }
    if (in.msg == GPU_UPLOAD) {
        if (!setup_ || ready_ || !binding(c,session_,options_.plan)) return refuse(ST_STALE_SESSION,"upload binding/state");
        const auto slot = c.get(4), offset = c.get(4), bytes = c.get(4);
        if (!c.ok || slot >= missing_.size() || !missing_[(size_t)slot] || bytes < 1 || bytes > kUploadChunkMax ||
            bytes != in.payload.size()-c.i) return refuse(ST_BAD_REQUEST,"upload index/length");
        const auto& asset = options_.plan.assets[(size_t)slot];
        if (uploading_ == SIZE_MAX) {
            if (offset) return refuse(ST_INCOMPLETE,"upload starts after zero");
            uploading_ = (size_t)slot; upload_.clear(); upload_.reserve((size_t)asset.bytes);
        }
        if (uploading_ != slot || offset != upload_.size() || offset > asset.bytes || bytes > asset.bytes-offset)
            return refuse(ST_BAD_REQUEST,"unordered or duplicate upload");
        upload_.insert(upload_.end(),in.payload.begin()+c.i,in.payload.end());
        counters_.transferred_bytes += bytes;
        // One blob plus one frame payload, and the cache's own verification buffer, conservatively.
        counters_.host_staging_peak = std::max(counters_.host_staging_peak,
            (uint64_t)upload_.capacity()+in.payload.capacity()+kHeaderBytes);
        if (upload_.size() == asset.bytes) {
            if (!cache_.publish(asset,upload_.data(),upload_.size(),err)) {out.status=ST_DIGEST_MISMATCH;return false;}
            cache_stats();
            missing_[(size_t)slot] = 0;
            std::vector<uint8_t>().swap(upload_); uploading_ = SIZE_MAX;
            counters_.cached_bytes = cache_.stats().cached_bytes;
        }
        out.payload = binding(session_,options_.plan); put(out.payload,slot,4); put(out.payload,offset+bytes,4);
        return true;
    }
    if (in.msg == GPU_READY) {
        if (!setup_ || ready_ || !binding(c,session_,options_.plan) || !c.done())
            return refuse(ST_STALE_SESSION,"READY binding/state");
        if (uploading_ != SIZE_MAX || std::find(missing_.begin(),missing_.end(),1) != missing_.end())
            return refuse(ST_INCOMPLETE,"assigned assets missing");
        if (!executor_->prepare(options_.plan,options_.runtime_reserve,cache_,counters_,err)) {
            executor_->release(counters_); out.status=ST_OVER_BUDGET; return false;
        }
        if(stop_.load() || now_ms()-setup_started_ >= (uint64_t)options_.setup_ms) {
            executor_->release(counters_); return refuse(ST_DEADLINE,"CUDA load completed after setup budget or stop");
        }
        // prepare synchronizes all copies and verifies allocations before READY is observable.
        cache_stats(); ready_ = true; out.payload = binding(session_,options_.plan); return true;
    }
    if (in.msg == GPU_INFER || in.msg == GPU_INFER_ROWS) {
        if ((in.msg == GPU_INFER_ROWS) != rows_) return refuse(ST_PROTOCOL,"CUDA reply mode was not negotiated");
        ++counters_.requests;
        if (!ready_) return refuse(ST_NOT_READY,"CUDA weights are not READY");
        InferRequest r;
        if (!decode_request(options_.plan,in.payload,r) || r.session != session_ || r.sequence != sequence_+1)
            return refuse(ST_STALE_SESSION,"inference binding/sequence/shape");
        if(options_.vulkan && r.T != 1) return refuse(ST_BAD_REQUEST,"Vulkan worker requires T1");
        uint64_t entries = 0;
        for (size_t i=0;i<r.mask.size();++i) {
            if (r.mask[i]>1 || (r.mask[i] && !options_.plan.owns(r.layer,r.ids[i])))
                return refuse(ST_NOT_OWNED,"inference requested unassigned expert");
            if (!std::isfinite(r.weights[i])) return refuse(ST_NONFINITE,"weight is nonfinite");
            entries += r.mask[i];
        }
        for (float v : r.activations) if (!std::isfinite(v)) return refuse(ST_NONFINITE,"activation is nonfinite");
        if (!entries) return refuse(ST_BAD_REQUEST,"empty CUDA request");
        if(rows_) {
            EntryRowsReply reply; reply.session=session_; reply.sequence=r.sequence;
            const auto begin=now_ns();
            if(!executor_->infer_rows(options_.plan,r,reply,err)) {out.status=ST_INTERNAL;return false;}
            counters_.service_ns+=now_ns()-begin;
            if(!encode_rows_reply(options_.plan,blob_digest(in.payload),r,reply,out.payload))
                return refuse(ST_INTERNAL,"CUDA executor row shape/indices/association/finiteness");
            sequence_=r.sequence; counters_.entries+=entries; return true;
        }
        InferReply reply; reply.session=session_; reply.sequence=r.sequence;
        const auto begin=now_ns();
        if (!executor_->infer(options_.plan,r,reply,err)) {out.status=ST_INTERNAL;return false;}
        counters_.service_ns += now_ns()-begin;
        if (reply.session != session_ || reply.sequence != r.sequence || reply.partials.size() != (size_t)r.T*options_.plan.n_embd)
            return refuse(ST_INTERNAL,"CUDA executor result shape/association");
        for(float v:reply.partials) if(!std::isfinite(v)) return refuse(ST_NONFINITE,"CUDA output is nonfinite");
        sequence_=r.sequence; counters_.entries+=entries;
        out.payload=encode_reply(options_.plan,blob_digest(in.payload),reply); return true;
    }
    if (in.msg == GPU_STATUS) {
        if (!setup_ || !binding(c,session_,options_.plan) || !c.done()) return refuse(ST_STALE_SESSION,"status binding");
        out.payload=binding(session_,options_.plan); put(out.payload,ready_?1:0,4);
        auto values=gpu_codec::counters(counters_); out.payload.insert(out.payload.end(),values.begin(),values.end()); return true;
    }
    if (in.msg == GPU_STOP) {
        if (!in.payload.empty()) return refuse(ST_BAD_REQUEST,"stop payload");
        reset(); return true;
    }
    return refuse(ST_PROTOCOL,"unsupported CUDA operation");
}
} // namespace strata::worker
