#include "strata/worker/gpu_client.hpp"
#include "wire_sock.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace strata::worker {
namespace {
uint64_t now_ms(){return(uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
}
GpuWorkerTier::~GpuWorkerTier(){close();}
void GpuWorkerTier::close(){sock_fd fd=(sock_fd)fd_;sock::close_fd(fd);fd_=~(uintptr_t)0;ready_=false;posted_=served_=false;reply_.partials.clear();rows_reply_={};cancelled_={};setup_end_=0;}
bool GpuWorkerTier::fail(const std::string& why,std::string& err){
    err=why;why_=why;dead_=true;++stats_.failures;close();return false;
}
bool GpuWorkerTier::exchange(uint16_t msg,const std::vector<uint8_t>& p,uint16_t want,WireFrame& reply,std::string& err){
    const auto cancel=[&]{return(cancelled_&&cancelled_())||(setup_end_&&now_ms()>=setup_end_);};
    int budget=deadline_ms_;
    if(setup_end_){const auto now=now_ms();if(now>=setup_end_)return fail("CUDA setup exceeded total deadline",err);
        budget=(int)std::min<uint64_t>((uint64_t)budget,setup_end_-now);}
    if(fd_==~(uintptr_t)0 || !gpu_codec::send(fd_,msg,ST_OK,p,budget,cancel,err))
        return fail("CUDA exchange failed: "+err,err);
    int response_wait=0;
    if(setup_end_){
        const auto now=now_ms();if(now>=setup_end_)return fail("CUDA setup exceeded total deadline",err);
        const auto remaining=setup_end_-now;
        budget=(int)std::min<uint64_t>((uint64_t)deadline_ms_,remaining);
        if(msg==GPU_SETUP || msg==GPU_READY)response_wait=(int)std::min<uint64_t>(INT32_MAX,remaining);
    }
    if(!gpu_codec::receive(fd_,reply,kGpuFrameMax,budget,cancel,err,response_wait))
        return fail("CUDA exchange failed: "+err,err);
    if(reply.msg!=want || reply.status!=ST_OK)return fail("CUDA worker rejected operation or negotiation: "+std::string(status_name(reply.status)),err);
    return true;
}
bool GpuWorkerTier::open(const Open& in,std::string& err){
    using namespace gpu_codec;
    if(dead_) {err="CUDA tier is dead; create a new engine session: "+why_;return false;}
    if(ready_){err="CUDA tier placement is already frozen";return false;}
    if(in.deadline_ms<1 || in.setup_ms<1 || !validate_gpu_plan(in.plan,err))return false;
    close();stats_={};remote_={};plan_=in.plan;rows_=in.entry_rows;cancelled_=in.cancelled;deadline_ms_=in.deadline_ms;
    const auto began=now_ms();setup_end_=began+(uint64_t)in.setup_ms;
    sock_fd fd=kBadFd;
    const auto cancel=[&]{return(cancelled_&&cancelled_())||now_ms()>=setup_end_;};
    if(!sock::connect_numeric_controlled(in.host,in.port,std::min(in.deadline_ms,in.setup_ms),cancel,fd,err))
        return fail("CUDA worker connection failed: "+err,err);
    fd_=(uintptr_t)fd;
    std::vector<uint8_t> payload;
    put(payload,kGpuWireVersion,4);put(payload,kGpuCapabilityCuda|(rows_?kGpuCapabilityEntryRows:0),4);put(payload,kGpuNumericsQ8_1,4);
    WireFrame response;
    if(!exchange(GPU_HELLO,payload,GPU_HELLO_ACK,response,err))return false;
    Cursor hello{response.payload};
    const auto version=hello.get(4),capability=hello.get(4),numerics=hello.get(4),chunk=hello.get(4);
    if(!hello.done() || version!=kGpuWireVersion || capability!=(kGpuCapabilityCuda|(rows_?kGpuCapabilityEntryRows:0)) || numerics!=kGpuNumericsQ8_1 ||
       !chunk || chunk>kUploadChunkMax)return fail("unsupported CUDA capability/numerical contract",err);
    std::vector<uint8_t> encoded;
    if(!encode_gpu_plan(plan_,encoded,err))return fail(err,err);
    payload.clear();uint64_t nonce=began^((uint64_t)(uintptr_t)this<<17);if(!nonce)nonce=1;
    put(payload,nonce);payload.insert(payload.end(),encoded.begin(),encoded.end());
    if(!exchange(GPU_SETUP,payload,GPU_SETUP_ACK,response,err))return false;
    Cursor setup{response.payload};session_=setup.get();setup.i=0;
    if(!binding(setup,session_,plan_))return fail("CUDA setup returned stale model/layout/placement binding",err);
    const auto count=setup.get(4);
    if(!setup.ok || count!=plan_.assets.size() || response.payload.size()-setup.i!=count)
        return fail("CUDA setup missing-set shape mismatch",err);
    const std::vector<uint8_t> missing(response.payload.begin()+setup.i,response.payload.end());
    std::vector<uint8_t> staging;
    for(size_t i=0;i<missing.size();++i){
        if(missing[i]>1)return fail("CUDA setup returned malformed missing-set",err);
        const auto& asset=plan_.assets[i];
        if(!missing[i]){stats_.cache_reused_bytes+=asset.bytes;continue;}
        if(cancel())return fail("CUDA acquisition cancelled or setup deadline",err);
        staging.resize((size_t)asset.bytes);
        if(!in.fetch || !in.fetch(asset.layer,asset.expert,staging.data(),(uint32_t)asset.bytes) ||
           blob_digest(staging)!=asset.digest)return fail("CUDA selected source blob missing or digest mismatch",err);
        stats_.host_staging_peak=std::max(stats_.host_staging_peak,(uint64_t)staging.capacity());
        for(size_t offset=0;offset<staging.size();){
            const size_t bytes=std::min<size_t>((size_t)chunk,staging.size()-offset);
            payload=binding(session_,plan_);put(payload,i,4);put(payload,offset,4);put(payload,bytes,4);
            payload.insert(payload.end(),staging.begin()+offset,staging.begin()+offset+bytes);
            if(!exchange(GPU_UPLOAD,payload,GPU_UPLOAD_ACK,response,err))return false;
            Cursor ack{response.payload};
            if(!binding(ack,session_,plan_))return fail("CUDA upload stale binding",err);
            const auto slot=ack.get(4),received=ack.get(4);
            if(!ack.done() || slot!=i || received!=offset+bytes)return fail("CUDA upload duplicate/missing acknowledgement",err);
            offset+=bytes;stats_.setup_network_bytes+=bytes;
        }
    }
    std::vector<uint8_t>().swap(staging);
    payload=binding(session_,plan_);
    if(!exchange(GPU_READY,payload,GPU_READY_ACK,response,err))return false;
    Cursor ready{response.payload};
    if(!binding(ready,session_,plan_) || !ready.done())return fail("CUDA READY returned stale binding",err);
    ready_=true;sequence_=0;stats_.session=session_;stats_.assigned_bytes=plan_.assigned_bytes();
    stats_.setup_ms=(double)(now_ms()-began);setup_end_=0;cancelled_={};
    if(!status(remote_,err))return false;
    if(remote_.assigned_bytes!=plan_.assigned_bytes() || remote_.cached_bytes!=plan_.assigned_bytes() ||
       remote_.gpu_weight_bytes!=plan_.assigned_bytes() || remote_.gpu_allocated_bytes<plan_.assigned_bytes())
        return fail("CUDA READY did not account every assigned verified GPU weight",err);
    return true;
}
bool GpuWorkerTier::post(int64_t layer,const float* x,const int32_t* ids,const float* weights,
                         const uint8_t* owned,int T,int K,std::string& err){
    if(posted_)return fail("CUDA post called before prior window finished",err);
    posted_=served_=false;reply_.partials.clear();rows_reply_={};
    if(dead_){err=why_;return false;}
    if(!ready_ || !x || !ids || !weights || !owned || layer<0 || layer>=plan_.n_layers || T<1 || T>kMaxT || K<1 || K>kMaxK)
        return fail("CUDA request invalid state, geometry or pointers",err);
    InferRequest r;r.session=session_;r.model_id=plan_.model_id;r.partition_id=plan_.placement_id.a;
    r.sequence=sequence_+1;r.layer=(uint16_t)layer;r.T=(uint8_t)T;r.K=(uint8_t)K;
    const size_t entries=(size_t)T*K;r.mask.resize(entries);r.ids.resize(entries);r.weights.resize(entries);
    uint64_t count=0;
    for(size_t i=0;i<entries;++i){
        if(owned[i]>1 || ids[i]<0 || ids[i]>=plan_.n_expert || !std::isfinite(weights[i]) ||
           (owned[i]&&!plan_.owns((int)layer,ids[i])))return fail("CUDA request ownership/ids/weights invalid",err);
        r.mask[i]=owned[i];r.ids[i]=(uint16_t)ids[i];r.weights[i]=weights[i];count+=owned[i];
    }
    if(!count)return true;
    r.activations.assign(x,x+(size_t)T*plan_.n_embd);
    for(float v:r.activations)if(!std::isfinite(v))return fail("CUDA request activation nonfinite",err);
    std::vector<uint8_t> payload;
    if(!gpu_codec::encode_request(plan_,r,payload))return fail("CUDA inference encode failed",err);
    const auto began=now_ms();WireFrame response;
    if(!exchange(rows_?GPU_INFER_ROWS:GPU_INFER,payload,rows_?GPU_INFER_ROWS_ACK:GPU_INFER_ACK,response,err))return false;
    if(rows_) {
        if(!gpu_codec::decode_rows_reply(plan_,blob_digest(payload),r,response.payload,rows_reply_))
            return fail("CUDA entry rows malformed, missing, duplicate, nonfinite or stale",err);
    } else {
        InferReply result;
        if(!gpu_codec::decode_reply(plan_,blob_digest(payload),r,response.payload,result))
            return fail("CUDA reply malformed, duplicate, missing, nonfinite or stale",err);
        reply_=std::move(result);
    }
    sequence_=r.sequence;posted_=true;
    ++stats_.requests;stats_.entries+=count;stats_.request_ms+=(double)(now_ms()-began);return true;
}
bool GpuWorkerTier::finish(std::string& err){
    if(dead_){err=why_;return false;}
    served_=posted_;posted_=false;return true;
}
bool GpuWorkerTier::status(GpuCounters& c,std::string& err){
    if(dead_ || !ready_){err=dead_?why_:"CUDA tier is not ready";return false;}
    WireFrame response;
    if(!exchange(GPU_STATUS,gpu_codec::binding(session_,plan_),GPU_STATUS_ACK,response,err))return false;
    gpu_codec::Cursor cursor{response.payload};
    if(!gpu_codec::binding(cursor,session_,plan_) || cursor.get(4)!=1 || !cursor.ok)
        return fail("CUDA status returned stale/nonready binding",err);
    std::vector<uint8_t> values(response.payload.begin()+cursor.i,response.payload.end());
    if(!gpu_codec::counters(values,c))return fail("malformed CUDA counters",err);
    remote_=c;return true;
}
void GpuWorkerTier::report(std::string& out)const{
    char line[1400];std::snprintf(line,sizeof line,
      "CUDA/q8_1 approximate %s model=%llx layout=%llx:%llx placement=%llx:%llx session=%llx "
      "assigned=%llu cached=%llu transferred=%llu reused=%llu persisted=%llu retained_disk=%llu verified=%llu host_staging_peak=%llu gpu_allocated=%llu gpu_peak=%llu gpu_scratch=%llu "
      "gpu_free_before=%llu gpu_free_after=%llu reserve=%llu requests=%llu entries=%llu failures=%llu setup_ms=%.1f request_ms=%.1f",
      dead_?"DEAD":"READY",(unsigned long long)plan_.model_id,(unsigned long long)plan_.layout_id.a,
      (unsigned long long)plan_.layout_id.b,(unsigned long long)plan_.placement_id.a,(unsigned long long)plan_.placement_id.b,
      (unsigned long long)session_,(unsigned long long)stats_.assigned_bytes,(unsigned long long)remote_.cached_bytes,
      (unsigned long long)stats_.setup_network_bytes,(unsigned long long)stats_.cache_reused_bytes,
      (unsigned long long)remote_.persisted_bytes,(unsigned long long)remote_.retained_disk_bytes,
      (unsigned long long)remote_.verified_bytes,
      (unsigned long long)std::max(stats_.host_staging_peak,remote_.host_staging_peak),
      (unsigned long long)remote_.gpu_allocated_bytes,(unsigned long long)remote_.gpu_peak_bytes,
      (unsigned long long)remote_.gpu_scratch_bytes,(unsigned long long)remote_.gpu_free_before,
      (unsigned long long)remote_.gpu_free_after,(unsigned long long)remote_.gpu_runtime_reserve,(unsigned long long)stats_.requests,
      (unsigned long long)stats_.entries,(unsigned long long)stats_.failures,stats_.setup_ms,stats_.request_ms);
    out=line;
}
} // namespace strata::worker
