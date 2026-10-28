// Model-free reply codec measurement. No sockets, assets, worker processes or native kernels.
#include "strata/worker/expert_wire.hpp"
#include "strata/worker/expert_worker.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

using namespace strata::worker;
namespace {
using Clock = std::chrono::steady_clock;
constexpr int H=2560, K=10, warmups=64, iterations=4096;
constexpr size_t scratch_limit=16u<<20;
const auto started=Clock::now();

void need(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
void deadline() {need(Clock::now()-started<std::chrono::seconds(29),"29s cooperative deadline");}
// Make all output bytes observable to the optimizer, without reading/logging them in timed loops.
void escape(const void* p) {asm volatile("" : : "g"(p) : "memory");}
void le(std::vector<uint8_t>& out,uint64_t v,int n) {
    for(int b=0;b<n;++b)out.push_back((uint8_t)(v>>(8*b)));
}
void f32(std::vector<uint8_t>& out,float v) {uint32_t u;std::memcpy(&u,&v,4);le(out,u,4);}
bool bits(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
}

struct Shape {
    InferRequest q;
    InferReply weighted,decoded_weighted;
    EntryRowsReply decoded_rows;
    std::vector<float> rows,central_rows,central_weights,combined;
    std::vector<uint8_t> payload,frame,copy,expected,central_mask,seen;
    explicit Shape(int T) {
        q.session=0x0102030405060708ull;q.sequence=0x1112131415161718ull;q.T=T;q.K=K;
        q.mask.assign(T*K,1);q.ids.resize(T*K);q.weights.resize(T*K);
        rows.resize((size_t)T*K*H);
        for(int i=0;i<T*K;++i){q.ids[i]=i%K;q.weights[i]=(i%2?-1.f:1.f)*0.125f;}
        for(size_t i=0;i<rows.size();++i)rows[i]=float(int(i%31)-15)*0.03125f;
        rows[0]=-0.0f;
        weighted.session=q.session;weighted.sequence=q.sequence;weighted.partials.resize(T*H);
        ExpertWorkerService::accumulate_partial(T,K,q.mask.data(),q.weights.data(),rows.data(),H,weighted.partials.data());
    }
    size_t scratch() const {
        return (rows.capacity()+central_rows.capacity()+central_weights.capacity()+combined.capacity()+
                q.weights.capacity()+q.activations.capacity()+weighted.partials.capacity()+
                decoded_weighted.partials.capacity()+decoded_rows.rows.capacity())*sizeof(float)+
            (q.ids.capacity()+decoded_rows.indices.capacity())*sizeof(uint16_t)+q.mask.capacity()+
            payload.capacity()+frame.capacity()+copy.capacity()+expected.capacity()+central_mask.capacity()+seen.capacity();
    }
    void bounded() const {need(scratch()<=scratch_limit,"scratch capacity exceeds16MiB");}
    bool encode(bool indexed) {
        if(indexed)return encode_infer_rows_ack(q,H,rows.data(),payload);
        encode_infer_ack(weighted,payload);return true;
    }
    bool frame_direct(bool indexed) {
        if(indexed)return encode_infer_rows_ack_frame(q,H,rows.data(),frame);
        encode_infer_ack_frame(weighted,frame);return true;
    }
    bool decode(bool indexed) {
        return indexed?decode_infer_rows_ack(payload.data(),payload.size(),H,q,decoded_rows):
            decode_infer_ack(payload.data(),payload.size(),H,q.T,decoded_weighted);
    }
    bool central(bool indexed) {
        if(indexed) {
            // Retained-buffer equivalent of group post's preparation and finish's scatter/reducer.
            // This does not measure group locks, I/O, scheduling or its tiny per-window seen allocation.
            central_rows.assign(rows.size(),0.f);central_weights.assign(q.weights.begin(),q.weights.end());
            central_mask.assign(q.mask.begin(),q.mask.end());seen.assign(q.mask.size(),0);
            for(size_t j=0;j<decoded_rows.indices.size();++j) {
                const size_t i=decoded_rows.indices[j];
                if(i>=seen.size() || !central_mask[i] || seen[i])return false;
                seen[i]=1;std::copy_n(decoded_rows.rows.data()+j*H,H,central_rows.data()+i*H);
            }
            if(seen!=central_mask)return false;
            combined.resize((size_t)q.T*H);
            ExpertWorkerService::accumulate_partial(q.T,K,central_mask.data(),central_weights.data(),central_rows.data(),H,combined.data());
        } else combined.assign(decoded_weighted.partials.begin(),decoded_weighted.partials.end());
        return std::all_of(combined.begin(),combined.end(),[](float v){return std::isfinite(v);});
    }
    void check(bool indexed) {
        need(encode(indexed),"encode refused fixture");
        expected.clear();le(expected,q.session,8);le(expected,q.sequence,8);
        if(indexed) {
            le(expected,kEntryRowsVersion,2);le(expected,q.T*K,2);
            for(int i=0;i<q.T*K;++i){le(expected,i,2);for(int h=0;h<H;++h)f32(expected,rows[(size_t)i*H+h]);}
        } else for(float v:weighted.partials)f32(expected,v);
        need(payload==expected,"exact independent little-endian payload mismatch");
        const auto msg=indexed?MSG_INFER_ROWS_ACK:MSG_INFER_ACK;
        encode_frame(frame,msg,ST_OK,payload.data(),payload.size());
        expected.clear();expected.insert(expected.end(),{'S','E','W','1'});le(expected,msg,2);
        le(expected,ST_OK,2);le(expected,payload.size(),8);expected.insert(expected.end(),payload.begin(),payload.end());
        need(frame==expected,"exact independent frame mismatch");
        need(frame_direct(indexed) && frame==expected,"direct frame differs from independent reference/old path");
        need(decode(indexed),"decode refused fixture");
        if(indexed) {
            need(decoded_rows.session==q.session && decoded_rows.sequence==q.sequence &&
                bits(decoded_rows.rows,rows) && std::signbit(decoded_rows.rows[0]),"row bits/signed zero/identity mismatch");
            need(decoded_rows.indices.size()==(size_t)q.T*K,"row index count mismatch");
            for(size_t i=0;i<decoded_rows.indices.size();++i)need(decoded_rows.indices[i]==i,"row index mismatch");
        } else need(decoded_weighted.session==q.session && decoded_weighted.sequence==q.sequence &&
                    bits(decoded_weighted.partials,weighted.partials),"weighted reply bits/identity mismatch");
        need(central(indexed) && bits(combined,weighted.partials),"central reconstruction bits mismatch");
        copy.resize(frame.size());std::memcpy(copy.data(),frame.data(),frame.size());
        need(copy==frame,"same-size block copy mismatch");bounded();
    }
};

void signed_zero_check() {
    InferReply original;original.session=1;original.sequence=2;original.partials={-0.0f,0.0f};
    std::vector<uint8_t> bytes;encode_infer_ack(original,bytes);
    need(bytes.size()==24 && bytes[19]==0x80 && bytes[23]==0,"weighted signed-zero exact bytes");
    InferReply decoded;need(decode_infer_ack(bytes.data(),bytes.size(),2,1,decoded) &&
        bits(original.partials,decoded.partials) && std::signbit(decoded.partials[0]),"weighted signed-zero decode");
}

template<class Work> void measure(Shape& s,bool indexed,const char* phase,size_t bytes,Work work) {
    for(int i=0;i<warmups;++i)need(work(),"warmup failed");
    s.bounded();const size_t retained=s.scratch();deadline();
    const auto begin=Clock::now();
    for(int i=0;i<iterations;++i){if(i%64==0)deadline();need(work(),"measurement failed");}
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();
    need(s.scratch()==retained,"buffer capacity grew after warmup");s.bounded();
    std::printf("CPU_ROW_CODEC T=%u K=10 H=2560 entries=%u mode=%s phase=%s warmup=64 measured=4096 "
        "bytes_per_iteration=%zu total_ns=%llu mean_ns=%.3f scratch_capacity_bytes=%zu\n",
        unsigned(s.q.T),unsigned(s.q.T*K),indexed?"rows":"weighted",phase,bytes,
        (unsigned long long)ns,double(ns)/iterations,s.scratch());
    std::fflush(stdout);
}
void benchmark(Shape& s,bool indexed) {
    s.check(indexed);const size_t payload=s.payload.size(),frame=s.frame.size();
    measure(s,indexed,"encode",payload,[&]{const bool ok=s.encode(indexed);escape(s.payload.data());return ok;});
    measure(s,indexed,"frame_copy",frame,[&]{encode_frame(s.frame,indexed?MSG_INFER_ROWS_ACK:MSG_INFER_ACK,
        ST_OK,s.payload.data(),s.payload.size());escape(s.frame.data());return true;});
    measure(s,indexed,"encode_then_frame",frame,[&]{
        if(!s.encode(indexed))return false;
        encode_frame(s.frame,indexed?MSG_INFER_ROWS_ACK:MSG_INFER_ACK,ST_OK,s.payload.data(),s.payload.size());
        escape(s.frame.data());return true;});
    measure(s,indexed,"encode_direct_frame",frame,[&]{const bool ok=s.frame_direct(indexed);escape(s.frame.data());return ok;});
    measure(s,indexed,"decode",payload,[&]{const bool ok=s.decode(indexed);
        escape(indexed?s.decoded_rows.rows.data():s.decoded_weighted.partials.data());return ok;});
    measure(s,indexed,"central_prepare_reduce",(indexed?s.rows.size():s.weighted.partials.size())*sizeof(float),[&]{const bool ok=s.central(indexed);
        escape(s.combined.data());return ok;});
    measure(s,indexed,"block_copy",frame,[&]{std::memcpy(s.copy.data(),s.frame.data(),frame);escape(s.copy.data());return true;});
    s.check(indexed); // correctness remains outside timing, including after the retained-buffer loops
}
}
int main(int argc,char** argv) {
    std::signal(SIGALRM,SIG_DFL);
    ::alarm(30); // Hard process ceiling; no child, socket, model allocation or external cleanup.
    const bool check_only=argc==2 && std::strcmp(argv[1],"--check")==0;
    if(argc!=1 && !check_only){std::fprintf(stderr,"usage: cpu_row_codec_bench [--check]\n");return 2;}
    try {
        signed_zero_check();
        for(int T:{1,2}) {
            Shape s(T);
            for(bool indexed:{false,true})if(check_only)s.check(indexed);else benchmark(s,indexed);
            deadline();
        }
        std::printf("CPU_ROW_CODEC_DONE ok=1 check_only=%d scratch_limit_bytes=%zu deadline_seconds=30 "
            "model=0 sockets=0 expert_kernels=0 phases_independent=1\n",check_only,scratch_limit);
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"CPU_ROW_CODEC_FAIL %s\n",e.what());return 1;}
}
