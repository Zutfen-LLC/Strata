// No model or socket. Format, bounded lifecycle and full-before-masked ordering.
#include "strata/worker/cpu_capture.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>
using namespace strata::worker;
namespace {
int failures=0;
void check(bool value,const char* message){if(!value){++failures;std::fprintf(stderr,"FAIL %s\n",message);}}
void set32(std::vector<uint8_t>& b,size_t at,uint32_t v){for(int i=0;i<4;++i)b[at+i]=(uint8_t)(v>>(8*i));}
uint32_t get32(const std::vector<uint8_t>& b,size_t at){uint32_t v=0;for(int i=0;i<4;++i)v|=(uint32_t)b[at+i]<<(8*i);return v;}
void resign(std::vector<uint8_t>& b){auto d=blob_digest(b.data(),b.size()-16);for(int i=0;i<8;++i){b[b.size()-16+i]=(uint8_t)(d.a>>(8*i));b[b.size()-8+i]=(uint8_t)(d.b>>(8*i));}}
std::vector<float> reference(const InferRequest& q){
    std::vector<float> out((size_t)q.T*32,0.f);
    for(int t=0;t<q.T;++t)for(int k=0;k<q.K;++k)if(q.mask[t*q.K+k])
        for(int h=0;h<32;++h)out[t*32+h]+=q.weights[t*q.K+k]*q.activations[t*32+h]*(q.ids[t*q.K+k]+1);
    return out;
}
}
int main(){
    const auto root=std::filesystem::temp_directory_path()/std::filesystem::path("strata-capture-"+std::to_string(getpid()));
    if(!std::filesystem::create_directory(root))return 2;
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::filesystem::remove_all(p);}} cleanup{root};
    SetupManifest m;m.model_id=7;m.n_layers=2;m.n_expert=4;m.n_embd=32;m.n_ff=32;m.gu_type=m.d_type=8;
    std::strcpy(m.quant_name,"Q8_0");m.partition={7,7};m.partition_id=blob_digest(m.partition).a;
    const auto path=(root/"capture.bin").string();std::string err;CpuCapture c;
    check(c.prepare(m,{7,8},{9,10},path,err,65536),"prepare");const auto capacity=c.storage_bytes();
    check(c.request("GEN 256 1 2 3",{1,2,3},256),"request binding");
    auto record=[&](int T,int layer){InferRequest q;q.session=1;q.sequence=1;q.model_id=7;q.partition_id=m.partition_id;
        q.T=(uint8_t)T;q.K=4;q.layer=(uint16_t)layer;
        q.mask.resize(T*4);q.ids.resize(T*4);q.weights.assign(T*4,.25f);q.activations.assign(T*32,2.f);
        for(int i=0;i<T*4;++i){q.ids[i]=i%4;q.mask[i]=i%4!=3;}
        c.slice(0);check(c.begin(q),"begin actual copied request");auto y=reference(q);check(c.finish(y.data()),"finish reply");};
    check(c.window(0,2,1),"multi-token prefill");record(2,0);record(2,1);
    check(c.window(2,1,2),"first-logit window");record(1,0);record(1,1);
    float logits[]={1,2,3};check(c.seal(logits,3),"seal");check(c.storage_bytes()==capacity,"fixed retained capacity");
    check(!std::filesystem::exists(path+".partial"),"no partial remains");
    CpuCapture other;check(!other.prepare(m,{7,8},{9,10},path,err),"no overwrite");
    CpuCaptureFile f;check(read_cpu_capture(path,f,err),"decode roundtrip");
    if(f.records.size()!=4)return 1;
    check(f.records.size()==4 && f.windows.size()==2 && f.records[0].request.T==2,"multi-token retained");
    const auto owners=cpu_capture_owners(m);check(owners[0]==0 && owners[2]==0 && owners[4]==1 && owners[7]==-1,"global alternating ordinal");
    CpuReplayReport result;size_t calls=0;
    check(replay_cpu_capture(f,[&](const InferRequest& q,std::vector<float>& y,std::string&){
        if(calls<4)check(q.mask==f.records[calls].request.mask,"all full requests before masked");++calls;y=reference(q);return true;
    },result,err),"replay");
    check(result.full_checked==4 && result.grouped_checked==4 && result.max_abs==0,"grouped reference");
    calls=0;check(!replay_cpu_capture(f,[&](const InferRequest& q,std::vector<float>& y,std::string&){++calls;y=reference(q);y[0]+=1;return true;},result,err),"full mismatch refuses grouping");
    check(calls==1 && result.grouped_checked==0,"mismatch stops immediately");
    std::ifstream in(path,std::ios::binary|std::ios::ate);std::vector<uint8_t> bytes((size_t)in.tellg());in.seekg(0);in.read((char*)bytes.data(),bytes.size());
    auto bad=bytes;bad[150]^=1;check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"corrupt digest refused");
    check(!decode_cpu_capture(bytes.data(),bytes.size()-1,f,err),"truncation refused");
    check(!decode_cpu_capture(bytes.data(),kCpuCaptureLimit+1,f,err),"oversize refused before access");
    bad=bytes;set32(bad,56,129);resign(bad);check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"window count bound");
    bad=bytes;set32(bad,136,8);resign(bad);check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"wrong first-logit row geometry");
    const size_t window=kCpuCaptureHeader+get32(bytes,16), event=window+28, payload=event+24;
    bad=bytes;set32(bad,window+8,2);resign(bad);check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"missing initial ordinal");
    bad=bytes;bad[payload+kInferFixed+3]=1;resign(bad);check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"GPU entry marked CPU refused");
    bad=bytes;set32(bad,payload+kInferFixed+2*4*7,0x7fc00000);resign(bad);
    check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"nonfinite activation refused");
    bad=bytes;const size_t missing=event+get32(bytes,event+4), missing_bytes=get32(bytes,missing+4);
    bad.erase(bad.begin()+missing,bad.begin()+missing+missing_bytes);
    set32(bad,12,(uint32_t)bad.size());set32(bad,20,3);resign(bad);
    check(!decode_cpu_capture(bad.data(),bad.size(),f,err),"omitted layer rejected despite valid digest/counts");
    CpuCapture small;check(small.prepare(m,{7,8},{9,10},(root/"small").string(),err,512),"small fixed capacity");
    check(!small.request("GEN 256 1 2 3",{1,2,3},256),"capacity refuses before work");
    CpuCapture lifecycle;check(lifecycle.prepare(m,{7,8},{9,10},(root/"lifecycle").string(),err,65536),"lifecycle prepare");
    check(lifecycle.request("GEN 256 1 2 3",{1,2,3},256),"lifecycle request");
    check(!lifecycle.window(1,2,1),"resumed/missing initial window refused");
    check(!lifecycle.seal(logits,3) && !std::filesystem::exists(root/"lifecycle"),"incomplete not published");
    CpuCapture gap;check(gap.prepare(m,{7,8},{9,10},(root/"gap").string(),err,65536),"gap prepare");
    check(gap.request("GEN 1 1 2 3",{1,2,3},1) && gap.window(0,2,1),"gap window");
    InferRequest missing_slice;missing_slice.model_id=7;missing_slice.partition_id=m.partition_id;missing_slice.T=missing_slice.K=1;
    missing_slice.ids={3};missing_slice.mask={0};missing_slice.weights={1};missing_slice.activations.assign(32,1);
    for(int l=0;l<2;++l){missing_slice.layer=(uint16_t)l;check(gap.begin(missing_slice) && gap.finish(nullptr),"partial slice");}
    check(!gap.window(2,1,2),"omitted token slice rejected before next window");
    CpuCapture zero;check(zero.prepare(m,{7,8},{9,10},(root/"zero").string(),err,65536),"zero prepare");
    check(zero.request("GEN 1 1",{1},1) && zero.window(0,1,2),"zero lifecycle");
    InferRequest q;q.model_id=7;q.partition_id=m.partition_id;q.T=q.K=1;q.layer=0;
    q.ids={3};q.mask={0};q.weights={1};q.activations.assign(32,1);
    check(zero.begin(q) && zero.finish(nullptr),"zero recorded without send");
    check(!zero.seal(logits,3),"omitted layer refuses seal");
    // A failed writer is latched; start a fresh complete zero-work file.
    CpuCapture allzero;check(allzero.prepare(m,{7,8},{9,10},(root/"allzero").string(),err,65536),"allzero prepare");
    check(allzero.request("GEN 1 1",{1},1) && allzero.window(0,1,2),"allzero lifecycle");
    for(int l=0;l<2;++l){q.layer=(uint16_t)l;check(allzero.begin(q) && allzero.finish(nullptr),"allzero layer");}
    check(allzero.seal(logits,3),"allzero seal");
    check(read_cpu_capture((root/"allzero").string(),f,err),"zero roundtrip");
    calls=0;check(replay_cpu_capture(f,[&](const InferRequest&,std::vector<float>&,std::string&){++calls;return false;},result,err) && !calls,"zero replay never exchanges");
    std::printf("CPU_CAPTURE_TEST failures=%d\n",failures);return failures?1:0;
}
