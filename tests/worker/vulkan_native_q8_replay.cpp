// One explicitly selected retained request; never a model/token campaign.
#ifdef STRATA_Q8_REPLAY_CUDA
#include "../../src/kernels/cuda/iq_kernels.cu"
#elif !defined(STRATA_Q8_REPLAY_CLIENT_ONLY)
#include "strata/worker/vulkan_executor.hpp"
#include "ggml-vulkan.h"
#include "native_q8_replay_timing.hpp"
#endif
#include "vulkan_native_q8_layout.hpp"
#include "strata/worker/cpu_capture.hpp"
#include "strata/worker/gpu_client.hpp"
#include "strata/worker/gpu_endpoint.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
using namespace strata::worker;
using namespace strata::worker::vulkan_detail;
namespace fs=std::filesystem;
std::string hexid(uint64_t v,int width=16){std::ostringstream o;o<<std::hex<<std::setfill('0')<<std::setw(width)<<v;return o.str();}
std::string key(const GpuPlan& p,const GpuAsset& a){return "sgpu1-"+hexid(p.model_id)+"-"+hexid(p.layout_id.a)+hexid(p.layout_id.b)+"-"+hexid(a.layer,4)+"-"+hexid(a.expert,4)+"-"+hexid(a.bytes)+"-"+hexid(a.digest.a)+hexid(a.digest.b)+".blob";}
void write(const std::string& path,const void* data,size_t n){std::ofstream f(path,std::ios::binary);require(bool(f.write(static_cast<const char*>(data),n)),"replay boundary write failed");}
#ifdef STRATA_Q8_REPLAY_CUDA
void cu(cudaError_t e){require(e==cudaSuccess,cudaGetErrorString(e));}
struct Device {void* p=nullptr;size_t n;explicit Device(size_t size):n(size){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}void put(const void* s){cu(cudaMemcpy(p,s,n,cudaMemcpyHostToDevice));}void dump(const std::string& path){std::vector<uint8_t> b(n);cu(cudaMemcpy(b.data(),p,n,cudaMemcpyDeviceToHost));write(path,b.data(),n);}};
#endif
#ifndef STRATA_Q8_REPLAY_CUDA
// The reference is an SGW1 row reply bound to the complete canonical request
// and plan. This digest detects mistakes, not a malicious evidence substitution.
std::vector<uint8_t> file_bytes(const fs::path& path,size_t limit) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);require(bool(in),"reference open failed");
    auto size=in.tellg();require(size>=0 && uint64_t(size)<=limit,"reference size invalid");
    std::vector<uint8_t> data(size_t(size),uint8_t{});in.seekg(0);
    require(bool(in.read(reinterpret_cast<char*>(data.data()),data.size())),"reference read failed");return data;
}
int worker_replay(int argc,char** argv) {
    require(argc==11 || argc==19,"usage: --worker HOST:PORT CAPTURE RECORD PLAN BLOB_DIR DIRECT_REPLY NEW_OUTPUT --deadline-ms N [--tls-ca FILE --tls-cert FILE --tls-key FILE --tls-peer DNS]");
    const fs::path output=argv[8];require(fs::create_directory(output),"output must be a new private directory");
    fs::permissions(output,fs::perms::owner_all,fs::perm_options::replace);
    try {
    GpuEndpoint endpoint;require(std::string(argv[2]).find('@')==std::string::npos && parse_gpu_endpoint(argv[2],endpoint) && !endpoint.vulkan,"worker endpoint must be numeric HOST:PORT without a backend tag");
    require(std::string(argv[9])=="--deadline-ms","explicit deadline required");
    std::string deadline_value=argv[10];require(!deadline_value.empty() && deadline_value.find_first_not_of("0123456789")==std::string::npos,"invalid deadline");
    const auto deadline=std::stoull(deadline_value);require(deadline && deadline<=600000,"deadline out of range");
    TlsConfig tls;
    if(argc==19) {
        require(std::string(argv[11])=="--tls-ca" && std::string(argv[13])=="--tls-cert" &&
            std::string(argv[15])=="--tls-key" && std::string(argv[17])=="--tls-peer","invalid TLS arguments");
        tls={argv[12],argv[14],argv[16],argv[18]};
    }
    std::string err;require(tls.validate(err),err);
    CpuCaptureFile f;GpuPlan p;require(read_cpu_capture(argv[3],f,err),err);require(read_gpu_plan(argv[5],p,err),err);
    const std::string record=argv[4];require(!record.empty() && record.find_first_not_of("0123456789")==std::string::npos,"invalid record");
    const auto ordinal=std::stoull(record);require(ordinal<f.records.size() && f.records[ordinal].sent,"record absent/not sent");
    auto q=f.records[ordinal].request;
    require(p.model_id==f.setup.model_id && p.n_embd==H && p.n_ff==F && p.gu_type==8 && p.d_type==8 &&
        p.n_layers==f.setup.n_layers && p.n_expert==f.setup.n_expert && !p.assets.empty(),"capture/plan identity or native geometry mismatch");
    for(size_t i=0;i<q.mask.size();++i)require(!q.mask[i] || p.owns(q.layer,q.ids[i]),"full request not covered by plan; refusing to remask");
    q.model_id=p.model_id;q.partition_id=p.placement_id.a;q.session=1;q.sequence=1;
    std::vector<uint8_t> request;require(gpu_codec::encode_request(p,q,request),"cannot encode retained request");
    const auto marker=file_bytes(fs::path(argv[7]).parent_path()/"DIRECT_PASS",256);
    require(std::string(marker.begin(),marker.end())=="direct replay complete\n","direct completion marker absent or invalid");
    const auto ref=file_bytes(argv[7],kGpuFrameMax);EntryRowsReply direct;
    require(gpu_codec::decode_rows_reply(p,blob_digest(request),q,ref,direct),"direct reference does not bind this plan and request");
    write((output/"direct.reply").string(),ref.data(),ref.size());
    write((output/"canonical.request").string(),request.data(),request.size());
        GpuWorkerTier::Open open;open.host=endpoint.host;open.port=endpoint.port;open.plan=p;
        open.vulkan=open.native_q8=open.entry_rows=true;open.deadline_ms=int(deadline);open.setup_ms=600000;open.tls=tls;
        open.fetch=[&](int64_t layer,int64_t expert,uint8_t* dst,uint32_t n){
            const auto* a=p.find(layer,expert);if(!a || a->bytes!=n)return false;
            std::ifstream in(fs::path(argv[6])/key(p,*a),std::ios::binary|std::ios::ate);
            if(!in || in.tellg()!=std::streamoff(n))return false;
            in.seekg(0);
            return bool(in.read(reinterpret_cast<char*>(dst),n)) && blob_digest(dst,n)==a->digest;
        };
        // Verify every source blob even if the service already has a cached copy.
        for(const auto& a:p.assets) {
            std::vector<uint8_t> blob(a.bytes);
            require(open.fetch(a.layer,a.expert,blob.data(),uint32_t(blob.size())),"staged blob missing or digest mismatch");
        }
        GpuWorkerTier tier;require(tier.open(open,err),err);GpuCounters before,after;require(tier.status(before,err),err);
        std::vector<int32_t> ids(q.ids.begin(),q.ids.end());
        require(tier.post(q.layer,q.activations.data(),ids.data(),q.weights.data(),q.mask.data(),q.T,q.K,err) && tier.finish(err) && tier.served(),err);
        const auto remote=tier.entry_rows();require(tier.status(after,err),err);
        write((output/"worker.indices").string(),remote.indices.data(),remote.indices.size()*sizeof(uint16_t));
        write((output/"worker.rows.f32").string(),remote.rows.data(),remote.rows.size()*sizeof(float));
        std::ofstream meta(output/"worker.status");
        auto status=[&](const char* label,const GpuCounters& c){meta<<label<<" assigned="<<c.assigned_bytes<<" cached="<<c.cached_bytes
            <<" transferred="<<c.transferred_bytes<<" verified="<<c.verified_bytes<<" weights="<<c.gpu_weight_bytes
            <<" allocated="<<c.gpu_allocated_bytes<<" scratch="<<c.gpu_scratch_bytes<<" allocations="<<c.gpu_allocation_count
            <<" requests="<<c.requests<<" entries="<<c.entries<<"\n";};
        meta<<"session="<<remote.session<<" sequence="<<remote.sequence<<"\n";status("before",before);status("after",after);
        require(bool(meta),"worker status write failed");meta.close();
        const auto entries=std::count(q.mask.begin(),q.mask.end(),uint8_t(1));
        require(before.gpu_allocated_bytes==after.gpu_allocated_bytes && before.gpu_weight_bytes==after.gpu_weight_bytes &&
            before.gpu_scratch_bytes==after.gpu_scratch_bytes && before.gpu_allocation_count==after.gpu_allocation_count &&
            before.assigned_bytes==after.assigned_bytes && before.cached_bytes==after.cached_bytes &&
            before.transferred_bytes==after.transferred_bytes && before.verified_bytes==after.verified_bytes &&
            after.requests>=before.requests && after.requests-before.requests==1 &&
            after.entries>=before.entries && after.entries-before.entries==uint64_t(entries),"worker lifecycle counters changed unexpectedly");
        require(remote.indices==direct.indices && remote.rows.size()==direct.rows.size() &&
            !std::memcmp(remote.rows.data(),direct.rows.data(),direct.rows.size()*sizeof(float)),"worker/direct indexed rows differ");
        const std::string pass="indexed rows and lifecycle counters match; same physical device/build/shaders require separate evidence\n";
        write((output/"PASS").string(),pass.data(),pass.size());return 0;
    } catch(const std::exception& e) {std::ofstream failure(output/"FAILURE");failure<<e.what()<<"\n";throw; }
}
#endif
int main(int argc,char** argv){try {
#ifndef STRATA_Q8_REPLAY_CUDA
    if(argc>1 && std::string(argv[1])=="--worker")return worker_replay(argc,argv);
#endif
#ifdef STRATA_Q8_REPLAY_CLIENT_ONLY
    throw std::runtime_error("CPU-only replay supports --worker only");
#else
    bool timing=false;size_t budget=128u<<20;
    for(int arg=10;arg<argc;++arg) {
        const std::string option=argv[arg];
        if(option=="--timing") {require(!timing,"duplicate timing option");timing=true;}
        else if(option=="--budget-bytes") {
            require(++arg<argc,"budget bytes required");const std::string value=argv[arg];
            require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid budget bytes");
            const auto parsed=std::stoull(value);require(parsed && parsed<=SIZE_MAX,"budget bytes out of range");budget=size_t(parsed);
        } else throw std::runtime_error("unknown replay option");
    }
    require(argc>=10 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device",
        "usage: --device INDEX --expect-device NAME CAPTURE RECORD ONE_LAYER_PLAN BLOB_DIR NEW_OUTPUT [--timing] [--budget-bytes N]");
#ifdef STRATA_Q8_REPLAY_CUDA
    require(!timing,"--timing is the Vulkan provider observation only");
#endif
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(60);
    struct Deadline {
        bool enabled;
        explicit Deadline(bool on):enabled(on){if(on){std::signal(SIGALRM,[](int){std::_Exit(124);});alarm(60);}}
        ~Deadline(){if(enabled)alarm(0);}
    } watchdog(timing);
    const std::string index=argv[2];require(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"invalid device index");
    const int device=std::stoi(index);const std::string expected=argv[4];require(!expected.empty(),"expected device required");
    CpuCaptureFile f;GpuPlan p;std::string error;require(read_cpu_capture(argv[5],f,error),error);require(read_gpu_plan(argv[7],p,error),error);
    const std::string record=argv[6];require(!record.empty() && record.find_first_not_of("0123456789")==std::string::npos,"invalid record index");
    const size_t ordinal=std::stoull(record);require(ordinal<f.records.size() && f.records[ordinal].sent,"record absent/not sent");
    auto q=f.records[ordinal].request;
    require(p.model_id==f.setup.model_id && p.n_embd==2560 && p.n_ff==640 && p.gu_type==8 && p.d_type==8 &&
        p.n_layers==f.setup.n_layers && p.n_expert==f.setup.n_expert && !p.assets.empty(),"capture/plan identity or native geometry mismatch");
    for(size_t j=0;j<q.mask.size();++j)require(!q.mask[j] || p.owns(q.layer,q.ids[j]),"full request not covered by plan; refusing to remask");
    q.model_id=p.model_id;q.partition_id=p.placement_id.a;q.session=1;q.sequence=1;
    require(q.activations.size()==size_t(q.T)*H && q.weights.size()==q.mask.size() && !q.packed,"replay requires raw F32 input");finite(q.activations);finite(q.weights);
    const auto t=native_q8_groups(p,q);
    require(fs::create_directory(argv[9]),"output must be a new private directory");fs::permissions(argv[9],fs::perms::owner_all,fs::perm_options::replace);const std::string prefix=(fs::path(argv[9])/"layer").string();
    auto reader=[&](const GpuAsset& a,std::vector<uint8_t>& b,std::string& err){
        std::ifstream in(fs::path(argv[8])/key(p,a),std::ios::binary|std::ios::ate);
        if(!in || in.tellg()!=std::streamoff(a.bytes)){err="replay staged blob missing/wrong size";return false;}
        b.resize(a.bytes);in.seekg(0);if(!in.read(reinterpret_cast<char*>(b.data()),b.size()) || blob_digest(b)!=a.digest){err="replay staged blob digest mismatch";return false;}return true;};
#ifdef STRATA_Q8_REPLAY_CUDA
    using namespace strata::kernels;
    cu(cudaSetDevice(device));cudaDeviceProp props{};cu(cudaGetDeviceProperties(&props,device));require(std::string(props.name).find(expected)!=std::string::npos,"unexpected CUDA device");
    require(split_off<8>() && !g_kq_split && !gu_split(8) && !g_grouped_v1 && !g_old_kernels && g_exp_phase==0,"native dispatch differs from implemented branch");
    auto layout=native_expert_layout(8,8,H,F);require(!layout.had_block && !expdec_gu_ok(layout) && !expdec_down_ok(layout),"unsupported native layout/dispatch");
    const size_t n=t.dst.size(),fa=(n*F*4+255)&~size_t(255);
    Device x(q.activations.size()*4),xq(q.T*(H/32)*36),ptr(t.experts.size()*8),starts(t.start.size()*4),ng(4),tok(n*4),dst(n*4),scratch(native_expert_scratch_bytes(n,F)),out(n*H*4);
    std::vector<std::unique_ptr<Device>> weights;std::vector<unsigned long long> pointers;
    for(auto slot:t.asset_slots){std::vector<uint8_t> b;require(reader(p.assets[slot],b,error),error);auto w=std::make_unique<Device>(b.size());w->put(b.data());pointers.push_back(reinterpret_cast<unsigned long long>(w->p));weights.push_back(std::move(w));}
    const int32_t count=t.experts.size();x.put(q.activations.data());ptr.put(pointers.data());starts.put(t.start.data());ng.put(&count);tok.put(t.tok.data());dst.put(t.dst.data());
    quantize_q8_1_rows(static_cast<float*>(x.p),q.T,H,xq.p,nullptr);
    native_expert_grouped(layout,static_cast<unsigned long long*>(ptr.p),static_cast<int32_t*>(starts.p),static_cast<int32_t*>(ng.p),static_cast<int32_t*>(dst.p),static_cast<int32_t*>(tok.p),count,n,xq.p,scratch.p,static_cast<float*>(out.p),nullptr);
    cu(cudaGetLastError());cu(cudaDeviceSynchronize());xq.dump(prefix+".input.q8_1");out.dump(prefix+".output.f32");
    for(const auto& item:{std::pair<const char*,size_t>{".gate.f32",0},{".up.f32",fa},{".hidden.q8_1",3*fa}}){const size_t bytes=std::string(item.first)==".hidden.q8_1"?n*(F/32)*36:n*F*4;std::vector<uint8_t> b(bytes);cu(cudaMemcpy(b.data(),static_cast<uint8_t*>(scratch.p)+item.second,bytes,cudaMemcpyDeviceToHost));write(prefix+item.first,b.data(),bytes);}
    std::cout<<"native_device="<<props.name<<" branch=native_gu_kernel<8>/swiglu_q8_1_entries_kernel/native_down_kernel<8>\n";
#else
    require(device<ggml_backend_vk_get_device_count(),"Vulkan device absent");
    Backend backend(ggml_backend_vk_init(device),ggml_backend_free);require(bool(backend),"Vulkan initialization failed");
    ggml_backend_dev_props props{};ggml_backend_dev_get_props(ggml_backend_get_device(backend.get()),&props);
    require(props.description && std::string(props.description).find(expected)!=std::string::npos,"unexpected Vulkan device");
    std::cout<<"vulkan_device="<<props.description<<" device_id="<<(props.device_id?props.device_id:"unknown")<<"\n";
    { std::ofstream device_file(prefix+".device");device_file<<"description="<<props.description<<"\ndevice_id="<<(props.device_id?props.device_id:"unknown")<<"\n";
      require(bool(device_file),"device evidence write failed"); }
    // Refuse a weights-only budget before any tensor workspace can be owned.
    auto small=create_vulkan_native_q8_observation(backend.get(),p.assigned_bytes(),prefix+".budget-reject");
    GpuCounters rejected;std::string budget_error;
    require(!small->prepare_source(p,reader,rejected,budget_error) && budget_error.find("budget")!=std::string::npos &&
        !rejected.gpu_allocated_bytes && !rejected.gpu_weight_bytes && !rejected.gpu_scratch_bytes && !rejected.gpu_allocation_count &&
        !vulkan_native_q8_allocation_snapshot(small.get(),rejected),"native Q8 weights-only budget refusal/rollback failed");
    small.reset();
    auto executor=create_vulkan_native_q8_observation(backend.get(),budget,prefix,prefix,timing);GpuCounters counters;EntryRowsReply reply;
    require(executor->prepare_source(p,reader,counters,error),error);
    if(timing) {
        auto infer=[&](EntryRowsReply& rows){const bool ok=executor->infer_rows(p,q,rows,error);++q.sequence;return ok;};
        auto snapshot=[&]{GpuCounters c;require(vulkan_native_q8_allocation_snapshot(executor.get(),c),"native Q8 allocation snapshot unavailable");return c;};
        require(native_q8_same_allocations(counters,snapshot()),"native Q8 prepared allocation snapshot differs");
        const auto result=native_q8_time_replay(infer,snapshot,deadline,error);reply=result.first_reply;
        std::cout<<std::setprecision(17)<<"timing_observation=true qualified=false quantizer_qualified=false metric=fenced_host_call_ms warmups=2 timed_calls=16 overall_deadline_seconds=60 capture_io_timed=false repeated_output_bits_stable=true owned_allocations_stable=true"
            <<" min_ms="<<result.minimum_ms<<" median_ms="<<result.median_ms<<" max_ms="<<result.maximum_ms
            <<" tensor_bytes="<<result.allocations.gpu_allocated_bytes<<" weight_bytes="<<result.allocations.gpu_weight_bytes
            <<" scratch_bytes="<<result.allocations.gpu_scratch_bytes<<" allocation_count="<<result.allocations.gpu_allocation_count<<" samples_ms=";
        for(size_t i=0;i<result.samples_ms.size();++i)std::cout<<(i?",":"")<<result.samples_ms[i];
        std::cout<<"\n";
    } else require(executor->infer_rows(p,q,reply,error),error);
    q.sequence=reply.sequence;
    std::vector<uint8_t> canonical,reference;
    require(gpu_codec::encode_request(p,q,canonical),"cannot encode retained request");
    require(gpu_codec::encode_rows_reply(p,blob_digest(canonical),q,reply,reference),"cannot bind direct rows");
    write(prefix+".direct.reply",reference.data(),reference.size());
    write(prefix+".canonical.request",canonical.data(),canonical.size());
    std::vector<uint8_t> encoded_plan;require(encode_gpu_plan(p,encoded_plan,error),error);
    write(prefix+".plan",encoded_plan.data(),encoded_plan.size());
    size_t at=0;for(size_t j=0;j<q.mask.size();++j)if(q.mask[j])require(at<reply.indices.size() && reply.indices[at++]==j,"restored reply association mismatch");require(at==reply.indices.size(),"extra reply rows");
    std::vector<float> packed_output(t.original.size()*H);
    std::ifstream output(prefix+".output.f32",std::ios::binary);
    require(bool(output.read(reinterpret_cast<char*>(packed_output.data()),packed_output.size()*4)) && output.peek()==EOF,
        "native Q8 captured down output shape/read failed");
    native_q8_check_restored_rows(reply,t,packed_output);
    GpuCounters actual;require(vulkan_native_q8_allocation_snapshot(executor.get(),actual) && native_q8_same_allocations(counters,actual),
        "native Q8 final actual allocation snapshot differs");
    std::cout<<"cache_accounting_observation=true qualified=false weights_only_budget_rejected=true restored_row_bits_checked=true"
        <<" tensor_bytes="<<actual.gpu_allocated_bytes<<" weight_bytes="<<actual.gpu_weight_bytes
        <<" scratch_bytes="<<actual.gpu_scratch_bytes<<" allocation_count="<<actual.gpu_allocation_count
        <<" host_staging_peak="<<counters.host_staging_peak<<"\n";
    executor->release(counters);GpuCounters after_release;
    require(!counters.gpu_allocated_bytes && !counters.gpu_weight_bytes && !counters.gpu_scratch_bytes && !counters.gpu_allocation_count &&
        !vulkan_native_q8_allocation_snapshot(executor.get(),after_release),"native Q8 release left owned allocations");
    std::cout<<"cache_release_observation=true owned_tensor_counters_zero=true backend_still_borrowed=true\n";
#endif
    std::ofstream meta(prefix+".metadata");native_q8_metadata_header(meta,ordinal,p,q);
    for(const auto& d:{blob_digest(q.activations.data(),q.activations.size()*4),blob_digest(q.ids.data(),q.ids.size()*sizeof(q.ids[0])),blob_digest(q.weights.data(),q.weights.size()*4),blob_digest(q.mask)}){meta<<hexid(d.a)<<hexid(d.b)<<' ';}meta<<'\n';
    for(const auto* a:{&t.experts,&t.asset_slots,&t.start,&t.dst,&t.tok,&t.original}){for(auto v:*a)meta<<v<<' ';meta<<'\n';}require(bool(meta),"metadata write failed");
#ifndef STRATA_Q8_REPLAY_CUDA
    const std::string direct_pass="direct replay complete\n";write((fs::path(argv[9])/"DIRECT_PASS").string(),direct_pass.data(),direct_pass.size());
#endif
    std::cout<<"one_layer_observation=true qualified=false quantizer_qualified=false record="<<ordinal<<" groups="<<t.experts.size()<<" entries="<<t.dst.size()<<'\n';return 0;
#endif
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
