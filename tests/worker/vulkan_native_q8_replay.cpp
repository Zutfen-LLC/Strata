// One explicitly selected retained request; never a model/token campaign.
#ifdef STRATA_Q8_REPLAY_CUDA
#include "../../src/kernels/cuda/iq_kernels.cu"
#else
#include "strata/worker/vulkan_executor.hpp"
#include "ggml-vulkan.h"
#endif
#include "vulkan_native_q8_layout.hpp"
#include "strata/worker/cpu_capture.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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
int main(int argc,char** argv){try {
    require(argc==10 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device",
        "usage: --device INDEX --expect-device NAME CAPTURE RECORD ONE_LAYER_PLAN BLOB_DIR NEW_OUTPUT");
    const std::string index=argv[2];require(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"invalid device index");
    const int device=std::stoi(index);const std::string expected=argv[4];require(!expected.empty(),"expected device required");
    CpuCaptureFile f;GpuPlan p;std::string error;require(read_cpu_capture(argv[5],f,error),error);require(read_gpu_plan(argv[7],p,error),error);
    const std::string record=argv[6];require(!record.empty() && record.find_first_not_of("0123456789")==std::string::npos,"invalid record index");
    const size_t ordinal=std::stoull(record);require(ordinal<f.records.size() && f.records[ordinal].sent,"record absent/not sent");
    auto q=f.records[ordinal].request;
    require(p.model_id==f.setup.model_id && p.n_embd==2560 && p.n_ff==640 && p.gu_type==8 && p.d_type==8 &&
        p.n_layers==f.setup.n_layers && p.n_expert==f.setup.n_expert && p.assets.size()<=24,"capture/plan identity or native geometry mismatch");
    for(const auto& a:p.assets)require(a.layer==q.layer,"replay requires one-layer frozen plan");
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
    auto executor=create_vulkan_native_q8_observation(backend.get(),128u<<20,prefix,prefix);GpuCounters counters;EntryRowsReply reply;
    require(executor->prepare_source(p,reader,counters,error),error);require(executor->infer_rows(p,q,reply,error),error);
    size_t at=0;for(size_t j=0;j<q.mask.size();++j)if(q.mask[j])require(at<reply.indices.size() && reply.indices[at++]==j,"restored reply association mismatch");require(at==reply.indices.size(),"extra reply rows");
    std::cout<<"tensor_bytes="<<counters.gpu_allocated_bytes<<" scratch_bytes="<<counters.gpu_scratch_bytes<<" host_staging_peak="<<counters.host_staging_peak<<"\n";
    executor->release(counters);
#endif
    std::ofstream meta(prefix+".metadata");native_q8_metadata_header(meta,ordinal,p,q);
    for(const auto& d:{blob_digest(q.activations.data(),q.activations.size()*4),blob_digest(q.ids.data(),q.ids.size()*sizeof(q.ids[0])),blob_digest(q.weights.data(),q.weights.size()*4),blob_digest(q.mask)}){meta<<hexid(d.a)<<hexid(d.b)<<' ';}meta<<'\n';
    for(const auto* a:{&t.experts,&t.asset_slots,&t.start,&t.dst,&t.tok,&t.original}){for(auto v:*a)meta<<v<<' ';meta<<'\n';}require(bool(meta),"metadata write failed");
    std::cout<<"one_layer_observation=true qualified=false quantizer_qualified=false record="<<ordinal<<" groups="<<t.experts.size()<<" entries="<<t.dst.size()<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
