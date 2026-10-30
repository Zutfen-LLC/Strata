// Include the unchanged production TU so runtime selectors and supplementary
// native SwiGLU are observable without adding a product API or rewriting kernels.
#include "../../src/kernels/cuda/iq_kernels.cu"
#include "native_q8_geometry_fixture.hpp"
namespace geo=native_q8_fixture::geometry;
using namespace native_q8_fixture;
namespace {
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct GeometryDevice {
    void* base=nullptr;void* p=nullptr;size_t size;
    explicit GeometryDevice(size_t n):size(n){checked(cudaMalloc(&base,n+32));p=static_cast<uint8_t*>(base)+16;checked(cudaMemset(base,0xa5,n+32));}
    ~GeometryDevice(){if(base)cudaFree(base);}GeometryDevice(const GeometryDevice&)=delete;
    void put(const void* src){checked(cudaMemcpy(p,src,size,cudaMemcpyHostToDevice));}
    Bytes get() const {Bytes out(size);checked(cudaMemcpy(out.data(),p,size,cudaMemcpyDeviceToHost));return out;}
    bool canaries() const {uint8_t ends[32];checked(cudaMemcpy(ends,base,16,cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(ends+16,static_cast<const uint8_t*>(p)+size,16,cudaMemcpyDeviceToHost));
        return std::all_of(ends,ends+32,[](uint8_t b){return b==0xa5;});}
};
}
int main(int argc,char** argv){try {
    need(argc==9 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--fixture" && std::string(argv[7])=="--output","geometry: use tools/native_q8_geometry.py");
    const auto input=geo::read_fixture(argv[6]);const auto weights=geo::weights();
    const std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"geometry: invalid device");
    const int device=std::stoi(index);checked(cudaSetDevice(device));cudaDeviceProp props{};checked(cudaGetDeviceProperties(&props,device));
    need(props.major==8 && props.minor==6,"geometry: this bounded native control requires sm_86");
    need(std::string(argv[4]).size() && std::string(props.name).find(argv[4])!=std::string::npos,"geometry: unexpected device");
    using namespace strata::kernels;
    // Inspect the actual variables/functions used by native_expert_grouped and
    // launch_gu/down. Never set their values to force a convenient branch.
    need(split_off<8>() && !g_kq_split && !gu_split(8) && !g_grouped_v1 && g_exp_phase==0,
        "geometry: native selectors do not select the implemented per-entry/fused-activation branch; do not override them");
    need(!g_old_kernels,"geometry: retained default control must not force old kernels");
    const auto layout=native_expert_layout(8,8,geo::H,geo::F);
    need(layout.bytes==geo::WEIGHT_BYTES && layout.bytes==weights.size() && layout.down_off==geo::DOWN_OFFSET && layout.had_block==0,
        "geometry: native layout mismatch");
    need(!expdec_gu_ok(layout) && !expdec_down_ok(layout),"geometry: EXPDEC unexpectedly covers this pair");
    const size_t fa=(geo::FLOATS*4+255)&~size_t(255),scratch_bytes=native_expert_scratch_bytes(geo::ENTRIES,geo::F);
    need(3*fa+geo::HQ_BYTES<=scratch_bytes,"geometry: hq scratch interval");
    char pci[32]{};checked(cudaDeviceGetPCIBusId(pci,sizeof(pci),device));
    std::ofstream dispatch(std::string(argv[8])+".dispatch.json");need(bool(dispatch),"geometry: dispatch receipt create failed");
    dispatch << "{\"basis\":\"runtime native selectors in unchanged included production TU, not profiler trace\","
        "\"h\":2560,\"f\":640,\"entries\":" << geo::ENTRIES << ",\"groups\":" << geo::GROUPS << ",\"grid_groups\":" << geo::GRID_GROUPS << ","
        "\"split_off_q8\":true,\"kq_split\":false,\"gu_split_q8\":false,\"grouped_v1\":false,\"old_kernels\":false,\"exp_phase\":0,"
        "\"gu_swiglu_requested\":" << (gu_swiglu_on()?"true":"false") << ","
        "\"gate_up\":\"native_gu_kernel<8>\",\"activation\":\"swiglu_q8_1_entries_kernel\",\"down\":\"native_down_kernel<8>\","
        "\"supplementary_hidden\":\"swiglu_entries_kernel; not consumed by down\"}\n";
    need(bool(dispatch),"geometry: dispatch receipt write failed");dispatch.close();
    GeometryDevice w(weights.size()),i(input.size()*4),x(geo::INPUT_BYTES),ptr(geo::GROUPS*8),starts(geo::STARTS.size()*4),
        groups(4),tok(geo::ENTRIES*4),dst(geo::ENTRIES*4),scratch(scratch_bytes),out(geo::OUTPUT_FLOATS*4),hidden(geo::FLOATS*4);
    const std::vector<unsigned long long> pointers(geo::GROUPS,reinterpret_cast<unsigned long long>(w.p));
    const int32_t ng=geo::GROUPS;
    w.put(weights.data());i.put(input.data());ptr.put(pointers.data());starts.put(geo::STARTS.data());groups.put(&ng);
    tok.put(geo::TOKENS_BY_ENTRY.data());dst.put(geo::DESTINATIONS.data());
    quantize_q8_1_rows((const float*)i.p,geo::TOKENS,geo::H,x.p,nullptr);
    native_expert_grouped(layout,(const unsigned long long*)ptr.p,(const int32_t*)starts.p,(const int32_t*)groups.p,
        (const int32_t*)dst.p,(const int32_t*)tok.p,geo::GROUPS,geo::ENTRIES,x.p,scratch.p,(float*)out.p,nullptr,geo::GRID_GROUPS);
    checked(cudaGetLastError());checked(cudaDeviceSynchronize());
    const Bytes packed=x.get();
    // Fused native activation does not materialize h. Run its actual separate
    // kernel only as an additional float diagnostic, after final output exists.
    swiglu_entries_kernel<<<(geo::FLOATS+255)/256,256>>>((const float*)scratch.p,
        (const float*)(static_cast<const uint8_t*>(scratch.p)+fa),(float*)hidden.p,geo::FLOATS);
    checked(cudaGetLastError());checked(cudaDeviceSynchronize());
    std::array<Bytes,6> result;result[0]=packed;result[1].resize(geo::FLOATS*4);result[2].resize(geo::FLOATS*4);result[4].resize(geo::HQ_BYTES);
    checked(cudaMemcpy(result[1].data(),scratch.p,result[1].size(),cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(result[2].data(),static_cast<const uint8_t*>(scratch.p)+fa,result[2].size(),cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(result[4].data(),static_cast<const uint8_t*>(scratch.p)+3*fa,result[4].size(),cudaMemcpyDeviceToHost));
    result[3]=hidden.get();result[5]=out.get();
    bool sentinels=true;for(const auto* allocation:{&w,&i,&x,&ptr,&starts,&groups,&tok,&dst,&scratch,&out,&hidden})sentinels&=allocation->canaries();
    sentinels&=w.get()==weights && i.get()==geo::bytes(input) && x.get()==packed;
    for(size_t e=0;e<geo::ENTRIES;++e)if(e<size_t(geo::STARTS.front()) || e>=size_t(geo::STARTS.back())) {
        for(auto item:{std::pair<size_t,size_t>{1,geo::F*4},{2,geo::F*4},{4,(geo::F/32)*36},{5,geo::H*4}})
            for(size_t b=e*item.second;b<(e+1)*item.second;++b)sentinels&=result[item.first][b]==0xa5;
    }
    geo::write_result(argv[8],false,sentinels,result);
    std::ofstream weights_out(std::string(argv[8])+".weights.bin",std::ios::binary);need(bool(weights_out),"geometry: weights receipt create failed");
    weights_out.write(reinterpret_cast<const char*>(weights.data()),weights.size());need(bool(weights_out),"geometry: weights receipt write failed");
    printf("native_model_geometry=true qualified=false quantizer_qualified=false device=%s pci=%s sm=%d%d sentinels=%d\n",props.name,pci,props.major,props.minor,int(sentinels));
    need(sentinels,"geometry: sentinel/input mutation; capture retained");return 0;
}catch(const std::exception& e){fprintf(stderr,"native geometry: %s\n",e.what());return 1;}}
