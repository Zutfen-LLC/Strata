// Native-consumer-only probe. No quantizer call and no new GPU kernel.
#include "native_q8_consumer_fixture.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
using namespace native_q8_fixture;
namespace c=native_q8_fixture::consumer;
namespace {
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
    void* p=nullptr;explicit Device(size_t bytes){checked(cudaMalloc(&p,bytes));}
    ~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;
};
void upload(void* d,const void* s,size_t n){checked(cudaMemcpy(d,s,n,cudaMemcpyHostToDevice));}
void read(void* d,const void* s,size_t n){checked(cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost));}
void save(std::ostream& out,const std::vector<float>& v){word(out,uint32_t(v.size()));for(float x:v)word(out,bits(x));}
size_t compare(const char* name,const std::vector<float>& v,size_t per_entry) {
    need(v.size()==per_entry*2,"consumer: output shape");size_t mismatches=0;bool signal=false;
    for(size_t i=0;i<per_entry;++i){need(std::isfinite(v[i]) && std::isfinite(v[per_entry+i]),"consumer: nonfinite native result");
        signal|=v[i]!=0.f;
        if(bits(v[i])!=bits(v[per_entry+i])){++mismatches;
            std::printf("stage=%s index=%zu native_input_bits=%08x observation_input_bits=%08x\n",name,i,bits(v[i]),bits(v[per_entry+i]));}}
    need(signal,"consumer: common-signal output degenerated to zero");
    std::printf("stage=%s compared=%zu raw_bit_mismatches=%zu signed_zeros_included=true\n",name,per_entry,mismatches);
    return mismatches;
}
}
int main(int argc,char** argv){try {
    need(argc==11 && std::string(argv[1])=="--device" && std::string(argv[3])=="--expect-device" &&
        std::string(argv[5])=="--native" && std::string(argv[7])=="--observation" && std::string(argv[9])=="--output",
        "consumer: run through tools/native_q8_consumer.py for SHA256 validation and provenance");
    const auto native=c::read_capture(argv[6],false),observation=c::read_capture(argv[8],true);
    c::validate_pair(native,observation);const auto weights=c::weights();
    std::string index=argv[2];need(!index.empty() && index.find_first_not_of("0123456789")==std::string::npos,"consumer: invalid device");
    {std::ofstream raw_weights(std::string(argv[10])+".weights.bin",std::ios::binary);
     need(bool(raw_weights),"consumer: cannot create weight receipt");
     raw_weights.write(reinterpret_cast<const char*>(weights.data()),weights.size());need(bool(raw_weights),"consumer: weight receipt write failed");}
    const int device=std::stoi(index);checked(cudaSetDevice(device));cudaDeviceProp props{};checked(cudaGetDeviceProperties(&props,device));
    need(std::string(argv[4]).size() && std::string(props.name).find(argv[4])!=std::string::npos,"consumer: unexpected device");
    char pci[32]{};checked(cudaDeviceGetPCIBusId(pci,sizeof(pci),device));
    using namespace strata::kernels;
    need(native_expert_supported(8,8,c::H,c::F),"consumer: native dimensions declined");
    const auto layout=native_expert_layout(8,8,c::H,c::F);
    need(layout.bytes==weights.size() && layout.gu_row==272 && layout.d_row==34 && layout.up_off==8704 &&
        layout.down_off==17408 && layout.tail_off==0 && layout.had_block==0,"consumer: native Q8 layout mismatch");
    const size_t scratch_bytes=native_expert_scratch_bytes(2,c::F),fa=(2*c::F*4+255)&~size_t(255);
    Device w(weights.size()),x(2*c::PREFIX_BYTES),pointer(8),start(8),groups(4),dst(8),tok(8),scratch(scratch_bytes),out(2*c::H*4);
    Bytes paired;paired.insert(paired.end(),native.input.begin(),native.input.begin()+c::PREFIX_BYTES);
    paired.insert(paired.end(),observation.input.begin(),observation.input.begin()+c::PREFIX_BYTES);
    const unsigned long long wp=reinterpret_cast<unsigned long long>(w.p);const int32_t starts[]={0,2},ng=1,ids[]={0,1};
    upload(w.p,weights.data(),weights.size());upload(x.p,paired.data(),paired.size());upload(pointer.p,&wp,8);
    upload(start.p,starts,8);upload(groups.p,&ng,4);upload(dst.p,ids,8);upload(tok.p,ids,8);
    checked(cudaMemset(scratch.p,0xff,scratch_bytes));checked(cudaMemset(out.p,0xff,2*c::H*4));
    // Unchanged public entry point receives CAPTURED Q8_1 bytes, without
    // requantization. One group, one finite expert, two paired entries.
    native_expert_grouped(layout,(const unsigned long long*)pointer.p,(const int32_t*)start.p,(const int32_t*)groups.p,
        (const int32_t*)dst.p,(const int32_t*)tok.p,1,2,x.p,scratch.p,(float*)out.p,nullptr);
    checked(cudaGetLastError());checked(cudaDeviceSynchronize());
    std::vector<float> gate(2*c::F),up(2*c::F),output(2*c::H);
    read(gate.data(),scratch.p,gate.size()*4);read(up.data(),static_cast<const uint8_t*>(scratch.p)+fa,up.size()*4);
    read(output.data(),out.p,output.size()*4);
    Bytes retained(paired.size());read(retained.data(),x.p,retained.size());need(retained==paired,"consumer: packed inputs changed");
    // Save every raw result before comparisons so a mismatch is not discarded.
    std::ofstream raw(argv[10],std::ios::binary);need(bool(raw),"consumer: cannot create raw receipt");
    word(raw,0x3143384e);save(raw,gate);save(raw,up);save(raw,output);need(bool(raw),"consumer: raw receipt write failed");raw.close();
    const size_t mismatches=compare("gate",gate,c::F)+compare("up",up,c::F)+compare("output",output,c::H);
    std::printf("native_consumer_only=true qualified=false observation_fp32_denorm_mode=unspecified original_byte_failure_preserved=true finite_synthetic_weights=true device=%s pci=%s entries=2 n_embd=256 n_ff=32 total_raw_bit_mismatches=%zu\n",props.name,pci,mismatches);
    return mismatches?1:0;
}catch(const std::exception& e){std::fprintf(stderr,"native consumer: %s\n",e.what());return 1;}}
