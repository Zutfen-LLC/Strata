#include "native_gdn_input_fixture.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <memory>
using namespace native_gdn_input_fixture;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Stream{cudaStream_t p=nullptr;Stream(){cu(cudaStreamCreate(&p));}~Stream(){if(p)cudaStreamDestroy(p);}Stream(const Stream&)=delete;};
struct Device{void*p=nullptr;explicit Device(size_t n){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
int main(int argc,char**argv){try {
    if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
    need(argc==11&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--weights"&&std::string(argv[7])=="--fixture"&&std::string(argv[9])=="--output","GDN input: invalid native arguments");
    auto [input,mode]=read_input(argv[8]);std::array<Bytes,2> w;for(size_t i=0;i<2;++i)w[i]=read_weights(std::filesystem::path(argv[6])/(std::string(names[i])+".bin"),i);
    std::string idx=argv[2];need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"GDN input: invalid device");
    cu(cudaSetDevice(std::stoi(idx)));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,std::stoi(idx)));
    need(std::string(argv[4]).size()&&std::string(prop.name).find(argv[4])!=std::string::npos&&prop.major==8&&prop.minor==6,"GDN input: unexpected native device");
    std::filesystem::path dir=argv[10];need(std::filesystem::create_directory(dir),"GDN input: output exists");
    constexpr uint32_t guard=0xa5a5a5a5;std::vector<uint32_t> x(q::elements+16,guard),packed(q::bytes/4+16,guard),stale(q::bytes/4+16,guard);
    std::memcpy(x.data()+8,input.data(),q::elements*4);std::fill(stale.begin()+8,stale.end()-8,0u);
    Device dx(x.size()*4),dq(packed.size()*4),ds(stale.size()*4);Stream stream;
    std::array<std::unique_ptr<Device>,2> dw,dy;std::array<std::vector<uint32_t>,2> wi,yi;
    for(size_t i=0;i<2;++i){wi[i].assign(w[i].size()/4+16,guard);std::memcpy(wi[i].data()+8,w[i].data(),w[i].size());yi[i].assign(rows[i]+16,guard);dw[i]=std::make_unique<Device>(wi[i].size()*4);dy[i]=std::make_unique<Device>(yi[i].size()*4);}
    Bytes first_q;std::array<Bytes,2> first_y;
    for(int pass=0;pass<2;++pass){
        cu(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(dq.p,packed.data(),packed.size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(ds.p,stale.data(),stale.size()*4,cudaMemcpyHostToDevice));
        for(size_t i=0;i<2;++i){cu(cudaMemcpy(dw[i]->p,wi[i].data(),wi[i].size()*4,cudaMemcpyHostToDevice));cu(cudaMemcpy(dy[i]->p,yi[i].data(),yi[i].size()*4,cudaMemcpyHostToDevice));}
        bool refused=false;try{strata::kernels::native_quantize_q8_1((float*)dx.p+8,(uint32_t*)dq.p+8,q::elements,1,nullptr);}catch(...){refused=true;}need(refused,"GDN input: null stream accepted");
        refused=false;try{strata::kernels::native_mmvq(-1,(uint32_t*)dw[0]->p+8,(uint32_t*)dq.p+8,(float*)((uint32_t*)dy[0]->p+8),q::elements,int(rows[0]),1,stream.p);}catch(...){refused=true;}need(refused,"GDN input: invalid MMVQ type accepted");
        std::vector<uint32_t>before_q(packed.size()),before_y(yi[0].size());cu(cudaMemcpy(before_q.data(),dq.p,before_q.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(before_y.data(),dy[0]->p,before_y.size()*4,cudaMemcpyDeviceToHost));need(before_q==packed&&before_y==yi[0],"GDN input: invalid call committed");
        strata::kernels::native_quantize_q8_1((float*)dx.p+8,(uint32_t*)dq.p+8,q::elements,1,stream.p);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
        std::vector<uint32_t> qread(packed.size());cu(cudaMemcpy(qread.data(),dq.p,qread.size()*4,cudaMemcpyDeviceToHost));Bytes packed_bytes(q::bytes);std::memcpy(packed_bytes.data(),qread.data()+8,q::bytes);pinned(packed_bytes,mode);packed_domain(packed_bytes);
        for(size_t i=0;i<8;++i)need(qread[i]==guard&&qread[qread.size()-1-i]==guard,"GDN input: packed canary");
        std::array<Bytes,2> output;
        for(size_t i=0;i<2;++i){
            strata::kernels::native_mmvq(8,(uint32_t*)dw[i]->p+8,(uint32_t*)dq.p+8,(float*)((uint32_t*)dy[i]->p+8),q::elements,int(rows[i]),1,stream.p);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
            std::vector<uint32_t> result(yi[i].size()),input_after(x.size()),weight_after(wi[i].size()),packed_after(qread.size());
            cu(cudaMemcpy(result.data(),dy[i]->p,result.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(input_after.data(),dx.p,x.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(weight_after.data(),dw[i]->p,wi[i].size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(packed_after.data(),dq.p,qread.size()*4,cudaMemcpyDeviceToHost));
            need(input_after==x&&weight_after==wi[i]&&packed_after==qread,"GDN input: input changed");for(size_t j=0;j<8;++j)need(result[j]==guard&&result[result.size()-1-j]==guard,"GDN input: output canary");
            output[i].resize(rows[i]*4);std::memcpy(output[i].data(),result.data()+8,output[i].size());
            cu(cudaMemcpy(dy[i]->p,yi[i].data(),yi[i].size()*4,cudaMemcpyHostToDevice));
            strata::kernels::native_mmvq(8,(uint32_t*)dw[i]->p+8,(uint32_t*)ds.p+8,(float*)((uint32_t*)dy[i]->p+8),q::elements,int(rows[i]),1,stream.p);cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
            cu(cudaMemcpy(result.data(),dy[i]->p,result.size()*4,cudaMemcpyDeviceToHost));need(std::memcmp(result.data()+8,output[i].data(),output[i].size())!=0,"GDN input: stale input not detected");
            for(size_t j=0;j<8;++j)need(result[j]==guard&&result[result.size()-1-j]==guard,"GDN input: stale output canary");
        }
        std::vector<uint32_t> stale_after(stale.size());cu(cudaMemcpy(stale_after.data(),ds.p,stale_after.size()*4,cudaMemcpyDeviceToHost));need(stale_after==stale,"GDN input: stale input changed");
        output_controls(output,mode);if(pass)need(packed_bytes==first_q&&output==first_y,"GDN input: repeat changed");first_q=packed_bytes;first_y=output;
        save(dir/(pass?"qx-repeat.bin":"qx.bin"),packed_bytes.data(),packed_bytes.size());for(size_t i=0;i<2;++i)save(dir/(std::string(names[i])+(pass?"-repeat.f32le":".f32le")),output[i].data(),output[i].size());
    }
    std::ofstream(dir/"report.json")<<"{\"actual_native_quantizer_and_mmvq\":true,\"one_packed_input\":true,\"stale_changes_output\":true,\"invalid_noncommit_retry\":true,\"immutable_canary_repeat\":true,\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
