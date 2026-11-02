#include "native_qsa_wo_compose_fixture.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
using namespace native_qsa_wo_compose;
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Stream{cudaStream_t p=nullptr;Stream(){cu(cudaStreamCreate(&p));}~Stream(){if(p)cudaStreamDestroy(p);}Stream(const Stream&)=delete;};
struct Device{void*p=nullptr;explicit Device(size_t n){cu(cudaMalloc(&p,n));}~Device(){if(p)cudaFree(p);}Device(const Device&)=delete;};
int main(int argc,char**argv){try{
    if(argc==5&&std::string(argv[1])=="--contract"){contract(argv[2],argv[3],argv[4]);return 0;}
    need(argc==11&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&
         std::string(argv[5])=="--weights"&&std::string(argv[7])=="--fixture"&&std::string(argv[9])=="--output","composition: invalid native arguments");
    auto ww=weights(argv[6]);auto input=q::read(argv[8]);std::string idx=argv[2];
    need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"composition: invalid device");
    cu(cudaSetDevice(std::stoi(idx)));cudaDeviceProp prop{};cu(cudaGetDeviceProperties(&prop,std::stoi(idx)));
    need(std::string(argv[4]).size()&&std::string(prop.name).find(argv[4])!=std::string::npos&&prop.major==8&&prop.minor==6,"composition: unexpected native device");
    std::filesystem::path dir=argv[10];need(std::filesystem::create_directory(dir),"composition: output exists");
    std::filesystem::copy_file(argv[8],dir/"fixture.bin");
    constexpr uint32_t guard=0xa5a5a5a5;
    std::vector<uint32_t> xi(q::elements+16,guard),wi(ww.size()/4+16,guard),qi(q::bytes/4+16,guard),yi(rows+16,guard),stale(q::bytes/4+16,guard);
    std::memcpy(xi.data()+8,input.data(),q::elements*4);std::memcpy(wi.data()+8,ww.data(),ww.size());
    std::fill(stale.begin()+8,stale.end()-8,0u);
    Device dx(xi.size()*4),dw(wi.size()*4),dq(qi.size()*4),dy(yi.size()*4),ds(stale.size()*4);Stream stream;
    Bytes first_q,first_y;
    for(int pass=0;pass<2;++pass){
        for(auto pair:{std::make_pair(dx.p,&xi),std::make_pair(dw.p,&wi),std::make_pair(dq.p,&qi),std::make_pair(dy.p,&yi),std::make_pair(ds.p,&stale)})
            cu(cudaMemcpy(pair.first,pair.second->data(),pair.second->size()*4,cudaMemcpyHostToDevice));
        bool refused=false;
        try { strata::kernels::native_quantize_q8_1((float*)dx.p+8,(uint32_t*)dq.p+8,q::elements,1,nullptr); }
        catch(const std::exception&) { refused=true; }
        need(refused,"composition: quantizer invalid stream accepted");
        refused=false;
        try { strata::kernels::native_mmvq(-1,(uint32_t*)dw.p+8,(uint32_t*)dq.p+8,(float*)((uint32_t*)dy.p+8),6144,int(rows),1,stream.p); }
        catch(const std::exception&) { refused=true; }
        need(refused,"composition: unsupported WO type accepted");
        std::vector<uint32_t> before_q(qi.size()),before_y(yi.size());
        cu(cudaMemcpy(before_q.data(),dq.p,qi.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(before_y.data(),dy.p,yi.size()*4,cudaMemcpyDeviceToHost));
        need(before_q==qi&&before_y==yi,"composition: invalid call committed");
        strata::kernels::native_quantize_q8_1((float*)dx.p+8,(uint32_t*)dq.p+8,q::elements,1,stream.p);
        cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
        std::vector<uint32_t> qread(qi.size());cu(cudaMemcpy(qread.data(),dq.p,qread.size()*4,cudaMemcpyDeviceToHost));
        Bytes packed(q::bytes);std::memcpy(packed.data(),qread.data()+8,packed.size());packed_domain(ww,packed);
        for(size_t i=0;i<8;++i)need(qread[i]==guard&&qread[qread.size()-1-i]==guard,"composition: packed canary changed");
        strata::kernels::native_mmvq(8,(uint32_t*)dw.p+8,(uint32_t*)dq.p+8,(float*)((uint32_t*)dy.p+8),6144,int(rows),1,stream.p);
        cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
        std::vector<uint32_t> yread(yi.size()),xread(xi.size()),wread(wi.size()),qafter(qi.size());
        cu(cudaMemcpy(yread.data(),dy.p,yread.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(xread.data(),dx.p,xread.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(wread.data(),dw.p,wread.size()*4,cudaMemcpyDeviceToHost));cu(cudaMemcpy(qafter.data(),dq.p,qafter.size()*4,cudaMemcpyDeviceToHost));
        need(xread==xi&&wread==wi&&qafter==qread,"composition: input, weights or producer image mutated");
        for(size_t i=0;i<8;++i)need(yread[i]==guard&&yread[yread.size()-1-i]==guard,"composition: output canary changed");
        Bytes result(rows*4);std::memcpy(result.data(),yread.data()+8,result.size());
        for(size_t i=0;i<rows;++i){float v;std::memcpy(&v,result.data()+i*4,4);(void)w::checked(v);}
        cu(cudaMemcpy(dy.p,yi.data(),yi.size()*4,cudaMemcpyHostToDevice));
        strata::kernels::native_mmvq(8,(uint32_t*)dw.p+8,(uint32_t*)ds.p+8,(float*)((uint32_t*)dy.p+8),6144,int(rows),1,stream.p);
        cu(cudaGetLastError());cu(cudaStreamSynchronize(stream.p));
        std::vector<uint32_t> bad(yi.size());cu(cudaMemcpy(bad.data(),dy.p,bad.size()*4,cudaMemcpyDeviceToHost));
        need(std::memcmp(bad.data()+8,result.data(),result.size())!=0,"composition: stale packed input did not change output");
        std::vector<uint32_t> stale_after(stale.size());
        cu(cudaMemcpy(stale_after.data(),ds.p,stale_after.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(xread.data(),dx.p,xread.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(wread.data(),dw.p,wread.size()*4,cudaMemcpyDeviceToHost));
        cu(cudaMemcpy(qafter.data(),dq.p,qafter.size()*4,cudaMemcpyDeviceToHost));
        need(stale_after==stale&&xread==xi&&wread==wi&&qafter==qread,"composition: stale test mutated inputs");
        for(size_t i=0;i<8;++i)need(bad[i]==guard&&bad[bad.size()-1-i]==guard,"composition: stale output canary changed");
        if(pass)need(packed==first_q&&result==first_y,"composition: reset/repeat changed");
        first_q=packed;first_y=result;
        save(dir/(pass?"qx-repeat.bin":"qx.bin"),packed.data(),packed.size());
        save(dir/(pass?"out-repeat.f32le":"out.f32le"),result.data(),result.size());
    }
    std::ofstream(dir/"report.json")<<"{\"actual_native_quantizer_and_mmvq\":true,\"same_stream\":true,\"producer_buffer_consumed\":true,\"stale_changes_output\":true,\"invalid_noncommit_retry\":true,\"immutable_canary_repeat\":true,\"qualified\":false}\n";
    return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
