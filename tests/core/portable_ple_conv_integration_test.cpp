#include "portable_native_hc_test_backend.hpp"
#include "portable_ple_fixture.hpp"
#include <iostream>
int main(int argc,char** argv){try{
    check(argc==2,"scratch root required");
    ple_fixture::Fixture fixture;std::vector<uint16_t> role(40960);
    for(size_t i=0;i<role.size();++i)role[i]=ggml_fp32_to_fp16(fixture.f32(5,i));
    auto& view=fixture.views.back();view.type=1;view.data=role.data();view.bytes=role.size()*2;
    std::string error;auto table=ple_fixture::small_table(error);check(bool(table),error);
    auto root=std::filesystem::path(argv[1])/("ple-conv-integration-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(root),"fresh root");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    original_compute=cpu->iface.graph_compute;cpu->iface.graph_compute=check_compute;
    PortableGdnOptions options;options.execution.threads=1;options.execution.accelerator=cpu.get();
    options.execution.request_accelerator=true;options.execution.allow_cpu_fallback=false;
    options.resident_attention_weights=true;options.ple_table=table;options.native_ple_conv_t1=true;
    options.native_ple_conv_receipt=(root/"candidate").string();options.state_budget+=PortableGdnOptions::native_ple_conv_host_bound;
    auto layer=PortableGdnLayer::create(1,fixture.views,options,error);check(bool(layer),error);
    check(layer->reset(error),error);auto initial=layer->ple_convolution_history();PortableRoute route;
    fail_ple_conv=true;check(!layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),"component refusal accepted");
    check(layer->tokens()==0&&layer->ple_convolution_history()==initial,"component refusal committed history");
    fail_ple_conv=false;observe_ple_conv=true;PortableGdnTrace trace;
    check(layer->step(19,fixture.input(0),ple_fixture::experts(),route,error,&trace),error);
    check(ple_conv_links>0&&actual_ple_conv_result,"suffix did not consume actual conv result bank");
    auto history=layer->ple_convolution_history();auto residual=layer->last_residual();
    for(size_t c=0;c<10240;++c){check(history[c*9+8]==trace.ple_normalized[c],"history append did not use actual normalized input");
        for(size_t j=0;j<8;++j)check(history[c*9+j]==0,"first history shift differs");}
    const auto& input=fixture.input(0);for(size_t c: {size_t(0),size_t(17),size_t(10239)}){
        float pre=trace.ple_normalized[c]*ggml_fp16_to_fp32(role[c*4+3]);
        float conv=pre/(1.f+std::exp(-pre));float gated=trace.ple_value[c%2560]*trace.ple_gate[c/2560];
        check(std::fabs(trace.ple_residual[c]-(input[c]+(gated+conv)))<1e-6f,"native tap/order control differs");
    }
    PortableExpertRows refuse=[](int64_t,const PortableRoute&,std::vector<float>&,std::string& e){e="intentional FFN failure";return false;};
    check(!layer->step(20,fixture.input(1),refuse,route,error),"FFN refusal accepted");
    check(layer->tokens()==1&&layer->ple_convolution_history()==history,"FFN refusal committed candidate history");
    check(layer->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    check(layer->token_history()==std::array<int32_t,2>{{19,20}},"retry token history differs");
    check(layer->reset(error),error);check(layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(layer->last_residual()==residual&&layer->ple_convolution_history()==history,"reset did not restore first result");
    std::puts("PLE conv producer, candidate history, refusal, retry and reset passed with host provider");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
