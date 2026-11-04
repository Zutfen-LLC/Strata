#include "portable_native_hc_test_backend.hpp"
#include "portable_ple_fixture.hpp"
#include <iostream>
int main(int argc, char** argv) { try {
    check(argc == 2, "scratch root required");
    ple_fixture::Fixture f; std::string error;
    {
        PortableModelOptions model;model.native_ple_key_t1=true;model.native_ple_key_receipt_prefix="ple-key";
        model.gdn_resident_weights=true;model.allow_cpu_fallback=false;
        validate_model_modes(model,1);auto off=model;off.native_ple_key_t1=false;
        for(size_t l=0;l<48;++l){PortableGdnOptions per;model_gdn_hc_options(model,l,per);
            check(per.native_ple_key_t1==(l==1),"PLE key leaked to another layer");
            check(model_hc_host_bound(model,l)-model_hc_host_bound(off,l)==(l==1?PortableGdnOptions::native_ple_key_host_bound:0),"model host admission differs");}
        auto bad=model;bad.allow_cpu_fallback=true;bool rejected=false;
        try{validate_model_modes(bad,1);}catch(const std::exception&){rejected=true;}
        check(rejected,"model accepted fallback for PLE key");
    }
    auto table = ple_fixture::small_table(error); check(bool(table), error);
    auto root = std::filesystem::path(argv[1]) / ("ple-key-integration-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(root), "fresh test root");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(cpu), "host provider"); original_compute = cpu->iface.graph_compute; cpu->iface.graph_compute = check_compute;
    PortableGdnOptions options; options.execution.threads=1; options.execution.accelerator=cpu.get();
    options.execution.request_accelerator=true; options.execution.allow_cpu_fallback=false;
    options.resident_attention_weights=true; options.ple_table=table;
    auto baseline=PortableGdnLayer::create(1,f.views,options,error); check(bool(baseline),error);
    options.native_ple_key_t1=true; options.native_ple_key_receipt=(root/"candidate").string();
    options.state_budget+=PortableGdnOptions::native_ple_key_host_bound;
    { auto bad=options;bad.execution.allow_cpu_fallback=true;check(!PortableGdnLayer::create(1,f.views,bad,error),"fallback accepted"); }
    { auto bad=options;bad.state_budget=baseline->state_bytes()+PortableGdnOptions::native_ple_key_host_bound-1;
      check(!PortableGdnLayer::create(1,f.views,bad,error),"short host admission accepted"); }
    observe_ple_key=true;
    auto layer=PortableGdnLayer::create(1,f.views,options,error);check(bool(layer),error);
    check(layer->weight_bytes()==baseline->weight_bytes(),"resident Q8 role was duplicated");
    check(layer->reset(error),error);PortableRoute route;auto before=layer->ple_convolution_history();
    fail_ple_key=true;check(!layer->step(19,f.input(0),ple_fixture::experts(),route,error),"failed projection accepted");
    check(layer->tokens()==0&&layer->native_ple_key_calls()==0&&layer->ple_convolution_history()==before&&layer->token_history()==std::array<int32_t,2>{{-1,-1}},"projection failure committed history");
    fail_ple_key=false;PortableGdnTrace first;
    check(layer->step(19,f.input(0),ple_fixture::experts(),route,error,&first),error);
    check(actual_ple_key_input&&actual_ple_key_output&&ple_key_links>0,"graph did not consume actual PLE key producer bank");
    check(layer->native_ple_key_calls()==1,"successful native key call not counted");
    check(get(actual_ple_key_input)==first.ple_embedding,"actual quantizer input differs from gathered PLE embedding");
    ple_fixture::reference(f,f.input(0),before,first,*layer);
    auto residual=layer->last_residual();auto history=layer->ple_convolution_history();auto tokens=layer->token_history();
    PortableExpertRows refuse=[](int64_t,const PortableRoute&,std::vector<float>&,std::string&e){e="intentional refusal";return false;};
    check(!layer->step(20,f.input(1),refuse,route,error),"FFN failure accepted");
    check(layer->tokens()==1&&layer->ple_convolution_history()==history&&layer->token_history()==tokens,"FFN failure committed PLE history");
    PortableGdnTrace second;check(layer->step(20,f.input(1),ple_fixture::experts(),route,error,&second),error);
    check(layer->token_history()==std::array<int32_t,2>{{19,20}},"retry history differs");
    ple_fixture::reference(f,f.input(1),history,second,*layer);
    check(layer->reset(error),error);check(layer->step(19,f.input(0),ple_fixture::experts(),route,error),error);
    check(layer->last_residual()==residual&&layer->ple_convolution_history()==history,"reset result differs");
    std::puts("PLE key graph ownership, native-format reference, failure noncommit, retry and history reset passed with host providers");return 0;
} catch(const std::exception&e) { std::cerr<<e.what()<<'\n';return 1; } }
