#include "portable_native_hc_test_backend.hpp"
#include "portable_ple_fixture.hpp"
#include "portable_native_ple_gate.hpp"
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
        auto gate_model=model;gate_model.native_ple_gate_t1=true;gate_model.native_ple_gate_receipt_prefix="ple-gate";validate_model_modes(gate_model,1);
        for(size_t l=0;l<48;++l){PortableGdnOptions per;model_gdn_hc_options(gate_model,l,per);check(per.native_ple_gate_t1==(l==1),"PLE gate leaked to another layer");check(model_hc_host_bound(gate_model,l)-model_hc_host_bound(model,l)==(l==1?PortableGdnOptions::native_ple_gate_host_bound:0),"PLE gate model admission differs");}
        {auto invalid=gate_model;invalid.gdn_causal_chunk=true;bool rejected=false;try{validate_model_modes(invalid,1);}catch(const std::exception&){rejected=true;}check(rejected,"PLE gate accepted chunk mode");}
        model.native_ple_value_t1=true;model.native_ple_value_receipt_prefix="ple-value";
        validate_model_modes(model,1);PortableGdnOptions both;model_gdn_hc_options(model,1,both);
        check(both.native_ple_key_t1&&both.native_ple_value_t1&&
              model_hc_host_bound(model,1)-model_hc_host_bound(off,1)==PortableGdnOptions::native_ple_key_host_bound+PortableGdnOptions::native_ple_value_host_bound,
              "combined PLE model options or admission differ");
    }
    auto table = ple_fixture::small_table(error); check(bool(table), error);
    auto root = std::filesystem::path(argv[1]) / ("ple-key-integration-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(root), "fresh test root");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(cpu), "host provider"); original_compute = cpu->iface.graph_compute; cpu->iface.graph_compute = check_compute;
    {const size_t bound=PortableNativePleGate::estimate(cpu.get());bool refused=false;
     try{PortableNativePleGate short_budget(cpu.get(),(root/"gate-short").string(),bound-1);}catch(const std::exception&){refused=true;}
     check(refused&&!std::filesystem::exists(root/"gate-short"),"PLE gate accepted short device admission");
     PortableNativePleGate exact_budget(cpu.get(),(root/"gate-exact").string(),bound);check(exact_budget.bytes()<=bound,"PLE gate exact device admission failed");}
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
    check(std::filesystem::file_size(root/"candidate/first-candidate/input.f32le")==10240 &&
          std::filesystem::file_size(root/"candidate/first-candidate/qx.bin")==2880 &&
          std::filesystem::file_size(root/"candidate/first-candidate/key.f32le")==40960,
          "graph-completed actual key capture missing");
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
    layer.reset();baseline.reset();
    options.native_ple_key_receipt=(root/"combined-key").string();
    options.native_ple_value_t1=true;options.native_ple_value_receipt=(root/"combined-value").string();
    options.state_budget+=PortableGdnOptions::native_ple_value_host_bound;
    observe_ple_value=true;auto combined=PortableGdnLayer::create(1,f.views,options,error);check(bool(combined),error);
    check(combined->reset(error),error);auto initial=combined->ple_convolution_history();
    fail_ple_value=true;check(!combined->step(19,f.input(0),ple_fixture::experts(),route,error),"value failure accepted after key");
    check(combined->tokens()==0&&combined->ple_convolution_history()==initial&&combined->token_history()==std::array<int32_t,2>{{-1,-1}},"value failure committed combined PLE history");
    fail_ple_value=false;PortableGdnTrace both;check(combined->step(19,f.input(0),ple_fixture::experts(),route,error,&both),error);
    check(ple_key_links>0&&ple_value_links>0&&actual_ple_key_input&&actual_ple_value_input&&get(actual_ple_key_input)==both.ple_embedding&&get(actual_ple_value_input)==both.ple_embedding,"combined key/value producer or gathered input differs");
    ple_fixture::reference(f,f.input(0),initial,both,*combined);
    const auto combined_residual=combined->last_residual(),combined_history=combined->ple_convolution_history();
    check(!combined->step(20,f.input(1),refuse,route,error),"combined FFN failure accepted");
    check(combined->tokens()==1&&combined->ple_convolution_history()==combined_history&&combined->token_history()==std::array<int32_t,2>{{-1,19}},"combined FFN failure committed history");
    check(combined->reset(error),error);check(combined->step(19,f.input(0),ple_fixture::experts(),route,error),error);
    check(combined->last_residual()==combined_residual&&combined->ple_convolution_history()==combined_history,"combined reset differs");
    combined.reset();observe_ple_key=false;observe_ple_value=false;actual_ple_key_output=nullptr;ple_value_bank=nullptr;
    options.native_ple_key_t1=false;options.native_ple_value_t1=false;options.native_ple_gate_t1=true;
    options.native_ple_gate_receipt=(root/"gate").string();options.state_budget+=PortableGdnOptions::native_ple_gate_host_bound;
    {auto bad=options;bad.execution.allow_cpu_fallback=true;check(!PortableGdnLayer::create(1,f.views,bad,error),"PLE gate accepted CPU fallback");}
    observe_ple_gate=true;auto gate_layer=PortableGdnLayer::create(1,f.views,options,error);check(bool(gate_layer),error);check(gate_layer->reset(error),error);
    auto initial_gate_history=gate_layer->ple_convolution_history();fail_ple_gate=true;
    check(!gate_layer->step(19,f.input(0),ple_fixture::experts(),route,error),"gate domain failure accepted");
    check(gate_layer->tokens()==0&&gate_layer->native_ple_gate_calls()==0&&gate_layer->ple_convolution_history()==initial_gate_history&&gate_layer->token_history()==std::array<int32_t,2>{{-1,-1}}&&!std::filesystem::exists(root/"gate/first-candidate"),"gate failure committed state or receipt");
    fail_ple_gate=false;PortableGdnTrace gate_trace;check(gate_layer->step(19,f.input(0),ple_fixture::experts(),route,error,&gate_trace),error);
    check(ple_gate_links>0&&actual_ple_gate_bank&&gate_layer->native_ple_gate_calls()==1,"value broadcast ignored actual gate bank");
    check(std::filesystem::file_size(root/"gate/first-candidate/key.f32le")==40960&&std::filesystem::file_size(root/"gate/first-candidate/query.f32le")==40960&&std::filesystem::file_size(root/"gate/first-candidate/gate.f32le")==16,"PLE gate candidate receipt missing");
    auto read_capture=[&](const char* name,size_t count){std::vector<float> data(count);std::ifstream file(root/"gate/first-candidate"/name,std::ios::binary);file.read(reinterpret_cast<char*>(data.data()),count*sizeof(float));check(file.gcount()==std::streamsize(count*sizeof(float))&&file.peek()==EOF,"PLE gate receipt span differs");return data;};
    check(read_capture("key.f32le",10240)==actual_ple_gate_key&&read_capture("query.f32le",10240)==actual_ple_gate_query&&actual_ple_gate_key==gate_trace.ple_key&&read_capture("gate.f32le",4)==gate_trace.ple_gate,"PLE gate receipt is not the executed producer/consumer data");
    ple_fixture::reference(f,f.input(0),initial_gate_history,gate_trace,*gate_layer);
    auto gate_residual=gate_layer->last_residual(),gate_history=gate_layer->ple_convolution_history();
    fail_ple_gate=true;check(!gate_layer->step(20,f.input(1),ple_fixture::experts(),route,error),"later gate domain failure accepted");fail_ple_gate=false;
    check(gate_layer->tokens()==1&&gate_layer->ple_convolution_history()==gate_history&&gate_layer->token_history()==std::array<int32_t,2>{{-1,19}},"later gate failure committed history");
    check(!gate_layer->step(20,f.input(1),refuse,route,error),"gate FFN failure accepted");
    check(gate_layer->tokens()==1&&gate_layer->ple_convolution_history()==gate_history&&gate_layer->token_history()==std::array<int32_t,2>{{-1,19}},"gate FFN failure committed history");
    check(gate_layer->step(20,f.input(1),ple_fixture::experts(),route,error),error);check(gate_layer->tokens()==2&&gate_layer->token_history()==std::array<int32_t,2>{{19,20}},"gate retry history differs");
    check(gate_layer->reset(error),error);check(gate_layer->step(19,f.input(0),ple_fixture::experts(),route,error),error);
    check(gate_layer->last_residual()==gate_residual&&gate_layer->ple_convolution_history()==gate_history,"gate reset/retry differs");
    gate_layer.reset();actual_ple_gate_bank=nullptr;ple_gate_links=0;actual_ple_key_output=nullptr;ple_value_bank=nullptr;
    options.native_ple_key_t1=true;options.native_ple_value_t1=true;options.native_ple_key_receipt=(root/"all-key").string();options.native_ple_value_receipt=(root/"all-value").string();options.native_ple_gate_receipt=(root/"all-gate").string();
    observe_ple_key=true;observe_ple_value=true;ple_key_links=0;ple_value_links=0;auto all=PortableGdnLayer::create(1,f.views,options,error);check(bool(all),error);check(all->reset(error),error);auto all_history=all->ple_convolution_history();PortableGdnTrace all_trace;
    check(all->step(19,f.input(0),ple_fixture::experts(),route,error,&all_trace),error);
    check(ple_key_links>0&&ple_value_links>0&&ple_gate_links>0&&actual_ple_key_output&&ple_value_bank&&actual_ple_gate_bank,"combined PLE producer and consumer banks not connected");
    check(actual_ple_gate_key==all_trace.ple_key&&all->native_ple_gate_calls()==1,"combined normalized PLE producer differs");
    ple_fixture::reference(f,f.input(0),all_history,all_trace,*all);auto all_after=all->ple_convolution_history();
    fail_ple_gate=true;check(!all->step(20,f.input(1),ple_fixture::experts(),route,error),"combined gate failure accepted");fail_ple_gate=false;
    check(all->tokens()==1&&all->ple_convolution_history()==all_after&&all->token_history()==std::array<int32_t,2>{{-1,19}},"combined gate refusal committed history");
    std::puts("PLE key/gate graph ownership, failure noncommit, retry and history reset passed with host providers");return 0;
} catch(const std::exception&e) { std::cerr<<e.what()<<'\n';return 1; } }
