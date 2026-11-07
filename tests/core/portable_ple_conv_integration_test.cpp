#include "portable_native_hc_test_backend.hpp"
#include "portable_ple_fixture.hpp"
#include <iostream>
int main(int argc,char** argv){try{
    check(argc==2,"scratch root required");
    check(!native_hc_boundary(0).defer_ffn&&!native_hc_boundary(1).pending_attention,
          "layer 0 must materialize its FFN write before PLE");
    {
        PortableModelOptions model;model.native_ple_conv_t1=true;model.native_ple_conv_receipt_prefix="ple-conv";
        model.gdn_resident_weights=true;model.allow_cpu_fallback=false;validate_model_modes(model,1);
        auto off=model;off.native_ple_conv_t1=false;
        for(size_t l=0;l<48;++l){PortableGdnOptions per;model_gdn_hc_options(model,l,per);
            check(per.native_ple_conv_t1==(l==1),"PLE conv leaked to another layer");
            check(model_hc_host_bound(model,l)-model_hc_host_bound(off,l)==(l==1?PortableGdnOptions::native_ple_conv_host_bound:0),"PLE conv admission differs");}
        model.allow_cpu_fallback=true;bool rejected=false;try{validate_model_modes(model,1);}catch(const std::exception&){rejected=true;}
        check(rejected,"PLE conv accepted CPU fallback");
        model.allow_cpu_fallback=false;model.native_ple_gate_t1=true;model.native_ple_gate_receipt_prefix="gate";
        model.native_ple_norm_t1=true;model.native_ple_norm_receipt_prefix="norm";validate_model_modes(model,1);
        auto no_norm=model;no_norm.native_ple_norm_t1=false;
        for(size_t l=0;l<48;++l){PortableGdnOptions per;model_gdn_hc_options(model,l,per);
            check(per.native_ple_norm_t1==(l==1),"PLE norm leaked to another layer");
            check(model_hc_host_bound(model,l)-model_hc_host_bound(no_norm,l)==(l==1?PortableGdnOptions::native_ple_norm_host_bound:0),"PLE norm host admission differs");}
        model.native_hc_serial=true;no_norm.native_hc_serial=true;
        check(model_hc_host_bound(model,1)-model_hc_host_bound(no_norm,1)==PortableGdnOptions::native_ple_norm_host_bound,"serial PLE norm host admission differs");
        model.native_hc_serial=false;
        model.native_ple_gate_t1=false;rejected=false;try{validate_model_modes(model,1);}catch(const std::exception&){rejected=true;}check(rejected,"PLE norm without native gate accepted");
    }
    ple_fixture::Fixture fixture;std::vector<uint16_t> role(40960);
    for(size_t i=0;i<role.size();++i)role[i]=ggml_fp32_to_fp16(fixture.f32(5,i));
    // The real Q8_0 GGUF stores this role in F32; native packing narrows it.
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
    check(layer->tokens()==0&&layer->native_ple_conv_calls()==0&&layer->ple_convolution_history()==initial,"component refusal committed history");
    fail_ple_conv=false;observe_ple_conv=true;PortableGdnTrace trace;
    check(layer->step(19,fixture.input(0),ple_fixture::experts(),route,error,&trace),error);
    check(layer->native_ple_conv_calls()==1,"successful convolution was not counted");
    check(ple_conv_links>0&&actual_ple_conv_result,"suffix did not consume actual conv result bank");
    check(std::filesystem::file_size(root/"candidate/first-candidate/history.f32le")==92160*4 &&
          std::filesystem::file_size(root/"candidate/first-candidate/normalized.f32le")==10240*4 &&
          std::filesystem::file_size(root/"candidate/first-candidate/weight.f16le")==81920 &&
          std::filesystem::file_size(root/"candidate/first-candidate/result.f32le")==10240*4,
          "actual convolution candidate receipt missing");
    {
        std::ifstream f(root/"candidate/first-candidate/weight.f16le",std::ios::binary);
        std::vector<uint16_t> captured(role.size());f.read(reinterpret_cast<char*>(captured.data()),captured.size()*2);
        check(bool(f)&&captured==role,"actual F32 role was not narrowed to the native F16 values");
    }
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
    check(layer->native_ple_conv_calls()==2,"completed candidate before FFN refusal was not counted");
    check(layer->tokens()==1&&layer->ple_convolution_history()==history,"FFN refusal committed candidate history");
    check(layer->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    check(layer->native_ple_conv_calls()==3,"retry convolution was not counted");
    check(layer->token_history()==std::array<int32_t,2>{{19,20}},"retry token history differs");
    check(layer->reset(error),error);check(layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(layer->last_residual()==residual&&layer->ple_convolution_history()==history,"reset did not restore first result");
    layer.reset();actual_ple_conv_result=nullptr;ple_conv_links=0;
    options.native_ple_conv_receipt=(root/"combined-conv").string();
    options.native_ple_gate_t1=true;options.native_ple_gate_receipt=(root/"combined-gate").string();
    options.state_budget+=PortableGdnOptions::native_ple_gate_host_bound;observe_ple_gate=true;
    auto combined=PortableGdnLayer::create(1,fixture.views,options,error);check(bool(combined),error);
    check(combined->reset(error),error);check(combined->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(ple_gate_links>0&&ple_conv_links>0,"combined gate/conv producer chain was not consumed");
    auto combined_history=combined->ple_convolution_history();
    check(!combined->step(20,fixture.input(1),refuse,route,error),"combined FFN refusal accepted");
    check(combined->tokens()==1&&combined->ple_convolution_history()==combined_history,"combined failure committed history");
    check(combined->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    combined.reset();actual_ple_conv_result=nullptr;
    options.native_ple_conv_receipt=(root/"shared-conv").string();options.native_ple_gate_receipt=(root/"shared-gate").string();
    options.resident_state=true;
    options.scratch=PortableGdnScratch::create(options.execution,error);check(bool(options.scratch),error);
    auto shared=PortableGdnLayer::create(1,fixture.views,options,error);check(bool(shared),error);
    check(shared->reset(error),error);
    auto stale_ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({4096,nullptr,true}),ggml_free);
    auto* stale=ggml_new_tensor_1d(stale_ctx.get(),GGML_TYPE_F32,10240);
    auto stale_buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(stale_ctx.get(),cpu.get()),ggml_backend_buffer_free);
    std::vector<float> stale_values(10240,88.f);ggml_backend_tensor_set(stale,stale_values.data(),0,stale_values.size()*4);
    ple_conv_simulated_stale=stale;
    check(!shared->step(19,fixture.input(0),ple_fixture::experts(),route,error),"stale scheduled source accepted");
    check(ple_conv_rewritten_node&&error.find("PLE conv: prefix copy differs: normalized byte ")!=std::string::npos,"scheduled-source refusal was not reproduced");
    check(shared->tokens()==0&&shared->native_ple_conv_calls()==0,"scheduled-source refusal committed state");
    ple_conv_rewritten_node->src[0]=ple_conv_previous_source;ple_conv_rewritten_node=nullptr;ple_conv_simulated_stale=nullptr;
    ple_conv_corrupt_hidden=true;ple_conv_hidden_fault_applied=false;
    check(!shared->step(19,fixture.input(0),ple_fixture::experts(),route,error),"corrupted hidden destination accepted");
    check(ple_conv_hidden_fault_applied&&error.find("PLE conv: prefix copy differs: hidden byte ")!=std::string::npos,"hidden corruption was not refused");
    check(shared->tokens()==0&&shared->native_ple_conv_calls()==0,"hidden corruption committed state");
    ple_conv_corrupt_hidden=false;ple_conv_hidden_fault_applied=false;ple_conv_hidden_stale=stale;
    check(shared->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(ple_conv_hidden_fault_applied&&ple_conv_rewritten_node,"hidden source rewrite was not exercised");
    ple_conv_rewritten_node->src[0]=ple_conv_previous_source;ple_conv_rewritten_node=nullptr;ple_conv_hidden_stale=nullptr;
    auto shared_history=shared->ple_convolution_history(),shared_residual=shared->last_residual();
    check(!shared->step(20,fixture.input(1),refuse,route,error),"resident FFN refusal accepted");
    check(shared->tokens()==1&&shared->ple_convolution_history()==shared_history,"resident failure committed candidate history");
    check(shared->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    check(shared->reset(error),error);check(shared->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(shared->ple_convolution_history()==shared_history&&shared->last_residual()==shared_residual,"shared scheduler rebind/reset differs");
    shared.reset();options.scratch.reset();options.resident_state=false;actual_ple_conv_result=nullptr;
    options.native_ple_conv_receipt=(root/"norm-conv").string();options.native_ple_gate_receipt=(root/"norm-gate").string();
    options.native_ple_norm_t1=true;options.native_ple_norm_receipt=(root/"norm").string();options.state_budget+=PortableGdnOptions::native_ple_norm_host_bound;
    observe_ple_norm=true;ple_norm_calls=0;ple_norm_conv_links=0;ple_norm_outputs={};
    auto normalized=PortableGdnLayer::create(1,fixture.views,options,error);check(bool(normalized),error);
    check(normalized->reset(error),error);auto before=normalized->ple_convolution_history();fail_ple_norm=true;
    check(!normalized->step(19,fixture.input(0),ple_fixture::experts(),route,error),"norm refusal accepted");
    check(normalized->tokens()==0&&normalized->ple_convolution_history()==before,"norm refusal committed state");
    fail_ple_norm=false;fail_ple_norm_stage=3;
    check(!normalized->step(19,fixture.input(0),ple_fixture::experts(),route,error),"gated norm refusal accepted");
    check(normalized->tokens()==0&&normalized->ple_convolution_history()==before&&normalized->native_ple_conv_calls()==0,"gated norm refusal committed state");
    fail_ple_norm_stage=0;ple_norm_calls=0;ple_norm_outputs={};
    check(normalized->step(19,fixture.input(0),ple_fixture::experts(),route,error,&trace),error);
    check(ple_norm_calls==3&&normalized->native_ple_norm_calls()==5&&ple_norm_conv_links>0,"norm to gate/convolution producer links differ");
    auto norm_history=normalized->ple_convolution_history();
    check(!normalized->step(20,fixture.input(1),refuse,route,error),"norm FFN failure accepted");
    check(normalized->tokens()==1&&normalized->ple_convolution_history()==norm_history,"norm FFN failure committed state");
    check(normalized->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    check(normalized->reset(error),error);check(normalized->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(normalized->ple_convolution_history()==norm_history,"norm reset differs");
    normalized.reset();actual_ple_conv_result=nullptr;ple_norm_outputs={};ple_norm_calls=0;ple_norm_conv_links=0;
    options.native_ple_conv_receipt=(root/"norm-shared-conv").string();options.native_ple_gate_receipt=(root/"norm-shared-gate").string();options.native_ple_norm_receipt=(root/"norm-shared").string();
    options.resident_state=true;options.scratch=PortableGdnScratch::create(options.execution,error);check(bool(options.scratch),error);
    auto norm_shared=PortableGdnLayer::create(1,fixture.views,options,error);check(bool(norm_shared),error);
    check(norm_shared->reset(error),error);check(norm_shared->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(ple_norm_calls==3&&ple_norm_conv_links>0,"shared norm producer links differ");
    auto shared_norm_history=norm_shared->ple_convolution_history();
    check(!norm_shared->step(20,fixture.input(1),refuse,route,error),"shared norm FFN failure accepted");
    check(norm_shared->tokens()==1&&norm_shared->ple_convolution_history()==shared_norm_history,"shared norm failure committed state");
    check(norm_shared->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    check(norm_shared->reset(error),error);check(norm_shared->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(norm_shared->ple_convolution_history()==shared_norm_history,"shared norm reset differs");
    const auto normalized_host_bytes=norm_shared->state_bytes();
    norm_shared.reset();options.scratch.reset();options.resident_state=false;
    PortableModelOptions hc_model;hc_model.native_ple_hc_read_t1=true;
    hc_model.native_ple_norm_t1=hc_model.native_ple_conv_t1=hc_model.native_ple_gate_t1=true;
    hc_model.native_ple_hc_receipt_prefix=(root/"post-ple-hc").string();
    hc_model.native_ple_conv_receipt_prefix=(root/"hc-conv").string();hc_model.native_ple_gate_receipt_prefix=(root/"hc-gate").string();hc_model.native_ple_norm_receipt_prefix=(root/"hc-norm").string();
    options.state_budget=normalized_host_bytes-PortableGdnOptions::native_ple_norm_host_bound-PortableGdnOptions::native_ple_conv_host_bound-PortableGdnOptions::native_ple_gate_host_bound;
    model_gdn_hc_options(hc_model,1,options);
    check(options.state_budget==normalized_host_bytes+PortableGdnOptions::native_hc_read_host_bound,"model-to-layer PLE HC allowance missing");
    {auto disabled=hc_model;disabled.native_ple_hc_read_t1=false;
     for(bool serial:{false,true}){hc_model.native_hc_serial=disabled.native_hc_serial=serial;
      for(size_t l=0;l<48;++l){PortableGdnOptions on,off;model_gdn_hc_options(hc_model,l,on);model_gdn_hc_options(disabled,l,off);
       const auto delta=l==1?PortableGdnOptions::native_hc_read_host_bound:0;
       check(on.state_budget-off.state_budget==delta,"PLE HC layer allowance scope differs");
       check(model_hc_host_bound(hc_model,l)-model_hc_host_bound(disabled,l)==delta+(serial && l==1 ? PortableFfnOptions::native_hc_pending_host_bound : 0),"global/layer PLE HC allowances disagree");}}}
    {auto short_budget=options;--short_budget.state_budget;
     check(!PortableGdnLayer::create(1,fixture.views,short_budget,error)&&error.find("host state budget exceeded")!=std::string::npos,"one-byte-short PLE HC host budget accepted");}
    for(const auto& v:fixture.views){
        if(v.name=="blk.1.hc_attn_inject.weight")expected_weights=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
        if(v.name=="blk.1.hc_attn_down.weight")expected_down=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
        if(v.name=="blk.1.hc_attn_up.weight")expected_up=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,3276800);
    }
    {auto bad=options;bad.native_hc_pending=true;check(!PortableGdnLayer::create(1,fixture.views,bad,error),"PLE HC pending write accepted");}
    {auto bad=options;bad.execution.graph_budget=1;check(!PortableGdnLayer::create(1,fixture.views,bad,error),"PLE HC graph budget ignored");}
    observe_ple_hc=true;actual_ple_conv_result=nullptr;
    auto hc=PortableGdnLayer::create(1,fixture.views,options,error);check(bool(hc),error);check(hc->reset(error),error);
    const auto pristine=hc->ple_convolution_history();fail_run=true;
    check(!hc->step(19,fixture.input(0),ple_fixture::experts(),route,error),"HC refusal accepted");
    check(hc->tokens()==0&&hc->ple_convolution_history()==pristine,"HC failure committed PLE history");fail_run=false;
    check(hc->step(19,fixture.input(0),ple_fixture::experts(),route,error,&trace),error);
    check(ple_hc_links>=2&&uploaded==trace.ple_residual&&trace.hc_residual==trace.ple_residual,"HC did not consume PLE result");
    check(hc->native_hc_pending_calls()==0,"materialized PLE reapplied pending write");
    const auto hc_history=hc->ple_convolution_history(),hc_residual=hc->last_residual();
    check(!hc->step(20,fixture.input(1),refuse,route,error),"HC FFN refusal accepted");
    check(hc->tokens()==1&&hc->ple_convolution_history()==hc_history,"HC FFN failure committed history");
    check(hc->step(20,fixture.input(1),ple_fixture::experts(),route,error),error);
    check(hc->reset(error),error);check(hc->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(hc->ple_convolution_history()==hc_history&&hc->last_residual()==hc_residual,"HC reset differs");
    hc.reset();
    // The serial model policy composes post-PLE HC with the actual FFN pending read.
    hc_model.native_hc_serial=true;hc_model.hc_serial_resident_weights=true;
    hc_model.native_ple_hc_receipt_prefix=(root/"composed-hc").string();
    hc_model.native_ple_conv_receipt_prefix=(root/"composed-conv").string();
    hc_model.native_ple_gate_receipt_prefix=(root/"composed-gate").string();
    hc_model.native_ple_norm_receipt_prefix=(root/"composed-norm").string();
    auto serial=options;model_gdn_hc_options(hc_model,1,serial);
    check(serial.native_hc_ffn_read && serial.resident_ffn_weights && !serial.native_hc_pending,
          "serial PLE FFN handoff policy differs");
    auto without_handoff=hc_model;without_handoff.native_ple_hc_read_t1=false;
    check(model_hc_host_bound(hc_model,1)-model_hc_host_bound(without_handoff,1)==
          PortableGdnOptions::native_hc_read_host_bound+PortableFfnOptions::native_hc_pending_host_bound,
          "serial PLE FFN host admission differs");
    for(const auto& v:fixture.views) if(v.name=="blk.1.hc_ffn_inject.weight")
        expected_ffn_inject=native_hc_q8_bf16(static_cast<const uint8_t*>(v.data),v.bytes,40960);
    auto composed=PortableGdnLayer::create(1,fixture.views,serial,error);check(bool(composed),error);
    check(composed->reset(error),error);
    auto before_composed=composed->ple_convolution_history();
    auto before_state=composed->recurrent_state(), before_history=composed->convolution_history();
    check(!composed->step(19,fixture.input(0),refuse,route,error),"composed FFN refusal accepted");
    check(composed->tokens()==0 && composed->ple_convolution_history()==before_composed &&
          composed->recurrent_state()==before_state && composed->convolution_history()==before_history,
          "composed failure committed candidate state");
    check(composed->step(19,fixture.input(0),ple_fixture::experts(),route,error,&trace),error);
    check(composed->native_ffn_hc_pending_calls()==2 && composed->native_hc_pending_calls()==0,
          "post-PLE FFN did not consume pending attention result");
    auto composed_history=composed->ple_convolution_history(),composed_residual=composed->last_residual();
    check(composed->reset(error),error);check(composed->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(composed->ple_convolution_history()==composed_history && composed->last_residual()==composed_residual,
          "composed reset changed result");
    composed.reset();
    // Full native GDN chain consumes the actual materialized PLE HC mix.
    hc_model.native_gdn_t1=hc_model.native_gdn_preparation_weights=hc_model.native_gdn_preparation_t1=true;
    hc_model.native_gdn_input_projection_t1=hc_model.native_gdn_output_projection_t1=true;
    hc_model.native_gdn_receipt_prefix=(root/"chain-gdn").string();
    hc_model.native_gdn_preparation_receipt_prefix=(root/"chain-prep").string();
    hc_model.native_gdn_input_projection_receipt_prefix=(root/"chain-input").string();
    hc_model.native_gdn_output_projection_receipt_prefix=(root/"chain-output").string();
    hc_model.native_ple_hc_receipt_prefix=(root/"chain-hc").string();
    hc_model.native_ple_conv_receipt_prefix=(root/"chain-conv").string();
    hc_model.native_ple_gate_receipt_prefix=(root/"chain-gate").string();
    hc_model.native_ple_norm_receipt_prefix=(root/"chain-norm").string();
    auto no_chain=hc_model;no_chain.native_ple_hc_read_t1=false;
    PortableGdnOptions no_layer;model_gdn_hc_options(no_chain,1,no_layer);
    check(!no_layer.native_recurrence_t1&&!no_layer.native_preparation_t1&&!no_layer.native_input_projection_t1&&!no_layer.native_output_projection_t1,"native GDN escaped PLE HC prerequisite");
    const auto expected_extra=PortableGdnOptions::native_recurrence_host_bound+PortableGdnOptions::native_preparation_host_bound+
        PortableGdnOptions::native_input_projection_host_bound+PortableGdnOptions::native_output_projection_host_bound;
    auto no_gdn=hc_model;no_gdn.native_gdn_t1=no_gdn.native_gdn_preparation_weights=no_gdn.native_gdn_preparation_t1=false;
    no_gdn.native_gdn_input_projection_t1=no_gdn.native_gdn_output_projection_t1=false;
    check(model_hc_host_bound(hc_model,1)-model_hc_host_bound(no_gdn,1)==expected_extra,"native PLE GDN host allowance missing");
    auto chain=options;chain.resident_state=true;model_gdn_hc_options(hc_model,1,chain);
    check(chain.native_recurrence_t1 && chain.native_preparation_t1 && chain.native_input_projection_t1 && chain.native_output_projection_t1,"native PLE GDN policy missing");
    {auto bad=chain;bad.native_ple_hc_read_t1=false;check(!PortableGdnLayer::create(1,fixture.views,bad,error),"native PLE GDN accepted missing materialized HC");}
    auto chain_layer=PortableGdnLayer::create(1,fixture.views,chain,error);check(bool(chain_layer),error);
    check(chain_layer->reset(error),error);
    const auto initial_ple=chain_layer->ple_convolution_history(), initial_gdn=chain_layer->recurrent_state(), initial_conv=chain_layer->convolution_history();
    fail_gdn_input=1;check(!chain_layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),"native input failure accepted");fail_gdn_input=0;
    check(chain_layer->tokens()==0 && chain_layer->ple_convolution_history()==initial_ple && chain_layer->recurrent_state()==initial_gdn && chain_layer->convolution_history()==initial_conv,"native input failure committed state");
    fail_gdn=true;check(!chain_layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),"native recurrence failure accepted");fail_gdn=false;
    check(chain_layer->tokens()==0 && chain_layer->ple_convolution_history()==initial_ple && chain_layer->recurrent_state()==initial_gdn && chain_layer->convolution_history()==initial_conv,"native recurrence failure committed state");
    observe_gdn_input=true;gdn_input_links=0;observe_gdn_output=true;gdn_output_quant_links=0;gdn_output_pending_links=0;
    check(chain_layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(gdn_input_links==2 && gdn_output_quant_links==1 && gdn_output_pending_links==1,"native GDN producer/consumer links differ");
    auto first_chain=chain_layer->last_residual(), first_state=chain_layer->recurrent_state(), first_ple=chain_layer->ple_convolution_history();
    check(!chain_layer->step(20,fixture.input(1),refuse,route,error),"native chain FFN refusal accepted");
    check(chain_layer->tokens()==1 && chain_layer->recurrent_state()==first_state && chain_layer->ple_convolution_history()==first_ple,"native chain FFN failure committed state");
    check(chain_layer->reset(error),error);check(chain_layer->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(chain_layer->last_residual()==first_chain && chain_layer->recurrent_state()==first_state && chain_layer->ple_convolution_history()==first_ple,"native chain reset differs");
    chain_layer.reset();
    auto constrained=chain;constrained.execution.graph_budget=1;
    check(!PortableGdnLayer::create(1,fixture.views,constrained,error),"native chain graph cap ignored");
    hc_model.native_gdn_receipt_prefix=(root/"shared-chain-gdn").string();
    hc_model.native_gdn_preparation_receipt_prefix=(root/"shared-chain-prep").string();
    hc_model.native_gdn_input_projection_receipt_prefix=(root/"shared-chain-input").string();
    hc_model.native_gdn_output_projection_receipt_prefix=(root/"shared-chain-output").string();
    hc_model.native_ple_hc_receipt_prefix=(root/"shared-chain-hc").string();
    hc_model.native_ple_conv_receipt_prefix=(root/"shared-chain-conv").string();
    hc_model.native_ple_gate_receipt_prefix=(root/"shared-chain-gate").string();
    hc_model.native_ple_norm_receipt_prefix=(root/"shared-chain-norm").string();
    auto shared_chain=options;shared_chain.resident_state=true;model_gdn_hc_options(hc_model,1,shared_chain);
    check(PortableGdnScratch::host_metadata_bound(true)>PortableGdnScratch::host_metadata_bound(),"additional metadata not priced");
    auto unreserved_model=hc_model;
    unreserved_model.native_ple_hc_receipt_prefix=(root/"unreserved-hc").string();
    unreserved_model.native_ple_conv_receipt_prefix=(root/"unreserved-conv").string();
    unreserved_model.native_ple_gate_receipt_prefix=(root/"unreserved-gate").string();
    unreserved_model.native_ple_norm_receipt_prefix=(root/"unreserved-norm").string();
    unreserved_model.native_gdn_receipt_prefix=(root/"unreserved-gdn").string();
    unreserved_model.native_gdn_preparation_receipt_prefix=(root/"unreserved-prep").string();
    unreserved_model.native_gdn_input_projection_receipt_prefix=(root/"unreserved-input").string();
    unreserved_model.native_gdn_output_projection_receipt_prefix=(root/"unreserved-output").string();
    auto unreserved=options;unreserved.resident_state=true;model_gdn_hc_options(unreserved_model,1,unreserved);
    unreserved.scratch=PortableGdnScratch::create(unreserved.execution,error);check(bool(unreserved.scratch),error);
    check(!PortableGdnLayer::create(1,fixture.views,unreserved,error) && error.find("additional scheduler not reserved")!=std::string::npos,
          "unreserved fourth scheduler accepted");
    shared_chain.scratch=PortableGdnScratch::create(shared_chain.execution,error,true);check(bool(shared_chain.scratch),error);
    auto leased=PortableGdnLayer::create(1,fixture.views,shared_chain,error);check(bool(leased),error);
    check(leased->reset(error),error);PortableGdnTrace leased_trace;
    check(leased->step(19,fixture.input(0),ple_fixture::experts(),route,error,&leased_trace),error);
    check(leased_trace.ffn_hc.original==leased_trace.ple_residual &&
          leased_trace.ffn_hc.previous_injection==leased_trace.hc_inject &&
          leased_trace.ffn_hc.previous_output==actual_native_wo,"shared chain FFN did not consume actual attention tuple");
    const auto published=leased->pending_write();check(!published.empty(),"shared chain omitted FFN pending tuple");
    check(leased->last_residual()==first_chain && leased->recurrent_state()==first_state && leased->ple_convolution_history()==first_ple,"shared chain differs");
    const auto owned_bytes=shared_chain.scratch->graph_bytes();check(owned_bytes>0 && owned_bytes<=shared_chain.execution.graph_budget,"shared scheduler ledger invalid");
    check(!leased->step(20,fixture.input(1),refuse,route,error),"shared chain FFN refusal accepted");
    check(leased->tokens()==1 && leased->recurrent_state()==first_state && leased->ple_convolution_history()==first_ple &&
          leased->pending_write().residual==published.residual && leased->pending_write().block==published.block &&
          leased->pending_write().injection==published.injection,"shared chain failure committed state or tuple");
    check(leased->reset(error),error);check(leased->pending_write().empty(),"shared chain reset kept pending tuple");
    check(leased->step(19,fixture.input(0),ple_fixture::experts(),route,error),error);
    check(leased->last_residual()==first_chain && leased->recurrent_state()==first_state &&
          leased->pending_write().residual==published.residual && leased->pending_write().block==published.block &&
          leased->pending_write().injection==published.injection,"shared chain reset/rebind differs");
    observe_gdn_input=observe_gdn_output=false;observe_ple_hc=false;
    std::puts("PLE conv producer, candidate history, refusal, retry and reset passed with host provider");return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
