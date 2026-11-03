#pragma once
#include "strata_vk_native_hc.hpp"
#include "native_hc_layout.hpp"
#include "ggml-alloc.h"
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
namespace strata::core::portable_detail {
// Opt-in serial owner. Graphs borrow these allocated leaves; no scheduler
// may reuse their storage between residual upload and component/consumer calls.
class PortableNativeHc {
    using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
    using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
    Context context_{nullptr,ggml_free};
    Buffer buffer_{nullptr,ggml_backend_buffer_free};
    std::unique_ptr<StrataVkNativeHc,decltype(&strata_vk_native_hc_destroy)> component_{nullptr,strata_vk_native_hc_destroy};
    std::unique_ptr<StrataVkNativeHcMix,decltype(&strata_vk_native_hc_mix_destroy)> mix_{nullptr,strata_vk_native_hc_mix_destroy};
    std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> pending_{nullptr,strata_vk_native_hc_pending_destroy};
    std::unique_ptr<StrataVkNativeHcPending,decltype(&strata_vk_native_hc_pending_destroy)> write_{nullptr,strata_vk_native_hc_pending_destroy};
    const ggml_tensor *norm_,*weight_,*down_,*up_;
    uint64_t position_=UINT64_MAX;
    size_t bytes_=0;
    uint64_t calls_=0,mix_calls_=0,pending_calls_=0,write_calls_=0;
    std::string receipt_dir_;
    bool first_recorded_=false,write_recorded_=false;
public:
    ggml_tensor *pending_input=nullptr,*previous_output=nullptr,*previous_inject=nullptr,*pending_gate=nullptr,*pending_exp=nullptr;
    ggml_tensor *write_block=nullptr,*write_candidate=nullptr,*write_gate=nullptr,*write_exp=nullptr;
    ggml_tensor *residual=nullptr,*normalized=nullptr,*rs=nullptr,*inject=nullptr;
    ggml_tensor *low_pre=nullptr,*low=nullptr,*low_exp=nullptr,*up_pre=nullptr,*gates=nullptr,*up_exp=nullptr,*mixed=nullptr;
    PortableNativeHc(ggml_backend_t backend,const ggml_tensor* norm,const ggml_tensor* weight,
                     size_t budget,const std::string& prefix,const ggml_tensor* down=nullptr,const ggml_tensor* up=nullptr,bool pending=false,bool final_write=false):norm_(norm),weight_(weight),down_(down),up_(up) {
        native_hc_environment();native_hc_need(backend&&!prefix.empty(),"native HC serial: backend/receipt prefix required");
        native_hc_need(bool(down)==bool(up),"native HC serial: paired down/up roles required");
        native_hc_need(!pending||down,"native HC pending: composed read required");
        native_hc_need(!final_write||pending,"native HC final write: pending FFN read required");
        context_.reset(ggml_init({16384,nullptr,true}));native_hc_need(bool(context_),"native HC serial: metadata");
        residual=ggml_new_tensor_3d(context_.get(),GGML_TYPE_F32,2560,4,1);
        normalized=ggml_new_tensor_2d(context_.get(),GGML_TYPE_F32,10240,1);
        rs=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
        inject=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
        for(auto* t:{residual,normalized,rs,inject}){ggml_set_input(t);ggml_set_output(t);}
        if(down_) {
            low_pre=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,320);
            low=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,320);
            low_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,320);
            up_pre=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240);
            gates=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240);
            up_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,10240);
            mixed=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,2560);
            for(auto* t:{low_pre,low,low_exp,up_pre,gates,up_exp,mixed}){ggml_set_input(t);ggml_set_output(t);}
        }
        if(pending) {
            pending_input=ggml_new_tensor_3d(context_.get(),GGML_TYPE_F32,2560,4,1);
            previous_output=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,2560);
            previous_inject=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            pending_gate=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            pending_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            for(auto* t:{pending_input,previous_output,previous_inject,pending_gate,pending_exp}){ggml_set_input(t);ggml_set_output(t);}
        }
        if(final_write) {
            write_block=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,2560);
            write_candidate=ggml_new_tensor_3d(context_.get(),GGML_TYPE_F32,2560,4,1);
            write_gate=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            write_exp=ggml_new_tensor_1d(context_.get(),GGML_TYPE_F32,4);
            for(auto* t:{write_block,write_candidate,write_gate,write_exp}){ggml_set_input(t);ggml_set_output(t);}
        }
        auto bound=ggml_backend_alloc_ctx_tensors_from_buft_size(context_.get(),ggml_backend_get_default_buffer_type(backend));
        native_hc_need(bound<=budget,"native HC serial: scratch graph budget");
        static std::atomic<uint64_t> next{0};std::string dir;
        for(int i=0;i<1024;++i){auto candidate=prefix+".owner-"+std::to_string(next.fetch_add(1));if(std::filesystem::create_directory(candidate)){dir=candidate;break;}}
        native_hc_need(!dir.empty(),"native HC serial: fresh receipt namespace");
        receipt_dir_=dir;
        component_.reset(strata_vk_native_hc_create(backend,(dir+"/effective.spv").c_str(),budget-bound));
        native_hc_need(bool(component_),"native HC serial: component unavailable");
        size_t metadata=strata_vk_native_hc_bytes(component_.get());
        native_hc_need(metadata<=budget-bound,"native HC serial: norm metadata budget");
        if(down_) {
            mix_.reset(strata_vk_native_hc_mix_create(backend,(dir+"/mix-effective.spv").c_str(),budget-bound-metadata));
            const size_t mix_bytes=strata_vk_native_hc_mix_bytes(mix_.get());
            native_hc_need(mix_bytes<=budget-bound-metadata,"native HC serial: mix metadata budget");
            metadata+=mix_bytes;
        }
        if(pending) {
            pending_.reset(strata_vk_native_hc_pending_create(backend,(dir+"/pending-effective.spv").c_str(),budget-bound-metadata));
            const size_t extra=strata_vk_native_hc_pending_bytes(pending_.get());
            native_hc_need(extra<=budget-bound-metadata,"native HC pending: metadata graph budget");metadata+=extra;
        }
        if(final_write) {
            write_.reset(strata_vk_native_hc_write_create(backend,(dir+"/write-effective.spv").c_str(),budget-bound-metadata));
            const size_t extra=strata_vk_native_hc_pending_bytes(write_.get());
            native_hc_need(extra<=budget-bound-metadata,"native HC final write: metadata budget");metadata+=extra;
        }
        native_hc_need(metadata<=budget-bound,"native HC serial: component graph budget");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(),backend));native_hc_need(bool(buffer_),"native HC serial: scratch allocation");
        const auto actual=ggml_backend_buffer_get_size(buffer_.get());native_hc_need(actual<=budget-metadata,"native HC serial: actual graph budget");
        bytes_=actual+metadata;
    }
    void diagnostic_position(uint64_t position){position_=position;}
    size_t bytes()const{return bytes_;}
    uint64_t calls()const{return calls_;}
    uint64_t mix_calls()const{return mix_calls_;}
    uint64_t pending_calls()const{return pending_calls_;}
    uint64_t write_calls()const{return write_calls_;}
    void run_write(){
        native_hc_need(write_&&write_calls_!=UINT64_MAX,"native HC final write: owner/counter required");
        try{strata_vk_native_hc_pending_run(write_.get(),residual,write_block,inject,write_candidate,write_gate,write_exp);}
        catch(...){record_input_refusal("write",{{"input",residual},{"block",write_block},{"inject",inject}});throw;}
        ++write_calls_;
    }
    void record_write_first(){
        if(write_recorded_||!write_calls_)return;
        for(auto pair:{std::pair{"input",residual},std::pair{"block",write_block},std::pair{"inject",inject},std::pair{"gate",write_gate},std::pair{"exp",write_exp},std::pair{"candidate",write_candidate}}){
            std::vector<float> v(size_t(ggml_nelements(pair.second)));ggml_backend_tensor_get(pair.second,v.data(),0,v.size()*4);
            auto* f=std::fopen((receipt_dir_+"/first-write-"+pair.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC final write: exclusive first receipt");
            const bool written=std::fwrite(v.data(),4,v.size(),f)==v.size();const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC final write: receipt failed");
        }write_recorded_=true;
    }
    // Only the first successful attention graph is observed. This is a bounded
    // diagnostic readback, not a weight extraction or numerical fallback.
    void record_first(ggml_tensor* producer_block=nullptr) {
        if(first_recorded_ || calls_==0 || (mix_&&mix_calls_==0))return;
        for(auto pair:{std::pair{"input.f32le",residual},std::pair{"normalized.f32le",normalized},
                       std::pair{"rs.f32le",rs},std::pair{"inject.f32le",inject}}) {
            std::vector<float> values(size_t(ggml_nelements(pair.second)));
            ggml_backend_tensor_get(pair.second,values.data(),0,values.size()*sizeof(float));
            auto* f=std::fopen((receipt_dir_+"/first-"+pair.first).c_str(),"wbx");
            native_hc_need(f,"native HC serial: exclusive initial receipt");
            const bool written=std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();
            const bool closed=std::fclose(f)==0;
            native_hc_need(written&&closed,"native HC serial: initial receipt write failed");
        }
        if(mix_)for(auto pair:{std::pair{"low-pre",low_pre},std::pair{"low",low},std::pair{"low-exp",low_exp},std::pair{"up-pre",up_pre},std::pair{"gates",gates},std::pair{"up-exp",up_exp},std::pair{"mixed",mixed}}) {
            std::vector<float> v(size_t(ggml_nelements(pair.second)));ggml_backend_tensor_get(pair.second,v.data(),0,v.size()*4);
            auto* f=std::fopen((receipt_dir_+"/first-"+pair.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC serial: initial mix receipt");
            const bool written=std::fwrite(v.data(),4,v.size(),f)==v.size();const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC serial: mix receipt write");
        }
        if(pending_)for(auto pair:{std::pair{"unapplied-residual",pending_input},std::pair{"previous-output",producer_block?producer_block:previous_output},std::pair{"previous-injection",previous_inject},std::pair{"pending-gate",pending_gate},std::pair{"pending-exp",pending_exp}}) {
            std::vector<float> v(size_t(ggml_nelements(pair.second)));ggml_backend_tensor_get(pair.second,v.data(),0,v.size()*4);
            auto* f=std::fopen((receipt_dir_+"/first-"+pair.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC pending: exclusive tuple receipt");
            const bool written=std::fwrite(v.data(),4,v.size(),f)==v.size();const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC pending: tuple receipt write");
        }
        first_recorded_=true;
    }
private:
    void record_input_refusal(const char* component,std::initializer_list<std::pair<const char*,const ggml_tensor*>> fields) noexcept {
        // Inputs are immutable for these kernels. Never expose outputs when
        // dispatch completion is unknown. Existing T1 host bounds cover this.
        try {
            const auto dir=receipt_dir_+"/input-refusal";
            native_hc_need(std::filesystem::create_directory(dir),"native HC: fresh input refusal namespace");
            size_t total=0;
            for(auto field:fields){
                const auto bytes=ggml_nbytes(field.second);native_hc_need(field.second->type==GGML_TYPE_F32&&bytes<=40960&&total+bytes<=51216,"native HC: bounded input refusal");total+=bytes;
                std::vector<uint32_t> data(bytes/4);ggml_backend_tensor_get(field.second,data.data(),0,bytes);
                auto*f=std::fopen((dir+"/"+field.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC: exclusive input refusal");
                const bool written=std::fwrite(data.data(),1,bytes,f)==bytes;const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC: input refusal write");
            }
            const std::string role=ggml_get_name(norm_),owner=std::filesystem::path(receipt_dir_).filename().string();
            native_hc_need((role+owner).find_first_of("\"\\\n\r")==std::string::npos,"native HC: input refusal identity");
            unsigned layer=0;native_hc_need(std::sscanf(role.c_str(),"blk.%u.",&layer)==1&&layer<48,"native HC: input refusal layer");
            const auto position=position_==UINT64_MAX?std::string("null"):std::to_string(position_);
            auto*f=std::fopen((dir+"/metadata.json").c_str(),"wx");native_hc_need(f,"native HC: exclusive input refusal metadata");
            const bool written=std::fprintf(f,"{\"schema\":1,\"component\":\"%s\",\"role\":\"%s\",\"owner\":\"%s\",\"layer\":%u,\"position\":%s,\"outputs_captured\":false,\"accepted\":false,\"payload_bytes\":%zu}\n",component,role.c_str(),owner.c_str(),layer,position.c_str(),total)>0;
            const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC: input refusal metadata");
        }catch(const std::exception&e){std::fprintf(stderr,"native HC input diagnostic incomplete: %s\n",e.what());}
         catch(...){std::fprintf(stderr,"native HC input diagnostic incomplete\n");}
    }
    void record_refusal(uint32_t status) {
        // Failure-only, current completed call. One tensor readback at a time
        // uses <=40960 bytes within the existing conservative HC host envelope.
        // No new GPU allocations, synchronization or accepted model candidate.
        const auto dir=receipt_dir_+"/refusal";
        native_hc_need(std::filesystem::create_directory(dir),"native HC: fresh refusal namespace");
        for(auto pair:{std::pair{"normalized",normalized},std::pair{"low-pre",low_pre},std::pair{"low",low},std::pair{"low-exp",low_exp},std::pair{"up-pre",up_pre},std::pair{"gates",gates},std::pair{"up-exp",up_exp},std::pair{"mixed",mixed}}){
            std::vector<float> values(size_t(ggml_nelements(pair.second)));
            ggml_backend_tensor_get(pair.second,values.data(),0,values.size()*4);
            auto*f=std::fopen((dir+"/"+pair.first+".f32le").c_str(),"wbx");native_hc_need(f,"native HC: exclusive refusal output");
            const bool written=std::fwrite(values.data(),4,values.size(),f)==values.size();const bool closed=std::fclose(f)==0;
            native_hc_need(written&&closed,"native HC: refusal write failed");
        }
        const std::string role=ggml_get_name(norm_);
        native_hc_need(role.find_first_of("\"\\\n\r") == std::string::npos,"native HC: refusal role identity");
        unsigned layer=0;native_hc_need(std::sscanf(role.c_str(),"blk.%u.",&layer)==1 && layer<48,"native HC: refusal layer identity");
        const auto owner=std::filesystem::path(receipt_dir_).filename().string();
        native_hc_need(owner.find_first_of("\"\\\n\r")==std::string::npos,"native HC: refusal owner identity");
        auto*f=std::fopen((dir+"/metadata.json").c_str(),"wx");native_hc_need(f,"native HC: exclusive refusal metadata");
        const std::string position=position_==UINT64_MAX?"null":std::to_string(position_);
        const bool written=std::fprintf(f,"{\"schema\":1,\"role\":\"%s\",\"owner\":\"%s\",\"layer\":%u,\"position\":%s,\"norm_call\":%llu,\"mix_successes\":%llu,\"status\":%u,\"stage_status\":[%u,%u,%u],\"all_dispatches_completed\":true,\"accepted\":false,\"payload_bytes\":177920}\n",role.c_str(),owner.c_str(),layer,position.c_str(),(unsigned long long)calls_,(unsigned long long)mix_calls_,status,status&255u,(status>>8)&255u,(status>>16)&255u)>0;
        const bool closed=std::fclose(f)==0;native_hc_need(written&&closed,"native HC: refusal metadata write failed");
    }
    void read(){
        native_hc_need(calls_!=UINT64_MAX&&mix_calls_!=UINT64_MAX,"native HC serial: call counter exhausted");
        try{strata_vk_native_hc_run(component_.get(),residual,norm_,weight_,normalized,rs,inject,1);}
        catch(...){record_input_refusal("norm",{{"input",residual}});throw;}
        ++calls_;
        if(mix_){
            try {strata_vk_native_hc_mix_run(mix_.get(),normalized,down_,up_,low_pre,low,low_exp,up_pre,gates,up_exp,mixed);}
            catch(const NativeHcMixDomainRefusal& e){
                try{record_refusal(e.status);}catch(const std::exception& diagnostic){std::fprintf(stderr,"native HC refusal diagnostic incomplete: %s\n",diagnostic.what());}
                throw;
            }
            catch(...){record_input_refusal("mix",{{"normalized",normalized}});throw;}
            ++mix_calls_;
        }
    }
public:
    void run(){native_hc_need(!pending_,"native HC pending: explicit tuple upload required");read();}
    void run_pending(ggml_tensor* producer_block=nullptr){
        native_hc_need(pending_&&pending_calls_!=UINT64_MAX&&calls_!=UINT64_MAX&&mix_calls_!=UINT64_MAX,"native HC pending: owner/counter required");
        auto* block=producer_block?producer_block:previous_output;
        try{strata_vk_native_hc_pending_run(pending_.get(),pending_input,block,previous_inject,residual,pending_gate,pending_exp);}
        catch(...){record_input_refusal("pending",{{"input",pending_input},{"block",block},{"inject",previous_inject}});throw;}
        ++pending_calls_;
        read();
    }
};
}
