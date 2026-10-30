#include "../../src/core/portable_memory.hpp"
#include <cstdio>
#include "ggml-cpu.h"
#include <limits>
using namespace strata::core;
namespace {
void check(bool b,const char* e){if(!b)throw std::runtime_error(e);}
PortableModelReport fresh(){
    PortableModelReport r;r.device_admission={0,0};r.device_scheduler={0,0};r.device_state={0,0};r.device_weights={0,0};
    r.device_workspace_bound={0,0};r.device_shared_conversion_bound={0,0};r.device_token_temporary_bound={0,0};return r;
}
}
int main(){
    try{
        constexpr size_t M=1u<<20;
        auto r=fresh();PortableMemoryUsage q;
        q.device_graph=M;q.device_extra_bound=2*M;q.device_token_temporary_bound=256*M;
        // Eight live fixed schedulers, but eight serial stack-scoped QSA phases.
        for(int i=0;i<8;++i)portable_detail::account_device_memory(r,0,512*M,q);
        check(r.device_admission[0]==280*M,"persistent sum plus one temporary peak");
        check(r.device_workspace_bound[0]==16*M && r.device_token_temporary_bound[0]==256*M,
              "private workspace must not be max-reduced with temporary");
        check(r.device_admission[1]==0,"independent physical-die ledger");
        auto other=q;other.device_token_temporary_bound=128*M;
        portable_detail::account_device_memory(r,1,512*M,other);
        check(r.device_admission[1]==131*M,"other die has independent peak");
        // A larger later phase replaces only the earlier peak contribution.
        q.device_token_temporary_bound=300*M;
        portable_detail::account_device_memory(r,0,512*M,q);
        check(r.device_admission[0]==327*M,"peak growth charged once");
        const auto before=r;
        bool refused=false;
        try{portable_detail::account_device_memory(r,0,328*M,q);}catch(const std::exception&){refused=true;}
        check(refused && r.device_admission==before.device_admission &&
              r.device_token_temporary_bound==before.device_token_temporary_bound,"cap refusal transactional");
        // Resident tensors/private expert workspace overlap QSA until layer returns.
        const size_t with_resident=r.device_admission[0]+160*M+32*M;
        check(with_resident==519*M && with_resident>512*M,"experts cannot max-share QSA temporary");
        auto shared=fresh();q={};q.device_graph=M;q.device_extra_bound=8*M;
        q.device_shared_conversion_bound=100*M;q.device_token_temporary_bound=20*M;
        for(int i=0;i<12;++i)portable_detail::account_device_memory(shared,0,512*M,q);
        check(shared.device_admission[0]==228*M && shared.device_workspace_bound[0]==96*M,
              "verified context conversion peak plus additive other private allowances");
        q.device_shared_conversion_bound=120*M;
        portable_detail::account_device_memory(shared,0,512*M,q);
        check(shared.device_shared_conversion_bound[0]==120*M && shared.device_admission[0]==257*M,
              "later conversion growth, not max of total scratch");
        auto weights=fresh();PortableMemoryUsage resident;
        resident.device_weights=65*M;resident.device_graph=20*M;resident.device_extra_bound=8*M;
        portable_detail::account_device_memory(weights,0,93*M,resident);
        check(weights.device_admission[0]==93*M && weights.device_weights[0]==65*M &&
              weights.device_scheduler[0]==20*M,"exact-cap resident weights separate from scheduler");
        const auto saved=weights;bool weight_refused=false;
        try{portable_detail::account_device_memory(weights,0,185*M,resident);}
        catch(const std::exception&){weight_refused=true;}
        check(weight_refused && weights.device_admission==saved.device_admission &&
              weights.device_weights==saved.device_weights && weights.device_scheduler==saved.device_scheduler &&
              weights.device_workspace_bound==saved.device_workspace_bound,"resident refusal preserves every ledger");
        check(weights.device_weights[1]==0,"resident weights remain per-die");
        auto states=fresh();PortableMemoryUsage state;
        state.device_state=4*M;state.device_graph=M;
        portable_detail::account_device_memory(states,0,10*M,state);
        portable_detail::account_device_memory(states,0,10*M,state);
        portable_detail::account_device_memory(states,1,5*M,state);
        check(states.device_state[0]==8*M && states.device_state[1]==4*M &&
              states.device_admission[0]==10*M && states.device_scheduler[0]==2*M,
              "persistent state sums once per layer, independently per die");
        const auto saved_state=states;refused=false;
        try{portable_detail::account_device_memory(states,0,14*M,state);}catch(const std::exception&){refused=true;}
        check(refused && states.device_state==saved_state.device_state &&
              states.device_admission==saved_state.device_admission && states.device_scheduler==saved_state.device_scheduler,
              "state admission refusal transactional");
        // A cap that admitted the previous QSA constructor bound must now
        // reject before any new layer/device storage is allocated. Preflight
        // uses the selected backend's aligned estimate, not just raw payload.
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
        PortableModelOptions opts; opts.context=3; opts.qsa_production=true;
        opts.gdn_resident_state=true;
        const size_t estimated=PortableQsaLayer::device_state_allocation_size(cpu.get(),opts.context);
        check(estimated>2048*opts.context+512*(opts.context/4+1)+512+2048+8*opts.context,
              "small-capacity estimate includes backend padding beyond raw payload");
        for (size_t layer : {size_t(3),size_t(7),size_t(47)})
            check(portable_detail::layer_state_preflight_bytes(opts,layer,cpu.get())==estimated,
                  "every QSA layer includes capacity/aligned state before allocation");
        check(portable_detail::layer_state_preflight_bytes(opts,0,cpu.get())==PortableGdnOptions{}.resident_state_budget,
              "GDN preflight bound retained");
        check(portable_detail::layer_state_preflight_bytes(opts,3,nullptr)==0,"no selected device has no state charge");
        opts.qsa_production=false;
        check(portable_detail::layer_state_preflight_bytes(opts,3,cpu.get())==0,"diagnostic QSA preflight unchanged");
        auto pending=fresh(); pending.device_admission[1]=17*M;
        const size_t legacy=280*M, exact_cap=pending.device_admission[1]+legacy+estimated;
        const auto pending_before=pending;
        portable_detail::preflight_device_memory(pending.device_admission[1],exact_cap,legacy+estimated);
        refused=false;
        try { portable_detail::preflight_device_memory(pending.device_admission[1],exact_cap-1,legacy+estimated); }
        catch(const std::exception&) {refused=true;}
        check(refused,"one byte below aligned QSA preflight bound rejected");
        refused=false;
        try { portable_detail::preflight_device_memory(pending.device_admission[1],exact_cap-estimated,legacy+estimated); }
        catch(const std::exception&) {refused=true;}
        check(refused,"old cap omitting QSA state rejected before allocation");
        check(pending.device_admission==pending_before.device_admission &&
              pending.device_state==pending_before.device_state,"preflight never mutates actual per-die ledger");
        PortableMemoryUsage actual;actual.device_state=estimated;actual.device_graph=M;
        portable_detail::account_device_memory(pending,1,exact_cap,actual);
        check(pending.device_state[1]==estimated && pending.device_admission[1]==17*M+estimated+M &&
              pending.device_admission[0]==0,"final actual state counted once on selected die");
        refused=false;
        try { portable_detail::preflight_device_memory(SIZE_MAX-1,SIZE_MAX,2); }
        catch(const std::exception&) {refused=true;}
        check(refused,"preflight addition overflow rejected by subtraction");
        // Refusal must retain the operands needed to diagnose a real load gate.
        std::string refusal;
        try { portable_detail::preflight_device_memory(200,300,101); }
        catch(const std::exception& e) { refusal=e.what(); }
        check(refusal.find("admitted_bytes=200")!=std::string::npos &&
              refusal.find("next_estimated_bytes=101")!=std::string::npos &&
              refusal.find("effective_limit_bytes=300")!=std::string::npos,
              "preflight refusal missing admission context");
        const portable_detail::DeviceAdmissionContext context{1,48,"output-head",320,16};
        refusal.clear();
        try { portable_detail::preflight_device_memory(200,300,101,&context); }
        catch(const std::exception& e) { refusal=e.what(); }
        for(const auto* field:{"slot=1","layer=48","module=output-head","admitted_bytes=200",
                               "next_estimated_bytes=101","policy_cap_bytes=320",
                               "effective_limit_bytes=300","reserve_bytes=16"})
            check(refusal.find(field)!=std::string::npos,"preflight diagnostic field missing");
        // Metadata must neither supply admission credit nor change exact boundaries.
        portable_detail::preflight_device_memory(200,300,100,&context);
        refused=false;
        try { portable_detail::preflight_device_memory(SIZE_MAX-1,SIZE_MAX,2,&context); }
        catch(const std::exception&) { refused=true; }
        check(refused,"diagnostic context preserves overflow refusal");
        auto overflow=fresh();overflow.device_admission[0]=std::numeric_limits<size_t>::max()-1;
        overflow.device_scheduler[0]=overflow.device_admission[0];
        q={};q.device_graph=2;refused=false;
        try{portable_detail::account_device_memory(overflow,0,std::numeric_limits<size_t>::max(),q);}catch(const std::exception&){refused=true;}
        check(refused,"overflow-safe cap subtraction");
        std::puts("portable_memory_admission_test passed: serial temporary peak, persistent/private sums, per-die caps, overlap and overflow");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
