#include "../../src/core/portable_memory.hpp"
#include <cstdio>
#include <limits>
using namespace strata::core;
namespace {
void check(bool b,const char* e){if(!b)throw std::runtime_error(e);}
PortableModelReport fresh(){
    PortableModelReport r;r.device_admission={0,0};r.device_scheduler={0,0};
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
        auto overflow=fresh();overflow.device_admission[0]=std::numeric_limits<size_t>::max()-1;
        overflow.device_scheduler[0]=overflow.device_admission[0];
        q={};q.device_graph=2;refused=false;
        try{portable_detail::account_device_memory(overflow,0,std::numeric_limits<size_t>::max(),q);}catch(const std::exception&){refused=true;}
        check(refused,"overflow-safe cap subtraction");
        std::puts("portable_memory_admission_test passed: serial temporary peak, persistent/private sums, per-die caps, overlap and overflow");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
