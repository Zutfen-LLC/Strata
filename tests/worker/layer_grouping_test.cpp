#include "strata/worker/layer_grouping.hpp"
#include <cstdio>
#include <sys/resource.h>
#include <unistd.h>
namespace G=strata::worker::diagnostic::grouping;
namespace {
void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
}
int main(){
    const rlimit memory{128ull<<20,128ull<<20},cpu{5,5},file{1ull<<20,1ull<<20};
    if(setrlimit(RLIMIT_AS,&memory)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_FSIZE,&file))return 2;
    alarm(10);
    try{
        check(G::seed({{4,false,false},{1,true,false},{0,true,true},{2,false,false}})==1,"earliest identical-input BIT boundary, not material boundary");
        check(G::seed({{1,false,false},{2,true,true}})==-1,"no invented common input");
        check(!G::exact({0.f},{-0.f}),"exact input means bytes, including signed zero");
        // Moving the small middle term into a separate partial loses it in one
        // entry-order reduction but preserves it in the other grouping.
        const std::vector<float> rows{16777216.f,1.f,-16777216.f},weights{1,1,1},shared{0};
        const auto b=G::reduce(rows,weights,{3,3,3},shared,G::Local::NativeFma);
        const auto g=G::reduce(rows,weights,{3,4,3},shared,G::Local::NativeFma);
        check(b.final[0]==0.f&&g.final[0]==1.f,"same rows, different CPU/GPU grouping");
        const auto residual=G::delta(b.final,g.final,b.final,g.final);
        check(residual.equal==1&&residual.actual_max==1&&residual.residual_max==0,"exact predicted versus actual delta");
        check(G::delta(b.final,g.final,b.final,b.final).residual_max==1,"wrong grouping cannot report reproduced delta");
        const std::vector<float> helper{7};
        const auto anchored=G::reduce({123}, {1}, {2}, shared,G::Local::NativeFma,&helper);
        check(anchored.helper[0]==7&&anchored.final[0]==7,"actual helper anchor does not invent per-expert rows");
        const auto local=G::local(rows,weights,{2},G::Local::NativeFma);
        check(local[0]==2,"shared added after routed reduction");
        check(G::local(rows,weights,{2},G::Local::Float64)[0]==3,"float64 fallback kept distinct from native FMA");
        const float a=1.f+0x1p-23f,c=-(1.f+0x1p-22f);
        check(G::local({c,a},{1,a},{0},G::Local::NativeFma)[0]==std::fma(a,a,c),"explicit FMA after rounded first product");
        bool refused=false;try{(void)G::reduce({1},{1},{5},{0},G::Local::NativeFma);}catch(...){refused=true;}
        check(refused,"invalid owner rejected");
        refused=false;try{(void)G::delta({0},{NAN},{0},{0});}catch(...){refused=true;}
        check(refused,"nonfinite delta rejected");
        std::puts("LAYER_GROUPING_FIXTURES_V1 PASS models=0 cuda=0");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
