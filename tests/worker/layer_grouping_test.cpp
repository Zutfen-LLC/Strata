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
        check(G::moved_cpu_to_gpu({0,3,2,3},{0,3,2,4})==3,"moved entry derived from owners, not expert number");
        refused=false;try{(void)G::moved_cpu_to_gpu({3,3},{4,4});}catch(...){refused=true;}
        check(refused,"multiple ownership changes cannot invent four supported arms");
        refused=false;try{(void)G::moved_cpu_to_gpu({0,3},{0,3});}catch(...){refused=true;}
        check(refused,"missing moved entry cannot use a hardcoded expert");
        refused=false;try{(void)G::moved_cpu_to_gpu({0,3},{4,3});}catch(...){refused=true;}
        check(refused,"a local-to-GPU change is outside the verified CPU-row experiment");
        const std::vector<float> fixed_local{3},fixed_helper{4};
        const auto cb=G::anchored(rows,weights,{3,3,3},fixed_local,fixed_helper);
        const auto cg=G::anchored(rows,weights,{3,4,3},fixed_local,fixed_helper);
        auto cuda_rows=rows;cuda_rows[1]=2;
        const auto gb=G::anchored(cuda_rows,weights,{3,3,3},fixed_local,fixed_helper);
        const auto gg=G::anchored(cuda_rows,weights,{3,4,3},fixed_local,fixed_helper);
        check(cb.local==fixed_local&&gg.local==fixed_local&&cb.helper==fixed_helper&&gg.helper==fixed_helper,"captured anchors reused without local/helper recomputation");
        check(cb.final[0]==7&&cg.final[0]==8&&gb.final[0]==9&&gg.final[0]==9,"four row/order cells with all other rows fixed");
        const auto effect=G::effects(cb.final,cg.final,gb.final,gg.final);
        check(effect.row_baseline[0]==2&&effect.row_distributed[0]==1&&effect.group_cpu[0]==1&&effect.group_cuda[0]==0,"conditional row/order effects");
        check(effect.row_main[0]==1.5&&effect.group_main[0]==.5&&effect.interaction[0]==-1,"main effects and difference-in-differences interaction");
        std::puts("LAYER_FOUR_WAY_FIXTURES_V1 PASS models=0 cuda=0");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
