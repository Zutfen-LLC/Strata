#include "native_hc_rows.hpp"
#include <cstdio>
#include <stdexcept>
using namespace strata::core;
using namespace strata::core::portable_detail;
static void check(bool x) { if (!x) throw std::runtime_error("row contract failed"); }
template<class F> static void reject(F f) { bool ok=false; try { f(); } catch (const std::exception&) { ok=true; } check(ok); }
static PortableHcPendingWrite tuple(float n, bool pending=true) {
    PortableHcPendingWrite v; v.residual.assign(10240,n);
    if (pending) { v.block.assign(2560,n+10); v.injection.assign(4,n+20); }
    return v;
}
int main() { try {
    NativeHcRows rows; auto a=tuple(1), b=tuple(2);
    rows.begin(2,10,2); reject([&]{rows.stage(1,11,101,b);});
    rows.stage(0,10,100,a); a.residual[0]=99; a.block[0]=99;
    reject([&]{rows.publish();}); check(rows.size()==0);
    rows.stage(1,11,101,b); rows.publish(); b.injection[0]=99;
    check(rows.size()==2 && rows.row(0).position==10 && rows.row(0).token==100 &&
          rows.row(0).tuple.residual[0]==1 && rows.row(0).tuple.block[0]==11 &&
          rows.row(1).position==11 && rows.row(1).token==101 && rows.row(1).tuple.injection[0]==22);
    reject([&]{rows.begin(2,13,1);}); reject([&]{rows.begin(3,12,1);});
    // A bad second row must not replace either row of the previous publication.
    rows.begin(2,12,2); rows.stage(0,12,102,tuple(3));
    auto bad=tuple(4); bad.injection.pop_back();
    reject([&]{rows.stage(1,13,103,bad);});
    reject([&]{rows.publish();});
    check(rows.size()==2 && rows.row(0).tuple.residual[0]==1 && rows.row(1).tuple.injection[0]==22);
    rows.abort();
    rows.begin(2,12,1); rows.stage(0,12,102,tuple(3));
    check(rows.row(0).token==100 && rows.row(1).token==101); rows.abort();
    check(rows.size()==2 && rows.row(0).tuple.block[0]==11);
    rows.begin(2,12,1); rows.stage(0,12,102,tuple(3)); rows.publish();
    check(rows.size()==1 && rows.row(0).token==102 && rows.row(0).tuple.block[0]==13);
    reject([&]{rows.row(1);}); rows.reset(); check(rows.size()==0);
    for (int layer : {0,1}) {
        rows.begin(layer,0,1); reject([&]{rows.stage(0,0,4,tuple(4));});
        rows.stage(0,0,4,tuple(4,false)); rows.publish(); check(rows.row(0).tuple.block.empty()); rows.reset();
    }
    rows.begin(2,0,1); reject([&]{rows.stage(0,0,4,tuple(4,false));}); rows.abort();
    reject([&]{rows.begin(48,0,1);}); reject([&]{rows.begin(2,UINT64_MAX,1);});
    rows.begin(2,0,2); rows.stage(0,0,5,tuple(5)); rows.stage(1,1,6,tuple(6)); rows.publish();
    check(rows.row(0).token==5 && rows.row(1).tuple.block[0]==16);
    std::puts("native HC bounded row ownership passed"); return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; } }
