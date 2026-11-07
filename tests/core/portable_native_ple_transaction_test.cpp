#include "../../src/core/portable_native_ple_transaction.hpp"
#include <cassert>
#include <stdexcept>
#include <iostream>
using Tx = strata::core::portable_detail::PortableNativePleTransaction;
template<class F> void fails(F f) { bool threw=false; try { f(); } catch (const std::exception&) { threw=true; } assert(threw); }
int main() {
    static_assert(Tx::host_bound(2, 2, 2) == 512 + (29*2 + 4*2)*sizeof(float) + 2*(sizeof(int32_t)+sizeof(std::array<int32_t,2>)));
    static_assert(Tx::host_bound(1, 2, 2) == Tx::host_bound(2, 2, 2));
    static_assert(Tx::host_bound_with_gather(1, 2, 2) == Tx::host_bound(1, 2, 2) + 5120*sizeof(float) + sizeof(std::array<uint32_t,16>));
    static_assert(Tx::host_bound_with_gather(2, 2, 2) == Tx::host_bound(2, 2, 2) + 7680*sizeof(float) + 2*sizeof(std::array<uint32_t,16>));
    fails([]{ (void)Tx::host_bound(0, 1, 1); });
    fails([]{ (void)Tx::host_bound(3, 1, 1); });
    fails([]{ (void)Tx::host_bound(1, 0, 1); });
    fails([]{ (void)Tx::host_bound(1, 1, 0); });
    fails([]{ (void)Tx::host_bound_with_gather(std::numeric_limits<size_t>::max(), 1, 1); });
    fails([]{ (void)Tx::host_bound(2, std::numeric_limits<size_t>::max(), 1); });
    fails([]{ (void)Tx::host_bound(2, 1, std::numeric_limits<size_t>::max()); });
    // CPU seam stands in for native full-T execution. Its canary formula is
    // deliberately test-only and exercises causal dilation 1, 3 and 9.
    bool broken=false, failure=false, bad_history=false, incomplete=false;
    auto exec = [&](const Tx::Request& r, const Tx::OutputView& o) {
        if (failure) throw std::runtime_error("downstream execution failed");
        std::copy(r.history.begin(), r.history.end(), o.history.begin());
        for (size_t t=0;t<r.tokens.size();++t) {
            for (size_t c=0;c<2;++c) {
                float n=float(r.tokens[t]*10+c+1);
                float y=n + o.history[9*c+8]*2 + o.history[9*c+6]*3 + o.history[9*c]*5;
                o.normalized[2*t+c]=n; o.residual[2*t+c]=y + r.residual_input[2*t+c];
            }
            for (size_t c=0;c<2;++c) {
                for (size_t j=0;j<8;++j) o.history[9*c+j]=o.history[9*c+j+1];
                o.history[9*c+8]=o.normalized[2*t+c];
            }
        }
        if (broken) o.residual[0]=std::numeric_limits<float>::quiet_NaN();
        if (bad_history) o.history[0]=123;
        if (incomplete) o.normalized[0]=std::numeric_limits<float>::quiet_NaN();
    };
    Tx a(2,2,4096,exec), b(2,2,4096,exec), foreign(2,2,4096,exec);
    auto x=a.stage(0,{1,2},std::vector<float>(4));
    assert((x.predecessors()==std::vector<std::array<int32_t,2>>{{-1,-1},{-1,1}}));
    assert((x.residual()==std::vector<float>{11,12,43,46}));
    assert(a.position()==0 && a.history()[8]==0);
    fails([&]{a.commit(x,false);}); assert(a.live(x));
    fails([&]{foreign.commit(x,true);});
    a.commit(x,true); assert(!a.live(x)); fails([&]{a.commit(x,true);});
    for (int i=1;i<=2;++i) { auto z=b.stage(i-1,{i},std::vector<float>(2)); b.commit(z,true); }
    assert(a.history()==b.history() && a.previous()==b.previous());
    auto y=a.stage(2,{3},std::vector<float>(2)); auto control=b.stage(2,{3},std::vector<float>(2));
    assert(y.residual()==control.residual()); assert((y.predecessors()[0]==std::array<int32_t,2>{1,2}));
    a.abort(y); assert(a.position()==2); fails([&]{a.commit(y,true);});
    b.commit(control,true);
    auto retry=a.stage(2,{3},std::vector<float>(2)); a.commit(retry,true); assert(a.history()==b.history());
    // Fill the nine-row window to exercise its oldest dilation tap.
    for (int i=4;i<=12;++i) { auto p=a.stage(i-1,{i},std::vector<float>(2)); a.commit(p,true); }
    auto p=a.stage(12,{13},std::vector<float>(2)); assert(p.residual()[0] == 131+121*2+101*3+41*5);
    a.reset(); assert(!a.live(p)); fails([&]{a.abort(p);}); assert(a.position()==0 && a.history()[8]==0);
    broken=true; fails([&]{a.stage(0,{4,5},std::vector<float>(4));}); broken=false;
    bad_history=true; fails([&]{a.stage(0,{4},std::vector<float>(2));}); bad_history=false;
    incomplete=true; fails([&]{a.stage(0,{4},std::vector<float>(2));}); incomplete=false;
    failure=true; fails([&]{a.stage(0,{4},std::vector<float>(2));}); failure=false;
    fails([&]{a.stage(1,{4},std::vector<float>(2));}); fails([&]{a.stage(0,{4,-2},std::vector<float>(4));});
    fails([&]{a.stage(0,{4},{});});
    fails([&]{a.stage(0,{4},{0.0f, std::numeric_limits<float>::infinity()});});
    auto input=a.stage(0,{4},{3.0f,4.0f});
    assert((input.residual()==std::vector<float>{44.0f,46.0f})); a.abort(input);
    assert(a.position()==0 && a.history()[8]==0);
    // The transaction's independently admitted storage has an exact formula;
    // test both boundaries and both multiplication overflow guards.
    constexpr size_t minimum = Tx::host_bound(2, 2, 2);
    size_t calls = 0;
    const void *token_bank=nullptr, *pred_bank=nullptr, *input_bank=nullptr, *norm_bank=nullptr, *residual_bank=nullptr;
    static_assert(!std::is_assignable<Tx::Span&, Tx::Span>::value, "spans cannot be replaced");
    auto counted = [&](const Tx::Request& r, const Tx::OutputView& o) {
        ++calls;
        if (calls == 1) { token_bank=r.tokens.data(); pred_bank=r.predecessors.data(); input_bank=r.residual_input.data(); norm_bank=o.normalized.data(); residual_bank=o.residual.data(); }
        else { assert(token_bank==r.tokens.data() && pred_bank==r.predecessors.data() && input_bank==r.residual_input.data() && norm_bank==o.normalized.data() && residual_bank==o.residual.data()); }
        assert(o.normalized.size()==r.tokens.size()*2 && o.history.size()==18 && o.residual.size()==r.tokens.size()*2);
        fails([&]{ o.normalized[o.normalized.size()] = 1; });
        fails([&]{ o.history[o.history.size()] = 1; });
        fails([&]{ o.residual[o.residual.size()] = 1; });
        fails([&]{ o.normalized[std::numeric_limits<size_t>::max()] = 1; });
        exec(r,o);
    };
    fails([&]{Tx too_small(2,2,minimum-1,counted);});
    assert(calls == 0);
    Tx exact(2,2,minimum,counted);
    fails([&]{exact.stage(0,{},{});});
    fails([&]{exact.stage(0,{1,2,3},std::vector<float>(6));});
    fails([&]{exact.stage(0,{1,2},std::vector<float>(3));});
    assert(calls == 0);
    auto admitted = exact.stage(0,{1,2},std::vector<float>(4));
    assert(calls == 1);
    exact.commit(admitted, true);
    failure=true; fails([&]{exact.stage(2,{3},std::vector<float>(2));}); failure=false;
    auto one=exact.stage(2,{3},std::vector<float>(2));
    assert(one.token_count()==1 && one.residual_count()==2);
    exact.abort(one);
    auto two=exact.stage(2,{3,4},std::vector<float>(4));
    fails([&]{ (void)one.residual(); }); fails([&]{ (void)admitted.tokens(); });
    fails([&]{ exact.commit(one,true); });
    assert(two.token_count()==2 && two.residual_count()==4);
    exact.commit(two, true);
    fails([&]{Tx overflow_channels(std::numeric_limits<size_t>::max(),1,
                                    std::numeric_limits<size_t>::max(),exec);});
    fails([&]{Tx overflow_width(1,std::numeric_limits<size_t>::max(),
                                std::numeric_limits<size_t>::max(),exec);});
    auto orphan = [&] { Tx producer(2,2,minimum,exec); return producer.stage(0,{1},std::vector<float>(2)); }();
    fails([&]{ (void)orphan.history(); });
    fails([&]{ exact.commit(orphan,true); });
    std::cout << "PLE transaction CPU seam: OK\n";
}
