#include "portable_native_layer0_composition.hpp"
#include <type_traits>
static_assert(!std::is_copy_constructible_v<strata::core::portable_detail::PortableNativeLayer0Rows>);
static_assert(!std::is_copy_assignable_v<strata::core::portable_detail::PortableNativeLayer0Rows>);
static_assert(!std::is_move_assignable_v<strata::core::portable_detail::PortableNativeLayer0Rows>);
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <vector>
using namespace strata::core;
using namespace strata::core::portable_detail;
static void check(bool ok){if(!ok)throw std::runtime_error("layer0 embedding test");}
int main() {
    std::vector<uint8_t> table(3*80*34);
    for(size_t row=0;row<3;++row) for(size_t b=0;b<80;++b) {
        auto* p=table.data()+(row*80+b)*34;
        p[0]=0; p[1]=0x3c; // IEEE half 1.0
        for(size_t k=0;k<32;++k) p[2+k]=uint8_t(int8_t(int(row*3+b%3+k%3)-4));
    }
    PortableTensorView view{"token_embd.weight",{2560,3},8,table.data(),table.size()};
    auto both=native_layer0_embedding(view,{2,1});
    auto a=native_layer0_embedding(view,{2}); auto b=native_layer0_embedding(view,{1});
    check(both.size()==20480 && std::equal(a.begin(),a.end(),both.begin()) && std::equal(b.begin(),b.end(),both.begin()+10240));
    check(a[0]==2.f && b[0]==-1.f);
    for(size_t r=0;r<2;++r)for(size_t s=1;s<4;++s)
        check(std::equal(both.begin()+r*10240,both.begin()+r*10240+2560,both.begin()+r*10240+s*2560));
    auto refuse=[&](const PortableTensorView& v,const std::vector<int32_t>& t){try{(void)native_layer0_embedding(v,t);}catch(const std::exception&){return true;}return false;};
    check(refuse(view,{3}) && refuse(view,{-1}) && refuse(view,{0,1,2}));
    auto bad=view;--bad.bytes;check(refuse(bad,{0}));
    table[0]=0;table[1]=0x7c;check(refuse(view,{0}));
}
