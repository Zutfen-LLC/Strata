#include "native_qsa_wo_domain.hpp"
#include <cassert>
#include <limits>
using namespace strata::core::portable_detail;
template<class F>void refuses(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}assert(failed);}
int main(){
    std::array<uint8_t,4*34> blocks{};
    blocks[34+1]=0x3c;blocks[2*34]=1;blocks[3*34+1]=0x7c;
    auto c=qsa_wo_scale_classes(blocks.data(),4,34);
    assert(c.zero==1&&c.normal==1&&c.subnormal==1&&c.nonfinite==1);
    refuses([&]{qsa_wo_require_scales(c,"test");});
    std::array<uint8_t,192*36> packed{};qsa_wo_packed_domain(packed.data(),packed.size());
    packed[36]=1;refuses([&]{qsa_wo_packed_domain(packed.data(),packed.size());});
    refuses([&]{qsa_wo_packed_domain(packed.data(),packed.size()-1);});
    std::vector<float> input(6144,1.f);qsa_wo_input_domain(input);
    input[1]=std::numeric_limits<float>::denorm_min();refuses([&]{qsa_wo_input_domain(input);});
    input[1]=std::numeric_limits<float>::infinity();refuses([&]{qsa_wo_input_domain(input);});
    input.assign(6144,std::numeric_limits<float>::max());refuses([&]{qsa_wo_input_domain(input);});
    input.assign(6144,0.f);qsa_wo_input_domain(input);
}
