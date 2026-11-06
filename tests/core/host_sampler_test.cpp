#include "strata/core/host_sampler.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
using namespace strata::core;
int main() {
    assert(host_philox_uniform(0,0)*16777216.0f==16079353);
    assert(host_philox_uniform(0,1)*16777216.0f==5293434);
    assert(host_philox_uniform(0x123456789abcdef0ULL,0x100000002ULL)*16777216.0f==13186932);
    strata::kernels::SamplerParams p; p.greedy=true;
    std::string error; int32_t token=-1;
    assert(sample_host_logits({-0.0f,0.0f,-1.0f},{},p,{},token,error) && token==0);
    p.penalty_last_n=3; p.penalty_repeat=2; p.penalty_freq=0.5f; p.penalty_present=0.5f;
    assert(sample_host_logits({4,2,-1},{0,0,99},p,{},token,error) && token==1);
    p.penalty_last_n=1;
    assert(sample_host_logits({4,2,-1},{0,0,99},p,{},token,error) && token==0);
    p.greedy=false; p.temperature=0;
    assert(sample_host_logits({0,1},{},p,{},token,error) && token==1);
    p.temperature=1; p.top_k=1;
    assert(sample_host_logits({0,1},{},p,{},token,error) && token==1);
    // Bias precedes repeat; negative values are multiplied, positive divided.
    p.greedy=true; p.penalty_last_n=4; p.penalty_repeat=2; p.penalty_freq=0; p.penalty_present=0;
    assert(sample_host_logits({-1,-1.5f},{0},p,{0,0},token,error) && token==1);
    assert(sample_host_logits({3,2},{0},p,{1,0},token,error) && token==0);
    p.penalty_repeat=1; p.penalty_freq=0.5f; p.penalty_present=0.25f;
    assert(sample_host_logits({2,1},{0,0,-1,99},p,{},token,error) && token==1);
    p.penalty_last_n=2;
    assert(sample_host_logits({2,1},{0,0,-1,99},p,{},token,error) && token==0);
    p.greedy=false; p.penalty_last_n=0; p.top_p=1; p.min_p=0; p.temperature=1;
    std::vector<float> equal(70,0);
    p.top_k=1; assert(sample_host_logits(equal,{},p,{},token,error) && token==0);
    p.top_k=63; assert(sample_host_logits(equal,{},p,{},token,error) && token==60);
    p.top_k=64; assert(sample_host_logits(equal,{},p,{},token,error) && token==61);
    p.top_k=0; assert(sample_host_logits(equal,{},p,{},token,error) && token==61);
    p.top_p=0; p.min_keep=3;
    assert(sample_host_logits(equal,{},p,{},token,error) && token==2);
    p.min_p=1; assert(sample_host_logits({1,0,0},{},p,{},token,error) && token==0);
    // top-p sees all four untempered weights before min-p; seed 0's
    // draw lands in the third survivor, which min-p-first would discard.
    p.top_k=4; p.top_p=0.75f; p.min_p=0.3f; p.min_keep=1; p.temperature=1;
    assert(sample_host_logits({std::log(0.4f),std::log(0.3f),std::log(0.2f),std::log(0.1f)}, {},p,{},token,error) && token==2);
    // Untempered probabilities .4,.3,.2,.1: top-p .6 retains IDs 0,1.
    // At temperature 2 their weights are sqrt(.4),sqrt(.3); seed 0's
    // u=16079353/16777216 selects 1. Temperature first would need three
    // survivors for top-p .6 and the same draw would instead select 2.
    p.top_p=0.6f; p.min_p=0; p.temperature=2;
    assert(sample_host_logits({std::log(0.4f),std::log(0.3f),std::log(0.2f),std::log(0.1f)}, {},p,{},token,error) && token==1);
    // Issue-53 once-only repeat oracle: two occurrences of ID 0 transform
    // 4/2 - 2*.25 - .5 = 1, tying ID 1. With counter 1, u=.315...
    // selects tie-first ID 0. Applying repeat twice yields 0 vs 1 and ID 1.
    p.top_p=1; p.temperature=1; p.counter=1; p.penalty_last_n=2;
    p.penalty_repeat=2; p.penalty_freq=0.25f; p.penalty_present=0.5f;
    assert(sample_host_logits({4,1},{0,0},p,{},token,error) && token==0);
    p.counter=0; p.penalty_last_n=0; p.penalty_repeat=1; p.penalty_freq=0; p.penalty_present=0;
    // Threshold equality survives; min_keep does not override min-p.
    p.top_p=0; p.min_keep=4; p.min_p=1;
    assert(sample_host_logits({1,1,0,0},{},p,{},token,error) && token==1);
    p.min_p=0; p.top_p=1; p.top_k=65;
    assert(sample_host_logits(equal,{},p,{},token,error) && token==61);
    p.temperature=0; p.top_k=1;
    assert(sample_host_logits({0,3,2},{},p,{},token,error) && token==1);
    p.top_p=2; token=42;
    assert(!sample_host_logits({0,1},{},p,{},token,error) && token==42);
}
