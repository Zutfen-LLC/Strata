#pragma once
#include "portable_ffn_fixture.hpp"
#include <cstring>
#include <limits>

// Existing fixture tolerances, fixed before running the multirow kernels.
// CPU F32 2e-5; Vulkan F32 3e-4; Q8 0.003. Record observed deltas separately.
inline void ffn_batch_fixture(PortableFfnOptions options) {
    Fixture f;
    std::string error;
    const PortableExpertRows single = [](int64_t, const PortableRoute& r, std::vector<float>& rows, std::string&) {
        rows.resize(64);
        for(size_t k=0;k<2;++k) for(size_t h=0;h<32;++h)
            rows[k*32+h]=float(r.ids[k]+1)*(0.03f+float(h)*0.001f);
        return true;
    };
    for(bool q8:{false,true}) {
        auto views=f.views;
        std::vector<std::vector<uint8_t>> quantized(9);
        if(q8) for(size_t i:{size_t(1),size_t(2),size_t(3),size_t(6),size_t(7),size_t(8)}) {
            quantized[i].resize(f.data[i].size()/32*34);
            ggml_quantize_chunk(GGML_TYPE_Q8_0,f.data[i].data(),quantized[i].data(),0,
                int64_t(f.data[i].size()/size_t(views[i].shape[0])),views[i].shape[0],nullptr);
            views[i].type=8;views[i].data=quantized[i].data();views[i].bytes=quantized[i].size();
        }
        const float tolerance=q8?0.003f:options.request_accelerator?3e-4f:2e-5f;
        double maximum=0, square=0, norm=0;
        size_t exact=0, elements=0;
        for(int64_t T=1;T<=8;++T) {
            auto batch=PortableFfn::create(f.g,0,views,options,error,T);
            check(bool(batch) && batch->tokens()==T,error);
            auto serial=PortableFfn::create(f.g,0,views,options,error);
            check(bool(serial),error);
            std::vector<float> input(size_t(T)*128), expected;
            std::vector<PortableRoute> reference;
            for(size_t t=0;t<size_t(T);++t) {
                // Distinct streams/tokens expose accidental normalization or broadcast across T.
                std::vector<float> row(128);
                for(size_t h=0;h<128;++h) row[h]=0.2f+float(int((h*7+t*11)%31)-15)*0.06f;
                std::copy(row.begin(),row.end(),input.begin()+t*128);
                PortableRoute r;
                check(serial->reset(row,error)&&serial->step(single,r,error),error);
                expected.insert(expected.end(),serial->residual().begin(),serial->residual().end());
                reference.push_back(r);
            }
            size_t calls=0;
            const PortableExpertRowsBatch provider=[&](int64_t l,const std::vector<PortableRoute>& routes,
                                                      std::vector<float>& rows,std::string& e) {
                ++calls;check(routes.size()==size_t(T),"batch callback T");
                for(const auto& r:routes) {
                    std::vector<float> token;check(single(l,r,token,e),e);
                    rows.insert(rows.end(),token.begin(),token.end());
                }
                return true;
            };
            std::vector<PortableRoute> routes;
            check(batch->reset(input,error)&&batch->step_batch(provider,routes,error),error);
            check(calls==1,"one real batch provider call");
            close(batch->residual(),expected,tolerance);
            for(size_t t=0;t<size_t(T);++t) {
                close(routes[t].activation,reference[t].activation,tolerance);
                close(routes[t].weights,reference[t].weights,tolerance);
                check(routes[t].ids==reference[t].ids,"batch route IDs");
            }
            for(size_t h=0;h<expected.size();++h) {
                const double delta=double(batch->residual()[h])-expected[h];
                maximum=std::max(maximum,std::abs(delta));square+=delta*delta;norm+=double(expected[h])*expected[h];
                ++elements;exact+=std::memcmp(&batch->residual()[h],&expected[h],sizeof(float))==0;
            }
            const auto work=batch->work();
            check(work.logical_rows==uint64_t(T)&&work.completed_chunks==1&&work.pre_graph_calls==1&&
                  work.provider_calls==1&&work.post_graph_calls==1,"graph executions distinct from logical rows");
            size_t multirow=0;
            for(const auto& p:batch->placements()) if(p.operation=="MUL_MAT") {
                check(p.shape.size()>=2 && p.shape[1]==T,"MUL_MAT has actual T columns");++multirow;
            }
            check(multirow==8,"all HC/router/shared matmuls covered");
            auto failed=[&](const PortableExpertRowsBatch& bad) {
                const auto before=batch->residual(); const auto prior=batch->work();
                check(!batch->step_batch(bad,routes,error)&&routes.empty(),"batch failure clears routes");
                check(std::memcmp(before.data(),batch->residual().data(),before.size()*sizeof(float))==0,
                      "batch failure cannot publish residual");
                check(batch->work().logical_rows==prior.logical_rows&&
                      batch->work().completed_chunks==prior.completed_chunks,"failed chunk has no completion receipt");
            };
            failed([](int64_t,const auto&,auto&,auto& e){e="controlled batch provider failure";return false;});
            failed([](int64_t,const auto&,auto& out,auto&){out={1};return true;});
            failed([&](int64_t l,const auto& r,auto& out,auto& e){
                provider(l,r,out,e);out[out.size()/2]=std::numeric_limits<float>::infinity();return true;
            });
            // A/B/A is a sublayer reset/replay check, not a recurrent-state claim.
            std::vector<float> b=input;for(auto& value:b)value=-value*0.7f;
            check(batch->reset(b,error)&&batch->step_batch(provider,routes,error),error);
            check(batch->reset(input,error)&&batch->step_batch(provider,routes,error),error);
            close(batch->residual(),expected,tolerance);
            auto tiny=options;tiny.graph_budget=1;
            check(!PortableFfn::create(f.g,0,views,tiny,error,T),"batch preallocation budget refusal");
        }
        std::printf("FFN batch T1-8 %s: max_abs=%.9g relative_l2=%.9g exact_elements=%zu/%zu; T-column graphs, one provider/chunk, slot merge and failure/reset passed\n",
                    q8?"Q8":"F32",maximum,std::sqrt(square/(norm+1e-30)),exact,elements);
    }
    for(int64_t T:{int64_t(0),int64_t(9)})
        check(!PortableFfn::create(f.g,0,f.views,options,error,T),"reject unsupported T before allocation");
}
