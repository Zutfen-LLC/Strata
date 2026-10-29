#include "portable_model_fixture.hpp"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iostream>
using namespace strata::core;
using namespace model_fixture;
namespace {
struct Trace {
    std::ostringstream text;
    std::streambuf* previous;
    Trace():previous(std::cerr.rdbuf(text.rdbuf())){}
    ~Trace(){std::cerr.rdbuf(previous);}
    std::vector<std::string> rows() {
        std::istringstream in(text.str());std::vector<std::string> result;std::string row;
        while(std::getline(in,row))if(row.rfind("expert_route ",0)==0)result.push_back(row);
        std::sort(result.begin(),result.end());return result;
    }
};
void near(const std::vector<float>& a,const std::vector<float>& b,double& maximum,double& square,double& norm) {
    check(a.size()==b.size(),"chunk full-vocabulary size");
    // Existing portable Q8 matrix gate, fixed before observing this execution.
    for(size_t i=0;i<a.size();++i) {
        const double delta=double(a[i])-b[i];
        check(std::isfinite(a[i])&&std::abs(delta)<=0.003*(1+std::abs(b[i])),"assembled chunk numerical gate");
        maximum=std::max(maximum,std::abs(delta));square+=delta*delta;norm+=double(b[i])*b[i];
    }
}
}
int main() {
    try {
        SparseModel fixture;PortableModelOptions options;options.context=10;
        options.prompt_head_elision=true;options.expert_timings=true;options.expert_record_routes=true;
        std::string error;const auto manifest=(fixture.root/"manifest").string();
        const std::vector<int32_t> prompt{5,248046,6,248044,248056,7,0,5,6};
        std::vector<float> requested,continuation;std::vector<std::string> reference_routes;
        PortableGeneration generation;
        size_t weights=0;
        {
            auto serial=PortableModelSession::open(fixture.paths,manifest,options,error);check(bool(serial),error);
            weights=serial->report().host_weights;
            {
                Trace trace;
                for(size_t t=0;t+1<prompt.size();++t)check(serial->advance(prompt[t],error),error);
                check(serial->step(prompt.back(),requested,error)&&serial->step(5,continuation,error),error);
                reference_routes=trace.rows();
            }
            check(serial->reset(error)&&serial->generate(prompt,1,generation,error),error);
        }
        double maximum=0,square=0,norm=0;
        for(size_t width=2;width<=8;++width) {
            options.prompt_chunk_size=width;
            auto session=PortableModelSession::open(fixture.paths,manifest,options,error);check(bool(session),error);
            check(session->report().chunk_memory_is_shape_envelope&&session->report().host_weights==weights,
                  "pre-admitted shape envelope retains exactly one dense weight owner");
            for(bool segmented:{false,true}) {
                check(session->reset(error),error);const auto before=session->model_work();
                std::vector<float> got,next;std::vector<std::string> routes;
                size_t chunks=0;
                {
                    Trace trace;size_t first=0;
                    while(first+1<prompt.size()) {
                        const size_t T=segmented&&first==0?1:std::min(width,prompt.size()-1-first);
                        check(session->advance_chunk({prompt.begin()+first,prompt.begin()+first+T},error),error);
                        check(!session->ended(),"input tokenizer/PLE EOS terminated chunk advancement");
                        first+=T;++chunks;
                    }
                    check(session->step(prompt.back(),got,error)&&session->step(5,next,error),error);
                    routes=trace.rows();
                }
                near(got,requested,maximum,square,norm);near(next,continuation,maximum,square,norm);
                check(routes==reference_routes,"assembled per-token/layer route IDs differ");
                const auto after=session->model_work();const size_t forwards=chunks+2;
                check(after.logical_tokens-before.logical_tokens==10&&after.prompt_chunks-before.prompt_chunks==chunks&&
                      after.prompt_chunk_tokens-before.prompt_chunk_tokens==8&&after.completed_forward_calls-before.completed_forward_calls==forwards,
                      "model logical versus forward counts");
                check(after.chunk_ffn_pre_calls-before.chunk_ffn_pre_calls==48*chunks&&
                      after.chunk_provider_calls-before.chunk_provider_calls==48*chunks&&
                      after.chunk_ffn_post_calls-before.chunk_ffn_post_calls==48*chunks&&
                      after.gdn_calls-before.gdn_calls==36*forwards&&after.qsa_calls-before.qsa_calls==12*forwards,
                      "actual model layer/pre/provider/post counters");
                check(after.heads_skipped-before.heads_skipped==8&&after.heads_executed-before.heads_executed==2,
                      "final requested/continuation heads remain T1");
                check(session->expert_work().batch_max_nt==width,"native ExpertJobMulti actual nt>1 through assembled model");
                // B between A replays; state must not leak across reset.
                check(session->reset(error)&&session->advance_chunk({6,5},error)&&session->step(7,next,error),error);
            }
            check(session->reset(error),error);PortableGeneration got;
            check(session->generate(prompt,1,got,error)&&got.tokens==generation.tokens&&got.stop==generation.stop&&
                  session->position()==10,"generate layer-major prompt and terminal feed parity");
            if(width==2) {
                for(const auto& tiny:std::vector<std::vector<int32_t>>{{5},{5,6},{5,6,7}}) {
                    check(session->reset(error)&&session->generate(tiny,0,got,error),error);
                    check(session->position()==tiny.size()&&got.tokens.empty(),"short prompt/T1-tail max-new zero");
                }
                check(session->reset(error),error);
                const auto before=session->model_work();
                const auto path=fixture.paths[3];const auto time=std::filesystem::last_write_time(path);
                const uint8_t corrupt=1,restored=0;
                fixture.patch(3,"blk.10.ffn_gate_exps.weight",0,&corrupt,1);
                std::filesystem::last_write_time(path,time+std::chrono::seconds(2));
                check(!session->advance_chunk({5,6},error)&&session->poisoned()&&session->position()==0&&
                      error.find("layer 10")!=std::string::npos&&error.find("digest mismatch")!=std::string::npos,"late chunk failure poisons without position commit");
                const auto failed=session->model_work();
                check(failed.prompt_chunks==before.prompt_chunks&&failed.prompt_chunk_tokens==before.prompt_chunk_tokens&&
                      failed.completed_forward_calls==before.completed_forward_calls&&failed.heads_skipped==before.heads_skipped,
                      "failed model chunk published completion receipts");
                fixture.patch(3,"blk.10.ffn_gate_exps.weight",0,&restored,1);
                std::filesystem::last_write_time(path,time+std::chrono::seconds(4));
                check(!session->advance_chunk({5,6},error),"poisoned chunk retry refused");
                std::vector<float> recovered;
                check(session->reset(error)&&session->advance_chunk({5,6},error)&&session->step(6,recovered,error),error);
            }
        }
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> selected(ggml_backend_cpu_init(),ggml_backend_free);
        options.accelerators={selected.get()};options.device_caps={1};options.prompt_chunk_size=2;
        check(!PortableModelSession::open(fixture.paths,manifest,options,error)&&
              error.find("next module exceeds separate die")!=std::string::npos,"combined chunk reservation fails before layer allocation");
        std::printf("portable_model_chunk_test passed: T2-8/tails/segmentation, exact routes, A/B/A, late poison/reset, pre-admission and native nt>1; max_abs=%.9g relative_l2=%.9g (existing Q8 gate)\n",maximum,std::sqrt(square/(norm+1e-30)));
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
