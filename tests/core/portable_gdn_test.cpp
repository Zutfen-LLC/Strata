#include "portable_layer_chunk_fixture.hpp"
#include "gguf_fixture.hpp"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "portable_gdn_fixture.hpp"
#include "portable_gdn_production.hpp"
#include "../../src/core/portable_gdn_finite.hpp"
#include <chrono>
#include <limits>
using namespace strata::core;
using namespace gdn_fixture;
namespace {
struct CaptureCausalState {
    static inline decltype(ggml_backend_i::graph_compute) original=nullptr;
    static inline ggml_backend_buffer_t state=nullptr;
    static inline bool corrupt_projection=false;
    static ggml_status compute(ggml_backend_t backend,ggml_cgraph* graph) {
        if(!state) for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* node=ggml_graph_node(graph,i);
            if(node->op==GGML_OP_GATED_DELTA_NET) {
                auto* root=node->src[5];while(root->view_src)root=root->view_src;
                state=root->buffer;break;
            }
        }
        const auto status=original(backend,graph);
        if(corrupt_projection && status==GGML_STATUS_SUCCESS) for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto* node=ggml_graph_node(graph,i);
            if(std::string(ggml_get_name(node))=="ssm_alpha.weight") {
                const float nonfinite=std::numeric_limits<float>::quiet_NaN();
                ggml_backend_tensor_set(node,&nonfinite,0,sizeof(nonfinite));break;
            }
        }
        return status;
    }
};
void finite_predicate() {
    const uint32_t valid[] = {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
                              0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u, 0x7f7fffffu, 0xff7fffffu};
    const uint32_t invalid[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
                                0x7f800001u, 0xff800001u, 0x7fffffffu, 0xffffffffu};
    const auto equivalent = [](const std::vector<float> &x) {
        const bool scalar = std::all_of(x.begin(), x.end(), [](float f) { return std::isfinite(f); });
        check(portable_detail::gdn_all_finite(x) == scalar, "bulk/scalar finite predicate differs");
    };
    // Every short tail and failure position, including either side of SIMD
    // widths. No padding or vector-capacity values may be read as elements.
    for (size_t n = 0; n <= 65; ++n) {
        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i) x[i] = std::bit_cast<float>(valid[i % std::size(valid)]);
        equivalent(x);
        for (size_t i = 0; i < n; ++i) {
            const float saved = x[i];
            for (uint32_t bits : invalid) { x[i] = std::bit_cast<float>(bits); equivalent(x); }
            if (i + 1 < n) {
                const float next = x[i + 1];
                x[i + 1] = std::bit_cast<float>(0x7f800001u);
                equivalent(x); // multiple failures, in vector body and tail
                x[i + 1] = next;
            }
            x[i] = saved;
        }
    }
    // All exponent values, both signs and representative mantissas, followed
    // by deterministic raw patterns. This includes quiet/signaling NaN payloads.
    std::vector<float> x;
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
        for (uint32_t sign : {0u, 0x80000000u})
            for (uint32_t mantissa : {0u, 1u, 0x3fffffu, 0x400000u, 0x7fffffu}) {
                x.assign(17, std::bit_cast<float>(sign | (exponent << 23) | mantissa));
                equivalent(x);
            }
    uint32_t bits = 1;
    x.resize(4097);
    for (float &f : x) { bits = bits * 1664525u + 1013904223u; f = std::bit_cast<float>(bits); }
    equivalent(x);
}
void finite_diagnostics(PortableGdnLayer &layer, const Fixture &f) {
    const auto state = layer.recurrent_state(), history = layer.convolution_history(), last = layer.last_residual();
    const auto count = layer.tokens();
    for (uint32_t bad : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u, 0xff800001u}) {
        auto input = f.input(0);
        input[3] = std::bit_cast<float>(bad);
        input.back() = std::bit_cast<float>(0x7f800000u);
        PortableRoute route;
        PortableGdnWork work;
        std::string error;
        bool dispatched = false;
        check(!layer.step(input, [&](int64_t, const PortableRoute &, std::vector<float> &, std::string &) {
            dispatched = true; return false;
        }, route, error, nullptr, &work), "bulk accepted nonfinite input");
        check(error == "portable GDN: nonfinite input/result" && !dispatched, "scalar failure diagnostic/order changed");
        check(layer.recurrent_state() == state && layer.convolution_history() == history &&
              layer.last_residual() == last && layer.tokens() == count, "nonfinite preflight changed committed state");
        check(work.candidate_readback_calls == 0 && work.candidate_finite_calls == 0 && work.state_commit_calls == 0,
              "nonfinite preflight entered candidate timing scopes");
    }
}
void candidate_reuse(PortableGdnLayer &layer, const Fixture &f) {
    std::string error;
    PortableRoute route;
    const auto reject=[](int64_t, const PortableRoute &, std::vector<float> &, std::string &e) {
        e="candidate reuse rejection";return false;
    };
    check(layer.reset(error),error);
    const auto memory=layer.memory_usage();
    const auto *state0=layer.recurrent_state().data(), *history0=layer.convolution_history().data();
    check(layer.step(f.input(0),experts(),route,error),error);
    const auto *state1=layer.recurrent_state().data(), *history1=layer.convolution_history().data();
    check(state1!=state0 && history1!=history0,"candidate and committed banks must not alias");
    const auto first_state=layer.recurrent_state(), first_history=layer.convolution_history(), first_last=layer.last_residual();
    for(int t=1;t<5;++t) {
        const auto before_state=layer.recurrent_state(), before_history=layer.convolution_history();
        const auto *state=layer.recurrent_state().data(), *history=layer.convolution_history().data();
        bool called=false;
        const auto provider=experts();
        check(layer.step(f.input(t),[&](int64_t l,const PortableRoute &r,std::vector<float> &rows,std::string &e) {
            called=true;
            check(layer.recurrent_state().data()==state && layer.convolution_history().data()==history &&
                  layer.recurrent_state()==before_state && layer.convolution_history()==before_history,
                  "attention candidate exposed before FFN success");
            return provider(l,r,rows,e);
        },route,error),error);
        check(called,"provider not reached");
        check(layer.recurrent_state().data()==(t%2?state0:state1) &&
              layer.convolution_history().data()==(t%2?history0:history1),"successful steps reuse exactly two banks");
        const auto current=layer.memory_usage();
        check(current.host_extra_bound==memory.host_extra_bound && current.device_state==memory.device_state,
              "candidate reuse stays within existing host/device ledger");
    }
    const auto continued_state=layer.recurrent_state(), continued_history=layer.convolution_history(), continued_last=layer.last_residual();
    const auto *committed=layer.recurrent_state().data();
    check(!layer.step(f.input(9),reject,route,error) && error=="candidate reuse rejection", "failed FFN accepted");
    check(layer.recurrent_state().data()==committed && layer.recurrent_state()==continued_state &&
          layer.convolution_history()==continued_history && layer.last_residual()==continued_last && layer.tokens()==5,
          "failed step exposed stale candidate bank");
    check(layer.reset(error) && layer.recurrent_state().data()==committed,error);
    check(layer.step(f.input(0),experts(),route,error),error);
    check(layer.recurrent_state()==first_state && layer.convolution_history()==first_history &&
          layer.last_residual()==first_last,"reset replay exposed rejected candidate data");
    for(int t=1;t<5;++t)check(layer.step(f.input(t),experts(),route,error),error);
    check(layer.recurrent_state()==continued_state && layer.convolution_history()==continued_history &&
          layer.last_residual()==continued_last,"repeated sequence with reused candidates changed");
}
}
int main() {
    try {
        finite_predicate();
        Fixture f;
        std::string error;
        PortableGdnOptions options;
        auto layer = PortableGdnLayer::create(0, f.views, options, error);
        check(bool(layer), error);
        // The trial flag alone preserves CPU allocation and CPU results.
        auto trial_options = options;
        trial_options.resident_state = true;
        trial_options.resident_attention_weights = true;
        auto trial = PortableGdnLayer::create(0, f.views, trial_options, error);
        check(bool(trial) && trial->memory_usage().device_state == 0 && trial->memory_usage().device_weights == 0, "CPU trial keeps host weights");
        auto fallback_options = trial_options;
        fallback_options.execution.request_accelerator = true;
        fallback_options.execution.allow_cpu_fallback = true;
        auto fallback = PortableGdnLayer::create(0, f.views, fallback_options, error);
        check(bool(fallback) && fallback->memory_usage().device_state == 0 && fallback->memory_usage().device_weights == 0,
              "missing requested backend preserves permitted CPU fallback");
        trial_options.execution.resident_dense_weights = true;
        // A borrowed CPU backend exercises the selected-backend allocation branch
        // without initializing GPU hardware; physical Vulkan tests are separate.
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> selected(ggml_backend_cpu_init(), ggml_backend_free);
        trial_options.execution.accelerator = selected.get();
        trial_options.execution.request_accelerator = true;
        auto allocated = PortableGdnLayer::create(0, f.views, trial_options, error);
        check(bool(allocated) && allocated->memory_usage().device_weights > 0, "selected backend owns weights");
        check(allocated->memory_usage().host_weights + allocated->memory_usage().device_weights == allocated->weight_bytes(),
              "actual aligned weight ledger reconciliation");
        check(allocated->memory_usage().device_state > 0 &&
              allocated->memory_usage().device_state <= trial_options.resident_state_budget, "aligned state ledger");
        auto refused_state = trial_options;
        refused_state.resident_state_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, refused_state, error) &&
              error.find("resident state budget") != std::string::npos, "state allocation admission refusal");
        for(bool batched:{false,true}) for(bool production:{false,true}) for(bool causal:{false,true}) {
            if(causal && (!batched || !production)) continue;
            auto chunk_opts=trial_options;
            chunk_opts.policy=production?PortableGdnPolicy::device_production:PortableGdnPolicy::eager_diagnostic;
            chunk_opts.batch_input_projections=batched;
            chunk_opts.causal_chunk=causal;
            auto chunk=PortableGdnLayer::create(0,f.views,chunk_opts,error);
            chunk_opts.batch_input_projections=false;
            chunk_opts.causal_chunk=false;
            auto serial=PortableGdnLayer::create(0,f.views,chunk_opts,error);
            check(bool(chunk)&&bool(serial),error);
            layer_chunk_fixture::exercise(*chunk,*serial,f,experts(),selected.get(),
                chunk_opts.execution.graph_budget,chunk_opts.state_budget,production?"GDN production CPU-selected":"GDN diagnostic resident CPU-selected",false,batched,causal);
        }
        // A projection submission failure occurs before any causal state commit.
        // Check execution, admission and recovery through the existing backend seam.
        {
            auto opts=trial_options; opts.batch_input_projections=true;
            auto candidate=PortableGdnLayer::create(0,f.views,opts,error);
            check(bool(candidate)&&candidate->prepare_chunk(3,error),error);
            const auto before=layer_chunk_fixture::snapshot(*candidate);
            std::vector<float> input;
            for(int t=0;t<3;++t) {const auto row=f.input(t);input.insert(input.end(),row.begin(),row.end());}
            const auto saved=input;
            const auto original=selected->iface.graph_compute;
            layer_chunk_fixture::PostFault::original=original;
            layer_chunk_fixture::PostFault::calls=0;layer_chunk_fixture::PostFault::fail_at=1;
            selected->iface.graph_compute=layer_chunk_fixture::PostFault::compute;
            bool called=false; std::vector<PortableRoute> routes;
            const bool ok=candidate->run_chunk({1,2,3},0,input,[&](auto,auto&,auto&,auto&) {called=true;return false;},routes,error);
            selected->iface.graph_compute=original;
            check(!ok&&!called&&candidate->poisoned()&&error=="portable GDN: input projection graph failed",
                  "projection failure must stop before recurrence/provider");
            check(candidate->input_projection_calls()==0&&candidate->input_projection_columns()==0&&input==saved&&routes.empty(),
                  "failed projection published a successful submission");
            layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*candidate),before);
            check(candidate->reset(error)&&!candidate->poisoned(),error);
            auto denied=opts;denied.execution.graph_budget=candidate->graph_bytes()-1;
            auto budgeted=PortableGdnLayer::create(0,f.views,denied,error);
            check(!budgeted||!budgeted->prepare_chunk(3,error),"projection graph cap must reject before execution");
        }
        {
            auto invalid=trial_options;invalid.causal_chunk=true;
            check(!PortableGdnLayer::create(0,f.views,invalid,error),"causal option requires production");
            invalid.policy=PortableGdnPolicy::device_production;
            check(!PortableGdnLayer::create(0,f.views,invalid,error),"causal option requires batched projections");
        }
        // The causal graph may execute before finite/provider validation, but
        // neither failed execution nor a failed final copy can publish a chunk.
        for(bool commit_failure:{false,true}) {
            auto opts=trial_options;opts.batch_input_projections=true;opts.causal_chunk=true;
            opts.policy=PortableGdnPolicy::device_production;
            auto candidate=PortableGdnLayer::create(0,f.views,opts,error);
            check(bool(candidate)&&candidate->prepare_chunk(3,error),error);
            const auto before=layer_chunk_fixture::snapshot(*candidate);
            std::vector<float> input;
            for(int t=0;t<3;++t){const auto row=f.input(t);input.insert(input.end(),row.begin(),row.end());}
            const auto saved=input;
            const auto compute=selected->iface.graph_compute;
            CaptureCausalState::state=nullptr;CaptureCausalState::original=compute;
            decltype(ggml_backend_buffer_i::set_tensor) set_tensor=nullptr;
            layer_chunk_fixture::PostFault::original=compute;
            layer_chunk_fixture::PostFault::calls=0;layer_chunk_fixture::PostFault::fail_at=1;
            selected->iface.graph_compute=commit_failure?CaptureCausalState::compute:layer_chunk_fixture::PostFault::compute;
            bool called=false;std::vector<PortableRoute> routes;
            const bool ok=candidate->run_chunk({1,2,3},0,input,[&](auto layer,const auto& rs,auto& rows,auto& e){
                called=true;
                for(const auto& r:rs){std::vector<float> row;check(experts()(layer,r,row,e),e);rows.insert(rows.end(),row.begin(),row.end());}
                check(CaptureCausalState::state!=nullptr,"causal committed state captured");
                set_tensor=CaptureCausalState::state->iface.set_tensor;
                CaptureCausalState::state->iface.set_tensor=[](ggml_backend_buffer_t,ggml_tensor*,const void*,size_t,size_t){
                    throw std::runtime_error("injected causal final state write failure");
                };
                return true;
            },routes,error);
            selected->iface.graph_compute=compute;
            if(set_tensor)CaptureCausalState::state->iface.set_tensor=set_tensor;
            if(commit_failure)check(error=="injected causal final state write failure","fault reached the final deferred state write");
            check(!ok&&candidate->poisoned()&&called==commit_failure&&input==saved&&routes.empty(),"causal execution/commit failure publication");
            check(candidate->causal_state_commits()==0,"failed final commit counted successful");
            layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*candidate),before);
            check(candidate->reset(error)&&!candidate->poisoned(),error);
        }
        {
            auto opts=trial_options;opts.batch_input_projections=true;opts.causal_chunk=true;
            opts.policy=PortableGdnPolicy::device_production;
            auto candidate=PortableGdnLayer::create(0,f.views,opts,error);
            check(bool(candidate)&&candidate->prepare_chunk(3,error),error);
            const auto before=layer_chunk_fixture::snapshot(*candidate);
            std::vector<float> input;
            for(int t=0;t<3;++t){const auto row=f.input(t);input.insert(input.end(),row.begin(),row.end());}
            const auto saved=input;const auto compute=selected->iface.graph_compute;
            CaptureCausalState::state=nullptr;CaptureCausalState::original=compute;CaptureCausalState::corrupt_projection=true;
            selected->iface.graph_compute=CaptureCausalState::compute;
            bool called=false;std::vector<PortableRoute> routes;
            const bool ok=candidate->run_chunk({1,2,3},0,input,[&](auto,auto&,auto&,auto&){called=true;return false;},routes,error);
            selected->iface.graph_compute=compute;CaptureCausalState::corrupt_projection=false;
            check(!ok&&!called&&candidate->poisoned()&&input==saved&&routes.empty()&&
                  error=="portable GDN: nonfinite input/result","bulk projection finite validation retained after causal computation");
            check(candidate->causal_chunk_calls()==1&&candidate->causal_residual_handoffs()==0&&candidate->causal_state_commits()==0,
                  "nonfinite projection cannot reach residual handoff or commit");
            layer_chunk_fixture::same(layer_chunk_fixture::snapshot(*candidate),before);
            check(candidate->reset(error)&&!candidate->poisoned(),error);
        }
        production_policy(f, trial_options);
        {
            auto opts = trial_options;
            opts.policy = PortableGdnPolicy::device_production;
            auto faulted = PortableGdnLayer::create(0, f.views, opts, error);
            check(bool(faulted), error);
            // Only the borrowed CPU test backend is changed. Restore the interface
            // before any cleanup; no production injection flag or kernel is added.
            const auto sync = selected->iface.synchronize;
            selected->iface.synchronize = [](ggml_backend_t) { throw std::runtime_error("injected reset sync failure"); };
            const bool reset_ok = faulted->reset(error);
            selected->iface.synchronize = sync;
            check(!reset_ok && faulted->poisoned() && error == "injected reset sync failure", "reset failure stays poisoned");
            PortableRoute rejected;
            check(!faulted->step(f.input(0), experts(), rejected, error) &&
                  error.find("poisoned") != std::string::npos, "retry after failed reset refused");
            check(faulted->reset(error) && !faulted->poisoned(), error);
            const auto compute = selected->iface.graph_compute;
            selected->iface.graph_compute = [](ggml_backend_t, ggml_cgraph *) { return GGML_STATUS_FAILED; };
            const bool step_ok = faulted->step(f.input(0), experts(), rejected, error);
            selected->iface.graph_compute = compute;
            check(!step_ok && faulted->poisoned() && faulted->tokens() == 0 && rejected.ids.empty(),
                  "backend compute failure poisons production");
            check(faulted->reset(error) && faulted->step(f.input(0), experts(), rejected, error), error);
        }
        PortableRoute route;
        PortableGdnTrace trace;
        PortableGdnWork host_work, resident_work;
        for (int t = 0; t < 3; ++t) {
            auto previous = layer->recurrent_state(), history = layer->convolution_history();
            check(layer->step(f.input(t), experts(), route, error, &trace, &host_work), error);
            reference(f, previous, history, trace, *layer);
            PortableRoute trial_route, allocated_route, fallback_route;
            check(fallback->step(f.input(t), experts(), fallback_route, error), error);
            check(trial->step(f.input(t), experts(), trial_route, error), error);
            PortableGdnTrace resident_trace;
            check(allocated->step(f.input(t), experts(), allocated_route, error, &resident_trace, &resident_work), error);
            candidate_work(host_work, t + 1, 0, false);
            candidate_work(resident_work, t + 1, t + 1, false);
            for (auto member : {&PortableGdnTrace::qkv, &PortableGdnTrace::convolution, &PortableGdnTrace::query,
                                &PortableGdnTrace::key, &PortableGdnTrace::value, &PortableGdnTrace::decay,
                                &PortableGdnTrace::beta, &PortableGdnTrace::attention})
                close(resident_trace.*member, trace.*member, 0, "resident state trace parity");
            close(fallback->last_residual(), layer->last_residual(), 0, "CPU fallback output unchanged");
            close(allocated_route.weights, route.weights, 0, "trial route weights unchanged");
            close(trial->last_residual(), layer->last_residual(), 0, "CPU trial output unchanged");
            close(allocated->last_residual(), layer->last_residual(), 0, "selected CPU allocation output unchanged");
            close(allocated->recurrent_state(), layer->recurrent_state(), 0, "selected allocation recurrence");
            check(allocated_route.ids == route.ids && trial_route.ids == route.ids, "trial route IDs unchanged");

            check(layer->tokens() == uint64_t(t + 1), "token count");
        }
        auto state = layer->recurrent_state(), history = layer->convolution_history(),
             residual = layer->last_residual();
        const auto allocated_state = allocated->recurrent_state();
        check(!allocated->step(f.input(3),
              [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) { return false; },
              route, error), "resident allocation rejects callback");
        close(allocated->recurrent_state(), allocated_state, 0, "resident allocation rollback");
        for (int failure = 0; failure < 3; ++failure) {
            check(!allocated->step(f.input(3),
                [failure](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &) -> bool {
                    if (failure == 0) throw std::runtime_error("callback threw");
                    if (failure == 1) return false;
                    rows.assign(25600, std::numeric_limits<float>::quiet_NaN());
                    return true;
                }, route, error, &trace), "resident callback failure accepted");
            check(allocated->recurrent_state() == state && allocated->convolution_history() == history &&
                  allocated->last_residual() == residual && allocated->tokens() == 3 && trace.attention.empty(),
                  "resident failure changed public state");
        }
        check(allocated->step(f.input(3), experts(), route, error, &trace), error);
        const auto after_failure = allocated->last_residual();
        check(allocated->reset(error) && allocated->tokens() == 0, "resident allocation reset");
        for (int t = 0; t < 4; ++t)
            check(allocated->step(f.input(t), experts(), route, error, &trace), error);
        close(allocated->last_residual(), after_failure, 0, "resident failure/reset replay");
        auto reject = [](int64_t, const PortableRoute &, std::vector<float> &, std::string &e) {
            e = "deliberate callback rejection";
            return false;
        };
        check(!layer->step(f.input(3), reject, route, error), "callback rejection");
        close(layer->recurrent_state(), state, 0, "state rollback");
        close(layer->convolution_history(), history, 0, "history rollback");
        close(layer->last_residual(), residual, 0, "residual rollback");
        check(layer->tokens() == 3, "counter rollback");
        check(!layer->step(
                  f.input(3),
                  [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) -> bool {
                      throw std::runtime_error("callback threw");
                  },
                  route, error),
              "throwing callback");
        close(layer->recurrent_state(), state, 0, "throw state rollback");
        close(layer->convolution_history(), history, 0, "throw history rollback");
        close(layer->last_residual(), residual, 0, "throw residual rollback");
        check(!layer->step(
                  f.input(3),
                  [](int64_t, const PortableRoute &, std::vector<float> &rows, std::string &) {
                      rows.assign(25600, std::numeric_limits<float>::quiet_NaN());
                      return true;
                  },
                  route, error),
              "nonfinite callback");
        close(layer->recurrent_state(), state, 0, "nonfinite state rollback");
        close(layer->convolution_history(), history, 0, "nonfinite history rollback");
        close(layer->last_residual(), residual, 0, "nonfinite residual rollback");
        // Exercise the shared finite helper at both boundaries and the middle.
        // Invalid input must fail before dispatch or any persistent-state commit.
        for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
            for (size_t at : {size_t(0), residual.size() / 2, residual.size() - 1}) {
                auto bad_input = f.input(3);
                bad_input[at] = invalid;
                bool dispatched = false;
                trace.attention = {1};
                route.ids = {1};
                check(!layer->step(bad_input,
                    [&](int64_t, const PortableRoute &, std::vector<float> &, std::string &) {
                        dispatched = true;
                        return true;
                    }, route, error, &trace), "nonfinite input accepted");
                check(error == "portable GDN: nonfinite input/result", "finite diagnostic changed");
                check(!dispatched && trace.attention.empty() && route.ids.empty(),
                      "nonfinite input dispatched or published trace/route");
                check(layer->recurrent_state() == state && layer->convolution_history() == history &&
                      layer->last_residual() == residual && layer->tokens() == 3,
                      "nonfinite input changed committed state");
            }
        }
        // Inject nonfinite attention weights into an isolated layer to exercise
        // the read-side guard with finite input, before provider dispatch.
        auto invalid_views = f.views;
        std::vector<float> invalid_decay(48, std::numeric_limits<float>::quiet_NaN());
        for (auto &view : invalid_views)
            if (view.name == "blk.0.ssm_a")
                view.data = invalid_decay.data();
        for (const auto &nonfinite_options : {options, trial_options}) {
            auto nonfinite_layer = PortableGdnLayer::create(0, invalid_views, nonfinite_options, error);
            check(bool(nonfinite_layer), error);
            const auto clean_state = nonfinite_layer->recurrent_state(),
                       clean_history = nonfinite_layer->convolution_history();
            const auto *committed=nonfinite_layer->recurrent_state().data();
            for(int attempt=0;attempt<2;++attempt) {
                bool nonfinite_dispatched = false;
                check(!nonfinite_layer->step(f.input(0),
                    [&](int64_t, const PortableRoute &, std::vector<float> &, std::string &) {
                        nonfinite_dispatched = true;
                        return true;
                    }, route, error, &trace), "nonfinite attention candidate accepted");
                check(error == "portable GDN: nonfinite input/result" && !nonfinite_dispatched,
                      "attention candidate finite guard or diagnostic changed: " + error);
                check(nonfinite_layer->recurrent_state() == clean_state &&
                      nonfinite_layer->convolution_history() == clean_history &&
                      nonfinite_layer->last_residual().empty() && nonfinite_layer->tokens() == 0 &&
                      trace.attention.empty() && nonfinite_layer->recurrent_state().data()==committed,
                    "nonfinite attention candidate changed committed state/trace");
                check(nonfinite_layer->reset(error),error);
            }
        } // finite guard stays active with state OFF and ON
        check(layer->step(f.input(3), experts(), route, error, &trace), error);
        reference(f, state, history, trace, *layer);
        check(layer->reset(error), error);
        check(layer->tokens() == 0 && layer->last_residual().empty(), "reset counter/output");
        for (float x : layer->recurrent_state())
            check(x == 0, "reset state");
        for (float x : layer->convolution_history())
            check(x == 0, "reset history");
        state = layer->recurrent_state();
        history = layer->convolution_history();
        check(layer->step(f.input(0), experts(), route, error, &trace), error);
        reference(f, state, history, trace, *layer);
        const auto first_output = layer->last_residual();
        // Signed zero and finite minimum/subnormal values still take the normal
        // path. Reset/replay must preserve every output and trace byte.
        auto finite_edges = f.input(0);
        finite_edges[0] = 0.0f;
        finite_edges[1] = -0.0f;
        finite_edges[finite_edges.size() / 2] = std::numeric_limits<float>::min();
        finite_edges.back() = std::numeric_limits<float>::denorm_min();
        check(layer->reset(error) && layer->step(finite_edges, experts(), route, error, &trace), error);
        const auto edge_state = layer->recurrent_state(), edge_history = layer->convolution_history(),
                   edge_residual = layer->last_residual();
        const auto edge_trace = trace;
        check(layer->reset(error) && layer->step(finite_edges, experts(), route, error, &trace), error);
        const auto same_bytes = [](const std::vector<float> &a, const std::vector<float> &b) {
            return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
        };
        check(same_bytes(layer->recurrent_state(), edge_state) &&
              same_bytes(layer->convolution_history(), edge_history) &&
              same_bytes(layer->last_residual(), edge_residual), "finite edge replay bytes changed");
        for (auto member : {&PortableGdnTrace::qkv, &PortableGdnTrace::convolution, &PortableGdnTrace::query,
                            &PortableGdnTrace::key, &PortableGdnTrace::value, &PortableGdnTrace::decay,
                            &PortableGdnTrace::beta, &PortableGdnTrace::attention})
            check(same_bytes(trace.*member, edge_trace.*member), "finite edge trace replay bytes changed");
        check(!PortableGdnLayer::create(1, f.views, options, error), "mismatched layer prefix rejection");
        check(!PortableGdnLayer::create(3, f.views, options, error), "QSA layer rejection");
        // Rebinding identical weights to a late layer must preserve the math
        // and pass the actual ID to the expert provider (two-digit prefixes).
        for (int target : {1, 46}) {
            auto late_views = f.views;
            for (auto &v : late_views)
                v.name = "blk." + std::to_string(target) + "." + v.name.substr(6);
            auto late = PortableGdnLayer::create(target, late_views, options, error);
            check(bool(late), error);
            check(late->step(f.input(0),
                [&](int64_t id, const PortableRoute &r, std::vector<float> &rows, std::string &e) {
                    check(id == target, "rebound GDN dispatch ID");
                    return experts()(0, r, rows, e);
                }, route, error), error);
            close(late->last_residual(), first_output, 0, "GDN rebinding without PLE");
            check(late->ple_convolution_history().empty(), "unconfigured PLE has no history");
        }
        for (int id = 0; id < 48; ++id)
            if (id % 4 != 3)
                check(!PortableGdnLayer::create(id, {}, options, error) &&
                      error.find("missing/extra") != std::string::npos, "all GDN IDs admitted");
        for (int id : {-1, 3, 7, 47, 48})
            check(!PortableGdnLayer::create(id, {}, options, error) &&
                  error.find("non-QSA") != std::string::npos, "GDN layer family guard");
        options.state_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, options, error), "state budget");
        options = {};
        options.execution.graph_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, options, error), "graph budget");
        options = {};
        options.execution.weight_budget = 1;
        check(!PortableGdnLayer::create(0, f.views, options, error), "weight budget");
        // Exact-shape synthetic GGUF loader; mapping closes before execution.
        const auto root = std::filesystem::temp_directory_path() /
                          ("strata-gdn-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(root), "fixture directory");
        struct Cleanup {
            std::filesystem::path p;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove_all(p, e);
            }
        } cleanup{root};
        std::vector<fixture::Kv> kv{fixture::str("general.architecture", "qwen4exp"),
                                    fixture::u32s("qwen4exp.ple.layers", {1})};
        for (auto pair : {std::pair<const char *, uint64_t>{"block_count", 48},
                          {"embedding_length", 2560},
                          {"expert_count", 512},
                          {"expert_used_count", 10},
                          {"expert_feed_forward_length", 640},
                          {"expert_shared_feed_forward_length", 640},
                          {"attention.head_count", 24},
                          {"attention.head_count_kv", 2},
                          {"ssm.state_size", 128},
                          {"ssm.group_count", 16},
                          {"ssm.time_step_rank", 48},
                          {"ssm.inner_size", 6144},
                          {"ssm.conv_kernel", 4},
                          {"hyper_connection.count", 4},
                          {"hyper_connection.low_rank", 320}})
            kv.push_back(fixture::u32("qwen4exp." + std::string(pair.first), pair.second));
        std::vector<fixture::Tensor> tensors;
        for (const auto &v : f.views) {
            fixture::Tensor t;
            t.name = v.name;
            t.type = v.type;
            for (auto d : v.shape)
                t.shape.push_back(uint64_t(d));
            tensors.push_back(t);
        }
        const auto path = root / "layer0.gguf";
        const auto written = fixture::write(path, kv, tensors);
        {
            std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
            for (size_t i = 0; i < f.views.size(); ++i) {
                file.seekp(std::streamoff(written.data_start + written.offsets[i]));
                file.write(static_cast<const char *>(f.views[i].data), std::streamsize(f.views[i].bytes));
            }
            check(bool(file), "payload write");
        }
        std::unique_ptr<PortableGdnLayer> bound;
        {
            auto model = strata::GgufModel::open(path.string());
            bound = PortableGdnLayer::from_gguf(model, 0, {}, error);
            check(bool(bound), error);
        }
        check(bound->step(f.input(0), experts(), route, error), error);
        close(bound->last_residual(), first_output, 0, "GGUF binding");
        kv[1] = fixture::u32s("qwen4exp.ple.layers", {0});
        fixture::write(root / "bad-ple.gguf", kv, {});
        {
            auto model = strata::GgufModel::open((root / "bad-ple.gguf").string());
            check(!PortableGdnLayer::from_gguf(model, 0, {}, error), "configured PLE0 rejection");
        }
        // Completed read/scan scopes survive a downstream rejection; no
        // device commit may run until the existing FFN validation succeeds.
        PortableGdnWork rejected;
        const auto committed_state = allocated->recurrent_state();
        check(!allocated->step(f.input(0),
            [](int64_t, const PortableRoute &, std::vector<float> &, std::string &) { return false; },
            route, error, nullptr, &rejected), "timed callback rejection");
        candidate_work(rejected, 1, 0, false);
        check(allocated->recurrent_state() == committed_state, "timed rejection changed committed state");
        const auto saved = rejected;
        check(allocated->reset(error), error);
        check(rejected.candidate_readback_ms == saved.candidate_readback_ms &&
              rejected.candidate_finite_ms == saved.candidate_finite_ms &&
              rejected.state_commit_calls == saved.state_commit_calls, "reset preserves caller timing totals");
        finite_diagnostics(*layer, f);
        finite_diagnostics(*allocated, f);
        candidate_reuse(*layer,f);
        candidate_reuse(*allocated,f);
        std::printf("portable_gdn_test: actual H2560/FF640/S128/Hk16/Hv48; independent multi-token state/convolution, "
                    "reset, rollback, budgets, PLE/QSA rejection passed; weights=%zu graphs=%zu host_state_bound=%zu\n",
                    layer->weight_bytes(), layer->graph_bytes(), layer->state_bytes());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
