#include "portable_model_fixture.hpp"
#include <cmath>
#include "ggml-cpu.h"
#include <cstdio>
using namespace strata::core;
using namespace model_fixture;
namespace {
void logits_reference(const std::vector<float> &logits, int32_t token) {
    const float first = token == 6 ? -3.0f : 2.0f;
    // Layer-0 expert 0 alone is nonzero. HC gates are 1/2, inject
    // scatter is 1, top-10 weights are 1/10. All other residual updates
    // are zero. These scalar dots do not call the product expert provider.
    const float input_rms = std::sqrt((2559.0f + first * first) / 2560.0f + 1e-6f);
    const float x = 0.5f * first / input_rms;
    const float expert = (x / (1 + std::exp(-x))) * (2 * x);
    const float residual = first + 0.1f * expert;
    const float output_rms = std::sqrt((2559.0f + residual * residual) / 2560.0f + 1e-6f);
    const float expected = 0.5f * residual / output_rms;
    check(logits.size() == SparseModel::V, "full vocabulary size");
    // Fixed before running: Q8 activation and intermediate quantization
    // produce rounding relative to scalar F32 dots in this reference.
    check(std::fabs(logits[0] - expected) <= 0.015f * (1 + std::fabs(expected)),
          "full model independent scalar logit reference");
    check(std::fabs(logits[1] + logits[0]) < 1e-5f, "opposite output rows");
    check(std::fabs(logits[248046] - 0.125f / output_rms) <= 0.005f, "independent generation-EOS output row");
    for (size_t i = 2; i < logits.size(); ++i)
        if (i != 248046)
            check(logits[i] == 0, "zero output rows");
}
} // namespace
int main() {
    try {
        SparseModel fixture;
        PortableModelOptions opts;
        opts.context = 3;
        std::string error;
        {
            std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
            for(int refusal=0;refusal<3;++refusal) {
                auto invalid=opts;invalid.native_router=true;invalid.native_router_receipt_prefix=(fixture.root/"router").string();
                if(refusal)invalid.accelerators={cpu.get()};
                if(refusal==1)invalid.native_router_receipt_prefix.clear();
                if(refusal==2)invalid.allow_cpu_fallback=true;
                check(!PortableModelSession::open(fixture.paths,(fixture.root/"manifest").string(),invalid,error) &&
                      error.find("native router requires")!=std::string::npos,"native router model preflight refusal");
            }
        }
        std::vector<std::vector<float>> reference;
        opts.expert_dispatch_oracle = true;
        {
            auto baseline = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
            check(bool(baseline), error);
            for (int32_t token : {5, 6}) {
                std::vector<float> logits;
                check(baseline->step(token, logits, error), error);
                reference.push_back(std::move(logits));
            }
            const auto untimed=baseline->model_work();
            check(untimed.native_router_calls==0 && untimed.native_router_rows==0,"default model credited native router work");
            check(untimed.heads_executed==2 && untimed.heads_skipped==0, "default exhaustive head counters without timers");
            check(untimed.gdn_phases.attention_exchange_ms==0 && untimed.gdn_phases.attention_compute_ms==0 &&
                  untimed.gdn_phases.ffn_ms==0 && untimed.gdn_phases.attention_exchange_calls==0 &&
                  untimed.gdn_phases.attention_compute_calls==0 && untimed.gdn_phases.ffn_calls==0,
                  "default disables GDN subphase capture");
            check(untimed.gdn_phases.candidate_readback_ms == 0 && untimed.gdn_phases.candidate_finite_ms == 0 &&
                  untimed.gdn_phases.state_commit_ms == 0 && untimed.gdn_phases.candidate_readback_calls == 0 &&
                  untimed.gdn_phases.candidate_finite_calls == 0 && untimed.gdn_phases.state_commit_calls == 0,
                  "default disables candidate read/finite/commit capture");
            check(untimed.step_ms==0 && untimed.gdn_ms==0 && untimed.qsa_ms==0 && untimed.head_ms==0 &&
                  untimed.step_calls==0 && untimed.gdn_calls==0 && untimed.qsa_calls==0 && untimed.head_calls==0,
                  "default model path has no new timing capture");
            check(baseline->expert_work().provider_ms == 0 && baseline->expert_work().provider_calls == 0,
                  "default provider path has no new timing capture");
        }
        opts.qsa_reuse_attention = true; // same-backend rows compare with default baseline above
        opts.gdn_resident_state = true; // absent accelerator preserves default logits
        opts.dense_resident_weights = true;
        opts.gdn_resident_weights = true; // CPU execution must preserve the default reference
        opts.expert_dispatch_oracle = false;
        opts.expert_timings = true;
        auto session = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
        check(bool(session), error);
        check(session->report().qsa_attention_resources == 1 && session->report().qsa_attention_host_scheduler > 0,
              "whole-model session owns one actual shared attention scheduler");
        check(session->report().gdn_scratch_resources == 1 && session->report().gdn_scratch_host_scheduler > 0,
              "whole-model session owns one actual shared GDN scratch family");
        check(session->position() == 0 && !session->poisoned() && !session->ended(), "fresh session");
        check(session->eos() == 248046, "generation EOS comes from tokenizer metadata, not PLE");
        check(session->report().mapped_bytes > (170ull << 30), "huge tables/experts are mapped");
        check(session->report().host_weights < (6ull << 30) &&
                  session->report().expert_staging == 10 * fixture.fmt.bytes,
              "bounded host copies/staging; no expert/PLE anonymous duplicate");
        std::vector<float> logits;
        check(session->step(5, logits, error), error);
        logits_reference(logits, 5);
        check(std::memcmp(logits.data(), reference[0].data(), logits.size() * sizeof(float)) == 0,
              "cold model logits byte identical to former staging provider");
        check(session->step(6, logits, error), error);
        check(std::memcmp(logits.data(), reference[1].data(), logits.size() * sizeof(float)) == 0,
              "repeated expert model logits byte identical to former staging provider");
        check(session->expert_work().hits == 480, "all-layer second token hits");
        check(session->reset(error) && session->step(5, logits, error), error);
        check(session->position() == 1, "successful position commit");
        // Reset retains immutable providers and verified rows; A/B/A must replay
        // identically without re-verifying the same selected expert views.
        const auto weights=session->report().host_weights;
        const auto verified=session->expert_work().verifications;
        for(int32_t token : {6, 5, 5}) {
            const auto phase_before=session->model_work();
            const auto before_reset = session->expert_work();
            check(session->reset(error), error);
            const auto after_reset = session->expert_work();
            const auto phase_reset=session->model_work();
            check(phase_before.step_ms==phase_reset.step_ms && phase_before.gdn_ms==phase_reset.gdn_ms &&
                  phase_before.qsa_ms==phase_reset.qsa_ms && phase_before.head_ms==phase_reset.head_ms &&
                  phase_before.gdn_provider_ms==phase_reset.gdn_provider_ms &&
                  phase_before.qsa_provider_ms==phase_reset.qsa_provider_ms, "reset preserves model totals");
            check(before_reset.pool_gu_ms == after_reset.pool_gu_ms &&
                  before_reset.pool_quant_ms == after_reset.pool_quant_ms &&
                  before_reset.pool_down_ms == after_reset.pool_down_ms &&
                  before_reset.pool_elapsed_ms == after_reset.pool_elapsed_ms,
                  "reset preserves pool accumulators without adding work");
            check(before_reset.provider_ms == after_reset.provider_ms &&
                  before_reset.provider_calls == after_reset.provider_calls,
                  "reset preserves provider timing and call count");
            check(session->step(token, logits, error), error);
            const auto after_step = session->expert_work();
            const auto phase_after=session->model_work();
            const auto &ga=phase_after.gdn_phases, &gb=phase_reset.gdn_phases;
            check(ga.attention_exchange_calls==gb.attention_exchange_calls+36 &&
                  ga.attention_compute_calls==gb.attention_compute_calls+36 && ga.ffn_calls==gb.ffn_calls+36,
                  "matched GDN subphase counts");
            check(ga.candidate_readback_calls == gb.candidate_readback_calls + 109 &&
                  ga.candidate_finite_calls == gb.candidate_finite_calls + 109 && ga.state_commit_calls == 0,
                  "36 layers * three candidate reads, plus layer-1 PLE; no CPU device commit");
            check(ga.candidate_readback_ms + ga.candidate_finite_ms + ga.attention_compute_ms <=
                      ga.attention_exchange_ms + 1e-6, "disjoint candidate scopes contained in attention");
            check(phase_before.gdn_phases.candidate_readback_ms == gb.candidate_readback_ms &&
                  phase_before.gdn_phases.candidate_finite_ms == gb.candidate_finite_ms &&
                  phase_before.gdn_phases.candidate_readback_calls == gb.candidate_readback_calls &&
                  phase_before.gdn_phases.candidate_finite_calls == gb.candidate_finite_calls,
                  "reset preserves candidate timing totals");
            for (double value : {ga.candidate_readback_ms,ga.candidate_finite_ms,ga.state_commit_ms,
                                 ga.attention_exchange_ms,ga.attention_compute_ms,ga.ffn_ms})
                check(std::isfinite(value) && value>=0, "finite nonnegative GDN timing");
            check(ga.attention_compute_ms<=ga.attention_exchange_ms &&
                  ga.attention_exchange_ms+ga.ffn_ms<=phase_after.gdn_ms &&
                  ga.ffn_ms>=phase_after.gdn_provider_ms, "nested GDN containment");
            check(phase_before.gdn_phases.attention_exchange_ms==gb.attention_exchange_ms &&
                  phase_before.gdn_phases.attention_compute_ms==gb.attention_compute_ms &&
                  phase_before.gdn_phases.ffn_ms==gb.ffn_ms &&
                  phase_before.gdn_phases.attention_exchange_calls==gb.attention_exchange_calls &&
                  phase_before.gdn_phases.attention_compute_calls==gb.attention_compute_calls &&
                  phase_before.gdn_phases.ffn_calls==gb.ffn_calls, "reset preserves GDN subphase totals");
            check(phase_after.step_calls==phase_reset.step_calls+1 && phase_after.gdn_calls==phase_reset.gdn_calls+36 &&
                  phase_after.qsa_calls==phase_reset.qsa_calls+12 && phase_after.head_calls==phase_reset.head_calls+1,
                  "one model/head and correct layer-family calls per token");
            check(phase_after.step_ms>=phase_after.gdn_ms+phase_after.qsa_ms+phase_after.head_ms &&
                  phase_after.gdn_ms>=phase_after.gdn_provider_ms && phase_after.qsa_ms>=phase_after.qsa_provider_ms &&
                  std::fabs(phase_after.gdn_provider_ms+phase_after.qsa_provider_ms-after_step.provider_ms)<1e-6,
                  "matched provider containment without double counting");
            check(session->model_work().step_ms==phase_after.step_ms, "reading model totals is idempotent");
            check(after_step.provider_calls == after_reset.provider_calls+48 &&
                  std::isfinite(after_step.provider_ms) && after_step.provider_ms > after_reset.provider_ms &&
                  after_step.provider_ms >= after_step.pool_elapsed_ms,
                  "one enclosing provider capture per layer callback");
            check(after_step.pool_elapsed_ms > after_reset.pool_elapsed_ms &&
                  after_step.pool_gu_ms >= after_reset.pool_gu_ms &&
                  after_step.pool_quant_ms >= after_reset.pool_quant_ms &&
                  after_step.pool_down_ms >= after_reset.pool_down_ms,
                  "pool phases accumulate across reset");
            const auto& expected=reference[token==6 ? 1 : 0];
            check(logits==expected, "A/B/A reset rows differ from explicit same-backend reference");
            check(session->expert_work().verifications==verified, "reset discarded verified CPU views");
            check(session->report().host_weights==weights, "weights changed across reset");
        }
        // Corrupt a selected expert in layer 10 after construction: layers
        // 0..9 commit before the digest error. The same token cannot retry.
        const size_t shard = 3;
        const std::string bad = "blk.10.ffn_gate_exps.weight";
        const uint8_t corrupt = 1, restored = 0;
        fixture.patch(shard, bad, 0, &corrupt, 1);
        check(!session->step(6, logits, error) && session->poisoned() && logits.empty() && session->position() == 1,
              "partial token poisons without advancing authoritative position");
        check(error.find("layer 10") != std::string::npos && error.find("digest mismatch") != std::string::npos,
              "partial failure reports layer and expert digest");
        fixture.patch(shard, bad, 0, &restored, 1);
        check(!session->step(6, logits, error) && error.find("poisoned") != std::string::npos, "no mixed-state retry");
        check(session->reset(error) && session->position() == 0 && !session->poisoned(), error);
        check(session->step(6, logits, error), error);
        logits_reference(logits, 6); // negative current token, not prior-token HC
        PortableGeneration result;
        check(session->generate({5}, 2, result, error), error);
        check(result.tokens == std::vector<int32_t>{0} && result.stop == PortableStop::context &&
                  session->position() == 3,
              "bounded greedy generation feeds its token and stops at context");
        check(!session->step(5, logits, error) && session->poisoned(), "context exhausted step poisons");
        check(session->reset(error), error);
        check(session->step(248044, logits, error, PortableTokenKind::generated), error);
        check(!session->ended(), "PLE boundary token is not generation EOS");
        check(session->generate({248046, 5}, 0, result, error), error);
        check(result.tokens.empty() && result.stop == PortableStop::max_new && session->position() == 3 &&
                  !session->ended(),
              "prompt tokenizer EOS is a consumed turn separator, not a generation stop");
        check(session->reset(error) && session->position() == 0 && !session->ended(), error);
        check(session->generate({7}, 3, result, error), error);
        check(result.tokens == std::vector<int32_t>{248046} && result.stop == PortableStop::eos &&
                  session->position() == 2 && session->ended(),
              "generated tokenizer EOS is fed once and stops generation");
        check(!session->step(5, logits, error) && session->poisoned(), "ended session requires explicit reset");
        check(session->reset(error) && !session->ended() && !session->poisoned(), error);
        check(!session->step(-1, logits, error) && session->poisoned(), "invalid token poisons");
        check(session->reset(error), error);
        check(!session->generate({}, 1, result, error) && !session->poisoned() && session->position() == 0,
              "generation preflight is side-effect free");
        bool gdn = false, qsa = false, ple = false, head = false;
        for (const auto &p : session->placements()) {
            gdn |= p.operation == "GATED_DELTA_NET";
            qsa |= p.operation == "ROPE";
            ple |= p.tensor == "ple_residual";
            head |= p.phase == "global-head";
        }
        check(gdn && qsa && ple && head, "all layer families/global head integrated");
        std::printf("portable_model_test: six sparse shards, real Q8 CPU experts, host PLE, all 48 layers, full "
                    "vocabulary scalar reference, fresh-token HC, late failure poison/reset, greedy/EOS/context "
                    "passed; mapped=%zu host_weights=%zu staging=%zu\n",
                    session->report().mapped_bytes, session->report().host_weights, session->report().expert_staging);
        // Eager reference results first; release its weights before opening the
        // production session. Preserve terminal-feed, context and input EOS.
        const std::vector<std::vector<int32_t>> prompts{{5}, {5,6}, {248046,5}, {5,6,7}};
        std::vector<PortableGeneration> eager;
        std::vector<uint64_t> positions;
        for (const auto &prompt : prompts) for (size_t limit : {size_t(0),size_t(1),size_t(3)}) {
            check(session->reset(error), error);
            PortableGeneration generation;
            check(session->generate(prompt, limit, generation, error), error);
            eager.push_back(generation);
            positions.push_back(session->position());
        }
        session.reset();
        opts.prompt_head_elision = true;
        session = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
        check(bool(session), error);
        size_t case_id=0;
        for (const auto &prompt : prompts) for (size_t limit : {size_t(0),size_t(1),size_t(3)}) {
            check(session->reset(error), error);
            const auto before=session->model_work();
            PortableGeneration generation;
            check(session->generate(prompt, limit, generation, error), error);
            check(generation.tokens==eager[case_id].tokens && generation.stop==eager[case_id].stop &&
                      session->position()==positions[case_id++], "head-elision generated/terminal behavior differs");
            const auto after=session->model_work();
            check(after.heads_skipped-before.heads_skipped==prompt.size()-1 &&
                      after.heads_executed-before.heads_executed==generation.tokens.size()+1 &&
                      after.head_calls-before.head_calls==generation.tokens.size()+1,
                  "only nonfinal prompt heads omitted, including max-new zero and context boundary");
        }
        // Explicit seam: exact final requested logits and future continuation
        // after A/B/A reset, and no stale global-head placement on skipped rows.
        for (int32_t token : {5,6,5}) {
            check(session->reset(error) && session->advance(token,error),error);
            check(session->position()==1 && !session->ended(), "advance commits one input position");
            for (const auto &p : session->placements())
                check(p.phase!="global-head", "skipped head reports stale execution placement");
            check(session->step(6,logits,error) && logits==reference[1], "final requested logits differ after advance/reset");
            check(session->step(5,logits,error) && logits==reference[0], "continuation logits differ after advance");
        }
        check(session->reset(error),error);
        const auto empty_before=session->model_work();
        check(!session->generate({},1,result,error) && !session->poisoned() && session->position()==0,
              "empty prompt remains side-effect free under elision");
        check(!session->generate({5},4,result,error) && !session->poisoned() && session->position()==0,
              "max-new bound preflight remains side-effect free under elision");
        check(session->model_work().heads_executed==empty_before.heads_executed &&
                  session->model_work().heads_skipped==empty_before.heads_skipped, "empty prompt executes nothing");
        check(!session->advance(-1,error) && session->poisoned(), "invalid state-only token poisons");
        check(session->reset(error),error);
        fixture.patch(shard,bad,0,&corrupt,1);
        check(!session->advance(5,error) && session->poisoned() && session->position()==0 &&
                  error.find("layer 10")!=std::string::npos, "state-only partial layer failure poisons");
        fixture.patch(shard,bad,0,&restored,1);
        check(!session->advance(5,error), "state-only poisoned retry rejected");
        check(session->reset(error) && session->advance(5,error) && session->step(6,logits,error) &&
                  logits==reference[1], "state-only reset recovery");
        session.reset(); // bound peak fixture memory before resident-ownership run
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> selected(ggml_backend_cpu_init(), ggml_backend_free);
        size_t gdn_state_only=0;
        for (bool qsa_only : {false, true}) {
        opts.qsa_production = qsa_only;
        opts.gdn_production = !qsa_only;
        opts.accelerators = {selected.get()};
        opts.device_caps = {64ull << 30}; // synthetic CPU backend; no hardware initialized
        session = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
        check(bool(session), error);
        if (!qsa_only) gdn_state_only=session->report().device_state[0];
        else check(session->report().device_state[0]==gdn_state_only+
                   12*PortableQsaLayer::device_state_allocation_size(selected.get(),opts.context),
                   "model actual QSA state ledger charged once after preflight");
        check(session->step(5, logits, error) && logits == reference[0], "production full model same-backend bytes");
        const auto work = session->model_work().gdn_phases;
        check(work.candidate_readback_calls == (qsa_only ? 109 : 36) && work.candidate_finite_calls == (qsa_only ? 109 : 36) &&
              work.state_commit_calls == 36, "production model residual-only reads");
        check(session->reset(error) && session->step(5, logits, error) && logits == reference[0],
              "production model reset/replay");
        session.reset();
        // Copied invalid output weights force a head failure after every layer ran.
        // Restore the immutable fixture file before execution; the head owns its copy.
        const uint16_t bad_scale = 0x7c00, good_scale = 0x3c00;
        fixture.patch(0, "output.weight", 0, &bad_scale, sizeof(bad_scale));
        session = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
        fixture.patch(0, "output.weight", 0, &good_scale, sizeof(good_scale));
        check(bool(session), error);
        for (int attempt = 0; attempt < 2; ++attempt) {
            check(session->advance(5,error) && !session->poisoned() && session->position()==1,
                  "omitted head has no logit check even with invalid head weights");
            check(!session->step(5, logits, error) && logits.empty() && session->poisoned() &&
                  session->position() == 1 && error.find("nonfinite logits") != std::string::npos,
                  "production head guard failure must poison without publishing logits/position");
            check(!session->step(5, logits, error) && error.find("poisoned") != std::string::npos,
                  "production head failure cannot retry");
            check(session->reset(error) && !session->poisoned(), error);
        }
        session.reset();
        }
        std::puts("portable_model_test: head elision boundaries, exact requested/continuation logits, A/B/A and poison/guard recovery passed");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
