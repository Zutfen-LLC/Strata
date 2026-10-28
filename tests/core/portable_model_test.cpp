#include "portable_model_fixture.hpp"
#include <cmath>
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
            check(untimed.gdn_phases.attention_exchange_ms==0 && untimed.gdn_phases.attention_compute_ms==0 &&
                  untimed.gdn_phases.ffn_ms==0 && untimed.gdn_phases.attention_exchange_calls==0 &&
                  untimed.gdn_phases.attention_compute_calls==0 && untimed.gdn_phases.ffn_calls==0,
                  "default disables GDN subphase capture");
            check(untimed.step_ms==0 && untimed.gdn_ms==0 && untimed.qsa_ms==0 && untimed.head_ms==0 &&
                  untimed.step_calls==0 && untimed.gdn_calls==0 && untimed.qsa_calls==0 && untimed.head_calls==0,
                  "default model path has no new timing capture");
            check(baseline->expert_work().provider_ms == 0 && baseline->expert_work().provider_calls == 0,
                  "default provider path has no new timing capture");
        }
        opts.gdn_resident_state = true; // absent accelerator preserves default logits
        opts.dense_resident_weights = true;
        opts.gdn_resident_weights = true; // CPU execution must preserve the default reference
        opts.expert_dispatch_oracle = false;
        opts.expert_timings = true;
        auto session = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
        check(bool(session), error);
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
            for (double value : {ga.attention_exchange_ms,ga.attention_compute_ms,ga.ffn_ms})
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
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
