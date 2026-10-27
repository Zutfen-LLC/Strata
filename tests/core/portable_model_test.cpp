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
    for (size_t i = 2; i < logits.size(); ++i)
        check(logits[i] == 0, "zero output rows");
}
} // namespace
int main() {
    try {
        SparseModel fixture;
        PortableModelOptions opts;
        opts.context = 3;
        std::string error;
        auto session = PortableModelSession::open(fixture.paths, (fixture.root / "manifest").string(), opts, error);
        check(bool(session), error);
        check(session->position() == 0 && !session->poisoned() && !session->ended(), "fresh session");
        check(session->report().mapped_bytes > (170ull << 30), "huge tables/experts are mapped");
        check(session->report().host_weights < (6ull << 30) &&
                  session->report().expert_staging == 10 * fixture.fmt.bytes,
              "bounded host copies/staging; no expert/PLE anonymous duplicate");
        std::vector<float> logits;
        check(session->step(5, logits, error), error);
        logits_reference(logits, 5);
        check(session->position() == 1, "successful position commit");
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
        check(session->generate({248044, 5}, 2, result, error), error);
        check(result.tokens.empty() && result.stop == PortableStop::eos && session->position() == 1 && session->ended(),
              "EOS is consumed and terminates prompt/generation");
        check(session->reset(error), error);
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
