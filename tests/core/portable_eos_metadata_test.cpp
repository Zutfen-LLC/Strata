#include "portable_model_fixture.hpp"
#include <algorithm>
#include <cstdio>
using namespace model_fixture;
using namespace strata::core;
int main() {
    try {
        SparseModel fixture(true);
        const auto missing_manifest = (fixture.root / "missing-manifest").string();
        auto metadata = SparseModel::metadata();
        metadata.push_back(fixture::u32("split.count", 6));
        metadata.push_back(fixture::u32("split.no", 0));
        metadata.push_back(fixture::u32("split.tensors.count", 1));
        std::string error;
        auto probe = [&](const std::vector<fixture::Kv> &kv) {
            fixture.write(0, kv);
            check(!PortableModelSession::open(fixture.paths, missing_manifest, {}, error),
                  "header-only fixture cannot load without a manifest");
        };
        probe(metadata);
        check(error == "cannot read bounded metadata: " + missing_manifest,
              "distinct artifact EOS values must reach manifest admission without reading weights: " + error);
        // Both roles independently require an in-vocabulary integer. No key
        // alias or equality rule is allowed to replace either role's validation.
        for (const char *key : {"tokenizer.ggml.eos_token_id", "qwen4exp.ple.eos_token_id"}) {
            for (int failure = 0; failure < 4; ++failure) {
                auto bad = metadata;
                auto at = std::find_if(bad.begin(), bad.end(), [&](const fixture::Kv &v) { return v.key == key; });
                check(at != bad.end(), "fixture key");
                if (failure == 0)
                    bad.erase(at);
                if (failure == 1)
                    *at = fixture::u32(key, SparseModel::V);
                if (failure == 2)
                    *at = fixture::f32(key, 248046.0f);
                if (failure == 3)
                    *at = fixture::str(key, "248046");
                probe(bad);
                check(error.find(key) != std::string::npos && error.find("integer token ID") != std::string::npos,
                      "role-specific EOS type/range guard: " + error);
            }
        }
        std::puts("portable_eos_metadata_test: artifact tokenizer EOS248046 / PLE EOS248044, independent integer/range "
                  "validation, header-only admission passed; no weight payload read");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
