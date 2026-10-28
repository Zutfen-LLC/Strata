#include "portable_model_fixture.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
using namespace model_fixture;
namespace {
std::string read(const std::filesystem::path &p) {
    std::ifstream in(p, std::ios::binary);
    check(bool(in), "missing regression artifact: " + p.string());
    return std::string(std::istreambuf_iterator<char>(in), {});
}
std::string quote(const std::string &s) {
    std::string out = "'";
    for (char c : s) out += c == '\'' ? "'\\''" : std::string(1, c);
    return out + "'";
}
}
int main(int argc, char **argv) {
    try {
        check(argc == 2, "CLI path required");
        SparseModel f;
        const auto run = [&](const std::string &name, bool timed) {
            const std::string base = (f.root / name).string();
            std::string cmd = quote(argv[1]) + " --model " + quote(f.paths[0]) + " --manifest " +
                quote((f.root / "manifest").string()) + " --tokens 5,7 --max-new 2 --context 4 --dump-logits " +
                quote(base) + (timed ? " --timings " + quote(base + "/timings.json") : "") +
                " > " + quote(base + "/stdout") + " 2> " + quote(base + "/stderr");
            std::filesystem::create_directory(base);
            const int rc = std::system(cmd.c_str());
            check(rc == 0, "CLI failed: " + read(base + "/stderr"));
        };
        run("default", false);
        run("timed", true);
        check(read(f.root / "default/stdout") == read(f.root / "timed/stdout"), "default/timed stdout differs");
        for (int p = 0; p < 3; ++p) {
            const auto name = "pos-" + std::to_string(p) + ".f32";
            check(read(f.root / "default" / name) == read(f.root / "timed" / name), "logit bytes differ");
        }
        const auto timing = read(f.root / "timed/timings.json");
        check(timing.find("\"prompt_count\":2,\"output_count\":1") != std::string::npos,
              "EOS must count actual generated steps rather than max-new");
        for (const auto *key : {"setup_ms", "prompt_ms", "decode_ms", "selection_prompt_ms", "selection_decode_ms", "terminal_feed_ms"}) {
            const auto begin = timing.find(std::string("\"") + key + "\":");
            check(begin != std::string::npos, "missing phase timer");
            const double value = std::stod(timing.substr(begin + std::string(key).size() + 3));
            check(std::isfinite(value) && value >= 0, "invalid monotonic duration");
        }
        check(timing.find("\"selection_prompt_steps\":1,\"selection_decode_steps\":1") != std::string::npos,
              "selection boundary step counts");
        std::puts("portable_timing_cli_test: default/timed stdout and logits identical; early EOS counters passed");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
