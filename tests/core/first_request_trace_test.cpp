#include "strata/core/first_request_trace.hpp"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

static void check(bool ok) { if (!ok) { std::fprintf(stderr, "first-request trace check failed\n"); std::exit(1); } }

int main(int argc, char** argv) {
    using strata::core::FirstRequestTrace;
    check(argc == 2);
    FirstRequestTrace trace;
    const int64_t prompt[] = {10, 20, 30};
    const int32_t window[] = {10, 20};
    check(trace.begin(prompt, 3));
    if (!FirstRequestTrace::enabled()) {
        check(!trace.active());
        check(trace.start_window(0, 2, window));
        check(trace.record(2, 0, 0, 2, 65536, nullptr)); // Disabled: no read, write or allocation.
        check(!std::filesystem::exists(argv[1]));
        return 0;
    }
    check(trace.active());
    check(trace.start_window(0, 2, window));
    check(!trace.finish(nullptr, 0)); // Cannot stop before the final prompt token window.
    check(trace.active());
    check(trace.start_window(2, 1, window));
    const float x[] = {1, 2, 3, 4};
    check(trace.record(2, 0, 0, 2, 2, x));
    std::vector<float> vocab(248320, 1.0f); // Real vocabulary exceeds 64k.
    check(trace.finish(vocab.data(), vocab.size()));
    check(!trace.active());
    const auto bytes = std::filesystem::file_size(argv[1]);
    check(trace.begin(prompt, 3)); // Only first request; no truncation or reactivation.
    check(!trace.active());
    check(trace.start_window(2, 1, window));
    check(std::filesystem::file_size(argv[1]) == bytes);
    FirstRequestTrace collision;
    check(!collision.begin(prompt, 3)); // Existing evidence cannot be replaced.
    check(std::filesystem::file_size(argv[1]) == bytes);
    // Bound enforcement latches failure, so finish cannot seal a partial capture.
    const auto bounded_path = std::string(argv[1]) + ".bounded";
#ifdef _WIN32
    _putenv_s("STRATA_FIRST_REQUEST_TRACE", bounded_path.c_str());
#else
    setenv("STRATA_FIRST_REQUEST_TRACE", bounded_path.c_str(), 1);
#endif
    FirstRequestTrace bounded;
    check(bounded.begin(prompt, 3));
    check(!bounded.record(2, 0, 0, 65536, 65536, x));
    check(!bounded.finish(x, 4));
    check(!bounded.good());
}
