// Fixed-memory RPC timing behavior. No sockets, model assets or worker pool are required.
#include "strata/worker/rpc_timing.hpp"

#include <cstdio>
#include <limits>

using namespace strata::worker;

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; }
    };

    RpcTimingCounters timing;
    timing.add_phase(RpcTimingPhase::ClientPrepare, 1000);
    timing.finish_call(1, 3, 2, 100, 80, true);
    check(timing.calls() == 0 && timing.report("disabled").empty(), "disabled timing is a no-op");

    timing.set_enabled(true);
    timing.add_phase(RpcTimingPhase::ClientPrepare, 1000); // <=1 us
    timing.add_phase(RpcTimingPhase::ClientPrepare, 1001); // <=2 us
    timing.add_phase(RpcTimingPhase::ClientPrepare, std::numeric_limits<uint64_t>::max() / 2); // overflow
    timing.finish_call(1, 3, 2, 100, 80, true);
    timing.finish_call(8, 10, 6, 200, 160, true);
    timing.finish_call(0, 0, 0, 16, 0, false); // malformed/empty or timeout: bounded failed attribution
    const auto& prep = timing.phase(RpcTimingPhase::ClientPrepare);
    check(prep.count == 3 && prep.buckets[0] == 1 && prep.buckets[1] == 1 &&
              prep.buckets.back() == 1,
          "histogram boundaries and overflow are stable");
    check(timing.calls() == 3 && timing.successes() == 2 && timing.failures() == 1,
          "success and failure calls have explicit boundaries");
    check(timing.request_bytes() == 316 && timing.reply_bytes() == 240 &&
              timing.tokens() == 9 && timing.entries() == 13 && timing.experts() == 8,
          "payload and request-shape attribution accumulates");
    check(timing.token_shapes()[0] == 1 && timing.token_shapes()[1] == 1 && timing.token_shapes()[8] == 1,
          "empty and maximum token shapes remain attributable");
    check(timing.report("test").find("p95_us=") != std::string::npos,
          "report exposes aggregate distributions, not samples");

    timing.reset();
    check(timing.enabled() && timing.calls() == 0 &&
              timing.phase(RpcTimingPhase::ClientPrepare).count == 0,
          "reset clears counters but preserves opt-in state");
    timing.set_enabled(false);
    timing.finish_call(2, 4, 3, 1, 1, false);
    check(timing.calls() == 0, "disable after reset restores the no-op path");

    std::printf("rpc_timing_test: failures=%d bytes=%zu\n", fails, sizeof(RpcTimingCounters));
    return fails ? 1 : 0;
}
