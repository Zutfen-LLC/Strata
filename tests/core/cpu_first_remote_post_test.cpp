// The same small submission function instantiated by expert_source.cpp.
// No model, CUDA, service or copied scheduler; dummy endpoints control completion.
#include "strata/core/cpu_first_remote_post.hpp"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef STRATA_NATIVE_EXPERTS
#error This standalone seam fixture must compile without native worker linkage
#endif

namespace {
int failures = 0;
void check(bool value, const char* what) {
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); }
}
struct Cpu {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    bool started = false, release = false, joined = false, expired = false, done = false;
    std::vector<float> copied, partial;
    ~Cpu() { finish(); }
    bool post(const std::vector<float>& input) {
        copied = input;  // Mirrors the real tier's ownership contract, not its math.
        thread = std::thread([this] {
            std::unique_lock<std::mutex> lock(mutex);
            started = true;
            cv.notify_all();
            if (!cv.wait_for(lock, std::chrono::seconds(1), [&] { return release; })) expired = true;
            partial = copied;
            done = true;
        });
        return true;
    }
    bool in_flight() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(1), [&] { return started; }) && !done && !joined;
    }
    void complete() {
        std::lock_guard<std::mutex> lock(mutex);
        release = true;
        cv.notify_all();
    }
    void finish() {
        if (thread.joinable()) thread.join();
        joined = true;
    }
};
void success() {
    Cpu cpu;
    std::vector<float> input{1.f, 2.f}, committed{99.f, 99.f};
    bool gpu_done = false;
    int drains = 0;
    const bool ok = strata::core::cpu_first_remote_post(
        [&] { return cpu.post(input); },
        [&] {
            check(cpu.in_flight() && !gpu_done, "CPU work is in flight before GPU completion");
            input.assign(2, -999.f); // CPU owns copied input while synchronous GPU post runs.
            cpu.complete();
            gpu_done = true;
            return true;
        }, [&] { ++drains; cpu.finish(); });
    check(ok && gpu_done && drains == 0, "success leaves the original finish path intact");
    cpu.finish();
    const std::vector<float> gpu{3.f, 4.f}, local{5.f, 6.f};
    if (ok) for (size_t i = 0; i < committed.size(); ++i) committed[i] = local[i] + (gpu[i] + cpu.partial[i]);
    const float expected[]{9.f, 12.f};
    check(std::memcmp(committed.data(), expected, sizeof expected) == 0, "copied inputs and declared GPU-then-CPU merge unchanged");
}
void failure(bool throws) {
    Cpu cpu;
    std::vector<float> input{7.f, 8.f}, committed{99.f, 99.f};
    int drains = 0;
    bool ok = false, caught = false;
    try {
        ok = strata::core::cpu_first_remote_post([&] { return cpu.post(input); },
            [&]() -> bool {
                check(cpu.in_flight(), "CPU submission preceded failed GPU post");
                // No release: CPU completion is bounded by the fixture's request deadline.
                if (throws) throw std::runtime_error("GPU post fixture failure");
                return false;
            }, [&] { ++drains; cpu.finish(); });
    } catch (const std::runtime_error&) { caught = true; }
    if (ok) committed = cpu.partial;
    check(!ok && caught == throws && drains == 1 && cpu.joined && cpu.expired,
          "GPU failure/exception drains the bounded CPU request before return");
    check(committed == std::vector<float>({99.f, 99.f}), "failure cannot reach the commit branch");
    input.assign(2, -1.f);
    check(cpu.partial == std::vector<float>({7.f, 8.f}), "joined request does not retain caller buffers");
}
}
int main() {
    success();
    failure(false);
    failure(true);
    int gpu_calls = 0, drains = 0;
    check(!strata::core::cpu_first_remote_post([] { return false; }, [&] { ++gpu_calls; return true; }, [&] { ++drains; }) &&
          gpu_calls == 0 && drains == 0, "CPU post refusal launches no GPU or nonexistent cleanup");
    std::printf("CPU_FIRST_REMOTE_POST_V1 cases=4 failures=%d models=0 cuda=0\n", failures);
    return failures ? 1 : 0;
}
