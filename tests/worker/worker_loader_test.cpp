#include "local_fixture.hpp"
#include "local_loader.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <set>

namespace {
const std::vector<std::pair<uint16_t, uint16_t>> slots{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

void open_sources(LocalSource& source, const Fixture& f) {
    std::string err;
    require(source.open(f.root.string(), f.m, err), err);
    for (uint32_t i = 0; i < f.m.files.size(); ++i) require(source.file(i, err) >= 0, err);
}
std::vector<LocalLoadTask> tasks_for(const Fixture& f, std::vector<uint8_t>& arena) {
    const size_t stride = f.fmt.bytes + 16;
    arena.assign(4 * stride, 0xcd);
    std::vector<LocalLoadTask> tasks;
    for (size_t i = 0; i < 4; ++i)
        tasks.push_back({(uint16_t)(i / 2), (uint16_t)(i % 2), arena.data() + i * stride + 8,
                         {true, true, true}});
    return tasks;
}
void check_arena(const Fixture& f, const std::vector<uint8_t>& arena) {
    const size_t stride = f.fmt.bytes + 16;
    for (size_t i = 0; i < 4; ++i) {
        const auto dst = arena.data() + i * stride + 8;
        check(!std::memcmp(dst, f.blobs[i].data(), f.fmt.bytes) &&
                  blob_digest(dst, f.fmt.bytes) == f.m.digests[i], "exact expert bytes and manifest digest");
        check(std::all_of(dst - 8, dst, [](uint8_t b) { return b == 0xcd; }) &&
                  std::all_of(dst + f.fmt.bytes, dst + f.fmt.bytes + 8,
                              [](uint8_t b) { return b == 0xcd; }), "disjoint arena guards remain intact");
    }
}
void direct_batches(Fixture& f, int threads) {
    LocalSource source;
    open_sources(source, f);
    std::vector<uint8_t> arena;
    auto tasks = tasks_for(f, arena);
    const auto ok = load_local_parallel(source, f.fmt, tasks, threads, {});
    check(ok.status == ST_OK && ok.completed == 4 && ok.bytes == 4 * f.fmt.bytes &&
              ok.verified_bytes == ok.bytes && ok.read_ns && ok.verify_ns, "bounded batch exact byte accounting");
    check_arena(f, arena);

    // Every lane reaches the first cancellation probe before any lane can proceed. Then interrupt
    // after some chunks, with all lanes outstanding. No disk-speed or sleep-based race is required.
    for (bool throws : {false, true}) {
        std::mutex mutex;
        std::condition_variable wake;
        std::set<std::thread::id> entered;
        std::atomic<int> probes{0};
        bool barrier_ok = true;
        auto cancelled = [&] {
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (entered.insert(std::this_thread::get_id()).second) {
                    wake.notify_all();
                    if (!wake.wait_for(lock, std::chrono::seconds(2), [&] { return entered.size() == (size_t)threads; }))
                        barrier_ok = false;
                }
            }
            if (++probes < 12) return false;
            if (throws) throw std::runtime_error("injected cancellation callback failure");
            return true;
        };
        const auto stopped = load_local_parallel(source, f.fmt, tasks, threads, cancelled);
        check(barrier_ok && entered.size() == (size_t)threads &&
                  stopped.status == (throws ? ST_INTERNAL : ST_DEADLINE) && stopped.completed < 4 &&
                  stopped.bytes > 0 && stopped.bytes < 4 * f.fmt.bytes,
              "active lanes cancel or catch failure and join with bounded partial accounting");
        const auto fresh = load_local_parallel(source, f.fmt, tasks, threads, {});
        check(fresh.status == ST_OK && fresh.completed == 4, "same arena/source safely reusable after all lanes join");
        check_arena(f, arena);
    }
    source.close_files(); // safe immediately on synchronous return, including failed batches

    // Truncate a descriptor after preflight, as distinct from a file rejected before threads start.
    fs::rename(f.root / "a.gguf", f.root / "a.saved");
    fs::copy_file(f.root / "a.saved", f.root / "a.gguf");
    {
        LocalSource changed;
        open_sources(changed, f);
        fs::resize_file(f.root / "a.gguf", 32);
        const auto bad = load_local_parallel(changed, f.fmt, tasks, threads, {});
        check(bad.status == ST_LOCAL_INVALID && bad.completed < 4,
              "truncated opened source fails stability checks and reaps concurrent readers");
    }
    fs::remove(f.root / "a.gguf");
    fs::rename(f.root / "a.saved", f.root / "a.gguf");
}

void service_lifecycle(Fixture& f, int threads) {
    auto options = f.options(); options.local_load_threads = threads; options.residency_ms = 60000;
    Service service(options);
    ExpertWorkerClient client;
    connect(client, service);
    std::string err;
    auto begin = [&] { require(client.setup_local(f.setup(), f.m.identity, false, err, true), err); };
    begin();
    check(!client.mark_ready(err), "READY refuses before required local bytes are verified");
    std::vector<uint8_t> hits;
    require(client.load_local(slots, hits, err), err);
    check(hits == std::vector<uint8_t>(4, 1), "ordered hit flags for 1/2/4 loader lanes");
    require(client.mark_ready(err), err);
    WorkerCacheInfo before;
    require(client.cache_status(before, err), err);
    check(before.complete && before.resident_bytes == 4 * f.fmt.bytes &&
              before.loaded_bytes == before.resident_bytes && before.verified_bytes == before.resident_bytes &&
              before.peak_arena_bytes == before.resident_bytes && before.arena_allocations == 1,
          "one exact final resident arena and exact loaded/verified counts");
    WorkerStatusInfo status;
    require(client.status(status, err), err);
    check(status.setup_weight_bytes == 0 && client.local_stats().bytes == before.resident_bytes &&
              client.local_stats().hits == 4 && client.local_stats().misses == 0,
          "local load sends zero network weights with exact local accounting");
    infer(client, f, 1, 8, 1);
    client.close(); connect(client, service); begin(); require(client.mark_ready(err), err);
    WorkerCacheInfo reused;
    require(client.cache_status(reused, err), err);
    check(client.cache_hit() && reused.arena_allocations == 1 && reused.loaded_bytes == before.loaded_bytes &&
              reused.verified_bytes == before.verified_bytes && reused.reused_bytes == before.resident_bytes,
          "resident reconnect does not read, hash or allocate another arena");
    infer(client, f, 0, 1, 1);
    require(client.unload(err), err);
    require(client.cache_status(reused, err), err);
    check(!reused.complete && reused.resident_bytes == 0, "explicit unload releases parallel-loaded arena");

    f.flip("a.gguf", f.m.layers[0][0].offset);
    begin();
    check(!client.load_local(slots, hits, err) && client.last_status() == ST_DIGEST_MISMATCH,
          "corrupt expert poisons the batch");
    check(!client.mark_ready(err), "parallel failure cannot expose partially READY state");
    require(client.cache_status(reused, err), err);
    check(!reused.complete && reused.state == 1, "failed batch is not a reusable resident partition");
    check(!client.load_local(slots, hits, err) && client.last_status() == ST_DUPLICATE,
          "failed batch cannot retry without fresh setup");
    f.flip("a.gguf", f.m.layers[0][0].offset);
    begin(); require(client.load_local(slots, hits, err), err); require(client.mark_ready(err), err);
    infer(client, f, 0, 2, 1);
    require(client.unload(err), err); client.close();
}

void missing_roles(Fixture& f, int threads) {
    fs::rename(f.root / "b.gguf", f.root / "b.saved");
    {
        auto options = f.options(); options.local_load_threads = threads;
        Service service(options);
        ExpertWorkerClient client; connect(client, service); setup(client, f);
        std::string err;
        std::vector<uint8_t> hits;
        require(client.load_local(slots, hits, err), err);
        const uint64_t present = 2 * f.fmt.bytes + 2 * (f.fmt.bytes - f.fmt.down_off);
        check(hits == std::vector<uint8_t>({1, 1, 0, 0}) && client.local_stats().bytes == present &&
                  client.local_stats().hits == 2 && client.local_stats().misses == 2,
              "missing roles preserve ordered misses and exact present-byte accounting");
        check(!client.mark_ready(err), "missing roles cannot become READY");
        for (int e = 0; e < 2; ++e)
            require(client.upload(1, e, f.blobs[2 + e].data(), f.fmt.bytes, err), err);
        require(client.mark_ready(err), err); infer(client, f, 1, 2, 1); client.close();
    }
    f.flip("c.gguf", f.m.layers[1][2].offset);
    {
        auto options = f.options(); options.local_load_threads = threads;
        Service service(options);
        ExpertWorkerClient client; connect(client, service); setup(client, f);
        std::string err;
        std::vector<uint8_t> hits;
        check(!client.load_local(slots, hits, err) && client.last_status() == ST_DIGEST_MISMATCH,
              "corrupt present role is still checked beside an absent role");
        check(!client.mark_ready(err), "corrupt incomplete slot cannot become READY"); client.close();
    }
    f.flip("c.gguf", f.m.layers[1][2].offset);
    fs::rename(f.root / "b.saved", f.root / "b.gguf");
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--threads", "use --threads 1|2|4");
        const int threads = std::stoi(argv[2]);
        require(threads == 1 || threads == 2 || threads == 4, "fixture threads must be 1, 2 or 4");
        Fixture fixture;
        direct_batches(fixture, threads);
        service_lifecycle(fixture, threads);
        missing_roles(fixture, threads);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL exception: %s\n", error.what()); ++failures;
    }
    std::printf("worker_loader_test: %d failures\n", failures);
    return failures ? 1 : 0;
}
