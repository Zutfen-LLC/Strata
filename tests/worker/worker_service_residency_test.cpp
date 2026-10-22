// Focused service-lifetime delta; real tiny worker, no model/CUDA assets.
#include "worker_binary_fixture.hpp"
#include <sys/resource.h>
namespace {
WorkerCacheInfo status(ExpertWorkerClient& c) {
    WorkerCacheInfo out; std::string error;
    require(c.cache_status(out, error), error);
    return out;
}
void setup(ExpertWorkerClient& c, const Fixture& f, bool resident) {
    std::string error;
    require(c.setup_local(f.setup(), f.m.identity, false, error, resident), error);
    if (!c.cache_hit()) {
        std::vector<uint8_t> hits;
        require(c.load_local({{0,0},{0,1},{1,0},{1,1}}, hits, error), error);
    }
    require(c.mark_ready(error), error);
}
void connect(ExpertWorkerClient& c, const Binary& b) {
    std::string error;
    require(c.connect_local("127.0.0.1", b.port, 1000, {}, error), error);
}
}
int main() {
    rlimit memory{512u * 1048576, 512u * 1048576}, cpu{10,10}, file{32u * 1048576,32u * 1048576};
    if (setrlimit(RLIMIT_AS,&memory) || setrlimit(RLIMIT_CPU,&cpu) || setrlimit(RLIMIT_FSIZE,&file)) return 2;
    alarm(30);
    try {
        Fixture f;
        auto invalid = f.options(); invalid.residency_forever = true; invalid.residency_ms = 1;
        ExpertWorkerService conflict(invalid); uint16_t port = 0; std::string error;
        check(!conflict.listen(port,error), "finite TTL plus forever is refused");
        invalid.residency_ms = 0; invalid.deadline_ms = 0;
        ExpertWorkerService unbounded_io(invalid);
        check(!unbounded_io.listen(port,error), "forever still requires bounded frame I/O");
        Binary b(f,0,1000,STRATA_WORKER_BIN,true,true);
        ExpertWorkerClient c; connect(c,b); setup(c,f,true);
        const auto cold = status(c);
        const auto session = c.session();
        check(cold.complete && cold.idle_remaining_ms == kResidencyNoExpiry, "explicit no-expiry status sentinel");
        c.close(); pause_ms(1200); connect(c,b);
        auto idle = status(c);
        check(!idle.session && idle.state == 0 && idle.complete && idle.resident_bytes == 4*f.fmt.bytes &&
              idle.idle_remaining_ms == kResidencyNoExpiry && !idle.expirations,
              "disconnect and idle retain complete arena with no finite TTL");
        setup(c,f,true); auto warm = status(c);
        check(c.cache_hit() && c.session() != session && warm.arena_allocations == 1 &&
              warm.loaded_bytes == cold.loaded_bytes && warm.verified_bytes == cold.verified_bytes,
              "fresh session reuses arena without new source load/hash/allocation");
        require(c.unload(error),error); auto empty = status(c);
        check(!empty.complete && !empty.resident_bytes && !empty.idle_remaining_ms,
              "explicit unload releases forever arena and leaves worker available");
        setup(c,f,false); c.close(); connect(c,b);
        check(!status(c).resident_bytes, "forever option never retains a nonresident session");
        setup(c,f,true); require(c.shutdown(error),error); c.close();
        check(b.wait_exit(), "explicit stop reaps worker with released arena");
        Binary finite(f,1000,1000); connect(c,finite); setup(c,f,true); c.close();
        pause_ms(1200); connect(c,finite); auto expired = status(c);
        check(!expired.resident_bytes && expired.expirations == 1, "finite TTL semantics remain finite");
        require(c.shutdown(error),error); c.close(); check(finite.wait_exit(), "finite worker cleanup");
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); ++failures; }
    std::printf("CPU_SERVICE_RESIDENCY_FIXTURES_V1 failures=%d models=0 cuda=0\n",failures);
    return failures ? 1 : 0;
}
