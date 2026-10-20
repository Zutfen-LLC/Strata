// Compile this source with the exact base client/header, or with the candidate client.
// In both cases only the child's process owns an arena. No in-process cross-version C++ ABI is assumed.
#include "worker_binary_fixture.hpp"
namespace {
void compatibility(Fixture& f, const char* executable) {
    Binary b(f, 0, 10000, executable, false); // no new CLI flag sent to the old worker
    ExpertWorkerClient c;
    std::string err;
    require(c.connect_local("127.0.0.1", b.port, 5000, {}, err), err);
#ifndef STRATA_COMPAT_OLD_CLIENT
    check(!c.setup_local(f.setup(), f.m.identity, false, err, true) &&
              c.last_status() == ST_LOCAL_UNAVAILABLE && !c.session(),
          "current resident negotiation refuses old/disabled worker without upload fallback");
    WorkerStatusInfo refused_status;
    require(c.status(refused_status,err),err);
    check(!refused_status.setup_weight_bytes && !refused_status.ready,
          "unsupported resident negotiation accepts zero network weights");
    c.close();
    require(c.connect_local("127.0.0.1", b.port, 5000, {}, err), err);
#endif
    require(c.setup(f.setup(), err), err);
    for (int l=0; l<2; ++l)
        for (int e=0; e<2; ++e)
            require(c.upload(l,e,f.blobs[l*2+e].data(),f.fmt.bytes,err),err);
    require(c.mark_ready(err),err);
    const auto uploaded=infer(c,f,1,4,1);
    WorkerStatusInfo status;
    require(c.status(status,err),err);
    check(status.ready && status.setup_weight_bytes==4*f.fmt.bytes,"cross-version upload and legacy STATUS");
    const auto old=c.session();
    c.close();
    require(c.connect_local("127.0.0.1",b.port,5000,{},err),err);
    require(c.setup_local(f.setup(),f.m.identity,false,err),err);
    check(c.session()!=old && !c.mark_ready(err),"cross-version reconnect has fresh incomplete session");
    std::vector<uint8_t> hits;
    require(c.load_local({{0,0},{0,1},{1,0},{1,1}},hits,err),err);
    require(c.mark_ready(err),err);
    const auto loaded=infer(c,f,1,4,1);
    require(c.status(status,err),err);
    check(loaded==uploaded && c.local_stats().bytes==4*f.fmt.bytes && status.setup_weight_bytes==4*f.fmt.bytes,
          "cross-version local-v1 adds zero network weights and preserves native bits");
    require(c.shutdown(err),err);
    c.close();
    check(b.wait_exit(),"cross-version worker exits cleanly");
}
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try {
        require(argc==1 || (argc==3 && std::string(argv[1])=="--peer"),"usage: worker_compat_test [--peer /absolute/worker]");
        Fixture f;
        compatibility(f,argc==3 ? argv[2] : STRATA_WORKER_BIN);
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL exception: %s\n",e.what()); ++failures; }
    std::printf("worker_compat_test: %d failures\n",failures);
    return failures ? 1 : 0;
}
