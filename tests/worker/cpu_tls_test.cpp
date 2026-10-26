// Disposable loopback PKI and synthetic native experts only. No configured fleet or models.
#include "local_fixture.hpp"
#include "tls_fixture.hpp"
#include "strata/worker/cpu_group.hpp"
#include <chrono>

namespace {
bool secure_connect(ExpertWorkerClient& c, Service& s, const TlsConfig& config, int ms=1000) {
    std::string err;
    return c.connect_transport("127.0.0.1",s.port,ms,{},config,false,err);
}
void refused(Service& s, const TlsConfig& config, const char* reason) {
    ExpertWorkerClient c; WorkerStatusInfo info; std::string err;
    const bool connected=secure_connect(c,s,config,250);
    check(!connected || !c.status(info,err),reason);
    c.close();
}
void upload(Fixture& f,TlsFixture& certs) {
    auto o=f.options();o.tls=certs.config("worker","coordinator.test");
    Service s(o);ExpertWorkerClient c;std::string err;
    require(secure_connect(c,s,certs.config("coordinator","worker.test")),"CPU upload TLS connect");
    require(c.setup(f.setup(),err),err);
    for(int l=0;l<2;++l)for(int e=0;e<2;++e)
        require(c.upload(l,e,f.blobs[l*2+e].data(),(uint32_t)f.fmt.bytes,err),err);
    require(c.mark_ready(err),err);const auto session=c.session();
    infer(c,f,0,2,1);infer(c,f,1,2,2);WorkerStatusInfo status;
    require(c.status(status,err),err);
    check(status.session==session && status.setup_weight_bytes==f.blobs.size()*f.fmt.bytes && status.requests==2,
          "persistent CPU TLS cold chunked upload/READY/repeated native inference");c.close();
}
void lifecycle(Fixture& f, TlsFixture& certs) {
    check(WorkerOptions{}.accepted_tcp_nodelay,"CPU accepted NODELAY defaults on");
    auto o=f.options();o.tls=certs.config("worker","coordinator.test");o.residency_ms=10000;
    Service s(o);ExpertWorkerClient c;std::string err;
    const auto client=certs.config("coordinator","worker.test");
    require(secure_connect(c,s,client),"CPU mutual TLS connect");
    require(c.setup_local(f.setup(),f.m.identity,false,err,true),err);
    std::vector<uint8_t> hits;
    require(c.load_local({{0,0},{0,1},{1,0},{1,1}},hits,err) && c.mark_ready(err),err);
    const auto session=c.session();
    const auto first=infer(c,f,0,2,1), second=infer(c,f,0,2,2);
    WorkerStatusInfo status; WorkerCacheInfo cache;
    require(c.status(status,err) && c.cache_status(cache,err),err);
    check(first==second && status.session==session && status.requests==2 && status.setup_weight_bytes==0,
          "persistent mTLS CPU setup/local load/READY/status/repeated infer with unchanged native bits");
    check(cache.complete && cache.resident_bytes==f.blobs.size()*f.fmt.bytes,"verified resident cache remains complete");
    c.close();
    require(secure_connect(c,s,client) && c.setup_local(f.setup(),f.m.identity,false,err,true),err);
    check(c.cache_hit() && c.reused_bytes()==f.blobs.size()*f.fmt.bytes,"new authenticated session reattaches resident cache");
    require(c.mark_ready(err),err);infer(c,f,1,2,1);c.close();
    auto bad=client;bad.peer_dns="wrong.test";refused(s,bad,"CPU rejects wrong worker SAN");
    refused(s,certs.config("rogue-coordinator","worker.test"),"CPU rejects untrusted coordinator");
    refused(s,certs.config("expired-coordinator","worker.test"),"CPU rejects expired coordinator");
    ExpertWorkerClient plain;require(plain.connect_host("127.0.0.1",s.port,200,err),err);
    check(!plain.status(status,err),"CPU mTLS listener refuses plaintext/missing client certificate");plain.close();
    // Certificate replacement is read at the next handshake; an existing session is not retroactively revoked.
    require(secure_connect(c,s,client),"valid reconnect after rejected peers");
    const auto cert=certs.root/"worker.crt", old=certs.root/"worker.saved";
    std::filesystem::rename(cert,old);
    std::filesystem::copy_file(certs.root/"rogue-worker.crt",cert);
    require(c.status(status,err),"existing TLS session survives certificate file replacement");c.close();
    refused(s,client,"next CPU handshake reloads changed certificate and fails closed");
    std::filesystem::remove(cert);std::filesystem::rename(old,cert);
    require(secure_connect(c,s,client),"restored certificate used on reconnect");c.close();
}
void rejection_cases(Fixture& f, TlsFixture& certs) {
    std::string err;
    for(const std::string stem:{"rogue-worker","expired-worker","cn-worker"}) {
        auto o=f.options();o.tls=certs.config(stem,"coordinator.test");
        if(stem=="expired-worker") {
            ExpertWorkerService invalid(o);uint16_t p=0;
            check(!invalid.listen(p,err) && err.find("expired")!=std::string::npos,"local expiry diagnostic before listening");
        } else {Service s(o);refused(s,certs.config("coordinator","worker.test"),"CPU rejects untrusted/CN-only worker");}
    }
    auto o=f.options();o.tls=certs.config("worker","wrong.test");
    {Service s(o);refused(s,certs.config("coordinator","worker.test"),"CPU worker checks exact coordinator SAN");}
    {Service s(f.options());refused(s,certs.config("coordinator","worker.test"),"CPU TLS client refuses plaintext worker without fallback");}
    ExpertWorkerClient c;
    check(!c.connect_transport("192.0.2.1",1,100,{}, {},false,err) && err.find("requires mTLS")!=std::string::npos,
          "remote CPU plaintext refused before connect");
    auto remote=f.options();remote.bind_ip="192.0.2.1";remote.allow_peer_ips={"192.0.2.2"};
    ExpertWorkerService invalid(remote);uint16_t port=0;
    check(!invalid.listen(port,err) && err.find("requires mTLS")!=std::string::npos,"old remote CPU configuration has clear migration error");
    auto incomplete=certs.config("worker","coordinator.test");incomplete.key_file.clear();remote=f.options();remote.tls=incomplete;
    ExpertWorkerService missing(remote);
    check(!missing.listen(port,err),"missing key fails before listen");
    o=f.options();o.tls=certs.config("worker","coordinator.test");o.deadline_ms=100;
    Service s(o);sock_fd fd=kBadFd;
    require(sock::connect_numeric_controlled("127.0.0.1",s.port,1000,{},fd,err),err);
    TlsSession tls;require(tls.handshake((uintptr_t)fd,certs.config("coordinator","worker.test"),false,1000,{},err),err);
    const uint8_t partial[]={0x53,0x45,0x57};
    require(tls.send_all(partial,sizeof partial,1000,{}),"partial CPU header sent");
    uint8_t byte=0;const auto began=std::chrono::steady_clock::now();
    check(!tls.recv_all(&byte,1,1000,{}),"CPU partial encrypted frame closes at finite deadline");
    check(std::chrono::steady_clock::now()-began<std::chrono::seconds(1),"partial frame cannot pin worker");
    tls.reset();sock::close_fd(fd);
    require(secure_connect(c,s,certs.config("coordinator","worker.test")),"worker accepts valid peer after partial frame deadline");
    c.close();
}
void group(Fixture& f,TlsFixture& certs) {
    auto o=f.options();o.tls=certs.config("worker","coordinator.test");
    Service first(o),second(o);
    std::vector<ExpertWorkerTier::Open> opens(2);
    for(size_t i=0;i<2;++i) {
        auto& in=opens[i];in.port=i?second.port:first.port;in.tls=certs.config("coordinator","worker.test");
        in.local=&f.m;in.entry_rows=true;in.n_layers=2;in.n_expert=2;in.n_embd=2560;in.n_ff=640;
        in.partition.assign(2,uint8_t(1u<<i));
    }
    ExpertWorkerGroup g;std::string err;
    require(g.open(opens,{3,3},err),err);
    float weights[]={.7f,.3f};int32_t ids[]={0,1};uint8_t mask[]={1,1};std::vector<float> x(2560);
    for(size_t i=0;i<x.size();++i)x[i]=.1f*(int(i%11)-5);
    for(int n=0;n<2;++n) {
        require(g.post(0,x.data(),ids,weights,mask,1,2,err) && g.finish(err),err);
        std::vector<float> direct;const uint8_t* blobs[]={f.blobs[0].data(),f.blobs[1].data()};
        require(ExpertWorkerService::direct_native_partial(f.fmt,1,2,mask,weights,x.data(),blobs,direct,err),err);
        check(g.partials() && !std::memcmp(g.partials(),direct.data(),direct.size()*sizeof(float)),
              "CPU mTLS coordinator group preserves original indexed-row reduction bits");
    }
    g.close();
}
}
int main() {
    ::alarm(60);
    try {TlsFixture certificates;Fixture f;upload(f,certificates);lifecycle(f,certificates);rejection_cases(f,certificates);group(f,certificates);}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    std::printf("CPU_TLS failures=%d models=0\n",failures);return failures?1:0;
}
