#include "tls_fixture.hpp"
#include "wire_sock.hpp"
#include <openssl/ssl.h>
#include <chrono>
#include <atomic>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
using namespace strata::worker;
namespace {
void require(bool ok, const std::string& why) { if(!ok) throw std::runtime_error(why); }
struct Pair {
    sock_fd listener=kBadFd, client=kBadFd, server=kBadFd;
    Pair() {
        uint16_t port=0; std::string err;
        require(sock::listen_loopback(port,listener,err),err);
        require(sock::connect_numeric_controlled("127.0.0.1",port,1000,{},client,err),err);
        require(sock::accept_one(listener,server,1000),"accept");
    }
    ~Pair(){sock::close_fd(client);sock::close_fd(server);sock::close_fd(listener);}
};
void verify_case(const TlsConfig& server_config,const TlsConfig& client_config,
                 bool accept_server,bool accept_client,const char* label) {
    Pair p; bool server_ok=false,client_ok=false;std::string server_error,client_error;
    std::thread server([&]{
        TlsSession tls; server_ok=tls.handshake(p.server,server_config,true,1000,{},server_error);
        if(server_ok){char byte=0; server_ok=tls.recv_all(&byte,1,1000,{})&&tls.send_all(&byte,1,1000,{});}
    });
    {
        TlsSession tls;client_ok=tls.handshake(p.client,client_config,false,1000,{},client_error);
        if(client_ok){char byte=42;client_ok=tls.send_all(&byte,1,1000,{})&&tls.recv_all(&byte,1,1000,{})&&byte==42;}
    }
    server.join();
    require(server_ok==accept_server && client_ok==accept_client,label);std::cout<<"PASS "<<label<<"\n";
}
void io_cases(const TlsFixture& fixture) {
    Pair p; bool ok=false;std::string err;
    std::vector<char> expected(512*1024);for(size_t i=0;i<expected.size();++i)expected[i]=char(i%251);
    // Small socket buffers and many TLS records force partial nonblocking writes/reads.
    int buffer=1024;setsockopt(p.client,SOL_SOCKET,SO_SNDBUF,&buffer,sizeof buffer);
    std::thread server([&]{TlsSession tls;std::string error;
        ok=tls.handshake(p.server,fixture.config("worker","coordinator.test"),true,2000,{},error);
        std::vector<char> bytes(expected.size());
        for(int request=0;request<3&&ok;++request){
            ok=tls.recv_all(bytes.data(),bytes.size(),5000,{}) && bytes==expected;
            if(ok)for(size_t offset=0;offset<bytes.size();offset+=4093)
                if(!tls.send_all(bytes.data()+offset,std::min(size_t(4093),bytes.size()-offset),5000,{})){ok=false;break;}
        }
    });
    TlsSession client;require(client.handshake(p.client,fixture.config("coordinator","worker.test"),false,2000,{},err),err);
    bool good=true;std::vector<char> output(expected.size());
    for(int request=0;request<3&&good;++request)
        good=client.send_all(expected.data(),expected.size(),5000,{}) && client.recv_all(output.data(),output.size(),5000,{}) && output==expected;
    server.join();require(good&&ok,"fragmentation and persistent reuse");
    std::cout<<"PASS partial TLS reads/writes, three large requests on one handshake\n";
    // Peer closes without close_notify: no signal termination, no partial success.
    sock::close_fd(p.server);char byte=0;
    require(!client.recv_all(&byte,1,200,{})&&!client.active(),"disconnect invalidates TLS");
    require(!client.send_all(&byte,1,200,{}),"failed stream cannot become plaintext");
    std::cout<<"PASS disconnect invalidates stream without fallback\n";
}
void deadline_cases(const TlsFixture& fixture) {
    for(bool server:{false,true}) {
        Pair p;TlsSession session;std::string err;const auto began=std::chrono::steady_clock::now();
        require(!session.handshake(server?p.server:p.client,fixture.config(server?"worker":"coordinator",server?"coordinator.test":"worker.test"),server,70,{},err),"stalled handshake refused");
        require(std::chrono::steady_clock::now()-began<std::chrono::seconds(1),"finite handshake deadline");
    }
    Pair p;bool ok=false;std::thread server([&]{TlsSession session;std::string err;
        ok=session.handshake(p.server,fixture.config("worker","coordinator.test"),true,1000,{},err);
        if(ok){char byte;const auto began=std::chrono::steady_clock::now();ok=!session.recv_all(&byte,1,70,{})&&!session.active()&&std::chrono::steady_clock::now()-began<std::chrono::seconds(1);}
    });
    TlsSession client;std::string err;const bool connected=client.handshake(p.client,fixture.config("coordinator","worker.test"),false,1000,{},err);
    server.join();require(connected&&ok,"finite I/O timeout");
    Pair cancelled;TlsSession session;
    require(!session.handshake(cancelled.client,fixture.config("coordinator","worker.test"),false,1000,[]{return true;},err),"handshake cancellation");
    std::cout<<"PASS handshake and I/O deadlines, cancellation\n";
}
void bounded_progress(const TlsFixture& fixture) {
    for(bool write:{false,true}) {
        Pair p;std::atomic<bool> done{false};bool server_ready=false;
        int buffer=1024;setsockopt(p.client,SOL_SOCKET,SO_SNDBUF,&buffer,sizeof buffer);
        std::thread server([&]{TlsSession tls;std::string err;
            server_ready=tls.handshake(p.server,fixture.config("worker","coordinator.test"),true,1000,{},err);
            while(server_ready&&!done.load()) {
                if(!write){char byte=1;if(!tls.send_all(&byte,1,200,{}))break;}
                sock::sleep_ms(write?1:20);
            }
        });
        TlsSession client;std::string err;
        const bool connected=client.handshake(p.client,fixture.config("coordinator","worker.test"),false,1000,{},err);
        std::vector<char> bytes(write?16*1024*1024:100);const auto began=std::chrono::steady_clock::now();
        const bool success=connected&&(write?client.send_all(bytes.data(),bytes.size(),90,{}):client.recv_all(bytes.data(),bytes.size(),90,{}));
        done.store(true);server.join();
        require(connected&&server_ready&&!success&&!client.active()&&
                std::chrono::steady_clock::now()-began<std::chrono::seconds(1),"progress cannot renew I/O deadline");
    }
    std::cout<<"PASS stalled write and trickle read retain total I/O deadlines\n";
}
void absent_certificate(const TlsFixture& fixture, bool legacy = false) {
    Pair p;bool accepted=true;std::thread server([&]{TlsSession session;std::string err;
        accepted=session.handshake(p.server,fixture.config("worker","coordinator.test"),true,700,{},err);
    });
    SSL_CTX* ctx=SSL_CTX_new(TLS_client_method());require(ctx,"client context");
    if(legacy)require(SSL_CTX_set_max_proto_version(ctx,TLS1_2_VERSION),"legacy test client");
    SSL* ssl=SSL_new(ctx);SSL_set_fd(ssl,p.client);
    const auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(500);
    while(std::chrono::steady_clock::now()<end) {
        const int result=SSL_connect(ssl);if(result==1)break;
        const int error=SSL_get_error(ssl,result);if(error!=SSL_ERROR_WANT_READ&&error!=SSL_ERROR_WANT_WRITE)break;
        sock::sleep_ms(1);
    }
    SSL_free(ssl);SSL_CTX_free(ctx);server.join();require(!accepted,"unauthenticated/legacy client rejected");
    std::cout<<(legacy?"PASS TLS 1.2 client rejected\n":"PASS absent client certificate rejected\n");
}
}
int main(){try{
    TlsFixture fixture;
    auto server=fixture.config("worker","coordinator.test"),client=fixture.config("coordinator","worker.test");
    verify_case(server,client,true,true,"mutual trusted exact identities");
    auto wrong=client;wrong.peer_dns="other.test";verify_case(server,wrong,false,false,"wrong worker identity rejected");
    wrong=server;wrong.peer_dns="other.test";verify_case(wrong,client,false,false,"wrong coordinator identity rejected");
    verify_case(fixture.config("rogue-worker","coordinator.test"),client,false,false,"untrusted worker rejected");
    verify_case(server,fixture.config("rogue-coordinator","worker.test"),false,false,"untrusted coordinator rejected");
    verify_case(fixture.config("expired-worker","coordinator.test"),client,false,false,"expired worker rejected");
    verify_case(server,fixture.config("expired-coordinator","worker.test"),false,false,"expired coordinator rejected");
    verify_case(fixture.config("cn-worker","coordinator.test"),client,false,false,"CN without SAN rejected");
    auto incomplete=client;incomplete.key_file.clear();std::string err;
    require(!incomplete.validate(err),"incomplete mTLS config rejected");
    for(const std::string name:{".test","*.test","worker..test","worker.test.","worker/test"}) {
        auto invalid=client;invalid.peer_dns=name;require(!invalid.validate(err),"peer must be exact DNS identity");
    }
    absent_certificate(fixture);absent_certificate(fixture,true);io_cases(fixture);deadline_cases(fixture);bounded_progress(fixture);return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}}
