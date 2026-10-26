#pragma once
// Disposable local test PKI only; private material never leaves this temporary directory.
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include "strata/worker/tls.hpp"

struct TlsFixture {
    std::filesystem::path root;
    EVP_PKEY* ca_key = nullptr;
    X509* ca = nullptr;
    static void need(bool ok) { if (!ok) throw std::runtime_error("test certificate fixture failed"); }
    static void extension(X509* cert, X509* issuer, int nid, const char* value) {
        X509V3_CTX ctx; X509V3_set_ctx(&ctx, issuer, cert, nullptr, nullptr, 0);
        auto* e = X509V3_EXT_conf_nid(nullptr, &ctx, nid, value);
        need(e && X509_add_ext(cert, e, -1)); X509_EXTENSION_free(e);
    }
    static X509* certificate(EVP_PKEY* key, const char* name, X509* issuer, EVP_PKEY* issuer_key,
                             bool expired = false, bool san = true) {
        static long serial = 0;
        auto* cert = X509_new(); need(cert);
        need(X509_set_version(cert, 2) && ASN1_INTEGER_set(X509_get_serialNumber(cert), ++serial));
        need(X509_gmtime_adj(X509_getm_notBefore(cert), expired ? -3600 : -60) &&
             X509_gmtime_adj(X509_getm_notAfter(cert), expired ? -60 : 3600));
        auto* subject = X509_get_subject_name(cert);
        need(X509_NAME_add_entry_by_txt(subject,"CN",MBSTRING_ASC,(const unsigned char*)name,-1,-1,0));
        need(X509_set_issuer_name(cert, issuer ? X509_get_subject_name(issuer) : subject) && X509_set_pubkey(cert,key));
        extension(cert,issuer?issuer:cert,NID_basic_constraints,issuer?"critical,CA:FALSE":"critical,CA:TRUE");
        extension(cert,issuer?issuer:cert,NID_key_usage,issuer?"critical,digitalSignature":"critical,keyCertSign,cRLSign");
        if (issuer) {
            extension(cert,issuer,NID_ext_key_usage,"serverAuth,clientAuth");
            if (san) extension(cert,issuer,NID_subject_alt_name,(std::string("DNS:")+name).c_str());
        }
        need(X509_sign(cert,issuer_key?issuer_key:key,EVP_sha256())>0); return cert;
    }
    void save(const std::string& stem, X509* cert, EVP_PKEY* key) {
        FILE* f = std::fopen((root/(stem+".crt")).c_str(),"wb"); need(f);
        need(PEM_write_X509(f,cert)); std::fclose(f);
        if (key) {
            f = std::fopen((root/(stem+".key")).c_str(),"wb"); need(f);
            need(::fchmod(fileno(f),0600)==0 && PEM_write_PrivateKey(f,key,nullptr,nullptr,0,nullptr,nullptr)); std::fclose(f);
        }
    }
    TlsFixture() {
        char path[]="/tmp/strata-test-mtls-XXXXXX"; auto* p=::mkdtemp(path); need(p); root=p;
        ca_key=EVP_EC_gen("prime256v1"); need(ca_key); ca=certificate(ca_key,"test CA",nullptr,nullptr); save("ca",ca,nullptr);
        auto* other_key=EVP_EC_gen("prime256v1"); need(other_key);
        auto* other=certificate(other_key,"other test CA",nullptr,nullptr); save("other-ca",other,nullptr);
        for(const std::string stem:{"worker","coordinator","expired-worker","expired-coordinator","rogue-worker","rogue-coordinator","cn-worker"}) {
            const bool worker=stem.find("worker")!=std::string::npos;
            const bool rogue=stem.find("rogue")==0;
            auto* key=EVP_EC_gen("prime256v1"); need(key);
            auto* cert=certificate(key,worker?"worker.test":"coordinator.test",rogue?other:ca,rogue?other_key:ca_key,
                                   stem.find("expired")==0,stem!="cn-worker");
            save(stem,cert,key); X509_free(cert); EVP_PKEY_free(key);
        }
        X509_free(other); EVP_PKEY_free(other_key);
    }
    ~TlsFixture() { X509_free(ca); EVP_PKEY_free(ca_key); std::filesystem::remove_all(root); }
    strata::worker::TlsConfig config(const std::string& stem, const std::string& peer) const {
        return {(root/"ca.crt").string(),(root/(stem+".crt")).string(),(root/(stem+".key")).string(),peer};
    }
};
