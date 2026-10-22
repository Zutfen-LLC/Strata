// Opt-in, one-request diagnostic records. Not a model/cache format.
#pragma once
#include "strata/worker/gpu_wire.hpp"
#include <cerrno>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace strata::worker::diagnostic {
constexpr size_t cap = 12u << 20;
static_assert(sizeof(float)==4,"capture requires float32");
inline std::vector<uint8_t> floats(const float* p, size_t n) {
    std::vector<uint8_t> out(n*sizeof(float)); if (n) std::memcpy(out.data(),p,out.size()); return out;
}
inline void section(std::vector<uint8_t>& out,const std::vector<uint8_t>& value) {
    gpu_codec::put(out,value.size()); out.insert(out.end(),value.begin(),value.end());
}
inline bool section(gpu_codec::Cursor& c,std::vector<uint8_t>& out) {
    const auto n=c.get();
    if (!c.ok || n>cap || c.i>c.p.size() || n>c.p.size()-c.i) return false;
    out.assign(c.p.begin()+c.i,c.p.begin()+c.i+(size_t)n); c.i+=(size_t)n; return true;
}
inline bool file(const std::string& directory,const char* name,std::vector<uint8_t>& bytes,bool write,std::string& err) {
    if (directory.empty() || directory[0]!='/' || bytes.size()>cap) {err="diagnostic absolute directory/size";return false;}
    const int dir=::open(directory.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    struct stat ds{};
    if (dir<0) {err="diagnostic directory open";return false;}
    if (::fstat(dir,&ds) || ds.st_uid!=::getuid() || (ds.st_mode&0077)) {::close(dir);err="diagnostic private owned directory";return false;}
    const int fd=::openat(dir,name,(write ? O_WRONLY|O_CREAT|O_EXCL : O_RDONLY)|O_NOFOLLOW|O_CLOEXEC|O_NONBLOCK,0600);
    ::close(dir); if (fd<0) {err="diagnostic file open (never replace)";return false;}
    struct stat before{},after{}; bool ok=::fstat(fd,&before)==0 && S_ISREG(before.st_mode);
    if (!write) {
        ok=ok && before.st_size>0 && (uint64_t)before.st_size<=cap;
        if (ok) bytes.resize((size_t)before.st_size);
    }
    size_t at=0;
    while (ok && at<bytes.size()) {
        const auto n=write ? ::write(fd,bytes.data()+at,bytes.size()-at) : ::read(fd,bytes.data()+at,bytes.size()-at);
        if (n<0 && errno==EINTR) continue;
        if (n<=0) {ok=false;break;} at+=(size_t)n;
    }
    if (write) ok=ok && ::fsync(fd)==0;
    else ok=ok && ::fstat(fd,&after)==0 && before.st_dev==after.st_dev && before.st_ino==after.st_ino &&
        before.st_size==after.st_size && before.st_mtim.tv_sec==after.st_mtim.tv_sec &&
        before.st_mtim.tv_nsec==after.st_mtim.tv_nsec && before.st_ctim.tv_sec==after.st_ctim.tv_sec && before.st_ctim.tv_nsec==after.st_ctim.tv_nsec;
    ok=(::close(fd)==0) && ok; if (!ok) err="diagnostic short/changed/nonregular file"; return ok;
}
struct Capture {
    uint32_t selected=0;
    std::vector<uint8_t> request,blob,rows,partial,input_q,hidden_q;
};
inline std::vector<uint8_t> encode(const Capture& c) {
    std::vector<uint8_t> out{'S','G','C','A','P','0','0','1'}; gpu_codec::put(out,c.selected,4);
    for (const auto* v : {&c.request,&c.blob,&c.rows,&c.partial,&c.input_q,&c.hidden_q}) section(out,*v);
    return out;
}
inline bool decode(const GpuPlan& p,const std::vector<uint8_t>& bytes,Capture& out,InferRequest& r,std::string& err) {
    if (bytes.size()>cap || bytes.size()<12 || std::memcmp(bytes.data(),"SGCAP001",8)) {err="capture header";return false;}
    gpu_codec::Cursor c{bytes}; c.i=8; out.selected=(uint32_t)c.get(4);
    for (auto* v : {&out.request,&out.blob,&out.rows,&out.partial,&out.input_q,&out.hidden_q})
        if (!section(c,*v)) {err="capture section bounds";return false;}
    if (!c.done() || !gpu_codec::decode_request(p,out.request,r) || r.T!=1 || out.selected>=r.K || !r.mask[out.selected] ||
        out.rows.size()!=(size_t)r.K*p.n_embd*4 || out.partial.size()!=(size_t)p.n_embd*4 ||
        out.input_q.size()!=(size_t)p.n_embd/32*36 || out.hidden_q.size()!=(size_t)p.n_ff/32*36) {err="capture geometry/binding";return false;}
    const auto* asset=p.find(r.layer,r.ids[out.selected]);
    if (!asset || out.blob.size()!=asset->bytes || out.blob.size()>kGpuBlobMax || blob_digest(out.blob)!=asset->digest) {err="capture resident blob identity";return false;}
    for (float x:r.activations) if (!std::isfinite(x)) {err="capture nonfinite input";return false;}
    for (float w:r.weights) if (!std::isfinite(w)) {err="capture nonfinite routing weight";return false;}
    return true;
}
struct Merge {
    std::vector<uint8_t> request,before,gpu,cpu,after,cpu_mask;
};
inline std::vector<uint8_t> encode(const Merge& c) {
    std::vector<uint8_t> out{'S','G','M','R','G','0','0','1'};
    for (const auto* v : {&c.request,&c.before,&c.gpu,&c.cpu,&c.after,&c.cpu_mask}) section(out,*v); return out;
}
inline bool decode(const std::vector<uint8_t>& bytes,Merge& out,std::string& err) {
    if (bytes.size()>cap || bytes.size()<8 || std::memcmp(bytes.data(),"SGMRG001",8)) {err="merge header";return false;}
    gpu_codec::Cursor c{bytes}; c.i=8;
    for (auto* v : {&out.request,&out.before,&out.gpu,&out.cpu,&out.after,&out.cpu_mask})
        if (!section(c,*v)) {err="merge section bounds";return false;}
    if (!c.done()) {err="merge trailing bytes";return false;} return true;
}
} // namespace strata::worker::diagnostic
