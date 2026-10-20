#include "strata/worker/local_source.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <map>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace strata::worker {
namespace {
using strata::kernels::cpu::NativeFmt;
bool basename(const std::string &s) {
    if (s.empty() || s.size() > 255 || s == "." || s == "..")
        return false;
    for (unsigned char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
              c == '_' || c == '-'))
            return false;
    return true;
}
void put(std::vector<uint8_t> &p, uint64_t v, int n) {
    for (int i = 0; i < n; ++i)
        p.push_back((uint8_t)(v >> (8 * i)));
}
struct Cursor {
    const std::vector<uint8_t> &p;
    size_t i = 0;
    bool ok = true;
    uint64_t get(size_t n) {
        if (n > p.size() - i) {
            ok = false;
            return 0;
        }
        uint64_t v = 0;
        for (size_t b = 0; b < n; ++b)
            v |= (uint64_t)p[i++] << (8 * b);
        return v;
    }
    std::string str(size_t n) {
        if (n > p.size() - i) {
            ok = false;
            return {};
        }
        std::string s((const char *)p.data() + i, n);
        i += n;
        return s;
    }
};
bool bytes(const std::string &path, std::vector<uint8_t> &p, size_t cap, std::string &err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0 || (uint64_t)f.tellg() > cap) {
        err = "cannot read bounded metadata: " + path;
        return false;
    }
    p.resize((size_t)f.tellg());
    f.seekg(0);
    if (!f.read((char *)p.data(), p.size())) {
        err = "short metadata: " + path;
        return false;
    }
    return true;
}
bool valid(const LocalManifest &m, std::string &err) {
    auto bad = [&] {
        err = "malformed local manifest: geometry, files, ranges or digest count";
        return false;
    };
    if (!m.n_layers || m.n_layers > kMaxLayers || !m.n_expert || m.n_expert > kMaxExperts || !m.n_embd ||
        m.n_embd > kMaxNEmbd || !m.n_ff || m.n_ff > kMaxNFf || m.gu_type != 8 || m.d_type != 8 ||
        m.files.empty() || m.files.size() > 64 || m.layers.size() != m.n_layers ||
        m.digests.size() != (size_t)m.n_layers * m.n_expert || m.role_digests.size() != m.digests.size())
        return bad();
    NativeFmt f;
    if (!strata::kernels::cpu::native_fmt(8, 8, m.n_embd, m.n_ff, f, err))
        return false;
    for (size_t i = 0; i < m.files.size(); ++i) {
        if (!basename(m.files[i].name) || !m.files[i].bytes || m.files[i].bytes > INT64_MAX)
            return bad();
        for (size_t j = 0; j < i; ++j)
            if (m.files[j].name == m.files[i].name)
                return bad();
    }
    const uint64_t per[] = {f.up_off, f.up_off, f.bytes - f.down_off};
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> ranges(m.files.size());
    for (const auto &l : m.layers)
        for (size_t r = 0; r < 3; ++r) {
            const auto &s = l[r];
            if (s.file >= m.files.size() || s.bytes != per[r])
                return bad();
            const uint64_t n = s.bytes * m.n_expert, size = m.files[s.file].bytes;
            if (s.offset > size || n > size - s.offset)
                return bad();
            ranges[s.file].push_back({s.offset, s.offset + n});
        }
    for (auto &rr : ranges) {
        std::sort(rr.begin(), rr.end());
        for (size_t i = 1; i < rr.size(); ++i)
            if (rr[i].first < rr[i - 1].second)
                return bad();
    }
    return true;
}
} // namespace
bool LocalManifest::matches(const SetupManifest &m) const {
    return m.model_id == identity.a && m.gu_type == gu_type && m.d_type == d_type && m.n_layers == n_layers &&
           m.n_expert == n_expert && m.n_embd == n_embd && m.n_ff == n_ff &&
           std::strcmp(m.quant_name, "Q8_0") == 0 && m.partition_id == blob_digest(m.partition).a;
}
bool write_local_manifest(const std::string &path, const LocalManifest &m, std::string &err) {
    if (!valid(m, err))
        return false;
    std::vector<uint8_t> p{'S', 'W', 'L', 'O', 'C', 'A', 'L', '1'};
    for (auto v : {m.n_layers, m.n_expert, m.n_embd, m.n_ff, m.gu_type, m.d_type})
        put(p, v, 2);
    put(p, m.index_digest.a, 8);
    put(p, m.index_digest.b, 8);
    put(p, m.files.size(), 4);
    for (const auto &f : m.files) {
        put(p, f.name.size(), 2);
        p.insert(p.end(), f.name.begin(), f.name.end());
        put(p, f.bytes, 8);
    }
    for (const auto &l : m.layers)
        for (const auto &r : l) {
            put(p, r.file, 4);
            put(p, r.offset, 8);
            put(p, r.bytes, 8);
        }
    for (const auto &d : m.digests) {
        put(p, d.a, 8);
        put(p, d.b, 8);
    }
    for (const auto &rr : m.role_digests)
        for (const auto &d : rr) {
            put(p, d.a, 8);
            put(p, d.b, 8);
        }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.write((const char *)p.data(), p.size())) {
        err = "cannot write local manifest: " + path;
        return false;
    }
    return true;
}
bool read_local_manifest(const std::string &path, LocalManifest &out, std::string &err) {
    std::vector<uint8_t> p;
    if (!bytes(path, p, kLocalManifestMax, err))
        return false;
    Cursor c{p};
    LocalManifest m;
    if (c.str(8) != "SWLOCAL1") {
        err = "unsupported local manifest version";
        return false;
    }
    m.n_layers = c.get(2);
    m.n_expert = c.get(2);
    m.n_embd = c.get(2);
    m.n_ff = c.get(2);
    m.gu_type = c.get(2);
    m.d_type = c.get(2);
    m.index_digest = {c.get(8), c.get(8)};
    const auto nf = c.get(4);
    if (!c.ok || nf > 64 || !nf || !m.n_layers || m.n_layers > kMaxLayers || !m.n_expert ||
        m.n_expert > kMaxExperts) {
        err = "malformed local manifest header";
        return false;
    }
    for (size_t i = 0; i < nf; ++i) {
        const auto n = c.get(2);
        if (n > 255) {
            err = "local filename too long";
            return false;
        }
        LocalFile f;
        f.name = c.str(n);
        f.bytes = c.get(8);
        m.files.push_back(f);
    }
    m.layers.resize(m.n_layers);
    for (auto &l : m.layers)
        for (auto &r : l) {
            r.file = c.get(4);
            r.offset = c.get(8);
            r.bytes = c.get(8);
        }
    m.digests.resize((size_t)m.n_layers * m.n_expert);
    for (auto &d : m.digests)
        d = {c.get(8), c.get(8)};
    m.role_digests.resize(m.digests.size());
    for (auto &rr : m.role_digests)
        for (auto &d : rr)
            d = {c.get(8), c.get(8)};
    if (!c.ok || c.i != p.size() || !valid(m, err)) {
        if (err.empty())
            err = "malformed local manifest length";
        return false;
    }
    m.identity = blob_digest(p);
    if (!m.identity.a) {
        err = "zero local identity";
        return false;
    }
    out = std::move(m);
    return true;
}
bool match_local_index(const LocalManifest &m, const std::string &pack, const std::string &shard,
                       const strata::kernels::cpu::ExpertLayout &lay, std::string &err) {
    std::vector<uint8_t> text;
    if (!bytes(pack + "/native_experts.txt", text, 128u << 10, err))
        return false;
    if (!lay.native || lay.n_layers != m.n_layers || lay.n_expert != m.n_expert ||
        lay.gguf_off.size() != 3u * m.n_layers || blob_digest(text) != m.index_digest) {
        err = "local manifest does not match native pack index or geometry";
        return false;
    }
    for (size_t l = 0; l < m.n_layers; ++l) {
        const auto &f = lay.fmt[l];
        if (f.gu_type != m.gu_type || f.d_type != m.d_type || f.n_embd != m.n_embd || f.n_ff != m.n_ff ||
            f.had_block) {
            err = "local manifest does not match native quantization";
            return false;
        }
        for (size_t r = 0; r < 3; ++r) {
            const size_t i = 3 * l + r;
            const auto &range = m.layers[l][r];
            const std::string name = lay.gguf_file.size() > i && !lay.gguf_file[i].empty()
                                         ? lay.gguf_file[i]
                                         : std::filesystem::path(shard).filename().string();
            if (!basename(name) || range.offset != lay.gguf_off[i] || m.files[range.file].name != name) {
                err = "local manifest does not match native shard slices";
                return false;
            }
        }
    }
    for (const auto &f : m.files) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(std::filesystem::path(shard).parent_path() / f.name, ec);
        if (ec || size != f.bytes) {
            err = "engine local manifest file inventory mismatch: " + f.name;
            return false;
        }
    }
    return true;
}
LocalSource::~LocalSource() {
    close_files();
#ifndef _WIN32
    if (root_ >= 0)
        ::close(root_);
#endif
}
void LocalSource::close_files() {
#ifndef _WIN32
    for (int f : files_)
        if (f >= 0)
            ::close(f);
#endif
    files_.assign(manifest_.files.size(), -3);
    stamps_.assign(manifest_.files.size(), {});
}
bool LocalSource::open(const std::string &root, const std::string &path, std::string &err) {
    LocalManifest m;
    return read_local_manifest(path, m, err) && open(root, m, err);
}
bool LocalSource::open(const std::string &root, const LocalManifest &m, std::string &err) {
#ifdef _WIN32
    err = "worker-local sources currently require POSIX regular-file reads";
    return false;
#else
    if (!valid(m, err))
        return false;
    close_files();
    if (root_ >= 0)
        ::close(root_);
    manifest_ = m;
    root_ = ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (root_ < 0) {
        err = "cannot open configured local root";
        return false;
    }
    close_files();
    return true;
#endif
}
int LocalSource::file(uint32_t index, std::string &err) {
#ifdef _WIN32
    err = "worker-local sources require POSIX";
    return -2;
#else
    if (index >= files_.size() || root_ < 0) {
        err = "local file index outside manifest";
        return -2;
    }
    int &fd = files_[index];
    const bool first = fd == -3;
    if (first) {
        fd = ::openat(root_, manifest_.files[index].name.c_str(),
                      O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
        if (fd < 0) {
            if (errno == ENOENT)
                return -1;
            err = "cannot open local regular file: " + manifest_.files[index].name;
            fd = -2;
            return -2;
        }
    }
    if (fd == -1)
        return -1;
    if (fd < 0) {
        err = "invalid local file";
        return -2;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uint64_t)st.st_size != manifest_.files[index].bytes) {
        err = "local file type or size mismatch: " + manifest_.files[index].name;
        return -2;
    }
    const std::array<uint64_t, 6> stamp{(uint64_t)st.st_dev,
                                        (uint64_t)st.st_ino,
#ifdef __APPLE__
                                        (uint64_t)st.st_mtimespec.tv_sec,
                                        (uint64_t)st.st_mtimespec.tv_nsec,
                                        (uint64_t)st.st_ctimespec.tv_sec,
                                        (uint64_t)st.st_ctimespec.tv_nsec
#else
                                        (uint64_t)st.st_mtim.tv_sec,
                                        (uint64_t)st.st_mtim.tv_nsec,
                                        (uint64_t)st.st_ctim.tv_sec,
                                        (uint64_t)st.st_ctim.tv_nsec
#endif
    };
    if (!first && stamp != stamps_[index]) {
        err = "local source changed during setup: " + manifest_.files[index].name;
        return -2;
    }
    stamps_[index] = stamp;
    return fd;
#endif
}
bool LocalSource::read(uint32_t index, uint64_t off, void *dst, size_t n, std::string &err) {
#ifdef _WIN32
    err = "worker-local sources require POSIX";
    return false;
#else
    const int fd = file(index, err);
    if (fd < 0 || off > manifest_.files[index].bytes || n > manifest_.files[index].bytes - off) {
        err = "invalid local read range: " + err;
        return false;
    }
    ssize_t got;
    do {
        got = ::pread(fd, dst, n, (off_t)off);
    } while (got < 0 && errno == EINTR);
    if (got != (ssize_t)n) {
        err = "short or failed local slice read";
        return false;
    }
    return true;
#endif
}
bool prepare_local_manifest(const std::string &pack, const std::string &shard, uint16_t nl, uint16_t ne,
                            LocalManifest &out, std::string &err, const std::function<bool()> &cancelled) {
    using namespace strata::kernels::cpu;
    if (!nl || nl > kMaxLayers || !ne || ne > kMaxExperts || !expert_layout_load(pack, nl, ne, err))
        return false;
    const auto &lay = expert_layout();
    if (!lay.native || lay.n_expert != ne || lay.gguf_off.size() != 3u * nl || lay.version < 2 ||
        lay.version > 4) {
        err = "preparation requires a complete native GGUF reference index v2..v4";
        return false;
    }
    LocalManifest m;
    m.n_layers = nl;
    m.n_expert = ne;
    m.n_embd = H;
    m.n_ff = FF;
    m.gu_type = m.d_type = 8;
    std::vector<uint8_t> text;
    if (!bytes(pack + "/native_experts.txt", text, 128u << 10, err))
        return false;
    m.index_digest = blob_digest(text);
    m.layers.resize(nl);
    m.digests.resize((size_t)nl * ne);
    m.role_digests.resize(m.digests.size());
    std::map<std::string, std::unique_ptr<strata::GgufFile>> files;
    const char *roles[] = {"gate", "up", "down"};
    try {
        for (int l = 0; l < nl; ++l) {
            const auto &f = lay.fmt[l];
            if (f.gu_type != 8 || f.d_type != 8 || f.had_block) {
                err = "only uniform Q8_0/Q8_0 is supported";
                return false;
            }
            const uint64_t per[] = {f.up_off, f.up_off, f.bytes - f.down_off};
            for (int r = 0; r < 3; ++r) {
                if (cancelled && cancelled()) {
                    err = "manifest preparation cancelled or expired";
                    return false;
                }
                const int i = 3 * l + r;
                const std::string name = lay.gguf_file.size() > (size_t)i && !lay.gguf_file[i].empty()
                                             ? lay.gguf_file[i]
                                             : std::filesystem::path(shard).filename().string();
                if (!basename(name)) {
                    err = "native index shard must be a basename";
                    return false;
                }
                auto &g = files[name];
                if (!g) {
                    g = std::make_unique<strata::GgufFile>(
                        (std::filesystem::path(shard).parent_path() / name).string());
                    m.files.push_back({name, g->file_size()});
                }
                const auto t = g->find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight");
                if (!t || t->type != 8 ||
                    t->shape !=
                        std::vector<uint64_t>{(uint64_t)(r < 2 ? H : FF), (uint64_t)(r < 2 ? FF : H), ne} ||
                    g->data_start() + t->offset != lay.gguf_off[i] ||
                    strata::tensor_payload_bytes(*t) != per[r] * ne || lay.gguf_off[i] > g->file_size() ||
                    per[r] * ne > g->file_size() - lay.gguf_off[i]) {
                    err = "native index does not match GGUF tensor header";
                    return false;
                }
                size_t fi = 0;
                while (m.files[fi].name != name)
                    ++fi;
                m.layers[l][r] = {(uint32_t)fi, lay.gguf_off[i], per[r]};
            }
        }
        if (!valid(m, err))
            return false;
        files.clear(); // release header mappings before reading payload through bounded pread calls
        LocalSource source;
        if (!source.open(std::filesystem::path(shard).parent_path().string(), m, err))
            return false;
        std::vector<uint8_t> blob(lay.fmt[0].bytes);
        uint64_t read = 0;
        for (int l = 0; l < nl; ++l)
            for (int e = 0; e < ne; ++e) {
                size_t dst = 0;
                for (int r = 0; r < 3; ++r) {
                    const auto &range = m.layers[l][r];
                    for (size_t i = 0; i < range.bytes;) {
                        if (cancelled && cancelled()) {
                            err = "manifest preparation cancelled or expired";
                            return false;
                        }
                        const auto n = std::min<size_t>(256u << 10, range.bytes - i);
                        if (!source.read(range.file, range.offset + (uint64_t)e * range.bytes + i,
                                         blob.data() + dst + i, n, err))
                            return false;
                        i += n;
                        read += n;
                    }
                    m.role_digests[(size_t)l * ne + e][r] = blob_digest(blob.data() + dst, range.bytes);
                    dst += range.bytes;
                }
                m.digests[(size_t)l * ne + e] = blob_digest(blob);
                if (e == ne - 1)
                    std::fprintf(stderr, "worker-manifest: layer %d/%d; expert bytes read %llu\n", l + 1, nl,
                                 (unsigned long long)read);
            }
        for (size_t i = 0; i < m.files.size(); ++i)
            if (source.file(i, err) < 0)
                return false;
    } catch (const std::exception &e) {
        err = e.what();
        return false;
    }
    out = std::move(m);
    return true;
}
} // namespace strata::worker
