#pragma once
// One fixed retained first-boundary packet. No generic model/trace replay interface.
inline constexpr uint8_t FirstBoundarySraSha[]={0x0a,0xae,0xb0,0x76,0x24,0x18,0xbc,0x15,0x0d,0x7c,0x4c,0xa2,0xe6,0xc1,0x68,0xe3,
        0x8b,0x50,0x69,0xfb,0x2d,0xf8,0x25,0xcb,0x94,0xdd,0x75,0x31,0x2a,0x6d,0x72,0x7d};
struct FirstBoundary {
    std::vector<uint8_t> prompt, expected, cpu_mask;
    std::vector<std::vector<uint8_t>> peers;
    std::vector<uint32_t> ids;
    std::vector<float> x, weights, a0, b, rows, sra_partial;
};
inline std::vector<uint8_t> boundary_bytes(const fs::path& path, size_t size) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(in && in.tellg()==std::streamoff(size),"BOUNDARY packet size mismatch");
    std::vector<uint8_t> bytes(size);in.seekg(0);
    require(bool(in.read((char*)bytes.data(),bytes.size())),"BOUNDARY short packet read");return bytes;
}
inline FirstBoundary read_boundary(const fs::path& path) {
    auto bytes=boundary_bytes(path,287220);size_t at=0;
    auto take=[&](void* out,size_t n){require(n<=bytes.size()-at,"BOUNDARY truncated packet");std::memcpy(out,bytes.data()+at,n);at+=n;};
    char magic[4];take(magic,4);require(!std::memcmp(magic,"SFB1",4),"BOUNDARY packet magic mismatch");
    uint32_t header[8];take(header,sizeof header);
    const uint32_t expected[]={2,10,2560,38,256,0,0,0};
    require(!std::memcmp(header,expected,sizeof header),"BOUNDARY T/K/H/prompt/GEN/window/layer/slice mismatch");
    uint8_t sha[32];take(sha,sizeof sha);
    require(!std::memcmp(sha,FirstBoundarySraSha,32),"BOUNDARY exporter SRA1 identity mismatch");
    FirstBoundary b;
    auto read=[&](auto& out,size_t count){out.resize(count);take(out.data(),count*sizeof(out[0]));};
    read(b.prompt,38*4);read(b.x,2*2560);read(b.ids,20);read(b.weights,20);
    read(b.expected,20);read(b.cpu_mask,20);b.peers.resize(4);
    for(auto& mask:b.peers)read(mask,20);
    read(b.a0,2*2560);read(b.b,2*2560);read(b.rows,20*2560);read(b.sra_partial,2*2560);
    require(at==bytes.size(),"BOUNDARY trailing bytes");
    for(const auto* v:{&b.x,&b.weights,&b.a0,&b.b,&b.rows,&b.sra_partial})
        require(std::all_of(v->begin(),v->end(),[](float x){return std::isfinite(x);}),"BOUNDARY nonfinite packet value");
    require(std::set<uint32_t>(b.ids.begin(),b.ids.end()).size()==20,"BOUNDARY distinct expert count mismatch");
    size_t gpu=0;
    for(size_t i=0;i<20;++i) {
        require(b.expected[i]==1 && b.cpu_mask[i]<=1,"BOUNDARY CPU ownership mismatch");
        size_t seen=b.cpu_mask[i];for(const auto& mask:b.peers){require(mask[i]<=1,"BOUNDARY nonbinary peer mask");seen+=mask[i];}
        require(seen==1,"BOUNDARY CPU/CUDA ownership gap or overlap");gpu+=!b.cpu_mask[i];
    }
    require(gpu==13,"BOUNDARY requires exactly 13 CUDA rows");return b;
}
inline void boundary_historic(const CpuCaptureFile& f,const FirstBoundary& b) {
    require(f.setup.n_embd==2560 && f.setup.n_ff==640 && f.setup.gu_type==8 && f.setup.d_type==8 &&
        f.prompt_tokens==38 && f.max_new==256 && blob_digest(b.prompt)==f.tokens && !f.records.empty() && !f.windows.empty(),
        "BOUNDARY historic prompt/geometry identity mismatch");
    const auto& w=f.windows[0];const auto& r=f.records[0];const auto& q=r.request;
    require(w.ordinal==1 && w.phase==1 && w.pos==0 && w.T==2 && r.window==1 && r.token_begin==0 && r.sent &&
        q.layer==0 && q.T==2 && q.K==10,"BOUNDARY historic record-0 window/layer/token-slice mismatch");
    require(q.activations.size()==b.x.size() && !std::memcmp(q.activations.data(),b.x.data(),b.x.size()*4),
        "BOUNDARY historic raw F32 input mismatch; SRA1 reuse stopped");
    require(q.ids.size()==20 && std::equal(q.ids.begin(),q.ids.end(),b.ids.begin()),"BOUNDARY historic router IDs mismatch");
    require(q.weights.size()==20 && !std::memcmp(q.weights.data(),b.weights.data(),80),"BOUNDARY historic weights mismatch");
    require(q.mask==b.expected,"BOUNDARY historic CPU-backed mask mismatch");
    require(r.partials.size()==b.a0.size() && !std::memcmp(r.partials.data(),b.a0.data(),b.a0.size()*4),
        "BOUNDARY historic partials do not match recorded A0 union; SRA1 reuse stopped");
    require(!std::memcmp(b.sra_partial.data(),b.a0.data(),b.a0.size()*4),"BOUNDARY SRA1 partial does not match A0 union");
    std::puts("BOUNDARY_IDENTITY raw_F32=1 ids=1 weights=1 mask=1 window_slice=1 historical_partials_vs_A0=1 SRA1_partials_vs_A0=1");
}
inline std::vector<float> boundary_reference(const FirstBoundary& b,const std::vector<float>& rows) {
    require(rows.size()==20*2560,"BOUNDARY reference row shape");std::vector<float> sum(2*2560);
    // Same independent scalar expression/order used by the existing group regression.
    for(int t=0;t<2;++t)for(int h=0;h<2560;++h) {
        float acc=0.f;
        for(int e=0;e<10;++e){const int i=t*10+e;const float product=b.weights[i]*rows[i*2560+h];acc=acc+product;}
        sum[t*2560+h]=acc;
    }
    return sum;
}
inline std::vector<float> boundary_reduce(const FirstBoundary& b,const std::vector<float>& rows,const char* label) {
    const auto reference=boundary_reference(b,rows);std::vector<float> production(2*2560),mixed;
    ExpertWorkerService::accumulate_partial(2,10,b.expected.data(),b.weights.data(),rows.data(),2560,production.data());
    require(metrics((std::string(label)+"_independent_vs_CPU_reducer").c_str(),reference,production),
        "BOUNDARY identical rows: independent vs CPU reducer mismatch");
    std::vector<EntryRowsReply> replies(4);std::vector<MixedRowsPeer> peers;
    for(size_t p=0;p<4;++p) {
        for(size_t i=0;i<20;++i)if(b.peers[p][i]) {
            replies[p].indices.push_back((uint16_t)i);
            replies[p].rows.insert(replies[p].rows.end(),rows.begin()+i*2560,rows.begin()+(i+1)*2560);
        }
        peers.push_back({replies[p],b.peers[p]});
    }
    std::string err;require(reduce_mixed_rows(2560,2,10,rows,b.cpu_mask,b.expected,b.weights.data(),peers,mixed,err),err);
    require(metrics((std::string(label)+"_identical_rows_CPU_vs_mixed_reducer").c_str(),production,mixed),
        "BOUNDARY identical rows: CPU vs mixed reducer mismatch");return production;
}
inline std::vector<float> boundary_cpu_gate(const CpuCaptureFile& f,const FirstBoundary& b) {
    boundary_historic(f,b);auto sum=boundary_reduce(b,b.rows,"CPU_rows");
    require(metrics("SRA1_rows_reconstruct_recorded_A0_union",b.a0,sum),"BOUNDARY CPU rows do not reconstruct A0; no fallback");
    return sum;
}
