// Offline synthetic wire integration. Ephemeral loopback only; never a configured worker/model.
#include "local_fixture.hpp"
#include "strata/worker/packed_activation.hpp"
#include <limits>

namespace strata::worker {
struct RpcWorkspaceTestAccess {
    static bool packet(ExpertWorkerClient& c,uint16_t msg,const std::vector<uint8_t>& payload,
                       uint16_t ack,uint16_t expected) {
        std::string err;WireFrame f;
        return c.send_frame(msg,ST_OK,payload,err) && c.recv_frame(ack,f,err) && f.status==expected;
    }
};
}
namespace {
const std::string qualification="123456789abcdef1";
InferRequest request(const Fixture& f,uint64_t session) {
    const auto m=f.setup();InferRequest q;q.session=session;q.model_id=m.model_id;q.partition_id=m.partition_id;
    q.sequence=1;q.T=2;q.K=2;q.ids={0,1,0,1};q.mask={1,1,0,0};q.weights={.75f,-.25f,.5f,.5f};
    q.activations.resize(5120);
    for(size_t i=0;i<q.activations.size();++i)q.activations[i]=.1f*(int(i%11)-5);
    return q;
}
void codec(Fixture& f) {
    std::string err;auto q=request(f,7);const uint8_t active[]={1,0};uint64_t quantized=0;
    q.packed=pack_activation(f.fmt,q.activations.data(),active,2,err,&quantized);
    require(bool(q.packed),err);check(quantized==1,"one native quantization per active token");
    std::vector<uint8_t> raw,packed,expected;
    require(encode_infer(q,2560,raw) && encode_infer(q,2560,packed,true),"raw/packed encode");
    put<uint64_t>(expected,q.session);put<uint64_t>(expected,q.model_id);put<uint64_t>(expected,q.partition_id);
    put<uint64_t>(expected,q.sequence);put<uint16_t>(expected,q.layer);expected.push_back(q.T);expected.push_back(q.K);
    expected.insert(expected.end(),q.mask.begin(),q.mask.end());for(auto id:q.ids)put<uint16_t>(expected,id);
    auto f32=[&](float v){uint32_t bits;std::memcpy(&bits,&v,4);put<uint32_t>(expected,bits);};
    for(float v:q.weights)f32(v);for(float v:q.activations)f32(v);
    check(raw==expected,"default raw wire exact independent scalar encoding");
    const size_t prefix=kInferFixed+q.mask.size()*7;
    check(packed.size()==prefix+8+kPackedRowBytes && raw.size()==prefix+5120*4,
          "exact wire sizes; inactive token omitted, metadata unchanged");
    InferRequest decoded;std::vector<uint8_t> restored;
    require(decode_infer(packed.data(),packed.size(),2,2,2560,decoded,true) &&
        restore_activation(f.fmt,*decoded.packed,restored),"production packed decode/restore");
    std::vector<uint8_t> native(2*strata::kernels::cpu::kNativeActBytes,0);
    strata::kernels::cpu::native_quant_act(f.fmt,q.activations.data(),native.data());
    check(native==restored,"production useful bytes, inactive scratch and padding exact native");
    auto rejects=[&](std::vector<uint8_t> bad){InferRequest out;check(!decode_infer(bad.data(),bad.size(),2,2,2560,out,true),"malformed packed request fails closed");};
    auto bad=packed;bad.pop_back();rejects(bad);bad=packed;bad.push_back(0);rejects(bad);
    for(size_t at:{prefix,prefix+2,prefix+4,prefix+6}){bad=packed;bad[at]^=1;rejects(bad);}
    for(uint16_t scale:{uint16_t(0x7c00),uint16_t(0xfc00),uint16_t(0x7e00)}) {
        bad=packed;bad[prefix+8]=(uint8_t)scale;bad[prefix+9]=(uint8_t)(scale>>8);rejects(bad);
    }
    bad=packed;bad[kInferFixed]=2;rejects(bad);
    bad=packed;bad[34]=0;rejects(bad);bad=packed;bad[35]=0;rejects(bad);
    bad=packed;bad[kInferFixed+q.mask.size()]=2;rejects(bad); // out-of-range expert
    auto fields=std::make_shared<PackedActivation>(*q.packed);fields->fields[0]=0;fields->fields[1]=0x80;fields->fields[2]=0xff;
    q.packed=fields;require(encode_infer(q,2560,packed,true),"signed zero fields encode");
    require(decode_infer(packed.data(),packed.size(),2,2,2560,decoded,true),"signed zero fields decode");
    check(decoded.packed->fields==fields->fields,"negative zero scale and signed-byte bits preserved");
    auto wrong=f.fmt;wrong.had_gu=true;check(!pack_activation(wrong,q.activations.data(),active,2,err),"rotated format ineligible");
    wrong=f.fmt;wrong.gu_act=15;check(!pack_activation(wrong,q.activations.data(),active,2,err),"unsupported format ineligible");
    wrong=f.fmt;--wrong.act_bytes;check(!restore_activation(wrong,*q.packed,restored),"wrong native geometry refused");
    check(!pack_activation(f.fmt,q.activations.data(),active,3,err),"T3 outside sampled contract falls back before reading inputs");
    std::fill(q.activations.begin(),q.activations.end(),std::numeric_limits<float>::max());
    check(!pack_activation(f.fmt,q.activations.data(),active,2,err) && err=="unsupported_scale","finite inputs with unsupported native scales select raw");
    q.activations[0]=std::numeric_limits<float>::infinity();
    check(!pack_activation(f.fmt,q.activations.data(),active,2,err),"nonfinite input never packed");
    check(!packed_qualification_valid("") && !packed_qualification_valid("0000000000000000") &&
        !packed_qualification_valid("123456789abcdeg1") && packed_qualification_valid(qualification),"explicit nonzero qualification ID syntax");
}
void transport(Fixture& f) {
    auto settings=f.options();settings.packed_qualification=qualification;Service service(settings);
    ExpertWorkerClient c;connect(c,service);std::string err;setup(c,f);
    std::vector<uint8_t> hits;require(c.load_local({{0,0},{0,1},{1,0},{1,1}},hits,err) && c.mark_ready(err),err);
    auto q=request(f,c.session());const uint8_t active[]={1,0};q.packed=pack_activation(f.fmt,q.activations.data(),active,2,err);
    std::vector<uint8_t> bytes;require(encode_infer(q,2560,bytes,true),"packed packet encode");
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bytes,MSG_INFER_ACK,ST_WRONG_STATE),"unnegotiated packed packet refused on receiver");
    require(c.negotiate_packed("123456789abcdef2",err),err);check(!c.packed_enabled(),"unqualified build pair selects raw");
    auto raw=q;raw.packed.reset();InferReply expected,got;RpcExchangeTiming raw_time{},packed_time{};require(c.infer(raw,expected,err,&raw_time),err);
    require(c.negotiate_packed(qualification,err) && c.packed_enabled(),err);
    q.sequence=2;require(c.infer(q,got,err,&packed_time),err);
    std::printf("PACKED_WIRE_COST raw_request_bytes=%llu packed_request_bytes=%llu raw_reply_bytes=%llu packed_reply_bytes=%llu raw_encode_send_ns=%llu packed_encode_send_ns=%llu raw_receive_decode_ns=%llu packed_receive_decode_ns=%llu\n",
        (unsigned long long)raw_time.request_bytes,(unsigned long long)packed_time.request_bytes,
        (unsigned long long)raw_time.reply_bytes,(unsigned long long)packed_time.reply_bytes,
        (unsigned long long)raw_time.encode_send_ns,(unsigned long long)packed_time.encode_send_ns,
        (unsigned long long)raw_time.receive_decode_ns,(unsigned long long)packed_time.receive_decode_ns);
    check(got.partials.size()==expected.partials.size() && !std::memcmp(got.partials.data(),expected.partials.data(),got.partials.size()*4),
        "actual packed transport equals raw native weighted reply bits");
    q.sequence=3;require(encode_infer(q,2560,bytes,true),"fresh packed packet");
    auto bad=bytes;bad.back()=0;bad.pop_back();
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bad,MSG_INFER_ACK,ST_BAD_REQUEST),"malformed packed packet never raw retry");
    auto changed=q;++changed.model_id;require(encode_infer(changed,2560,bad,true),"stale packet encode");
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bad,MSG_INFER_ACK,ST_STALE_SESSION),"packed identity validation preserved");
    q.weights[0]=std::numeric_limits<float>::quiet_NaN();require(encode_infer(q,2560,bad,true),"nonfinite weight packet encode");
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bad,MSG_INFER_ACK,ST_NONFINITE),"packed finite weight validation preserved");
    q.weights[0]=.75f;q.sequence=2;require(encode_infer(q,2560,bad,true),"replay encode");
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bad,MSG_INFER_ACK,ST_BAD_REQUEST),"packed replay ordering preserved");
    q.sequence=3;q.T=2;q.K=10;q.mask.assign(20,1);q.ids.assign(20,0);q.weights.assign(20,.1f);
    require(encode_infer(q,2560,bad,true)==false,"packed rows must cover newly active token");
    const uint8_t both[]={1,1};q.packed=pack_activation(f.fmt,q.activations.data(),both,2,err);
    require(encode_infer(q,2560,bad,true),"repeated expert packet encode");
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bad,MSG_INFER_ACK,ST_BAD_REQUEST),"packed per-expert count limit preserved");
    c.close();check(!c.packed_enabled(),"packed capability reset on disconnect");
    connect(c,service);auto m=f.setup();m.partition={1,1};m.partition_id=blob_digest(m.partition).a;
    require(c.setup_local(m,f.m.identity,false,err),err);hits.clear();
    require(c.load_local({{0,0},{1,0}},hits,err) && c.mark_ready(err) && c.negotiate_packed(qualification,err),err);
    q=request(f,c.session());q.partition_id=m.partition_id;q.packed=pack_activation(f.fmt,q.activations.data(),active,2,err);
    require(encode_infer(q,2560,bad,true),"unowned packet encode");
    check(RpcWorkspaceTestAccess::packet(c,MSG_INFER_PACKED,bad,MSG_INFER_ACK,ST_NOT_OWNED),"packed ownership validation preserved");
    c.set_setup_cancelled([]{return true;});
    check(!c.negotiate_packed(qualification,err) && !c.packed_enabled(),"cancelled probe fails without raw fallback");
    c.set_setup_cancelled({});
}
}
int main() {
    std::signal(SIGALRM,SIG_DFL);::alarm(180);
    try {Fixture fixture;codec(fixture);transport(fixture);}
    catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());++failures;}
    std::printf("CPU_PACKED_WIRE failures=%d models=0\n",failures);return failures?1:0;
}
