#pragma once
#include "native_router_fixture.hpp"
namespace native_router_fixture {
inline void validate_trace_fixture(const Fixture& f) {
    validate(f);const auto expected=synthetic();
    need(f.tokens==8 && f.logits.size()==expected.logits.size(),"router trace requires original T8 fixture");
    for(size_t i=0;i<f.logits.size();++i)need(bits(f.logits[i])==bits(expected.logits[i]),"router trace requires exact original fixture bits");
}
inline void complete_trace(const std::string& dir) {
    std::ofstream out(dir+"/trace.complete.json");need(bool(out),"router trace completion create");
    out<<"{\"schema\":1,\"tokens\":8,\"row\":2,\"words\":26,\"fixture_exact\":true,\"trace_guards_ok\":true}\n";
    need(bool(out),"router trace completion write");
}
inline void write_trace(const std::string& dir,const std::array<uint32_t,26>& trace) {
    std::ofstream out(dir+"/trace.f32",std::ios::binary);need(bool(out),"router trace output create");
    for(auto x:trace)word(out,x);need(bool(out),"router trace output write");
    // Preserve raw bytes first, including unwritten/invalid traces on failure.
    for(auto x:trace)need(std::isfinite(value(x)) && value(x)>0,"router trace invalid/unwritten field; raw retained");
    need(trace[0]==bits(8.f),"router trace wrong maximum; raw retained");
}
}
