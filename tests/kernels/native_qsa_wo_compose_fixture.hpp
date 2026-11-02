#pragma once
// Composition fixtures reuse the replay input framing and the frozen WO weights.
// Literal producer and WO controls remain independent standalone fixtures.
#ifndef STRATA_PROJECTION_Q8_REPLAY
#define STRATA_PROJECTION_Q8_REPLAY 1
#endif
#include "native_qsa_projection_q8_fixture.hpp"
#include "native_q8_wo_fixture.hpp"

namespace native_qsa_wo_compose {
namespace q = native_projection_q8;
namespace w = native_q8_wo_fixture;
using native_q8_fixture::Bytes;
using native_q8_fixture::need;
constexpr size_t rows = 2560;

inline Bytes weights(const std::string& path) {
    auto actual=w::load(path,rows*strata::core::portable_detail::q8_wo_row_bytes);
    auto expected=w::synthetic(rows).weights;
    need(actual==expected,"composition: frozen synthetic WO weights differ");
    return actual;
}
inline void packed_domain(const Bytes& weights,const Bytes& packed) {
    // Checks the actual Q8_1 packed scales and the WO intermediate domain.
    w::Fixture f{rows,weights,packed};w::validate(f);
}
inline void contract(const std::string& weights_path,const std::string& input_path,const std::string& path) {
    (void)weights(weights_path);(void)q::read(input_path);
    need(!std::filesystem::exists(path),"composition: contract exists");
    std::ofstream o(path);o<<"{\"fields\":{\"qx.bin\":6912,\"out.f32le\":10240},\"qualified\":false}\n";
    need(bool(o),"composition: contract write");
}
inline void save(const std::filesystem::path& path,const void* data,size_t size) {
    need(!std::filesystem::exists(path),"composition: output exists");
    std::ofstream o(path,std::ios::binary);o.write(static_cast<const char*>(data),size);
    need(bool(o),"composition: output write");
}
inline std::vector<float> deterministic() {
    std::vector<float> values(q::elements);
    for(size_t i=0;i<values.size();++i) values[i]=float(int((i*37)%251)-125)*0.125f;
    q::validate(values);return values;
}
}
