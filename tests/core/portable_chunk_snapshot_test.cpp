#include "portable_layer_chunk_fixture.hpp"
#include <limits>
using namespace layer_chunk_fixture;
namespace {
Snapshot sample() {
    Snapshot s{3,{0.5f,-0.25f},{},{}};
    append_field(s,"token_history",std::vector<int32_t>{7,9});
    append_field(s,"recurrent_state",std::vector<float>{1,2});
    append_field(s,"convolution_history",std::vector<float>{3,4});
    append_field(s,"ple_convolution_history",std::vector<float>{5});
    return s;
}
void rejects(const Snapshot& a,const Snapshot& b,bool last,const char* field) {
    const auto detail=snapshot_difference(a,b,last,"controlled-regression");
    check(detail.find(field)!=std::string::npos,"diagnostic omitted changed field");
    bool rejected=false;
    try {same(a,b,last,"controlled-regression");}
    catch(const std::runtime_error& e) {rejected=std::string(e.what())=="chunk committed snapshot differs";}
    check(rejected,"original exact snapshot assertion was weakened");
}
}
int main() {
    try {
        const auto reference=sample();same(reference,reference);
        auto actual=reference;actual.tokens=4;rejects(actual,reference,true,"tokens_actual=4 tokens_reference=3");
        for(const auto& field:reference.fields) {
            actual=reference;actual.history[field.offset]^=1;
            rejects(actual,reference,true,(std::string("field=")+field.name).c_str());
        }
        actual=reference;actual.last[0]=std::nextafter(actual.last[0],1.f);
        rejects(actual,reference,true,"field=last_output");
        // Output tolerance is handled elsewhere; last=false must still omit it.
        same(actual,reference,false);
        actual=reference;actual.last[0]=std::numeric_limits<float>::quiet_NaN();
        rejects(actual,reference,true,"actual_nonfinite=1");
        actual=reference;actual.last={0.5f};rejects(actual,reference,true,"actual_elements=1 reference_elements=2");
        actual=reference;actual.last[0]=0.f;auto negative=actual;negative.last[0]=-0.f;
        rejects(actual,negative,true,"finite_pair_max_abs=0"); // bit difference is still a failure
        std::puts("portable_chunk_snapshot_test passed: field diagnostics preserve exact token/history/state/output gates");
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
