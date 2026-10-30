#include "native_router_trace_fixture.hpp"
#include <chrono>
using namespace native_router_fixture;
int main(){try {
    auto f=synthetic();validate_trace_fixture(f);
    bool refused=false;f.logits[0]=1.f;
    try{validate_trace_fixture(f);}catch(...){refused=true;}need(refused,"trace changed fixture accepted");
    f=synthetic();f.tokens=1;f.logits.resize(512);refused=false;
    try{validate_trace_fixture(f);}catch(...){refused=true;}need(refused,"trace changed launch geometry accepted");
    const auto temporary=std::filesystem::temp_directory_path()/(
        "native-router-trace-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    need(std::filesystem::create_directory(temporary),"trace test directory collision");
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}} cleanup{temporary};
    std::array<uint32_t,26> trace;trace.fill(bits(1.f));trace[0]=bits(8.f);
    write_trace(temporary.string(),trace);
    need(std::filesystem::file_size(temporary/"trace.f32")==104,"trace exact byte count");
    {std::ifstream in(temporary/"trace.f32",std::ios::binary);for(auto expected:trace)need(word(in)==expected,"trace word layout");}
    trace[3]=0xa5a5a5a5;refused=false;
    try{write_trace(temporary.string(),trace);}catch(...){refused=true;}need(refused,"unwritten trace accepted");
    {std::ifstream in(temporary/"trace.f32",std::ios::binary);for(auto expected:trace)need(word(in)==expected,"failed raw trace lost");}
    printf("trace exact-fixture/T8 checks and raw 26-word output pass; no GPU execution\n");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
