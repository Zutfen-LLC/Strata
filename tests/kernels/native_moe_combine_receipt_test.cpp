#include "native_moe_combine_fixture.hpp"
#include <chrono>
using namespace native_moe_fixture;
struct Temporary {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("native-moe-combine-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary(){need(std::filesystem::create_directory(path),"combine temporary directory unavailable");}
    ~Temporary(){std::error_code error;std::filesystem::remove_all(path,error);}
};
template<class Function> void rejects(Function function) {
    bool rejected=false;try{function();}catch(const std::runtime_error&){rejected=true;}
    need(rejected,"combine negative control accepted");
}
int main(){try {
    Temporary temporary;const auto f=synthetic();const auto expected=reference(f);const auto before=padded(f);auto after=before;
    for(size_t i=0;i<expected.size();++i)after[3][prefixes[3]+i]=bits(expected[i]);
    const auto prefix=[&](const char* name){return (temporary.path/name).string();};
    need(check_and_write(prefix("exact"),f,before,after)==0,"combine exact host receipt failed");
    auto changed=after;changed[3][prefixes[3]+13]^=1;
    need(check_and_write(prefix("bit"),f,before,changed)==1,"combine raw-bit mismatch waived");
    changed=after;changed[0][prefixes[0]]^=1;
    rejects([&]{check_and_write(prefix("input"),f,before,changed);});
    changed=after;changed[3].back()^=1;
    rejects([&]{check_and_write(prefix("sentinel"),f,before,changed);});
    changed=after;changed[3][prefixes[3]]=0x7f800000;
    rejects([&]{check_and_write(prefix("nonfinite"),f,before,changed);});
    const auto fixture=prefix("fixture.bin");write_fixture(fixture,f);const auto loaded=read_fixture(fixture);
    need(loaded.tokens==f.tokens,"combine fixture token roundtrip");
    for(size_t i=0;i<f.parts.size();++i)need(bits(f.parts[i])==bits(loaded.parts[i]),"combine parts bit roundtrip");
    for(size_t i=0;i<f.weights.size();++i)need(bits(f.weights[i])==bits(loaded.weights[i]),"combine weights bit roundtrip");
    for(size_t i=0;i<f.shared.size();++i)need(bits(f.shared[i])==bits(loaded.shared[i]),"combine shared bit roundtrip");
    rejects([&]{write_fixture(fixture,f);});
    {std::ofstream extra(fixture,std::ios::app|std::ios::binary);extra.put(0);}
    rejects([&]{read_fixture(fixture);});
    puts("combine fixture/receipts: exact bits, bit-flip failure, immutable inputs, padding sentinel, roundtrip and overwrite/trailing-data refusal; no GPU");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
