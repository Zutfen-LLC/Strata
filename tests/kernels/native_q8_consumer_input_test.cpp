#include "native_q8_consumer_fixture.hpp"
#include "native_q8_gu_fixture.hpp"
#include <chrono>
#include <filesystem>
using namespace native_q8_fixture;
namespace c=native_q8_fixture::consumer;
namespace {
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}need(rejected,"consumer accepted malformed input");}
struct Temporary {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("native-q8-consumer-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temporary(){need(std::filesystem::create_directory(path),"temporary directory creation failed");}
    ~Temporary(){std::error_code error;std::filesystem::remove_all(path,error);}
};
}
int main(){try {
    // Synthetic parser fixtures, NOT native/Vulkan execution results.
    auto native=blank(832);native.input.assign(936,0);native.hidden.assign(3328,0);native.separate.assign(936,0);
    std::copy(native.separate.begin()+4*36,native.separate.begin()+18*36,native.fused.begin()+4*36);
    native.input[3*36+1]=0x3c;native.input[3*36+4]=127;
    std::fill(native.input.begin()+184,native.input.begin()+216,127);
    auto observation=native;std::fill(observation.input.begin()+184,observation.input.begin()+216,0);
    c::validate_pair(native,observation);
    auto bad=observation;bad.input[216]=1;rejects([&]{c::validate_pair(native,bad);});
    bad=observation;bad.input[180]=1;rejects([&]{c::validate_pair(native,bad);});
    bad=observation;bad.hidden[0]=1;rejects([&]{c::validate_pair(native,bad);});
    bad=observation;bad.empty[0]=0;rejects([&]{c::validate_pair(native,bad);});
    bad=observation;bad.input.pop_back();rejects([&]{c::validate_pair(native,bad);});
    auto zero=native;zero.input[3*36+1]=zero.input[3*36+4]=0;
    bad=zero;std::fill(bad.input.begin()+184,bad.input.begin()+216,0);
    rejects([&]{c::validate_pair(zero,bad);});
    const auto weights=c::weights();need(weights.size()==26112,"finite expert byte count");
    for(size_t i=0;i<weights.size();i+=34){const uint16_t d=uint16_t(weights[i])|(uint16_t(weights[i+1])<<8);
        need(d==0x1000 || d==0x8000,"unexpected synthetic finite scale");
        for(size_t j=2;j<34;++j)need(weights[i+j]==1 || weights[i+j]==255,"unexpected synthetic weight code");}
    Temporary tmp;const auto a=(tmp.path/"native").string(),b=(tmp.path/"observation").string();
    write_result(a,native);write_observation_result(b,observation,true);
    c::validate_pair(c::read_capture(a,false),c::read_capture(b,true));
    rejects([&]{c::read_capture(a,true);});rejects([&]{c::read_capture(b,false);});
    write_observation_result(b,observation,false);rejects([&]{c::read_capture(b,true);});
    write_observation_result(b,observation,true);
    {std::fstream file(b,std::ios::binary|std::ios::in|std::ios::out);file.seekp(8);word(file,3328);}
    rejects([&]{c::read_capture(b,true);}); // wrong first section/size
    write_observation_result(b,observation,true);std::filesystem::resize_file(b,std::filesystem::file_size(b)-1);
    rejects([&]{c::read_capture(b,true);});
    write_observation_result(b,observation,true);{std::ofstream file(b,std::ios::binary|std::ios::app);file.put('x');}
    rejects([&]{c::read_capture(b,true);});
    const auto oracle=(tmp.path/"oracle").string(),weight_path=(tmp.path/"weights").string();
    {std::ofstream file(oracle,std::ios::binary);word(file,0x3143384e);
     for(size_t count:{size_t(64),size_t(64),size_t(512)}){word(file,uint32_t(count));for(size_t i=0;i<count;++i)word(file,i%4==0?0x80000000u:0x3f800000u);}}
    auto decoded=gu::read_oracle(oracle);need(decoded.gate[0]==0x80000000u,"oracle zero-sign bits lost");
    {std::fstream file(oracle,std::ios::binary|std::ios::in|std::ios::out);file.seekp(4);word(file,32);}
    rejects([&]{gu::read_oracle(oracle);});
    {std::ofstream file(weight_path,std::ios::binary);file.write(reinterpret_cast<const char*>(weights.data()),weights.size());}
    need(gu::read_weights(weight_path)==weights,"native recipe receipt mismatch");
    {std::ofstream file(weight_path,std::ios::binary|std::ios::app);file.put('x');}
    rejects([&]{gu::read_weights(weight_path);});
    std::puts("consumer input/finite-weight tests passed; no CUDA or numerical consumer proof");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
