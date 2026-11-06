#include "../../src/core/portable_stdio.hpp"
#include "../../src/core/portable_expert_timer.hpp"
#include <fstream>
#include <iostream>
using namespace strata::core::stdio_detail;
struct FakeClock {
    using time_point = std::chrono::time_point<FakeClock, std::chrono::milliseconds>;
    static inline int reads = 0;
    static inline int ticks = 0;
    static time_point now() { ++reads; return time_point(std::chrono::milliseconds(ticks)); }
};
int main() {
    double elapsed = 0; uint64_t calls = 0;
    {
        strata::core::portable_detail::ScopedExpertTimer<FakeClock> timer(false, elapsed, calls);
        FakeClock::ticks = 3;
    }
    if(FakeClock::reads || elapsed || calls) return 4;
    {
        strata::core::portable_detail::ScopedExpertTimer<FakeClock> timer(true, elapsed, calls);
        FakeClock::ticks += 7;
    }
    try {
        strata::core::portable_detail::ScopedExpertTimer<FakeClock> timer(true, elapsed, calls);
        FakeClock::ticks += 5;
        throw std::runtime_error("fake executor failure");
    } catch(const std::runtime_error&) {}
    if(elapsed != 12 || calls != 2 || FakeClock::reads != 4) return 5;
    auto valid = parse("GEN 3 0,12,99", 3, 100);
    if (valid.max_new != 3 || valid.tokens != std::vector<int32_t>({0, 12, 99}))
        return 1;
    for (const auto &bad : std::vector<std::string>{"", "QUIT", "GEN 4 0", "GEN 1 ", "GEN -1 0", "GEN 0 100",
                                                    "GEN 0 1,", "GEN 0 ,1", "GEN 0 1 2", "GEN 0 1,2,3,4", "GEN 0 1\r",
                                                    "GEN 0 99999999999999999999999", std::string(4097, '0')}) {
        try {
            parse(bad, 3, 100);
            return 2;
        } catch (const std::runtime_error &) {
        }
    }
    if(request_limit(256)!=4096 || request_limit(4096)!=45064) return 6;
    std::string long_request="GEN 4096 ";
    for(size_t i=0;i<4096;++i) { if(i) long_request+=',';long_request+="2147483647"; }
    if(long_request.size()!=request_limit(4096)) return 7;
    LineFramer long_frame(4096);
    for(char c:long_request) if(long_frame.append(c)) return 8;
    if(!long_frame.append('\n')) return 9;
    auto framed=long_frame.take();
    auto long_valid=parse(framed,4096,size_t(INT32_MAX)+1);
    if(long_valid.max_new!=4096 || long_valid.tokens.size()!=4096 || long_valid.tokens.back()!=INT32_MAX) return 10;
    std::string realistic="GEN 0 ";
    for(size_t i=0;i<4096;++i) { if(i) realistic+=',';realistic+="248046"; }
    if(parse(realistic,4096,248047).tokens.size()!=4096) return 11;
    auto rejects=[](const std::string &s,size_t context,size_t vocab){try{(void)parse(s,context,vocab);return false;}catch(const std::runtime_error&){return true;}};
    if(!rejects(long_request+"0",4096,size_t(INT32_MAX)+1) ||
       !rejects(realistic+",0",4096,248047) || !rejects("GEN 0 2147483648",4096,size_t(INT32_MAX)+2) ||
       !rejects("GEN 0 248047",4096,248047) || !rejects("GEN 0 123 456",4096,248047) ||
       !rejects("GEN 0 123,,456",4096,248047) || !rejects(std::string(4097,'0'),256,100)) return 12;
    bool refused=false;try{LineFramer f(4096);for(size_t i=0;i<=request_limit(4096);++i) f.append('0');}catch(const std::runtime_error&){refused=true;}if(!refused)return 13;
    refused=false;try{LineFramer f(4096);for(char c:std::string("GEN 0 1"))f.append(c);(void)f.eof();}catch(const std::runtime_error&){refused=true;}if(!refused)return 14;
    {LineFramer f(256);for(size_t i=0;i<4096;++i)if(f.append('0'))return 16;refused=false;try{f.append('0');}catch(const std::runtime_error&){refused=true;}if(!refused)return 17;}
    // A caller must parse completely before reset/mutation, even when framing succeeds.
    int resets=0;try{LineFramer f(4096);for(char c:std::string("GEN 0 1,\n"))if(f.append(c)){(void)parse(f.take(),4096,100);++resets;}}catch(const std::runtime_error&){}if(resets)return 15;
    const auto path = std::filesystem::temp_directory_path() / "portable-stdio-source-test";
    {
        std::ofstream f(path);
        f << "one";
    }
    Sources source({path.string()});
    source.check();
    {
        std::ofstream f(path);
        f << "different";
    }
    try {
        source.check();
        return 3;
    } catch (const std::runtime_error &) {
    }
    std::filesystem::remove(path);
    std::cout << "bounded preflight and immutable source guard passed\n";
}
