#include "strata/core/remote_route_counts.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3)return 1;
    using strata::core::remote_route_counts_supported;
    if(!remote_route_counts_supported(false,false,false,5,true,true,true) ||
       !remote_route_counts_supported(true,true,true,0,false,false,false) ||
       remote_route_counts_supported(true,true,true,1,false,false,false) ||
       remote_route_counts_supported(true,false,true,0,false,false,false) ||
       remote_route_counts_supported(true,true,false,0,false,false,false) ||
       remote_route_counts_supported(true,true,true,0,true,false,false) ||
       remote_route_counts_supported(true,true,true,0,false,true,false) ||
       remote_route_counts_supported(true,true,true,0,false,false,true))return 5;
    setenv("STRATA_REMOTE_ROUTE_COUNTS",argv[1],1);
    if(argc==3) {
        strata::core::RemoteRouteCounts counts;counts.begin(2,512);
        if(std::string(argv[2])=="threads") {
            std::vector<std::thread> threads;
            for(int t=0;t<4;++t)threads.emplace_back([&]{const int32_t ids[]={1,2};for(int i=0;i<1000;++i)counts.record(0,ids,1,2);});
            for(auto& t:threads)t.join();counts.finish(213,"stop");
            std::ifstream in(argv[1]);std::string header;std::getline(in,header);
            return header.find("complete=1 calls=4000")!=std::string::npos ? 0 : 4;
        }
        for(int i=0;i<65537;++i) {const int32_t ids[]={i/512,i%512};counts.record(0,ids,1,2);}
        counts.finish(213,"stop");
        std::ifstream in(argv[1]);std::string header;std::getline(in,header);
        return header.find("complete=0 calls=65536")!=std::string::npos ? 0 : 3;
    }
    strata::core::RemoteRouteCounts counts;counts.begin(2,16);
    const int32_t a[]={1,2,2,3},b[]={1,2};
    counts.record(0,a,2,2);counts.record(0,a,2,2);counts.record(1,b,1,2);counts.finish(213,"stop");
    std::ifstream in(argv[1]);std::string text((std::istreambuf_iterator<char>(in)),{});
    if(text.find("complete=1 calls=3 generated=213 finish=stop patterns=2 layers=2 experts=16")==std::string::npos ||
       text.find("0 2 2 2 1 2 2 3\n")==std::string::npos || text.find("1 1 2 1 1 2\n")==std::string::npos)return 2;
    counts.begin(2,16);counts.record(0,a,2,2);counts.finish(1,"stop");
    return 0;
}
