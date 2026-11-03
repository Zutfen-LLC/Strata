#include "native_head_hc_fixture.hpp"
#include <cstdio>
int main(int argc,char**argv){try{if(argc==2&&std::string(argv[1])=="--contract"){puts("native_head_hc_fixture_v1 cases=zero,sparse,dense,cancel,tails");return 0;}if(argc==4&&std::string(argv[1])=="--make"){head_hc_fixture::write(argv[3],head_hc_fixture::make(argv[2]));return 0;}if(argc==3&&std::string(argv[1])=="--check"){(void)head_hc_fixture::read(argv[2]);return 0;}throw std::invalid_argument("head fixture: --make CASE FILE or --check FILE");}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
