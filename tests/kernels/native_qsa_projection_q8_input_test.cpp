#include "native_qsa_projection_q8_fixture.hpp"
using namespace native_projection_q8;
int main(int argc,char**argv){try{
    if(argc==4 && std::string(argv[1])=="--synthetic"){write(argv[3],synthetic(std::stoul(argv[2])));return 0;}
    if(argc==3 && std::string(argv[1])=="--validate"){(void)read(argv[2]);return 0;}
    if(argc==4 && std::string(argv[1])=="--contract"){contract(read(argv[2]),argv[3]);return 0;}
    if(argc==4 && std::string(argv[1])=="--check"){auto x=read(argv[2]);auto q=load(argv[3]);pinned(q);return 0;}
    if(argc==4 && std::string(argv[1])=="--host-control"){auto x=read(argv[2]);save(argv[3],quantize(x));return 0;}
    if(argc==4 && std::string(argv[1])=="--trace"){trace(read(argv[2]),argv[3]);return 0;}
    need(argc==1,"projection q8: control arguments");
    for(unsigned i=0;i<5;++i){auto x=synthetic(i);auto q=quantize(x);pinned(q);q[2*36+5]^=1;bool rejected=false;try{pinned(q);}catch(...){rejected=true;}need(rejected,"projection q8: altered pin accepted");}
    for(auto bad:{std::vector<float>(elements-1),std::vector<float>(elements+1)}){bool rejected=false;try{validate(bad);}catch(...){rejected=true;}need(rejected,"projection q8: shape accepted");}
    for(float value:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::denorm_min()}){auto x=synthetic(0);x[300]=value;bool rejected=false;try{validate(x);}catch(...){rejected=true;}need(rejected,"projection q8: unsupported input accepted");}
    {auto x=synthetic(0);x[300]=std::numeric_limits<float>::max();x[301]=x[300];bool rejected=false;try{validate(x);}catch(...){rejected=true;}need(rejected,"projection q8: overflowing sum accepted");}
    {auto x=synthetic(0);x[300]=0x1p-126f;x[301]=1e8f;bool rejected=false;try{validate(x);}catch(...){rejected=true;}need(rejected,"projection q8: subnormal ratio accepted");}
    {auto x=synthetic(0);x[64]=126.f;bool rejected=false;try{validate(x);}catch(...){rejected=true;}need(rejected,"projection q8: changed pin input accepted");}
    puts("projection q8 pinned layout, ties, XOR order, half and finite clamp controls pass");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
