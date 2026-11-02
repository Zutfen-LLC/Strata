#include "native_q8_wo_fixture.hpp"
using namespace native_q8_wo_fixture;
int main(int argc,char**argv){try{
 if(argc==5 && std::string(argv[1])=="--validate"){(void)read_fixture(argv[2],argv[3],std::stoull(argv[4]));return 0;}
 if(argc==6 && std::string(argv[1])=="--contract"){auto f=read_fixture(argv[2],argv[3],std::stoull(argv[4]));contract(f,argv[5]);return 0;}
 if(argc==4 && std::string(argv[1])=="--synthetic"){
  size_t rows=std::stoull(argv[2]);auto f=synthetic(rows);literal_controls(f);validate(f);std::filesystem::path dir=argv[3];need(std::filesystem::create_directory(dir),"WO: output exists");save(dir/"weights.bin",f.weights);save(dir/"qx.bin",f.input);return 0;}
 need(argc==1,"WO: invalid input arguments");auto f=synthetic(9);literal_controls(f);validate(f);
 {auto g=f;for(size_t b=0;b<blocks;++b)set16(g.input,b*36+2,0x0400);for(size_t row=0;row<f.rows;++row)need(bits(reference(f,row).value)==bits(reference(g,row).value),"WO: unused original-sum metadata affected result");}
 {Q8WoSpan w{(void*)1,100,100},x{(void*)1,200,40},y{(void*)2,120,40};q8_wo_require_nonoverlap(w,x,y);q8_wo_require_budget(100,100);
  bool fail=false;try{q8_wo_require_nonoverlap(w,Q8WoSpan{(void*)1,199,2},y);}catch(...){fail=true;}need(fail,"WO: accepted overlap");
  fail=false;try{q8_wo_require_budget(101,100);}catch(...){fail=true;}need(fail,"WO: accepted budget");q8_wo_require_nonoverlap(w,x,y);q8_wo_require_budget(100,100);}
 {bool fail=false;try{q8_wo_require_layout(9,9*q8_wo_row_bytes-4,q8_wo_input_bytes,36);}catch(...){fail=true;}need(fail,"WO: accepted short weights");}
 for(auto bad:{size_t(0),size_t(2561)}){bool fail=false;try{q8_wo_require_layout(bad,bad*q8_wo_row_bytes,q8_wo_input_bytes,bad*4);}catch(...){fail=true;}need(fail,"WO: accepted invalid rows");}
 {size_t checked_fragments=0;for(size_t b=0;b<blocks;++b)for(size_t q=0;q<8;++q){int dot=0;for(size_t j=0;j<4;++j){size_t lane=q*4+j;int x=s8(f.input[b*36+4+lane]);int w=s8(f.weights[8*q8_wo_row_bytes+b*34+2+lane]);need(x!=0&&w!=0,"WO: zero in coverage lane");dot+=x*w;}need(dot>0,"WO: inactive DP4A fragment");++checked_fragments;}need(checked_fragments==blocks*8,"WO: incomplete fragment coverage");}
 for(auto h:{uint16_t(0x0001),uint16_t(0x8001),uint16_t(0x7c00),uint16_t(0x7e00)}){auto g=f;set16(g.weights,0,h);bool fail=false;try{validate(g);}catch(...){fail=true;}need(fail,"WO: accepted unsupported scale");}
 for(auto h:{uint16_t(0x0001),uint16_t(0x8001)}){auto g=f;set16(g.input,0,h);bool fail=false;try{validate(g);}catch(...){fail=true;}need(fail,"WO: accepted subnormal input half scale");}
 puts("WO independent literal, native partition, finite-domain and layout controls pass");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
