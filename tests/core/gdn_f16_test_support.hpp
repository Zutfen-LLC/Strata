#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>
#include "ggml-backend.h"
namespace gdn_f16_test {
inline void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
// Four nonzero rows. Each uses the first Q8_0 block. The last case is a
// halfway product: 3 * 1.0009765625 rounds to the even half 3.00390625.
inline void fill(std::vector<uint8_t>& role,size_t k){
 size_t stride=k/32*34;
 auto set=[&](size_t row,uint16_t scale,int a,int b){auto*p=role.data()+row*stride;p[0]=scale&255;p[1]=scale>>8;p[2]=uint8_t(a);p[3]=uint8_t(b);};
 set(0,0x3c00,-2,1);set(1,0x3800,-128,3);set(2,0x0001,-128,1);set(3,0x3c01,3,0);
}
inline std::vector<float> input(size_t k,size_t rows,int phase){
 std::vector<float> x(k*rows);
 if(phase==0){x[0]=1.0f+0x1p-11f;x[1]=2; if(rows==2){x[k]=-3;x[k+1]=4;}}
 if(phase==1){x[0]=5;x[1]=-2;}
 if(phase==2){x[0]=-1;x[1]=6;if(rows==2){x[k]=2;x[k+1]=-4;}}
 return x;
}
inline void check(ggml_tensor* t,size_t n,size_t rows,int phase){
 std::vector<float> v(n*rows);ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(float));
 float a[2]{},b[2]{};if(phase==0){a[0]=1;b[0]=2;a[1]=-3;b[1]=4;}if(phase==1){a[0]=5;b[0]=-2;}if(phase==2){a[0]=-1;b[0]=6;a[1]=2;b[1]=-4;}
 for(size_t r=0;r<rows;++r){
  need(v[r*n]==-2*a[r]+b[r],"F16 expected signed row");
  need(v[r*n+1]==-64*a[r]+1.5f*b[r],"F16 expected negative code");
  need(v[r*n+2]==-0x1p-17f*a[r]+0x1p-24f*b[r],"F16 expected subnormal scale");
  need(v[r*n+3]==3.00390625f*a[r],"F16 expected weight tie");
  for(size_t j=4;j<n;++j)need(v[r*n+j]==0,"F16 zero row");
 }
}
template<class Owner,class Output> void exercise(Owner& owner,size_t k,const std::vector<size_t>& columns,Output output){
 need(owner.method()==strata::core::portable_detail::PortableGdnProjectionMethod::native_f16,"F16 method");
 for(int phase=0;phase<3;++phase){size_t rows=phase==1?1:2;auto x=input(k,rows,phase);auto original=x;owner.run(x,rows);need(x==original,"F16 input modified");
  for(size_t i=0;i<columns.size();++i)check(output(i,rows),columns[i],rows,phase);
 }
 need(owner.graph_attempts()==3,"F16 graph count");
 auto bad=input(k,2,0);bad[k]=std::numeric_limits<float>::infinity();bool failed=false;try{owner.run(bad,2);}catch(const std::exception&){failed=true;}need(failed&&!owner.valid(),"F16 nonfinite refusal");
 failed=false;try{output(0,2);}catch(const std::exception&){failed=true;}need(failed,"F16 stale output");
 bad[k]=65520.0f;failed=false;try{owner.run(bad,2);}catch(const std::exception&){failed=true;}need(failed&&!owner.valid(),"F16 overflow refusal");
 failed=false;try{owner.run(bad,3);}catch(const std::exception&){failed=true;}need(failed&&!owner.valid(),"F16 shape refusal");
 auto good=input(k,1,1);owner.run(good,1);need(owner.valid(),"F16 recovery");for(size_t i=0;i<columns.size();++i)check(output(i,1),columns[i],1,1);
}
}
