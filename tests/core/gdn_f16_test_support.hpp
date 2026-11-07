#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <limits>
#include "ggml-backend.h"
#include "ggml-cpu.h"
namespace gdn_f16_test {
inline void need(bool x,const char*m){if(!x)throw std::runtime_error(m);}
// Four early rows plus a distinct last-row, last-block sentinel. The
// halfway product 3 * 1.0009765625 rounds to even half 3.00390625.
inline void fill(std::vector<uint8_t>& role,size_t k,int late_code){
 size_t stride=k/32*34;
 auto set=[&](size_t row,uint16_t scale,int a,int b){auto*p=role.data()+row*stride;p[0]=scale&255;p[1]=scale>>8;p[2]=uint8_t(a);p[3]=uint8_t(b);};
 set(0,0x3c00,-2,1);set(1,0x3800,-128,3);set(2,0x0001,-128,1);set(3,0x3c01,3,0);
 auto*last=role.data()+role.size()-34;last[0]=0;last[1]=0x38;last[33]=uint8_t(late_code);
}
inline std::vector<float> input(size_t k,size_t rows,int phase){
 std::vector<float> x(k*rows);
 if(phase==0){x[0]=1.0f+0x1p-11f;x[1]=2; if(rows==2){x[k]=-3;x[k+1]=4;}}
 if(phase==1){x[0]=5;x[1]=-2;}
 if(phase==2){x[0]=-1;x[1]=6;if(rows==2){x[k]=2;x[k+1]=-4;}}
 x[k-1]=phase==0?2:phase==1?4:-5;if(rows==2)x[2*k-1]=phase==0?-3:6;
 return x;
}
inline void check(ggml_tensor* t,size_t n,size_t rows,int phase,size_t role){
 // Independent expected products for scales 0.5 and codes -7, +11, +5.
 const float late[3][3]={{-7,10.5f,-14},{11,-16.5f,22},{5,-7.5f,10}};
 const float last[3][2]={{17.5f,-21},{-27.5f,33},{-12.5f,15}};
 std::vector<float> v(n*rows);ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(float));
 float a[2]{},b[2]{};if(phase==0){a[0]=1;b[0]=2;a[1]=-3;b[1]=4;}if(phase==1){a[0]=5;b[0]=-2;}if(phase==2){a[0]=-1;b[0]=6;a[1]=2;b[1]=-4;}
 for(size_t r=0;r<rows;++r){
  need(v[r*n]==-2*a[r]+b[r],"F16 expected signed row");
  need(v[r*n+1]==-64*a[r]+1.5f*b[r],"F16 expected negative code");
  need(v[r*n+2]==-0x1p-17f*a[r]+0x1p-24f*b[r],"F16 expected subnormal scale");
  need(v[r*n+3]==3.00390625f*a[r],"F16 expected weight tie");
  for(size_t j=4;j+1<n;++j)need(v[r*n+j]==0,"F16 zero row");
  need(role<3 && v[r*n+n-1]==(phase==2?last[role][r]:late[role][phase==1?2:r]),"F16 late row/block sentinel");
 }
}
template<class Owner,class Output> void exercise(ggml_backend_t backend,Owner& owner,size_t k,const std::vector<size_t>& columns,Output output,size_t first_role=0){
 need(owner.method()==strata::core::portable_detail::PortableGdnProjectionMethod::native_f16,"F16 method");
 for(int phase=0;phase<3;++phase){size_t rows=phase==1?1:2;auto x=input(k,rows,phase);auto original=x;owner.run(x,rows);need(x==original,"F16 input modified");
  for(size_t i=0;i<columns.size();++i)check(output(i,rows),columns[i],rows,phase,first_role+i);
 }
 need(owner.graph_attempts()==3,"F16 graph count");
 auto bad=input(k,2,0);bad[k]=std::numeric_limits<float>::infinity();bool failed=false;try{owner.run(bad,2);}catch(const std::exception&){failed=true;}need(failed&&!owner.valid(),"F16 nonfinite refusal");
 failed=false;try{output(0,2);}catch(const std::exception&){failed=true;}need(failed,"F16 stale output");
 bad[k]=65520.0f;failed=false;try{owner.run(bad,2);}catch(const std::exception&){failed=true;}need(failed&&!owner.valid(),"F16 overflow refusal");
 failed=false;try{owner.run(bad,3);}catch(const std::exception&){failed=true;}need(failed&&!owner.valid(),"F16 shape refusal");
 auto good=input(k,1,1);owner.run(good,1);need(owner.valid(),"F16 recovery");for(size_t i=0;i<columns.size();++i)check(output(i,1),columns[i],1,1,first_role+i);
 // CPU calls this after a computed node and returns GGML_STATUS_ABORTED.
 // No assertion is made about writes performed before the abort.
 struct AbortScope {ggml_backend_t b;size_t calls=0;explicit AbortScope(ggml_backend_t backend):b(backend){ggml_backend_cpu_set_abort_callback(b,[](void* p){++*static_cast<size_t*>(p);return true;},&calls);}~AbortScope(){ggml_backend_cpu_set_abort_callback(b,nullptr,nullptr);}};
 const size_t before=owner.graph_attempts();
 {AbortScope abort(backend);failed=false;try{owner.run(input(k,2,2),2);}catch(const std::exception& e){failed=std::string(e.what()).find("graph failure")!=std::string::npos;}
  need(failed&&abort.calls>0&&owner.graph_attempts()==before+1&&!owner.valid(),"F16 graph abort invalidation");
  failed=false;try{output(0,1);}catch(const std::exception&){failed=true;}need(failed,"F16 graph abort stale output");}
 owner.run(good,1);need(owner.valid(),"F16 graph abort recovery");for(size_t i=0;i<columns.size();++i)check(output(i,1),columns[i],1,1,first_role+i);
}
}
