// QSA low-level CPU component mocks adapted from portable_native_hc_test_backend.hpp.
#include "../kernels/native_qsa_norm_rope_fixture.hpp"
#include "strata_vk_native_qsa_norm_rope.hpp"
namespace { bool fail_qsa_norm=false;size_t qsa_norm_calls=0;std::vector<float>qsa_last_raw;
#ifdef STRATA_MODEL_ROWS_FIXTURE
bool owned_fail_indexer_create=false;size_t owned_norm_live=0,owned_attn_q_seen=0,owned_attn_k_seen=0;
#endif
}
namespace strata::core::portable_detail {
struct StrataVkNativeQsaNormRope{uint64_t calls=0;};
StrataVkNativeQsaNormRope* strata_vk_native_qsa_norm_rope_create(ggml_backend_t,const char*path,size_t budget){check(budget>=512,"QSA component metadata budget");std::ofstream(path)<<"mock, not shader evidence\n";
#ifdef STRATA_MODEL_ROWS_FIXTURE
 ++owned_norm_live;
#endif
 return new StrataVkNativeQsaNormRope;}
void strata_vk_native_qsa_norm_rope_destroy(StrataVkNativeQsaNormRope*p)noexcept{
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(p)--owned_norm_live;
#endif
 delete p;}
size_t strata_vk_native_qsa_norm_rope_bytes(const StrataVkNativeQsaNormRope*){return 512;}
uint64_t strata_vk_native_qsa_norm_rope_calls(const StrataVkNativeQsaNormRope*p)noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_qsa_norm_rope_run(StrataVkNativeQsaNormRope*p,ggml_tensor*const t[6],uint32_t cols,uint32_t rows,uint32_t stride,uint32_t theta){
 check(t[0]->flags&GGML_TENSOR_FLAG_OUTPUT,"actual QSA projection lifetime missing OUTPUT");
 native_qsa_nr::Fixture f;f.cols=cols;f.rows=rows;f.stride=stride;f.theta=theta;f.x=get(t[0]);f.gamma=get(t[1]);f.pos.resize(rows);ggml_backend_tensor_get(t[2],f.pos.data(),0,rows*4);native_qsa_nr::validate(f);qsa_last_raw=f.x;
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(f.gamma[0]==1.25f){check(cols==128&&rows==4,"owned QSA query norm binding");++owned_qnorm_seen;}
 if(f.gamma[0]==1.5f){check(cols==256&&rows==24,"owned attention query norm binding");++owned_attn_q_seen;}
 if(f.gamma[0]==.5f){check(cols==256&&rows==2,"owned attention key norm binding");++owned_attn_k_seen;}
#endif
 ++p->calls;++qsa_norm_calls;if(fail_qsa_norm)return 1;
 // Independent host primitive provider, not Vulkan/CUDA numerical evidence.
 std::vector<float>d(rows*130);
 for(size_t r=0;r<rows;++r){std::array<float,256>a{};for(size_t c=0;c<cols;++c)a[c]=f.x[r*stride+c]*f.x[r*stride+c];for(int off:{16,8,4,2,1}){auto b=a;for(size_t c=0;c<256;++c)a[c]=b[c]+b[c^off];}std::array<float,32>b{};for(size_t c=0;c<8;++c)b[c]=a[c*32];for(int off:{16,8,4,2,1}){auto old=b;for(size_t c=0;c<32;++c)b[c]=old[c]+old[c^off];}d[r*130]=b[0]/float(cols)+1e-6f;d[r*130+1]=1.f/std::sqrt(d[r*130]);
  for(size_t c=0;c<32;++c){float freq=::powf(native_qsa_nr::value(theta),float(c)),angle=float(f.pos[r])*freq;d[r*130+2+c]=freq;d[r*130+34+c]=angle;d[r*130+66+c]=::cosf(angle);d[r*130+98+c]=::sinf(angle);}}
 auto out=native_qsa_nr::control(f,d);put(t[3],out.normalized);put(t[4],out.rotated);put(t[5],d);return 0;
}
}
#include "../kernels/native_qsa_indexer_fixture.hpp"
#include "strata_vk_native_qsa_indexer.hpp"
namespace {bool fail_qsa_indexer=false;std::vector<uint32_t>last_indexer_candidate;bool observe_qsa_rows=false;size_t qsa_row_bf_calls=0,qsa_row_idx_calls=0,qsa_row_score_calls=0,qsa_row_select_calls=0,qsa_row_attn_calls=0;std::string qsa_rows_fail;int64_t qsa_rows_fail_position=-1;std::vector<uint32_t>qsa_row_candidate_3,qsa_row_before_4;}
namespace strata::core::portable_detail {
struct StrataVkNativeQsaIndexer{uint64_t calls=0;};
StrataVkNativeQsaIndexer* strata_vk_native_qsa_indexer_create(ggml_backend_t,const char*path,size_t budget){check(budget>=512,"QSA indexer metadata budget");std::ofstream(path)<<"mock, not GPU proof\n";
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(owned_fail_indexer_create)throw std::runtime_error("injected QSA indexer factory");
#endif
 return new StrataVkNativeQsaIndexer;}
void strata_vk_native_qsa_indexer_destroy(StrataVkNativeQsaIndexer*p)noexcept{delete p;}
size_t strata_vk_native_qsa_indexer_bytes(const StrataVkNativeQsaIndexer*){return 512;}
uint64_t strata_vk_native_qsa_indexer_calls(const StrataVkNativeQsaIndexer*p)noexcept{return p?p->calls:0;}
uint32_t strata_vk_native_qsa_indexer_run(StrataVkNativeQsaIndexer*p,ggml_tensor*const t[5],uint32_t cells,int32_t pos,int32_t base,uint32_t theta){
 if(observe_qsa_rows)++qsa_row_idx_calls;check(t[0]->flags&GGML_TENSOR_FLAG_OUTPUT,"actual raw indexer projection lifetime missing OUTPUT");
 native_qsa_idx::Fixture f;f.cells=cells;f.base=base;f.first=pos;f.steps=1;f.theta=theta;f.raw=get(t[0]);f.gamma=get(t[1]);f.before.resize(qsa_idx_words(cells));ggml_backend_tensor_get(t[2],f.before.data(),0,f.before.size()*4);native_qsa_idx::validate(f);
#ifdef STRATA_MODEL_ROWS_FIXTURE
 if(f.gamma[0]==.75f)++owned_knorm_seen;
#endif
 ++p->calls;if(fail_qsa_indexer)return 1;

 std::vector<float>d(386);native_qsa_nr::Fixture n;n.cols=128;n.rows=1;n.stride=128;n.gamma=f.gamma;n.x.resize(128);n.pos={pos==0?0:base+4*(pos/4)};
 for(size_t i=0;i<128;++i){d[i]=native_qsa_idx::rounded(f.raw[i]);float sum=pos==0?d[i]:native_qsa_idx::value(f.before[i]);for(int j=1;j<4;++j)sum=sum+(pos==0||j==3?d[i]:native_qsa_idx::value(f.before[j*128+i]));n.x[i]=std::fma(.25f,sum,0.f);if(pos==0||pos%4==3)d[128+i]=n.x[i];}
 if(pos==0||pos%4==3){std::vector<float>nd(130);auto z=native_qsa_nr::control(n,nd);nd[0]=z.arg[0];nd[1]=1.f/std::sqrt(nd[0]);for(size_t i=0;i<32;++i){float freq=::powf(native_qsa_idx::value(theta),float(i)),angle=float(n.pos[0])*freq;nd[2+i]=freq;nd[34+i]=angle;nd[66+i]=::cosf(angle);nd[98+i]=::sinf(angle);}std::copy(nd.begin(),nd.end(),d.begin()+256);}
 auto out=native_qsa_idx::control(f,0,f.before,d);if(observe_qsa_rows&&pos==3)qsa_row_candidate_3=out;if(observe_qsa_rows&&pos==4)qsa_row_before_4=f.before;last_indexer_candidate=out;ggml_backend_tensor_set(t[3],out.data(),0,out.size()*4);put(t[4],d);if(observe_qsa_rows&&qsa_rows_fail=="indexer"&&pos==qsa_rows_fail_position)throw std::runtime_error("indexer write then fail");return 0;
}
}
#include "../kernels/native_qsa_select_fixture.hpp"
#include "strata_vk_native_qsa_select.hpp"
namespace {bool fail_qsa_score=false,fail_qsa_select=false;}
namespace strata::core::portable_detail {
struct StrataVkNativeQsaSelect{uint64_t scores=0,selections=0;};
StrataVkNativeQsaSelect* strata_vk_native_qsa_select_create(ggml_backend_t,const char*s,const char*t,size_t budget){check(budget>=1024,"QSA score/selection metadata budget");std::ofstream(s)<<"mock, not GPU proof\n";std::ofstream(t)<<"mock, not GPU proof\n";return new StrataVkNativeQsaSelect;}
void strata_vk_native_qsa_select_destroy(StrataVkNativeQsaSelect*p)noexcept{delete p;}
size_t strata_vk_native_qsa_select_bytes(const StrataVkNativeQsaSelect*){return 1024;}
uint64_t strata_vk_native_qsa_score_calls(const StrataVkNativeQsaSelect*p)noexcept{return p?p->scores:0;}
uint64_t strata_vk_native_qsa_select_calls(const StrataVkNativeQsaSelect*p)noexcept{return p?p->selections:0;}
uint32_t strata_vk_native_qsa_scores_run(StrataVkNativeQsaSelect*p,ggml_tensor*const t[5],uint32_t cells){
 if(observe_qsa_rows)++qsa_row_score_calls;check((t[2]->flags&GGML_TENSOR_FLAG_OUTPUT)||(t[2]->op==GGML_OP_NONE&&t[2]->buffer&&t[2]->data),"actual projected query lifetime missing");native_qsa_ss::Fixture f;f.cells=cells;ggml_backend_tensor_get(t[3],f.step.data(),0,16);f.pooled=get(t[0]);f.dead=get(t[1]);f.query=get(t[2]);f.seed.assign(qsa_select_blocks(cells),0);
 check(last_indexer_candidate.size()==qsa_idx_words(cells),"missing actual indexer candidate producer");check(std::memcmp(f.pooled.data(),last_indexer_candidate.data()+qsa_idx_pool,(last_indexer_candidate.size()-qsa_idx_pool-1)*4)==0,"score used old pooled state");check(std::memcmp(f.dead.data(),last_indexer_candidate.data()+384,512)==0,"score lost actual retained dead/spare");++p->scores;if(fail_qsa_score)return 1;auto out=native_qsa_ss::score_control(f);ggml_backend_tensor_set(t[4],out.data(),0,out.size()*4);if(observe_qsa_rows&&qsa_rows_fail=="score"&&f.step[0]==qsa_rows_fail_position)throw std::runtime_error("score write then fail");return 0;
}
uint32_t strata_vk_native_qsa_select_run(StrataVkNativeQsaSelect*p,ggml_tensor*const t[3],uint32_t cells){
 if(observe_qsa_rows)++qsa_row_select_calls;native_qsa_ss::Fixture f;f.cells=cells;f.compute=0;ggml_backend_tensor_get(t[1],f.step.data(),0,16);f.pooled.assign(qsa_select_blocks(cells)*128,0);f.dead.assign(128,0);f.query.assign(512,0);f.seed.resize(qsa_select_blocks(cells));ggml_backend_tensor_get(t[0],f.seed.data(),0,f.seed.size()*4);++p->selections;if(fail_qsa_select)return 1;auto out=native_qsa_ss::select_control(f,f.seed);ggml_backend_tensor_set(t[2],out.data(),0,out.size()*4);if(observe_qsa_rows&&qsa_rows_fail=="select"&&f.step[0]==qsa_rows_fail_position)throw std::runtime_error("select write then fail");return 0;
}
}


#include "../kernels/native_qsa_attention_fixture.hpp"
#include "strata_vk_native_qsa_attention.hpp"
namespace {bool fail_qsa_attention=false;std::vector<float>actual_native_gated;}
namespace strata::core::portable_detail {
struct StrataVkNativeQsaAttention{std::array<uint64_t,3>calls{};};
StrataVkNativeQsaAttention* strata_vk_native_qsa_attention_create(ggml_backend_t,const char*a,const char*b,const char*c,size_t budget){check(budget>=3072,"QSA attention metadata budget");for(auto p:{a,b,c})std::ofstream(p)<<"mock, not GPU proof\n";return new StrataVkNativeQsaAttention;}
void strata_vk_native_qsa_attention_destroy(StrataVkNativeQsaAttention*p)noexcept{delete p;}
size_t strata_vk_native_qsa_attention_bytes(const StrataVkNativeQsaAttention*){return 3072;}
uint64_t strata_vk_native_qsa_attention_calls(const StrataVkNativeQsaAttention*p,unsigned i)noexcept{return p&&i<3?p->calls[i]:0;}
uint32_t strata_vk_native_qsa_attention_run(StrataVkNativeQsaAttention*p,ggml_tensor*const t[13],uint32_t cells){if(observe_qsa_rows)++qsa_row_attn_calls;native_qsa_attn::Fixture f;f.cells=cells;auto n=native_qsa_attn::counts(cells);for(size_t i=0;i<7;++i){f.in[i].resize(n[i]);check(ggml_nbytes(t[i])==n[i]*4,"actual attention typed transport span");ggml_backend_tensor_get(t[i],f.in[i].data(),0,n[i]*4);}++p->calls[0];if(fail_qsa_attention)return 0x10001;auto out=native_qsa_attn::control(f);++p->calls[1];++p->calls[2];for(size_t i=0;i<6;++i)ggml_backend_tensor_set(t[i+7],out.out[i].data(),0,out.out[i].size()*4);actual_native_gated.resize(6144);std::memcpy(actual_native_gated.data(),out.out[5].data(),24576);if(observe_qsa_rows&&qsa_rows_fail=="attention"&&int32_t(f.in[5][0])==qsa_rows_fail_position)throw std::runtime_error("attention write then fail");return 0;}
}
