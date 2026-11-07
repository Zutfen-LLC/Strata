#pragma once
// Standalone ordinary pending-HC -> causal QSA -> WO. This is pre-FFN
// output, not a post-FFN snapshot and not full-model chunk admission.
// Component ownership is transferred. Their backend, source weight buffers,
// producer and snapshots are borrowed and must remain alive and unmodified
// during synchronous run. Do not retain mutable aliases to transferred owners.
// Callbacks may have external side effects; those cannot be rolled back.
#include "portable_native_hc_composed.hpp"
#include "portable_native_ffn_snapshot.hpp"
#include <variant>
#include <type_traits>
#include "portable_native_qsa_qkv_rows.hpp"
#include "portable_native_qsa_indexer_rows.hpp"
#include "portable_native_qsa_norm_rope.hpp"
#include "portable_native_qsa_indexer.hpp"
#include "portable_native_qsa_select.hpp"
#include "portable_native_qsa_attention.hpp"
#include "portable_native_qsa_wo_rows.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
namespace strata::core::portable_detail {
class PortableNativeQsaComposition {
 using Anchor=std::variant<std::monostate,std::shared_ptr<const PortableGdnSnapshot>,std::shared_ptr<const PortableNativeFfnSnapshot>>;
 static bool anchor_live(const Anchor& a,int layer,uint64_t position){
  return std::visit([&](const auto& p)->bool {using T=std::decay_t<decltype(p)>;if constexpr(std::is_same_v<T,std::monostate>)return false;else return p&&p->live()&&p->layer()+1==layer&&p->position()==position;},a);
 }
 template<class Producer,class Snapshot> static bool anchor_matches(const Anchor& a,const Producer& p,int layer,uint64_t pos){auto v=std::get_if<std::shared_ptr<const Snapshot>>(&a);return v&&*v&&(*v)->valid_for(p,layer,pos);}
 static bool anchored(const Anchor& a){return a.index()!=0;}
 struct Access {bool valid=true;uint64_t first=0,generation=0;size_t rows=0;std::vector<float>output,residual,injection;std::vector<int32_t>ids;};
public:
 class View {
  std::weak_ptr<Access> access_;
  explicit View(const std::shared_ptr<Access>&a):access_(a){}friend class PortableNativeQsaComposition;
  std::shared_ptr<Access> check()const{auto p=access_.lock();if(!p||!p->valid)throw std::runtime_error("QSA composition: stale view");return p;}
 public:
  uint64_t first()const{return check()->first;}uint64_t generation()const{return check()->generation;}size_t rows()const{return check()->rows;}
  std::vector<float> output()const{return check()->output;}std::vector<float> residual()const{return check()->residual;}std::vector<float> injection()const{return check()->injection;}
  std::vector<int32_t> selected(size_t row)const{auto p=check();if(row>=p->rows)throw std::runtime_error("QSA composition: selected row");return {p->ids.begin()+row*2051,p->ids.begin()+(row+1)*2051};}
 };
 using Consumer=std::function<void(const View&)>;
 struct Seed {
  uint64_t position=0;std::vector<uint16_t>keys,values;std::vector<uint32_t>indexer;
 private:
  // Only state() can issue a nonempty provenance anchor. A caller restoring
  // edited payloads is responsible for their correspondence to that prefix.
  Anchor anchor_;
  friend class PortableNativeQsaComposition;
 };
private:
 using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
 std::unique_ptr<PortableNativeHcComposed> hc_;std::unique_ptr<PortableNativeQsaQkvRows> qkv_;std::unique_ptr<PortableNativeQsaIndexerRows> projections_;
 std::unique_ptr<PortableNativeQsaNormRope> norm_;std::unique_ptr<PortableNativeQsaIndexer> indexer_;std::unique_ptr<PortableNativeQsaSelect> select_;
 std::unique_ptr<PortableNativeQsaAttention> attention_;std::unique_ptr<PortableNativeQsaWoRows> wo_;
 uint32_t cells_;int layer_;uint64_t position_=0,generation_=0;bool running_=false,poisoned_=false;
 std::vector<uint16_t>keys_,values_;std::vector<uint32_t>indexer_state_;std::shared_ptr<Access> published_;
 // Holds the last genuine predecessor snapshot, including its shared lifetime
 // guard. This prevents pointer reuse from impersonating a prefix generation.
 Anchor anchor_;
 static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
 static ggml_tensor* row(ggml_context*c,ggml_tensor*t,size_t offset,size_t count){
  need(t&&t->type==GGML_TYPE_F32&&offset<=size_t(ggml_nelements(t))&&count<=size_t(ggml_nelements(t))-offset,"QSA composition: row span");
  auto*v=ggml_view_1d(c,t,int64_t(count),offset*sizeof(float));need(v&&ggml_backend_view_init(v)==GGML_STATUS_SUCCESS,"QSA composition: row view");ggml_set_output(v);return v;
 }
 static void copy_half(ggml_tensor*t,std::vector<uint16_t>&dst,size_t offset){std::vector<float>v(512);need(t&&ggml_nbytes(t)==v.size()*4&&offset<=dst.size()&&v.size()<=dst.size()-offset,"QSA composition: KV row");ggml_backend_tensor_get(t,v.data(),0,v.size()*4);for(size_t i=0;i<v.size();++i){need(std::isfinite(v[i]),"QSA composition: nonfinite KV");auto h=ggml_fp32_to_fp16(v[i]);need(std::isfinite(ggml_fp16_to_fp32(h)),"QSA composition: FP16 KV overflow");dst[offset+i]=h;}}
 void validate_seed(const Seed&s)const{
  need(s.position<=cells_&&s.keys.size()==size_t(cells_)*512&&s.values.size()==s.keys.size()&&s.indexer.size()==qsa_idx_words(cells_),"QSA composition: seed spans/position");
  for(auto*v:{&s.keys,&s.values})for(auto h:*v)need(std::isfinite(ggml_fp16_to_fp32(h)),"QSA composition: nonfinite seed KV");
  for(size_t i=0;i+1<s.indexer.size();++i){float x;std::memcpy(&x,&s.indexer[i],4);need(qsa_wo_normal_or_zero(x),"QSA composition: invalid seed indexer float");if(i<384){float y=ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));need(std::isfinite(y)&&std::memcmp(&x,&y,4)==0,"QSA composition: unrounded seed tail");}}
  need(s.indexer.back()==(s.position>=4?4*((s.position-1)/4):0),"QSA composition: seed block position");
  need(s.position ? anchor_live(s.anchor_,layer_,s.position-1) : !anchored(s.anchor_),"QSA composition: seed provenance");
  if(!s.position){for(auto h:s.keys)need(h==0,"QSA composition: nonzero initial key");for(auto h:s.values)need(h==0,"QSA composition: nonzero initial value");for(auto x:s.indexer)need(x==0,"QSA composition: nonzero initial indexer");}
 }
public:
 // Requested owned payload bound. Includes committed and candidate full KV and
 // indexer images, attention's active KV copies, two publications, transient
 // mixed/gated/output reads and row metadata. Excludes component scratch,
 // callbacks, caller copies, allocator and driver internals. The fixed 1 MiB
 // allowance includes the retained anchor's immutable tuple payload and one
 // incoming anchor while publication is prepared. Other caller-held snapshots
 // and their storage are excluded.
 static size_t host_bound(uint32_t cells){need(cells>=1&&cells<=4096,"QSA composition: capacity");return ple_host_add(1u<<20,ple_host_add(ple_host_mul(size_t(cells),size_t(6)*512*sizeof(uint16_t)),ple_host_mul(size_t(qsa_idx_words(cells)),size_t(3)*sizeof(uint32_t))));}
 PortableNativeQsaComposition(int layer,uint32_t cells,size_t host_budget,
  std::unique_ptr<PortableNativeHcComposed>hc,std::unique_ptr<PortableNativeQsaQkvRows>qkv,std::unique_ptr<PortableNativeQsaIndexerRows>projections,
  std::unique_ptr<PortableNativeQsaNormRope>norm,std::unique_ptr<PortableNativeQsaIndexer>indexer,std::unique_ptr<PortableNativeQsaSelect>select,
  std::unique_ptr<PortableNativeQsaAttention>attention,std::unique_ptr<PortableNativeQsaWoRows>wo)
  :hc_(std::move(hc)),qkv_(std::move(qkv)),projections_(std::move(projections)),norm_(std::move(norm)),indexer_(std::move(indexer)),select_(std::move(select)),attention_(std::move(attention)),wo_(std::move(wo)),cells_(cells),layer_(layer){
  need(layer>=3&&layer<=47&&layer%4==3&&host_bound(cells)<=host_budget&&hc_&&qkv_&&projections_&&norm_&&indexer_&&select_&&attention_&&wo_,"QSA composition: layer, components or budget");
  need(indexer_->capacity()==cells&&select_->capacity()==cells&&attention_->capacity()==cells,"QSA composition: component capacities");
  keys_.assign(size_t(cells)*512,0);values_.assign(size_t(cells)*512,0);indexer_state_.assign(qsa_idx_words(cells),0);
 }
 PortableNativeQsaComposition(const PortableNativeQsaComposition&)=delete;PortableNativeQsaComposition&operator=(const PortableNativeQsaComposition&)=delete;
 ~PortableNativeQsaComposition(){if(published_)published_->valid=false;}
 int layer()const{return layer_;}uint64_t position()const{return position_;}uint64_t generation()const{return generation_;}bool poisoned()const{return poisoned_;}
 bool provenance_live()const{return !anchored(anchor_) || (position_&&anchor_live(anchor_,layer_,position_-1));}
 uint32_t capacity()const{return cells_;}
 bool available()const{return !running_&&!poisoned_&&generation_!=UINT64_MAX&&provenance_live();}
 Seed state()const{Seed s;s.position=position_;s.keys=keys_;s.values=values_;s.indexer=indexer_state_;s.anchor_=anchor_;return s;}
 void reset(){need(!running_&&generation_!=UINT64_MAX,"QSA composition: reset active/generation");Seed s;s.keys.resize(size_t(cells_)*512);s.values.resize(size_t(cells_)*512);s.indexer.resize(qsa_idx_words(cells_));reset(s);}
 // Nonzero seeds must come from state() and retain its live producer-generation
 // anchor; the caller must preserve the captured payload and matching prefix.
 // Mutated payloads cannot be authenticated. reset() starts an unbound sequence.
 void reset(const Seed&s){need(!running_&&generation_!=UINT64_MAX,"QSA composition: reset active/generation");validate_seed(s);auto k=s.keys,v=s.values;auto idx=s.indexer;auto provenance=s.anchor_;hc_->reset();if(published_)published_->valid=false;published_.reset();keys_.swap(k);values_.swap(v);indexer_state_.swap(idx);anchor_.swap(provenance);position_=s.position;++generation_;poisoned_=false;}
private:
 template<class Producer,class Snapshot>
 void run_impl(uint64_t first,const Producer&producer,const std::vector<std::shared_ptr<const Snapshot>>&snapshots,const Consumer&consumer,const Consumer&prepare){
  need(!running_&&!poisoned_&&bool(consumer)&&first==position_&&generation_!=UINT64_MAX&&(snapshots.size()==1||snapshots.size()==2)&&first<=cells_&&snapshots.size()<=cells_-first,"QSA composition: unavailable/position/rows");
  const size_t count=snapshots.size();
  // Refuse mixed histories before any HC or component dispatch. Hold exact
  // handles independently of a caller-mutated snapshot vector.
  std::vector<std::shared_ptr<const Snapshot>> held=snapshots;
  for(size_t i=0;i<count;++i)need(held[i]&&held[i]->valid_for(producer,layer_,first+i),"QSA composition: snapshot identity");
  need(!anchored(anchor_)||anchor_matches<Producer,Snapshot>(anchor_,producer,layer_,first-1),"QSA composition: predecessor generation differs from committed history");
  Anchor next_anchor=held.back();running_=true;struct Guard{bool&b;~Guard(){b=false;}}guard{running_};
  std::shared_ptr<Access> candidate;std::vector<uint16_t>next_keys,next_values;std::vector<uint32_t>next_indexer;
  try{
   hc_->run(layer_,first,producer,held,[&](const PortableNativeHcComposed::View&h,uint32_t n){
    need(n==count&&h.rows()==count,"QSA composition: HC rows");auto mixed=h.read(PortableNativeHcComposed::View::Field::mixed);auto residual=h.read(PortableNativeHcComposed::View::Field::residual);auto injection=h.read(PortableNativeHcComposed::View::Field::inject);
    qkv_->run(mixed,count);projections_->run(mixed,count);next_keys=keys_;next_values=values_;next_indexer=indexer_state_;
    Context ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"QSA composition: row metadata");std::vector<float>gated(count*6144);std::vector<int32_t>ids(count*2051);
    for(size_t r=0;r<count;++r){const auto pos=first+r;
     auto*q=row(ctx.get(),qkv_->output(0,count),r*12288,12288);auto*k=row(ctx.get(),qkv_->output(1,count),r*512,512);auto*v=row(ctx.get(),qkv_->output(2,count),r*512,512);
     auto*ik=row(ctx.get(),projections_->output(0,count),r*128,128);auto*iq=row(ctx.get(),projections_->output(1,count),r*512,512);
     norm_->run({q,k,iq},pos);copy_half(norm_->output[1],next_keys,pos*512);copy_half(v,next_values,pos*512);
     next_indexer=indexer_->run_before(ik,pos,next_indexer);auto chosen=select_->run(norm_->output[2],next_indexer,pos);
     std::copy(chosen.ids.begin(),chosen.ids.end(),ids.begin()+r*2051);
     std::vector<uint16_t>active_keys(next_keys.begin(),next_keys.begin()+(pos+1)*512),active_values(next_values.begin(),next_values.begin()+(pos+1)*512);
     auto result=attention_->run(norm_->output[0],q,active_keys,active_values,chosen.ids,pos);need(result.gated.size()==6144,"QSA composition: attention output");std::copy(result.gated.begin(),result.gated.end(),gated.begin()+r*6144);
    }
    wo_->run(gated,count);std::vector<float>out(count*2560);ggml_backend_tensor_get(wo_->output(count),out.data(),0,out.size()*4);
    candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{});candidate->first=first;candidate->generation=generation_;candidate->rows=count;candidate->output=std::move(out);candidate->residual=std::move(residual);candidate->injection=std::move(injection);candidate->ids=std::move(ids);
    consumer(View(candidate));
   },[&](const PortableNativeHcComposed::View&,uint32_t n){need(candidate&&candidate->valid&&n==count&&next_keys.size()==keys_.size()&&next_values.size()==values_.size()&&next_indexer.size()==indexer_state_.size(),"QSA composition: candidate incomplete");if(prepare)prepare(View(candidate));
    for(size_t i=0;i<count;++i)need(held[i]&&held[i]->valid_for(producer,layer_,first+i),"QSA composition: snapshot invalidated during consumption");
    need(!anchored(anchor_)||anchor_matches<Producer,Snapshot>(anchor_,producer,layer_,first-1),"QSA composition: predecessor generation invalidated");});
  }catch(...){if(candidate)candidate->valid=false;poisoned_=true;throw;}
  // HC has finished snapshot revalidation and published. All subsequent state
  // changes are no-throw swaps/assignments, with no user code in between.
  keys_.swap(next_keys);values_.swap(next_values);indexer_state_.swap(next_indexer);anchor_.swap(next_anchor);if(published_)published_->valid=false;published_=std::move(candidate);position_+=count;++generation_;
 }
public:
 void run(uint64_t first,const PortableGdnLayer& producer,const std::vector<std::shared_ptr<const PortableGdnSnapshot>>& snapshots,const Consumer& consumer,const Consumer& prepare={}){run_impl(first,producer,snapshots,consumer,prepare);}
 void run(uint64_t first,const PortableNativeFfnComposition& producer,const std::vector<std::shared_ptr<const PortableNativeFfnSnapshot>>& snapshots,const Consumer& consumer,const Consumer& prepare={}){run_impl(first,producer,snapshots,consumer,prepare);}
};
}
