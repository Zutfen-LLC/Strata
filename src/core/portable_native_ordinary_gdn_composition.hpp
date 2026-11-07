#pragma once
// Ordinary committed FFN -> HC -> GDN rows. Not model chunk admission.
// Borrowed components and producer must remain alive and unmodified during calls.
#include "portable_native_ffn_snapshot.hpp"
#include "portable_native_gdn_input_rows.hpp"
#include "portable_native_gdn_core_rows.hpp"
#include "portable_native_gdn_output_rows.hpp"
namespace strata::core::portable_detail {
class PortableNativeOrdinaryGdnComposition {
 struct Access {bool valid=true;uint64_t position=0,generation=0;size_t rows=0;std::vector<float>output,residual,injection;};
public:
 class View {
  std::weak_ptr<Access> access_;
  explicit View(const std::shared_ptr<Access>& a):access_(a){} friend class PortableNativeOrdinaryGdnComposition;
  std::shared_ptr<Access> check()const{auto a=access_.lock();if(!a||!a->valid)throw std::runtime_error("ordinary GDN: stale view");return a;}
 public:
  uint64_t position()const{return check()->position;}uint64_t generation()const{return check()->generation;}size_t rows()const{return check()->rows;}
  std::vector<float> output()const{return check()->output;}std::vector<float> residual()const{return check()->residual;}std::vector<float> injection()const{return check()->injection;}
 };
 using Consumer=std::function<void(const View&)>;
 struct Seed {
  uint64_t position=0;std::vector<float> state,history;
 private:
  std::shared_ptr<const PortableNativeFfnSnapshot> anchor_;
  friend class PortableNativeOrdinaryGdnComposition;
 };
private:
 std::unique_ptr<PortableNativeHcComposed> hc_;PortableNativeGdnInputRows& input_;std::unique_ptr<PortableNativeGdnCoreRows> core_;PortableNativeGdnOutputRows& output_;
 int layer_;uint64_t position_=0,generation_=0;bool running_=false,poisoned_=false;
 std::shared_ptr<Access> published_;std::shared_ptr<const PortableNativeFfnSnapshot> anchor_;
 static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
public:
 // Two publications, simultaneous HC/readback/candidate state and two retained
 // tuple payloads. Component buffers and caller-retained snapshots are excluded.
 static constexpr size_t host_bound(){return 2*(2*(2560+10240+4)*sizeof(float)+512)+2*(2560+6144)*sizeof(float)+(786432+30720)*sizeof(float)+(1u<<20);}
 PortableNativeOrdinaryGdnComposition(int layer,std::unique_ptr<PortableNativeHcComposed> hc,PortableNativeGdnInputRows& input,std::unique_ptr<PortableNativeGdnCoreRows> core,PortableNativeGdnOutputRows& output,size_t budget)
  :hc_(std::move(hc)),input_(input),core_(std::move(core)),output_(output),layer_(layer){
  need(layer>=2&&layer<=47&&layer%4!=3&&hc_&&core_&&!core_->candidate()&&!core_->poisoned()&&budget>=host_bound(),"ordinary GDN: admission");
  // An unanchored sequence starts at zero even if the transferred core was
  // previously seeded. Nonzero prefixes require a provenance-bearing Seed.
  core_->reset();
 }
 PortableNativeOrdinaryGdnComposition(const PortableNativeOrdinaryGdnComposition&)=delete;
 PortableNativeOrdinaryGdnComposition& operator=(const PortableNativeOrdinaryGdnComposition&)=delete;
 ~PortableNativeOrdinaryGdnComposition(){if(published_)published_->valid=false;}
 int layer()const{return layer_;}uint64_t position()const{return position_;}uint64_t generation()const{return generation_;}bool poisoned()const{return poisoned_;}
 bool provenance_live()const{return !anchor_||(position_&&anchor_->live()&&anchor_->layer()+1==layer_&&anchor_->position()==position_-1);}
 bool available()const{return !running_&&!poisoned_&&generation_!=UINT64_MAX&&!core_->candidate()&&!core_->poisoned()&&provenance_live();}
 PortableGdnAbMethod ab_method()const{return core_->ab_method();}
 std::vector<float> committed_state()const{return core_->committed_state();}
 std::vector<float> committed_history()const{return core_->committed_history();}
 Seed state()const{Seed s;s.position=position_;s.state=core_->committed_state();s.history=core_->committed_history();s.anchor_=anchor_;return s;}
 void reset(){Seed s;s.state.assign(786432,0.f);s.history.assign(30720,0.f);reset(s);}
 void reset(const Seed& s){
  need(!running_&&generation_!=UINT64_MAX,"ordinary GDN: reset active/generation");PortableNativeGdnCoreRows::validate_seed(s.state,s.history);
  need(s.position?(s.anchor_&&s.anchor_->live()&&s.anchor_->layer()+1==layer_&&s.anchor_->position()==s.position-1):!s.anchor_,"ordinary GDN: seed provenance");
  if(!s.position){for(float x:s.state)need(x==0.f,"ordinary GDN: initial state");for(float x:s.history)need(x==0.f,"ordinary GDN: initial history");}
  auto anchor=s.anchor_;poisoned_=true;if(published_)published_->valid=false;published_.reset();hc_->reset();core_->reset(s.state,s.history);anchor_.swap(anchor);position_=s.position;++generation_;poisoned_=false;
 }
 void run(uint64_t first,const PortableNativeFfnComposition& producer,const std::vector<std::shared_ptr<const PortableNativeFfnSnapshot>>& snapshots,const Consumer& consumer,const Consumer& prepare={}){
  need(!running_&&!poisoned_&&!core_->candidate()&&!core_->poisoned()&&consumer&&(snapshots.size()==1||snapshots.size()==2)&&first==position_&&first<=UINT64_MAX-snapshots.size()&&generation_!=UINT64_MAX,"ordinary GDN: unavailable/position/rows");
  auto held=snapshots;const size_t count=held.size();
  for(size_t i=0;i<count;++i)need(held[i]&&held[i]->valid_for(producer,layer_,first+i),"ordinary GDN: snapshot identity");
  need(!anchor_||anchor_->valid_for(producer,layer_,first-1),"ordinary GDN: predecessor generation");
  auto next_anchor=held.back();running_=true;struct Guard{bool& b;~Guard(){b=false;}}guard{running_};std::shared_ptr<Access> candidate;
  try{
   hc_->run(layer_,first,producer,held,[&](const PortableNativeHcComposed::View& h,uint32_t n){
    need(n==count&&h.rows()==count,"ordinary GDN: HC rows");auto mixed=h.read(PortableNativeHcComposed::View::Field::mixed);auto residual=h.read(PortableNativeHcComposed::View::Field::residual);auto inject=h.read(PortableNativeHcComposed::View::Field::inject);
    input_.run(mixed,count);core_->stage(mixed,input_.output(0,count),input_.output(1,count),uint32_t(count));auto y=core_->output();output_.run(y,count);
    std::vector<float> result(count*2560);ggml_backend_tensor_get(output_.output(count),result.data(),0,result.size()*sizeof(float));
    candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{});candidate->position=first;candidate->generation=generation_;candidate->rows=count;candidate->output=std::move(result);candidate->residual=std::move(residual);candidate->injection=std::move(inject);consumer(View(candidate));
   },[&](const PortableNativeHcComposed::View&,uint32_t n){
    need(candidate&&candidate->valid&&core_->candidate()&&n==count,"ordinary GDN: missing candidate");if(prepare)prepare(View(candidate));
    for(size_t i=0;i<count;++i)need(held[i]&&held[i]->valid_for(producer,layer_,first+i),"ordinary GDN: source invalidated");
    need(!anchor_||anchor_->valid_for(producer,layer_,first-1),"ordinary GDN: predecessor invalidated");
   });
  }catch(...){if(candidate)candidate->valid=false;core_->invalidate();poisoned_=true;throw;}
  core_->publish_prepared();anchor_.swap(next_anchor);if(published_)published_->valid=false;published_=std::move(candidate);position_+=count;++generation_;
 }
};
}
