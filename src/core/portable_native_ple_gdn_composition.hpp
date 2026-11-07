#pragma once
// Standalone materialized PLE -> HC -> GDN rows; no model chunk admission.
// PLE and projection owners are borrowed; the core owner is transferred.
// Their borrowed backends, roles and storage must outlive this composition.
// Borrowed dependencies must not be operated on independently while attached,
// or concurrently/reentrantly with a composition operation.
#include "portable_native_ple_composition.hpp"
#include "portable_native_gdn_input_rows.hpp"
#include "portable_native_gdn_core_rows.hpp"
#include "portable_native_gdn_output_rows.hpp"
namespace strata::core::portable_detail {
class PortableNativePleGdnComposition {
 struct Access {bool valid=true;uint64_t position=0,generation=0;std::vector<int32_t>tokens;std::vector<float>output,residual,injection;};
public:
 using Handle=PortableNativePleComposition::Handle;
 class View {
  std::weak_ptr<Access> access_;
  explicit View(const std::shared_ptr<Access>&a):access_(a){}friend class PortableNativePleGdnComposition;
  std::shared_ptr<Access> check()const{auto a=access_.lock();if(!a||!a->valid)throw std::runtime_error("PLE GDN: stale view");return a;}
 public:
  uint64_t position()const{return check()->position;}uint64_t generation()const{return check()->generation;}
  std::vector<int32_t> tokens()const{return check()->tokens;}
  std::vector<float> output()const{return check()->output;}
  std::vector<float> residual()const{return check()->residual;}
  std::vector<float> injection()const{return check()->injection;}
 };
 using Consumer=std::function<void(const View&)>;
private:
 PortableNativePleComposition&ple_;PortableNativeGdnInputRows&input_;std::unique_ptr<PortableNativeGdnCoreRows> core_;PortableNativeGdnOutputRows&output_;
 std::shared_ptr<Access> published_;uint64_t position_,generation_=0;bool running_=false,poisoned_=false;size_t host_budget_;
 static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
public:
 // Two simultaneous owner-held publications: output/residual/injection and
 // tokens. Covers requested payload plus two bounded shared control blocks.
 // Also includes live HC mixed and recurrence output readbacks, plus small
 // token copies. Caller copies, component scratch, roles, allocator overhead
 // and backend/driver internals are excluded; not a process bound.
 static constexpr size_t host_bound(){return 2*(2*(2560+10240+4)*sizeof(float)+2*sizeof(int32_t)+512)+2*(2560+6144)*sizeof(float)+4*sizeof(int32_t)+(786432+30720)*sizeof(float);}
 PortableNativePleGdnComposition(PortableNativePleComposition&ple,PortableNativeGdnInputRows&input,std::unique_ptr<PortableNativeGdnCoreRows> core,PortableNativeGdnOutputRows&output,size_t host_budget)
  :ple_(ple),input_(input),core_(std::move(core)),output_(output),position_(ple.position()),host_budget_(host_budget){need(host_bound()<=host_budget_&&core_&&!core_->candidate()&&!core_->poisoned(),"PLE GDN: admission/core state");}
 PortableNativePleGdnComposition(const PortableNativePleGdnComposition&)=delete;PortableNativePleGdnComposition&operator=(const PortableNativePleGdnComposition&)=delete;
 ~PortableNativePleGdnComposition(){if(published_)published_->valid=false;}
 uint64_t position()const{return position_;}bool poisoned()const{return poisoned_;}
 std::vector<float> committed_state()const{return core_->committed_state();}
 std::vector<float> committed_history()const{return core_->committed_history();}
 // Optional preparation may perform downstream work that can still fail.
 // Neither callback may mutate the borrowed dependencies.
 void run(const Handle&handle,const Consumer&consumer,const Consumer&prepare={}){
  need(!running_&&!poisoned_&&bool(consumer)&&!core_->candidate()&&!core_->poisoned(),"PLE GDN: unavailable");
  const auto first=handle.position(),gen=handle.generation();auto tokens=handle.tokens();const size_t count=tokens.size();
  need((count==1||count==2)&&first==position_&&ple_.position()==position_&&first<=UINT64_MAX-count&&generation_!=UINT64_MAX,"PLE GDN: position/rows/generation");
  running_=true;struct Guard{bool&x;~Guard(){x=false;}} guard{running_};std::shared_ptr<Access> candidate;
  try{
   ple_.commit_to_hc(handle,[&](const PortableNativeHcComposed::View&hc,uint32_t n){
    need(n==count&&hc.rows()==count&&handle.position()==first&&handle.generation()==gen&&handle.tokens()==tokens,"PLE GDN: boundary identity");
    auto mixed=hc.read(PortableNativeHcComposed::View::Field::mixed);auto residual=hc.read(PortableNativeHcComposed::View::Field::residual);auto inject=hc.read(PortableNativeHcComposed::View::Field::inject);
    input_.run(mixed,count);core_->stage(mixed,input_.output(0,count),input_.output(1,count),uint32_t(count));auto y=core_->output();output_.run(y,count);
    auto result=std::vector<float>(count*2560);ggml_backend_tensor_get(output_.output(count),result.data(),0,result.size()*4);
    candidate=std::allocate_shared<Access>(PleOwnerAllocator<Access>{});candidate->position=first;candidate->generation=generation_;candidate->tokens=tokens;candidate->output=std::move(result);candidate->residual=std::move(residual);candidate->injection=std::move(inject);
    need(candidate->tokens.capacity()==count&&candidate->output.capacity()==count*2560&&candidate->residual.capacity()==count*10240&&candidate->injection.capacity()==count*4,"PLE GDN: candidate capacity");
    consumer(View(candidate));
   },[&](const PortableNativeHcComposed::View&,uint32_t n){
    need(candidate&&candidate->valid&&core_->candidate()&&n==count,"PLE GDN: missing candidate");
    if(prepare)prepare(View(candidate));
    need(candidate->valid&&core_->candidate()&&handle.position()==first&&handle.generation()==gen&&handle.tokens()==tokens&&ple_.position()==first,"PLE GDN: publication preparation");
   });
  }catch(...){if(candidate)candidate->valid=false;core_->invalidate();poisoned_=true;throw;}
  // HC has performed the last throwing PLE commit and published its bank.
  // All following actions are non-allocating and no-throw. No user code runs
  // between the producer commit and the coordinated GDN bank swap.
  core_->publish_prepared();if(published_)published_->valid=false;published_=std::move(candidate);position_+=count;++generation_;
 }
 void reset(std::array<int32_t,2>previous={-1,-1},uint64_t position=0){
  need(!running_&&generation_!=UINT64_MAX,"PLE GDN: reset active/generation");
  std::vector<float>state(786432,0.f),history(30720,0.f);reset(previous,position,state,history);
 }
 void reset(std::array<int32_t,2>previous,uint64_t position,const std::vector<float>&state,const std::vector<float>&history){
  need(!running_&&generation_!=UINT64_MAX,"PLE GDN: reset active/generation");PortableNativeGdnCoreRows::validate_seed(state,history);
  poisoned_=true;if(published_)published_->valid=false;published_.reset();ple_.reset(previous,position);core_->reset(state,history);position_=position;++generation_;poisoned_=false;
 }
};
}
