#include "portable_gdn_fixture.hpp"
#include "ggml-backend.h"
#include "strata/core/layer0_observer.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>
using namespace strata::core;
using namespace gdn_fixture;
namespace {
struct Result {std::vector<float> residual,state,history;std::vector<uint8_t> mixed,inject;};
std::vector<uint8_t> field(const std::filesystem::path& path,uint32_t wanted) {
 std::ifstream f(path,std::ios::binary);f.seekg(72);uint32_t id,count;
 while(f.read(reinterpret_cast<char*>(&id),4)&&f.read(reinterpret_cast<char*>(&count),4)) {
  if(id==UINT32_MAX)break;
  std::vector<uint8_t> v(size_t(count)*4);check(bool(f.read(reinterpret_cast<char*>(v.data()),v.size())),"field read");if(id==wanted)return v;
 }
 throw std::runtime_error("observer field absent");
}
Result run(const Fixture& fixture,const std::filesystem::path& path,bool shared,bool observed,ggml_backend_t backend=nullptr,const std::vector<float>* supplied=nullptr) {
#ifndef _WIN32
 if(observed){setenv("STRATA_LAYER0_OBSERVER",path.c_str(),1);setenv("STRATA_LAYER0_BINDING",std::string(64,'a').c_str(),1);}
 else {unsetenv("STRATA_LAYER0_OBSERVER");unsetenv("STRATA_LAYER0_BINDING");}
 Layer0Observer observer;unsetenv("STRATA_LAYER0_OBSERVER");unsetenv("STRATA_LAYER0_BINDING");
 PortableGdnOptions options;options.execution.threads=1;options.execution.accelerator=backend;options.execution.request_accelerator=backend!=nullptr;options.execution.allow_cpu_fallback=false;std::string error;
 if(shared)options.scratch=PortableGdnScratch::create(options.execution,error);
 check(!shared||bool(options.scratch),error);
 // Establish a same-shaped cached allocation with no observer output lifetimes.
 auto other_views=fixture.views;for(auto& v:other_views)v.name.replace(0,6,"blk.4.");
 std::unique_ptr<PortableGdnLayer> other;
 if(shared){other=PortableGdnLayer::create(4,other_views,options,error);check(bool(other),error);}
 if(observed)options.execution.layer0_observer=&observer;
 auto layer=PortableGdnLayer::create(0,fixture.views,options,error);check(bool(layer),error);check(layer->reset(error),error);
 const auto input=supplied?*supplied:fixture.input(0);
 if(observed){observer.begin(5);observer.put(Layer0Observer::embedding,input.data(),input.size());observer.put(Layer0Observer::state_before,layer->recurrent_state().data(),786432);observer.put(Layer0Observer::history_before,layer->convolution_history().data(),30720);}
 PortableRoute route;check(layer->step(input,experts(),route,error),error);
 Result r{layer->last_residual(),layer->recurrent_state(),layer->convolution_history(),{}, {}};
 if(observed){std::array<int32_t,512> owns{};observer.put(Layer0Observer::primary_ownership,owns.data(),512);observer.put(Layer0Observer::helper_ownership,owns.data(),512);observer.finish();r.mixed=field(path,Layer0Observer::attention_mixed);r.inject=field(path,Layer0Observer::attention_inject);}
 return r;
#else
 return {};
#endif
}
}
int main(){try {
#ifndef _WIN32
 Fixture fixture;const auto root=std::filesystem::temp_directory_path()/("strata-l0gdn-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directory(root);
 const auto fresh=run(fixture,root/"private.bin",false,true);
 const auto reused=run(fixture,root/"shared.bin",true,true);
 check(fresh.inject==reused.inject,"observer injection must survive an ordinary cached GDN allocation plan");
 check(fresh.mixed==reused.mixed,"observer mixed must survive an ordinary cached GDN allocation plan");
 const auto disabled=run(fixture,root/"disabled.bin",true,false);
 check(!std::filesystem::exists(root/"disabled.bin"),"disabled observer created output");
 check(fresh.residual==reused.residual&&fresh.residual==disabled.residual,"observer changes numerical residual");
 check(fresh.state==reused.state&&fresh.state==disabled.state,"observer changes committed state");
 check(fresh.history==reused.history&&fresh.history==disabled.history,"observer changes history");
 std::filesystem::remove_all(root);
#endif
 puts("ordinary-plan to observed-plan liveness and enabled/disabled GDN outputs passed");return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
