#include "native_q8_wo_fixture.hpp"
#include "strata_vk_q8_wo.hpp"
#include "ggml-vulkan.h"
#include "ggml-alloc.h"
#include <memory>
using namespace native_q8_wo_fixture;
using namespace strata::core::portable_detail;
int main(int argc,char**argv){try{
 if(argc==6&&std::string(argv[1])=="--contract"){auto f=read_fixture(argv[2],argv[3],std::stoull(argv[4]));contract(f,argv[5]);return 0;}
 need(argc==15&&std::string(argv[1])=="--device"&&std::string(argv[3])=="--expect-device"&&std::string(argv[5])=="--weights"&&std::string(argv[7])=="--input"&&std::string(argv[9])=="--rows"&&std::string(argv[11])=="--budget"&&std::string(argv[13])=="--output","WO: invalid Vulkan arguments");
 auto f=read_fixture(argv[6],argv[8],std::stoull(argv[10]));size_t budget=std::stoull(argv[12]);std::string idx=argv[2];need(!idx.empty()&&idx.find_first_not_of("0123456789")==std::string::npos,"WO: invalid device");size_t device=std::stoull(idx);
 need(device<size_t(ggml_backend_vk_get_device_count()),"WO: missing device");
 std::filesystem::path dir=argv[14];need(std::filesystem::create_directory(dir),"WO: output exists");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_vk_init(device),ggml_backend_free);
 need(bool(backend)&&std::string(argv[4]).size()&&std::string(ggml_backend_dev_description(ggml_backend_get_device(backend.get()))).find(argv[4])!=std::string::npos,"WO: unexpected device");
 std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({16384,nullptr,true}),ggml_free);need(bool(ctx),"WO: metadata");
 auto make=[&](ggml_type type,size_t n){auto* b=ggml_new_tensor_1d(ctx.get(),type,n+16);return std::pair{b,ggml_view_1d(ctx.get(),b,n,32)};};
 auto [wb,w]=make(GGML_TYPE_I32,f.weights.size()/4);auto [xb,x]=make(GGML_TYPE_I32,f.input.size()/4);auto [yb,y]=make(GGML_TYPE_F32,f.rows);
 auto*shortx=ggml_view_1d(ctx.get(),xb,f.input.size()/4-1,32);auto*badw=ggml_view_1d(ctx.get(),wb,f.weights.size()/4,32);badw->type=GGML_TYPE_F32;
 auto*alias=ggml_view_1d(ctx.get(),wb,f.rows,32);alias->type=GGML_TYPE_F32;auto*aliasx=ggml_view_1d(ctx.get(),wb,f.input.size()/4,32);
 size_t bound=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),ggml_backend_get_default_buffer_type(backend.get()));need(bound<=budget,"WO: aligned tensor budget");
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);need(bool(buffer),"WO: tensor allocation");
 size_t allocated=ggml_backend_buffer_get_size(buffer.get());need(allocated<=budget,"WO: actual tensor budget");
 std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)>stage(strata_vk_q8_wo_create(backend.get(),(dir/"effective.spv").c_str(),budget),strata_vk_q8_wo_destroy);
 need(bool(stage),"WO: candidate owner");
 need(strata_vk_q8_wo_bytes(stage.get())<=budget-allocated,"WO: owner and tensor budget");
 std::unique_ptr<StrataVkQ8Wo,decltype(&strata_vk_q8_wo_destroy)>low(strata_vk_q8_wo_create(backend.get(),(dir/"low-budget.spv").c_str(),strata_vk_q8_wo_bytes(stage.get())),strata_vk_q8_wo_destroy);
 need(strata_vk_q8_wo_bytes(low.get())<=budget-allocated-strata_vk_q8_wo_bytes(stage.get()),"WO: test owner budget");
 std::vector<uint32_t>iw(f.weights.size()/4+16,0xa5a5a5a5),ix(f.input.size()/4+16,0xa5a5a5a5),iy(f.rows+16,0xa5a5a5a5);
 std::memcpy(iw.data()+8,f.weights.data(),f.weights.size());std::memcpy(ix.data()+8,f.input.data(),f.input.size());
 auto reset=[&](){ggml_backend_tensor_set(wb,iw.data(),0,iw.size()*4);ggml_backend_tensor_set(xb,ix.data(),0,ix.size()*4);ggml_backend_tensor_set(yb,iy.data(),0,iy.size()*4);};
 auto contents=[&](ggml_tensor*t,size_t n){std::vector<uint32_t>v(n);ggml_backend_tensor_get(t,v.data(),0,n*4);return v;};
 auto unchanged=[&](){return contents(wb,iw.size())==iw&&contents(xb,ix.size())==ix&&contents(yb,iy.size())==iy;};
 reset();for(int which=0;which<5;++which){bool refused=false;try{
  strata_vk_q8_wo_run(which==3?low.get():stage.get(),which==1?badw:w,which==0?shortx:which==4?aliasx:x,which==2?alias:y,f.rows);
 }catch(...){refused=true;}need(refused&&unchanged(),"WO: invalid call committed");}
 low.reset();
 std::vector<uint32_t> first;for(int pass=0;pass<2;++pass){reset();strata_vk_q8_wo_run(stage.get(),w,x,y,f.rows);auto rw=contents(wb,iw.size()),rx=contents(xb,ix.size()),ry=contents(yb,iy.size());
  need(rw==iw&&rx==ix,"WO: input mutated");for(size_t i=0;i<8;++i)need(ry[i]==0xa5a5a5a5&&ry[ry.size()-1-i]==0xa5a5a5a5,"WO: output canary changed");
  std::vector<uint32_t>out(ry.begin()+8,ry.begin()+8+f.rows);std::vector<float>values(f.rows);std::memcpy(values.data(),out.data(),f.rows*4);output_literals(values);if(pass)need(out==first,"WO: repeat differs");first=out;
  Bytes data(f.rows*4);std::memcpy(data.data(),out.data(),data.size());save(dir/(pass?"out-repeat.f32le":"out.f32le"),data);
 }
 std::ofstream(dir/"report.json")<<"{\"actual_vulkan_wo_candidate\":true,\"immutable_canary_retry\":true,\"tensor_buffer_bytes\":"<<allocated<<",\"owner_bytes\":"<<strata_vk_q8_wo_bytes(stage.get())<<",\"qualified\":false}\n";return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
