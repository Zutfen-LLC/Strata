#include "portable_gdn_projection_probe.hpp"
using namespace strata::core;
int main(){
    try{
        portable_detail::Backend selected(ggml_backend_cpu_init(),ggml_backend_free);
        portable_detail::require(bool(selected),"selected CPU required");
        portable_detail::require(gdn_projection_probe::run(selected.get(),false),"CPU projection diagnostic baseline differs");
        portable_detail::require(gdn_projection_probe::run(selected.get(),true),"CPU projection F32 control differs");
        portable_detail::require(gdn_projection_probe::run(selected.get(),false,true),"CPU projection strided F32 control differs");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
