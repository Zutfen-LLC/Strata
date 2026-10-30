#pragma once
#include "strata/core/layer0_observer.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
namespace strata::core {
// Allocated before native expert sizing, so cudaMemGetInfo prices these bytes.
// Copies are graph nodes; host reads occur only after run's existing completion.
class NativeLayer0Observer {
    Layer0Observer writer_;
    float* device_=nullptr;
    int device_id_=0;
    std::array<size_t,18> offsets_{};
    std::array<bool,18> available_{};
public:
    std::array<int32_t,512> helper{}; // actual helper membership, filled after cache admission
    NativeLayer0Observer() {
        size_t at=0;
        for(size_t i=0;i<offsets_.size();++i){offsets_[i]=at;at+=Layer0Observer::counts[i];}
        if(at>Layer0Observer::limit/4)throw std::runtime_error("layer0 observer: device layout exceeds bound");
        cudaGetDevice(&device_id_);
        size_t free=0,total=0;
        if(device_id_!=0 || cudaMemGetInfo(&free,&total)!=cudaSuccess ||
           free<(512u<<20)+Layer0Observer::limit ||
           cudaMalloc(reinterpret_cast<void**>(&device_),Layer0Observer::limit)!=cudaSuccess)
            throw std::runtime_error("layer0 observer: primary snapshot admission failed; reserve unchanged");
    }
    ~NativeLayer0Observer(){if(device_){int previous=0;cudaGetDevice(&previous);cudaSetDevice(device_id_);cudaFree(device_);cudaSetDevice(previous);}}
    bool finished() const{return writer_.finished();}
    bool snapshot(Layer0Observer::Field field,const void* source,cudaStream_t stream,std::string& error) {
        const size_t id=size_t(field);
        if(id>=offsets_.size() || !source || cudaMemcpyAsync(device_+offsets_[id],source,
            Layer0Observer::counts[id]*4,cudaMemcpyDeviceToDevice,stream)!=cudaSuccess) {
            error="layer0 observer: graph snapshot failed";return false;
        }
        available_[id]=true;return true;
    }
    // Caller has already synchronized the successful original window. No new wait.
    bool flush(int32_t token,const int32_t* primary,std::string& error) {
        if(finished())return true;
        try {
            writer_.begin(token);
            writer_.put(Layer0Observer::helper_ownership,helper.data(),helper.size());
            if(primary){std::array<int32_t,512> owns{};for(size_t i=0;i<owns.size();++i)owns[i]=primary[i]>=0?1:0;
                writer_.put(Layer0Observer::primary_ownership,owns.data(),owns.size());}
            for(size_t i=0;i<available_.size();++i)if(available_[i]) {
                std::vector<float> host(Layer0Observer::counts[i]);
                if(cudaMemcpy(host.data(),device_+offsets_[i],host.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess)
                    throw std::runtime_error("layer0 observer: completed snapshot download failed");
                if(i==Layer0Observer::state_before || i==Layer0Observer::state_after) {
                    std::vector<uint32_t> canonical(host.size());
                    layer0_native_state_view(host.data(),canonical.data(),host.size());
                    writer_.put(Layer0Observer::Field(i),canonical.data(),canonical.size());
                } else writer_.put(Layer0Observer::Field(i),host.data(),host.size());
            }
            writer_.finish();return true;
        } catch(const std::exception& e){error=e.what();return false;}
    }
};
}
