#pragma once
#include <cstdlib>
#include <cstring>
using cudaStream_t=void*;
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaMemcpyDeviceToDevice=1,cudaMemcpyDeviceToHost=2;
inline size_t observer_mock_free=2ull<<30;
inline int observer_mock_allocations=0,observer_mock_async=0,observer_mock_downloads=0;
inline bool observer_mock_fail_copy=false;
inline int cudaGetDevice(int* p){*p=0;return 0;}
inline int cudaSetDevice(int){return 0;}
inline int cudaMemGetInfo(size_t* f,size_t* t){*f=observer_mock_free;*t=2ull<<30;return 0;}
inline int cudaMalloc(void** p,size_t n){*p=std::malloc(n);if(*p)++observer_mock_allocations;return *p?0:1;}
inline int cudaFree(void* p){if(p)--observer_mock_allocations;std::free(p);return 0;}
inline int cudaMemcpyAsync(void* d,const void* s,size_t n,int kind,cudaStream_t){
    if(observer_mock_fail_copy)return 1;
    if(kind!=cudaMemcpyDeviceToDevice)return 1;
    ++observer_mock_async;std::memcpy(d,s,n);return 0;
}
inline int cudaMemcpy(void* d,const void* s,size_t n,int kind){
    if(kind!=cudaMemcpyDeviceToHost)return 1;
    ++observer_mock_downloads;std::memcpy(d,s,n);return 0;
}
