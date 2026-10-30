#include <immintrin.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
static __m256 dot4(const int8_t* w,const int8_t* x) {
    const __m256i a=_mm256_loadu_si256((const __m256i*)w);
    const __m256i b=_mm256_loadu_si256((const __m256i*)x);
    const __m256i p=_mm256_maddubs_epi16(_mm256_sign_epi8(a,a),_mm256_sign_epi8(b,a));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p,_mm256_set1_epi16(1)));
}
int main() {
    alignas(32) int8_t w0[32]={11,-1},x0[32]={-93,4};
    alignas(32) int8_t w1[32]={127,9},x1[32]={8,1};
    const float h=1.f+1.f/1024.f;
    const __m256 a=_mm256_fmadd_ps(_mm256_set1_ps(1.f),dot4(w0,x0),_mm256_setzero_ps());
    const __m256 d=dot4(w1,x1),s=_mm256_set1_ps(h*h);
    const __m256 fused=_mm256_fmadd_ps(s,d,a);
    const __m256 separate=_mm256_add_ps(_mm256_mul_ps(s,d),a);
    uint32_t fb,sb; float f=_mm_cvtss_f32(_mm256_castps256_ps128(fused)),v=_mm_cvtss_f32(_mm256_castps256_ps128(separate));
    std::memcpy(&fb,&f,4);std::memcpy(&sb,&v,4);
    std::printf("AVX2 two-block Q8 lane: fused=%a bits=%08x separate=%a bits=%08x\n",f,fb,v,sb);
    return !(fb==0x3b401000 && sb==0x3b400000);
}
