// Test-only LLVM libc non-FMA/RNE support adaptation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfenv>
#include <cerrno>
#include <limits>
#include <type_traits>
#define LIBC_NAMESPACE_DECL strata_llvm_upstream
#define LIBC_INLINE inline
#define LIBC_INLINE_VAR inline
#define LIBC_CONSTEXPR
#define LIBC_UNLIKELY(x) (x)
#define LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY 1
namespace strata_llvm_upstream {
enum class Sign {NEG};
namespace fputil {
template<class T>struct FPBits {
 using U=std::conditional_t<sizeof(T)==8,uint64_t,uint32_t>;U u;
 static constexpr int FRACTION_LEN=sizeof(T)==8?52:23,EXP_BIAS=sizeof(T)==8?1023:127;
 static constexpr U FRACTION_MASK=(U(1)<<FRACTION_LEN)-1;
 explicit FPBits(T x){std::memcpy(&u,&x,sizeof x);}explicit FPBits(U b):u(b){}
 U uintval()const{return u;}T get_val()const{T x;std::memcpy(&x,&u,sizeof x);return x;}
 static FPBits one(){return FPBits(T(1));}static FPBits inf(Sign){return FPBits(-std::numeric_limits<T>::infinity());}
 static FPBits quiet_nan(){return FPBits(std::numeric_limits<T>::quiet_NaN());}
 bool is_signaling_nan()const{return false;}
};
template<class T>inline T multiply_add(T x,T y,T z){volatile T prod=x*y;return prod+z;}
template<class T>inline T round_result_slightly_down(T x){return x;}
template<class T>inline T round_result_slightly_up(T x){return x;}
inline void set_errno_if_required(int){}inline void raise_except_if_required(int){}
}}
