// Compile as an ordinary caller, without the low-level fixture accessor.
#include "portable_native_ffn_composition.hpp"
#include <type_traits>
using namespace strata::core::portable_detail;

template<class T, class = void> struct exposes_unchecked_tag : std::false_type {};
template<class T> struct exposes_unchecked_tag<T, std::void_t<typename T::TestOnlyHc>> : std::true_type {};
using Owner = PortableNativeFfnComposition;
static_assert(!exposes_unchecked_tag<Owner>::value, "unchecked HC tag must not be public");
static_assert(!std::is_constructible_v<Owner, PortableNativePleGdnComposition&,
              std::unique_ptr<PortableNativeHcComposed>, PortableNativeFfnRows&, size_t>);
static_assert(!std::is_constructible_v<Owner, PortableNativeQsaComposition&,
              std::unique_ptr<PortableNativeHcComposed>, PortableNativeFfnRows&, size_t>);
static_assert(std::is_constructible_v<Owner, PortableNativeQsaComposition&, ggml_backend_t,
              const std::vector<strata::core::PortableTensorView>&, PortableNativeFfnRows&,
              size_t, size_t, const char*>);
static_assert(std::is_constructible_v<Owner, PortableNativePleGdnComposition&, ggml_backend_t,
              const strata::GgufModel&, PortableNativeFfnRows&, size_t, size_t, const char*>);
int main() { return 0; }
