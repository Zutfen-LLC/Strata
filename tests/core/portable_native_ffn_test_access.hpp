#pragma once
// Low-level fixture access; production code uses the named-role constructors.
#include "portable_native_ffn_composition.hpp"
namespace strata::core::portable_detail {
struct PortableNativeFfnCompositionTestAccess {
    template<class Source>
    static std::unique_ptr<PortableNativeFfnComposition> make_unique(Source& source, std::unique_ptr<PortableNativeHcComposed> hc,
                                                                      PortableNativeFfnRows& rows, size_t host_budget) {
        return std::unique_ptr<PortableNativeFfnComposition>(new PortableNativeFfnComposition(PortableNativeFfnComposition::TestOnlyHc{}, source, std::move(hc), rows, host_budget));
    }
    template<class Source>
    static PortableNativeFfnComposition make(Source& source, std::unique_ptr<PortableNativeHcComposed> hc,
                                             PortableNativeFfnRows& rows, size_t host_budget) {
        return PortableNativeFfnComposition(PortableNativeFfnComposition::TestOnlyHc{}, source,
                                            std::move(hc), rows, host_budget);
    }
};
}
