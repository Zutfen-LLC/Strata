#include "../../src/core/portable_device_budget.hpp"
#include <iostream>
#include <limits>
using namespace strata::core::portable_detail;
void check(bool value) { if (!value) throw std::runtime_error("device budget assertion failed"); }
template<class F> void rejects(F f) {
    bool refused = false;
    try { f(); } catch (const std::exception&) { refused = true; }
    check(refused);
}
int main() {
    try {
        constexpr size_t M = 1u << 20;
        // A ceiling is not an allocation. Baseline 9 MiB must not reject a
        // 7550 MiB ledger at a 7680 MiB ceiling on an 8 GiB card.
        auto small = device_admission_limit(7680*M, 8183*M, 8192*M, 512*M);
        check(small == 7671*M && 7550*M <= small);
        auto large = device_admission_limit(11776*M, 12279*M, 12288*M, 512*M);
        check(large == 11767*M);
        check(device_admission_limit(7000*M, 8183*M, 8192*M, 512*M) == 7000*M);
        rejects([&]{device_admission_limit(11776*M,8183*M,8192*M,512*M);});
        rejects([&]{device_admission_limit(7680*M,8193*M,8192*M,512*M);});
        rejects([&]{device_admission_limit(7680*M,512*M,8192*M,512*M);});
        rejects([&]{device_admission_limit(7680*M,0,0,512*M);});
        rejects([&]{device_mib(0);});
        rejects([&]{device_mib(-1);});
        rejects([&]{device_mib(std::numeric_limits<int64_t>::max());});
        check(device_mib(11776)==11776*M);
        check(device_admission_limit(SIZE_MAX,SIZE_MAX,SIZE_MAX,1)==SIZE_MAX-1);
        std::cout << "portable_device_budget_test passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
