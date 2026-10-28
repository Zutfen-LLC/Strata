#include "../../src/core/portable_stdio.hpp"
#include <fstream>
#include <iostream>
using namespace strata::core::stdio_detail;
int main() {
    auto valid = parse("GEN 3 0,12,99", 3, 100);
    if (valid.max_new != 3 || valid.tokens != std::vector<int32_t>({0, 12, 99}))
        return 1;
    for (const auto &bad : std::vector<std::string>{"", "QUIT", "GEN 4 0", "GEN 1 ", "GEN -1 0", "GEN 0 100",
                                                    "GEN 0 1,", "GEN 0 ,1", "GEN 0 1 2", "GEN 0 1,2,3,4", "GEN 0 1\r",
                                                    "GEN 0 99999999999999999999999", std::string(4097, '0')}) {
        try {
            parse(bad, 3, 100);
            return 2;
        } catch (const std::runtime_error &) {
        }
    }
    const auto path = std::filesystem::temp_directory_path() / "portable-stdio-source-test";
    {
        std::ofstream f(path);
        f << "one";
    }
    Sources source({path.string()});
    source.check();
    {
        std::ofstream f(path);
        f << "different";
    }
    try {
        source.check();
        return 3;
    } catch (const std::runtime_error &) {
    }
    std::filesystem::remove(path);
    std::cout << "bounded preflight and immutable source guard passed\n";
}
